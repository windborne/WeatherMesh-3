import torch
import time 
import numpy as np 
from utils import *
import copy
import itertools
from meshes import *
from collections import defaultdict
from pandas import date_range
from functools import reduce

class TrainingSample():
    def __init__(self):
        # Format for inputs / outputs is 
        #    a list (# encoders / decoders) 
        #       of two lists (mesh_id, tensors) 
        #           of lists (# of meshes):
        # [
        #   [
        #     [mesh_ids, ...], [tensors, ...]
        #   ],
        #   [
        #     [mesh_ids, ...], [tensors, ...]
        #   ],
        #   ...
        # ]
        self.inputs = []
        self.outputs = []
        self.additional_inputs = defaultdict(list)
        # List of unix timesteps (order matched)
        # First timestep is the input timestep (t0), the rest are output timesteps
        self.timestamps = [] 
    
    def update(self):
        assert len(self.inputs) + len(self.outputs) == len(self.timestamps)
        self.t0 = max(self.timestamps[:len(self.inputs)])
        self.dts = [int((t-self.t0)/3600) for t in self.timestamps[1:]]

    def get_x_t0(self, encoders=None):
        if encoders is not None:
            output_x_t0 = []
            string_ids = self.inputs[0][0]
            for encoder in encoders:
                assert encoder.mesh.string_id in string_ids, f"Encoder {encoder.__class__.__name__}'s mesh ({encoder.mesh.string_id}) not found in sample (input string_ids: {string_ids})"
                
                mesh_idx = string_ids.index(encoder.mesh.string_id)
                output_x_t0.append(self.inputs[0][1][mesh_idx])
            
            # For the moment add timestep to the end (not sure if this is actually necessary, but its convention rn)
            output_x_t0.append(torch.tensor([self.timestamps[0]]))
            return output_x_t0
        
        # Basic use of get_x_t0 (where we assume there is only one encoder, so we can just match it easily)
        else:
            return self.inputs[0][1] + [torch.tensor([self.timestamps[0]])]

    # Doesn't have a basic usage yet (on purpose, basic usage may want to be its own function)
    # Outputs data in format:
    # [
    #    [decoder1_tensor_at_timestamp1, decoder2_tensor_at_timestamp1, ..., timestamp1],
    #    [decoder1_tensor_at_timestamp2, decoder2_tensor_at_timestamp2, ..., timestamp2],
    #    ...
    # ]
    def get_y_ts(self, decoders):
        output_y_ts = []
        for index_timestamp, timestamp in enumerate(self.timestamps[1:]):
            output_y_ts.append([])
            string_ids = self.outputs[index_timestamp][0]
            for decoder in decoders:
                assert decoder.mesh.string_id in string_ids, f"Decoder {decoder.__class__.__name__}'s mesh ({decoder.mesh.string_id}) not found in sample (output string_ids: {string_ids})"

                mesh_idx = string_ids.index(decoder.mesh.string_id)
                out = self.outputs[index_timestamp][1][mesh_idx]
                output_y_ts[-1].append(out)
            output_y_ts[-1].append(torch.tensor([timestamp]))
        return output_y_ts

    def get_additional_inputs(self):
        return self.additional_inputs

class DataConfig():
    def __init__(self,inputs=[],outputs=[],conf_to_copy=None,**kwargs):
        # Main Config
        self.inputs = inputs 
        self.outputs = outputs
        self.timesteps = [24]

        self.random_timestep_subset = False # Tracks the number of random timesteps we are sampling from (basically, don't touch)
        
        # can also pass single datetime in a list
        self.requested_train_dates = [date_range('1970','1980',freq='D'),date_range('1970','1980',freq='D')]
        self.requested_val_dates = None
        self.validate_every = -1
        self.only_at_z = None
        self.simulate_realtime = False # This only works for DA datasets right now (ObsDatasets)
        
        # Realtime + ens_nums
        self.realtime = False
        self.ens_nums = None
        
        # Largely unused variables?
        self.use_rh = False
        self.clamp = 13
        self.clamp_output=None

        if conf_to_copy is not None:
            self.__dict__.update(copy.deepcopy(conf_to_copy.__dict__))
            
        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a DataConfig attribute"
            setattr(self,k,v)
            
        for dataset in (self.inputs + self.outputs):
            if dataset.mesh.shape() == -1: print(ORANGE("WARNING: Mesh shape is -1, shape will not be checked (specify mesh shape in meshes.py for your specific mesh type)"))
        self.update()

    def update(self):
        if len(self.outputs) == 0:
            self.outputs = self.inputs
        self.proc_start_t = time.time()

        if self.clamp_output is None:
            self.clamp_output = self.clamp
        ts0 = [0] + self.timesteps
        
        assert ts0 == sorted(ts0)
        if self.requested_train_dates.__class__.__name__ != "list":
            self.requested_train_dates = [self.requested_train_dates]
        if self.requested_val_dates.__class__.__name__ != "list":
            self.requested_val_dates = [self.requested_val_dates]
        if len(self.requested_train_dates) == 1 and self.requested_train_dates[0].__class__.__name__ == "datetime":
            self.requested_train_dates = [[self.requested_train_dates[0].timestamp()*1e9]]

        def _to_unix(x): 
            return np.concatenate(x).astype('int64') // 10**9 if len(x) > 0 and x[0] is not None else np.array([],dtype=np.int64)
        self.requested_train_unix = _to_unix(self.requested_train_dates)
        self.requested_val_unix = _to_unix(self.requested_val_dates)

class WeatherDataset():
    def __init__(self, config):
        self.config = config 
        
    def __len__(self):
        return self.sample_array.shape[0]
    
    def __getitem__(self, idx):
        if not hasattr(self, 'sample_array'):
            assert False, "You need to call check_for_dates before you can get items"
        
        # Keep trying to load the sample
        while True:
            try:
                return self.get_sample(idx)
            except Exception as e:
                unix = self.sample_unix_times[idx]
                debug_info = f"\nDEBUG INFO:\nIndex: {idx}\nDataset size: {len(self)}\nUnix: {unix}\n"
                raise type(e)(str(e) + debug_info) from e

    def unflatten(self,l):
        out = []
        j=0
        for d in self.sample_descriptor:
            out.append(l[j:j+len(d[1])])
            j += len(d[1])
        return out

    def get_sample(self,idx):
        ts = self.sample_array[idx]
        ts = self.unflatten(ts)
            
        if self.config.random_timestep_subset:
            ts = [ts[0]] + [ts[i+1] for i in sorted(torch.randperm(len(ts)-2)[:self.config.random_timestep_subset-1])] + [ts[-1]] # the first one is the input, last one we always want cause it's the max
        
        sample = TrainingSample()
        # times = [[unix times] (encoder timestep 0), [unix_times] (decoder timestep 1), [unix_times] (decoder timestep 2), ...]
        first_unix = None
        for index_t, times in enumerate(ts):
            instances = [[], []]
            for index_mesh, unix_t in enumerate(times):
                if first_unix is None: first_unix = unix_t
                dataset = self.sample_descriptor[index_t][1][index_mesh]
                data = dataset.load_data(unix_t, is_output=index_t > 0)
                if dataset.mesh.shape() != -1: assert all([x==y or x==-1 for x, y in zip(dataset.mesh.shape(), data.shape)]), f"{dataset.__class__.__name__} expects a tensor of shape: {dataset.mesh.shape()} but received a tensor of shape: {data.shape}"
                instances[0].append(dataset.mesh.string_id)
                instances[1].append(data)
                
            if index_t == 0:
                sample.inputs.append(instances)
            else:
                sample.outputs.append(instances)
            sample.timestamps.append(unix_t)
        sample.update()
        return sample
    
    def is_validation_idx(self, idx):
        return idx in self.val_sample_indices
    
    def is_validation_timestep(self, unix):
        idx = self.get_idx_from_unix(unix)
        return self.is_validation_idx(idx)
    
    def get_idx_from_unix(self, unix):
        o = np.where(self.sample_unix_times == unix)[0]
        assert len(o) == 1, f"Multiple indices found for unix: {unix} {self.sample_unix_times} {len(self.sample_unix_times)}"
        return o[0]

    def check_for_dates(self):
        # Instance = One weather state
        # Sample   = All instances across relevant timesteps 
        unix_dates = np.sort(np.concatenate([self.config.requested_train_unix, self.config.requested_val_unix]))
        self.datasets = copy.deepcopy(self.config.inputs)

        # Only gather times at these z values 
        only_at_z = range(0, 24, 3)
        if self.config.only_at_z is not None:
            only_at_z = set(self.config.only_at_z)
            
            # Make sure we also have the timesteps
            for z, timestep in zip(self.config.only_at_z, self.config.timesteps):
                additional_z = ( z + timestep ) % 24
                only_at_z.add(additional_z)
                
            only_at_z = sorted(list(only_at_z))
            
        self.sample_descriptor = [(0, self.config.inputs)] # Inputs
        
        instance_unix_time_offsets = [0] * len(self.config.inputs)
        self.instance_position_to_dataset_position = list(range(len(self.config.inputs)))
        if not self.config.realtime:
            self.datasets += self.config.outputs
            self.sample_descriptor += [(t, self.config.outputs) for t in self.config.timesteps] # Outputs
            instance_unix_time_offsets += list(itertools.chain.from_iterable([[dt*3600]*len(self.config.outputs) for dt in self.config.timesteps]))
            self.instance_position_to_dataset_position += [len(self.config.inputs)+i for i in range(len(self.config.outputs))]*len(self.config.timesteps)
        
        # This assert should be theoretically impossible to break given the code above, but in either case it should be true
        assert len(instance_unix_time_offsets) == len(self.instance_position_to_dataset_position), "Length of instance_unix_time_offsets and instance_position_to_dataset_position must be equal. Not a consistent number of instances represented in the sample"
        
        unix_datemin = get_date(np.min(unix_dates))
        unix_datemax = get_date(np.max(unix_dates))
        
        all_unix_times_i_could_want = np.array([unix_dates + 3600*h for h in only_at_z], dtype=np.int64).flatten()
        all_val_i_could_want = np.array([self.config.requested_val_unix + 3600*h for h in only_at_z], dtype=np.int64).flatten()
        all_train_i_could_want = np.array([self.config.requested_train_unix + 3600*h for h in only_at_z], dtype=np.int64).flatten()
        assert np.intersect1d(all_val_i_could_want, all_train_i_could_want).shape[0] == 0, "Validation and training dates should not overlap"

        dataset_unix_times = []
        for dataset in self.datasets:
            # all_unix_times_i_could_want represents the total number of dates available from model config
            # dataset.get_loadable_times(unix_datemin, unix_datemax) represents the total number of dates available for the specific dataset type
            loadable_times = dataset.get_loadable_times(unix_datemin, unix_datemax)
            assert not dataset.is_required or len(loadable_times) > 0, f"No loadable times for dataset: {dataset}"
            shared_unix_times = np.intersect1d(all_unix_times_i_could_want, loadable_times)

            if len(shared_unix_times) == 0 and dataset.is_required:
                print("No shared unix times for dataset: ", dataset)
                print("all_unix_times_i_could_want: ", all_unix_times_i_could_want)
                print("dataset.get_loadable_times(unix_datemin, unix_datemax): ", loadable_times)
                raise ValueError("No shared unix times for dataset: ", dataset)
            
            dataset_unix_times.append(shared_unix_times)
            
        # Normalizes dataset unix dates across instances
        num_instances = len(instance_unix_time_offsets)
        gathered_unix_times = [
            dataset_unix_times[self.instance_position_to_dataset_position[i]] - instance_unix_time_offsets[i]
            for i in range(num_instances) 
            if self.datasets[self.instance_position_to_dataset_position[i]].is_required
        ]
        sample_unix_times = reduce(np.intersect1d, gathered_unix_times)
        self.sample_unix_times = sample_unix_times
        assert len(sample_unix_times) > 0, f"No data found for"
        
        train_unix_times = np.intersect1d(all_train_i_could_want, sample_unix_times)
        val_unix_times = np.intersect1d(all_val_i_could_want, sample_unix_times)
        self.train_sample_indices = np.where(np.isin(sample_unix_times, train_unix_times))[0]
        self.val_sample_indices = np.where(np.isin(sample_unix_times, val_unix_times))[0]
        assert np.intersect1d(self.train_sample_indices, self.val_sample_indices).shape[0] == 0, "Training and validation dates should not overlap"

        assert len(train_unix_times) > 0, f"No training data found for training dates {self.config.requested_train_dates}"
        if self.config.requested_val_dates[0] is not None: assert len(val_unix_times) > 0, f"No validation data found for validation dates {self.config.requested_val_dates}"
        assert len(self.train_sample_indices) + len(self.val_sample_indices) == len(sample_unix_times), f"Training and validation indices should cover all sample indices. {len(self.train_sample_indices)} + {len(self.val_sample_indices)} != {len(sample_unix_times)}"

        sample_array = np.zeros((num_instances,len(sample_unix_times)),dtype=np.int64)
        
        for i in range(num_instances):
            sample_array[i] = sample_unix_times + instance_unix_time_offsets[i]
            assert np.all(np.isin(sample_array[i], dataset_unix_times[self.instance_position_to_dataset_position[i]])) or not self.datasets[self.instance_position_to_dataset_position[i]].is_required, f"Something is fucked"
        
        sample_array = sample_array.T
        assert sample_array.shape[1] == len(self.instance_position_to_dataset_position), f"{sample_array.shape} vs {self.instance_position_to_dataset_position}"
        self.sample_array = sample_array 
        if sample_array.shape[0] > 1: print(f"📅📅 Total Min dh {np.diff(sample_array[:,0]).min() // 3600}hr, Max dh {np.diff(sample_array[:,0]).max() // 3600}hr, Mean dh {np.diff(sample_array[:,0]).mean() / 3600}hr",only_rank_0=True)
