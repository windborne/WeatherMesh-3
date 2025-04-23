#JACK_CHANGE_LATER (the whole bloody file needs to be cleaned)
# this file exists becasue I'm sick of always importing torch every time 
# I need utils for ever script so that fast stuff is all going here
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
import os
import time
import numpy as np
import copy
import random

D = lambda *x: datetime(*x, tzinfo=timezone.utc)

def RED(text): return f"\033[91m{text}\033[0m"
def GREEN(text): return f"\033[92m{text}\033[0m"
def YELLOW(text): return f"\033[93m{text}\033[0m"
def BLUE(text): return f"\033[94m{text}\033[0m"
def MAGENTA(text): return f"\033[95m{text}\033[0m"
def CYAN(text): return f"\033[96m{text}\033[0m"
def WHITE(text): return f"\033[97m{text}\033[0m"
def ORANGE(text): return f"\033[38;5;214m{text}\033[0m"

ASNI_COLORS = {
    'RED' : '\033[91m',
    'GREEN' : '\033[92m',
    'YELLOW' : '\033[93m',
    'BLUE' : '\033[94m',
    'MAGENTA' : '\033[95m',
    'CYAN' : '\033[96m',
    'WHITE' : '\033[97m',
    'END' : '\033[0m',
    'ORANGE': '\033[38;5;214m',
}

default_hres_config = None
try:
    with open("hres/hres_utils.py") as f:
        t = f.read().replace("default_config", "default_hres_config")
        exec(t)
except:
    pass

class HourlyConfig:
    def __init__(self, conf_to_copy=None, **kwargs):
        # Couple hourly variables (assumes a dt schedule since )
        self.use_hourly = False 
        self.rank_separation = -1 # GPU rank separation (which gpus do only 6hr prediction (<) and which do 6hr + 1hr prediction (>=))
        
        # Number of timesteps to take from inbetween 6 hour timesteps (i.e. 1,2,3,4,5 -> 2,5 or 1,3 or 2,4 ...)
        # Only used when schedule dts is False (since we can't change timesteps + when using schedule dts, random subsampling will natively handle this for us)
        self.num_hourly = 2 
        
        # Settings for max timestep taken from inbetween 6 hour timesteps 
        # Only used when schedule dts is True (since we can change timesteps)
        self.hourly_max_timestep = 3 # Max hour to take samples from (i.e. 1,2,3,4,5 -> 1,2,3)
        self.final_hourly_max_timestep = 5 # Max hour to take samples from at the end of training (i.e. 1,2,3,4,5 -> 1,2,3,4,5)
        self.steps_till_max_hourly_timestep = 5000 # Num of steps before we're using the full [1,5] timestep range for hourly
        
        # General settings
        self.max_dt = 24 # We only want to run hourly up until 24 hours
        self.curr_max_dt = self.max_dt
        
        if conf_to_copy is not None:
            self.__dict__.update(copy.deepcopy(conf_to_copy.__dict__))
        for k, v in kwargs.items():
            assert hasattr(self, k), f"Unknown HourlyConfig option: {k}"
            setattr(self, k, v)
    
    # Computes the max timestep to take from inbetween 6 hour timesteps (i.e. 1,2,3,4,5 -> 1,2,3 at step 0 vs 1,2,3,4,5 at step self.steps_till_max_hourly_timestep)
    def compute_hourly_max_timestep(self, n_step, cosine_period):
        self.steps_till_max_hourly_timestep = min(self.steps_till_max_hourly_timestep, cosine_period)
        decimal_max_timestep = self.hourly_max_timestep + (self.final_hourly_max_timestep - self.hourly_max_timestep) * (min(n_step, self.steps_till_max_hourly_timestep)/self.steps_till_max_hourly_timestep)
        return int(decimal_max_timestep) 

    # Generates the actual timestep used for 
    def generate_timesteps(self, processor_dts, schedule_dts=False, step=None, schedule_warmup=None):
        assert 1 in processor_dts, "Coupled hourly requires processor_dts to include 1"
        assert 6 in processor_dts, "Coupled hourly assumes 6 hr processor as well lol, not real general atm I know..."
        assert 0 <= self.num_hourly and self.num_hourly < 6, "Can only take a max of 5 timestemps between (0,6) exclusive (i.e. you set num_hourly too large)"
        assert self.final_hourly_max_timestep <= 5, "Can only take a max of 5 timesteps between (0,6) exclusive (i.e. you set final_hourly_max_dt too large)"
        
        six_hour_timesteps = list(range(0,min(self.curr_max_dt, self.max_dt)+1,6))
        if schedule_dts: # Assume the subsampling happens via schedule_dts
            assert step is not None and schedule_warmup is not None, "If using schedule_dts, need to pass in step and schedule_warmup"
            if step < schedule_warmup:
                return [1] # We only want to train up to 1 timestep (first hour) for the first schedule_warmup steps
            hourly_timesteps = [six_hour_timesteps[i] + j for i in range(len(six_hour_timesteps) - 1) for j in range(1, self.hourly_max_timestep + 1)]
            return hourly_timesteps # sorted(six_hour_timesteps + hourly_timesteps) if we want functionality for gpus to do both 6hr and 1hr processing 
        else:
            hourly_timesteps = [six_hour_timesteps[i] + j for i in range(len(six_hour_timesteps) - 1) for j in random.sample(range(1, 6), self.num_hourly)]
            return sorted(hourly_timesteps)

class LRScheduleConfig:
    def __init__(self, conf_to_copy=None, **kwargs):
        # GENERAL LR VARIABLES
        self.lr = 2e-4
        self.warmup_end_step = 1000 # Number of steps before warmup is done
        
        # RESTART VARIABLES
        self.restart_warmup_end_step = 100 # Number of steps before restart warmup is done
        
        # WSD-S LR VARIABLES (https://arxiv.org/pdf/2410.05192)
        self.wsds_en = False # WSD-S enable
        self.wsds_min_lr_factor = 1/10 # Difference in size between min and max learning rate, the literature used 0.1
        self.wsds_decay_length = 1_000 # Steps in the decay period (literature suggests ~10% of time should be spent in decay)
        self.wsds_decay_frequency = 10_000 # How many steps between decay periods 
        
        # COSINE LR VARIABLES
        self.cosine_en = True
        self.cosine_period = 45_000
        self.cosine_bottom = 5e-8
        
        # SCHEDULE DTS VARIABLES
        # Schedule dts is a way to increase the max dt over time
        self.schedule_dts = False
        self.schedule_dts_warmup_end_step = 1000 # End of the warmup period (training on only processor_dt for the first 1000 steps)
        self.min_dt_min = 0
        self.min_dt_max = 0 # Latest timestep we want the dt schedule to start at
        self.max_dt_max = 144 # Latest timestep we want the dt schedule to reach by end of step limit below
        self.steps_till_max_dt_max = 45_000 # Timesteps required to reach latest timestep (if cosine schedule < this, will use cosine schedule)
        
        # TIMESTEP SUBSET VARIABLES
        self.num_random_subset_min = 2 # How many random timesteps to use at the start of training
        self.num_random_subset_max = 8 # How many random timesteps to use at the end of step limit below
        self.steps_till_num_random_subset_max = 45_000 # Timesteps required to reach most subsets
        
        # MISC (NOT THAT USEFUL?)
        self.step_offset = 0 # Offset for step 
        self.div_factor = 4 
        
        if conf_to_copy is not None:
            self.__dict__.update(copy.deepcopy(conf_to_copy.__dict__))
        for k, v in kwargs.items():
            assert hasattr(self, k), f"Unknown LRSchedulerConfig option: {k}"
            setattr(self, k, v)

    def computeLR(self, step, n_step_since_restart=None):
        is_restart = n_step_since_restart != step and n_step_since_restart is not None

        # Warmup learning rate
        step = max(step + self.step_offset,0)
        lr = np.interp(step+1, [0, self.warmup_end_step], [0, self.lr])
        
        if step > self.warmup_end_step:
            if self.cosine_en:
                assert not self.wsds_en, "Can't have both cosine and wsds LR schedules enabled"
                cycle = self.warmup_end_step + self.cosine_period
                n = step // cycle
                step_modc = step % cycle
                
                lr = np.interp(step_modc+1, [0, self.warmup_end_step], [0, self.lr / (self.div_factor**n)]) # should use og warmup step to get og lr curve
                cstep = step_modc - self.warmup_end_step
                lr = lr * (np.cos(cstep/self.cosine_period *np.pi)+1)/2
                if self.cosine_bottom is not None:
                    if n > 0:
                        lr = self.cosine_bottom
                    else:
                        lr = max(lr, self.cosine_bottom)
            elif self.wsds_en:
                # Technically the paper says we don't want to start a decay after a loss spike (which yknow, is valid) 
                # but we're not doing that here cause too complex and questionable how much it would actually help given that the decay period is a, yknow, decay period 
                # (thus we likely converge to valley anyways even if loss spike leads to large gradient step in wrong direction)
                period_step = step % self.wsds_decay_frequency
                if period_step < self.wsds_decay_frequency - self.wsds_decay_length:
                    lr = lr
                else:
                    period_fraction = (period_step - (self.wsds_decay_frequency - self.wsds_decay_length))/ self.wsds_decay_length
                    lr = period_fraction * (1/(self.wsds_min_lr_factor * self.lr)) + (1 - period_fraction)*(1/self.lr)
                    lr = 1/lr
            else:
                assert False, "LR type not yet implemented" # Used to default to lr = self.lr
        
        # Restart warmup
        if is_restart and n_step_since_restart < self.restart_warmup_end_step:
            return lr * n_step_since_restart / self.restart_warmup_end_step
        return lr

    def computeMaxDT(self,step,proc_dt = 6, slow=False):
        if not self.schedule_dts:
            return 0
        if type(proc_dt) == list:
            if len(proc_dt) > 1:
                assert 1 in proc_dt, "Multiple dts is only supported for coupled_hourly"
                assert len(proc_dt) == 2, "coupled hourly can only take in one more timestep (for the moment)"
                proc_dt = max(proc_dt)
            else:
                assert len(proc_dt) == 1, "proc_dt must be a single integer for random timesteps"
                proc_dt = proc_dt[0]
        if hasattr(self,'max_dt_func') and self.max_dt_func is not None:
            max_dt = self.max_dt_func(step)
        else:
            if step < self.schedule_dts_warmup_end_step:
                if slow:
                    return 24#4*proc_dt
                else:
                    return proc_dt
            self.steps_till_max_dt_max = min(self.steps_till_max_dt_max, self.cosine_period)
            max_dt = self.max_dt_min + (self.max_dt_max - self.max_dt_min)/self.steps_till_max_dt_max**2 * min(step,self.steps_till_max_dt_max)**2
        max_dt = max_dt // proc_dt * proc_dt
        return int(max_dt)
    
    def computeMinDT(self,step,proc_dt = 6):
        if not self.schedule_dts:
            return 0
        if type(proc_dt) == list:
            assert len(proc_dt) == 1, "need to implement multiple dts for min dt"
            proc_dt = proc_dt[0]
        if hasattr(self,'min_dt_func') and self.min_dt_func is not None:
            min_dt = self.min_dt_func(step)
        else:
            self.steps_till_max_dt_max = min(self.steps_till_max_dt_max, self.cosine_period)
            min_dt = self.min_dt_min + (self.min_dt_max - self.min_dt_min)/self.steps_till_max_dt_max**2 * min(step,self.steps_till_max_dt_max)**2
        min_dt = min_dt // proc_dt * proc_dt
        return int(min_dt)
    
    def computeNumRandomSubset(self,step, slow=False):
        if not self.schedule_dts:
            return 1
        if hasattr(self,'num_random_subset_func') and self.num_random_subset_func is not None:
            num_random_subset = self.num_random_subset_func(step)
        else:
            if step < self.schedule_dts_warmup_end_step:
                if slow:
                    return 3
                else:
                    return 1
            self.steps_till_num_random_subset_max = min(self.steps_till_num_random_subset_max, self.cosine_period)
            num_random_subset = self.num_random_subset_min + (self.num_random_subset_max - self.num_random_subset_min)/self.steps_till_num_random_subset_max * min(step,self.steps_till_num_random_subset_max)
        return int(num_random_subset)
    
    def get_max_lr_schedule(self):
        if self.cosine_en:
            return self.cosine_period
        elif self.wsds_en:
            return 'WSDS'
        else:
            return 'unknown_lr_schedule'
    
    def make_plots(self, step_range=None, save_path="ignored/plots/lr_sched.png"):
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        if step_range is None:
            step_range = self.cosine_period
        _, axs = plt.subplots(4,1,figsize=(6,10))
        axs[0].plot(np.arange(0,step_range,100),np.vectorize(self.computeLR)(np.arange(0,step_range,100))); axs[0].grid(); axs[0].set_title("LR") ; axs[0].set_xlabel("Step")
        axs[1].plot(np.arange(0,step_range,100),np.vectorize(self.computeMinDT)(np.arange(0,step_range,100))); axs[1].grid(); axs[1].set_title("Min DT") ; axs[1].set_xlabel("Step")
        axs[2].plot(np.arange(0,step_range,100),np.vectorize(self.computeMaxDT)(np.arange(0,step_range,100))); axs[2].grid(); axs[2].set_title("Max DT") ; axs[2].set_xlabel("Step")
        axs[3].plot(np.arange(0,step_range,100),np.vectorize(self.computeNumRandomSubset)(np.arange(0,step_range,100))); axs[3].grid(); axs[3].set_title("Num Random Subset") ; axs[3].set_xlabel("Step")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

class WeatherTrainerConfig():
    def __init__(self,conf_to_copy=None,**kwargs):
        # Debugging configs
        self.nope = False # Whether to log/save run to actual run directory
        self.log_to_tmp = False # Same as self.nope but more verbose lol
        self.quit = False # Whether to quit before training
        self.print_ram_usage = False 
        self.profile = False # Whether to profile the training loop using torch.profiler.profile()
        
        # Resume configs
        self.resume = False
        self.resume_select = ''
        self.strict_load = True
        self.new_resume_folder = False
        self.reset_optimizer = False
        self.reset_steps_on_resume = False
        
        # Gradient config
        self.HALF = True # Half precision
        self.HALF_gradient_clip = 4.0 # Gradient clipping (for half precision)
        self.initial_gradscale = 65536.0
        self.actually_scale_gradients = True # If we actually want to scale the gradient by gradscaler 
        
        # Optimizer configs
        #   General settings
        self.save_optimizer = True
        self.optim = 'shampoo' # Defaults to adam if we're using not using DDP 
        #   Adam settings
        self.adam = SimpleNamespace()
        self.adam.betas= (0.9, 0.99)
        self.adam.weight_decay= 0.001
        #   Shampoo settings
        self.shampoo = SimpleNamespace()
        self.shampoo.version = 'old'
        self.shampoo.dim = 8192
        self.shampoo.num_trainers = -1 # defaults to # gpus
        self.shampoo.start_preconditioning_step = 1 # When to start shampoo preconditioning
        
        # LR + dts schedule config
        self.lr_sched = LRScheduleConfig()
        self.slow_start = False # uses more timesteps (3) for the first 1k steps to avoid big jumps
        
        # Hourly config
        self.hourly = HourlyConfig()
        
        # Logging configs
        self.no_logging = False # Whether to log to tensorboard
        self.tb_prefix = '' # Prefix to add before run name
        self.run_folder_suffix = '' # Suffix to add after run folder name
        self.log_every = 25 # How often to log to tensorboard
        self.log_step_every = 25
        self.log_dir_base = '/huge/deep'
        self.save_every = 100 # How often to save model
        self.save_imgs_every = 1_000 # How often to save training images
        
        # Cluster (training) configs
        self.gpus = '0'
        self.num_workers = 2 # Number of dataloading workers
        self.prefetch_factor = 2 # Number of batches to prefetch for dataloading
        self.pin_memory = True # Whether to pin memory for dataloading (makes GPU usage faster)
        self.timeout = timedelta(minutes=10) # Timeout for init_process_group
        
        # Loss weighing
        self.latent_l2 = 0 # L2 on y_gpu? (i.e. total_loss += self.conf.latent_l2 * y_gpus["latent_l2"])
        self.dt_loss_weights_override = {} # See trainer.py for more details on dt_loss_weighing
        self.dt_loss_beta = 0.995
        
        # Useful misc configs
        self.find_unused_params = False
        self.ignore_train_safegaurd = False # Related to test vs training set separation (represents the year 2020)
        self.use_tf32 = False # Allows the use of tf32 via cuda.matmul.allow_tf32
        self.compute_Bcrit_every = np.nan # Allows backward pass to compute Bcrit every x steps
        self.already_init_process_group = False # If we already init_process_group (useful for coupled hourly)
        self.use_reduced_loss = False # If we should use reduced loss across gpus instead of normal loss
        
        # Misc variables (not entirely sorted yet)
        self.batch_size = 1
        self.activity = ''
        self.name = ''
        self.disregard_buffer_checksum = False
        self.steamroll_over_mismatched_dims = False
        self.rerun_sample = 1 
        self.console_log_path = None
        self.skip_audit = False # audit_inputs() is never called in train.py so this is prob not needed?        
        self.diffusion = False
        self.bboxes = None # I think used for diffusion?
        self.do_da_compare = False # Useful for DA?
        
        
        # POTENTIALLY DEPRECATED BELOW (don't add to it, pls, add to the unsorted above!!)
        # Drop path (potentially deprecated as well?)
        self.drop_path = False
        self.drop_sched = SimpleNamespace()
        self.drop_sched.drop_max = 0.2
        self.drop_sched.drop_min = 0.0
        self.drop_sched.iter_offset = -816_300
        self.drop_sched.ramp_length = 200_000
        
        # Coupled (potentially deprecated as well? I can't find it used really anywhere)
        self.coupled_pointy_weight = 0.1
        
        # Pointy (potentially deprecated as well? I can't find it used really anywhere)
        self.use_point_dataset = False
        self.point_batch_size = 16

        # Potentially not used anymore? Looks like they're not used anymore (checked by Jack)
        self.yolo = False
        self.dimprint = False
        self.diff_loss = 0.0
        self.TIME = False 
        self.seed = 0
        self.clamp = 13 # already defined in data.py config
        self.clamp_output = None # already defined in data.py config
        self.only_at_z = None # already defined in data.py config
        self.N_epochs = 1_000_000_000
        self.cpu_only = False
        self.shuffle = True
        self.on_cloud = False

        self.broke_vram = False
        
        if conf_to_copy is not None:
            self.__dict__.update(copy.deepcopy(conf_to_copy.__dict__))
        for k,v in kwargs.items():
            assert hasattr(self,k), f"Unknown config option: {k}"
            setattr(self,k,v)

def am_i_torchrun():
    return 'TORCHELASTIC_ERROR_FILE' in os.environ.keys()

