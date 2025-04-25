import torch
import torch.nn as nn
import matepoint
from types import SimpleNamespace
from model_latlon.transformer3d import SlideLayers3D
from utils import MAGENTA

def simple_gen_todo(dts, processor_dts):
    """
    Generates a todo dictionary for the model based on the given dts and processor_dts.
    This todo dictionary is used to determine which forwards the model should call and in which order.
    
    Args:
        dts (list): A list of integers representing the dts for which the model should be called.
        processor_dts (list): A list of integers representing the processor dts for which the model is capable of.
    
    Returns:
        todo_dictionary (dict): A dictionary where the keys are the dts and the values are the todo strings necessary to achieve that dt.
    """
    todo_dictionary = {}
    
    for dt in dts:
        remaining_dt = dt
        
        # Encoder
        todo = "E,"
        
        # Processor
        for processor_dt in reversed(processor_dts):
            processor_dt_iterations = remaining_dt // processor_dt
            
            todo += f"P{processor_dt}," * processor_dt_iterations
            remaining_dt -= processor_dt * processor_dt_iterations
            if remaining_dt == 0:
                break
        assert remaining_dt == 0, f"Unable to fit processor_dts {processor_dts} into dt {dt}. The remaining dt is {remaining_dt}"
        
        # Decoder
        todo += "D"
        
        todo_dictionary[dt] = todo
    
    return todo_dictionary

class ForecastModel(nn.Module):
    def __init__(self, config, encoders=[], decoders=[], processors=None):
        super(ForecastModel, self).__init__()
        self.config = config
        
        self.encoders = nn.ModuleList(encoders)
        if processors is None:
            processors = self.assume_default_processors()
        self.processors = nn.ModuleDict(processors)
        self.decoders = nn.ModuleList(decoders)

        print(MAGENTA(f"Initializing model. 'model.config.checkpoint_type' is {self.config.checkpoint_type}"))
        
    def assume_default_processors(self):
        """
        Defines default processors for the model if no processors were given (assumes processor_dts is defined) and generates processors corresponding to those dts.
        """    
        assert self.config.processor_dts is not None, "processor_dts must be defined in some capacity"
        
        processors = {}
        for i,dt in enumerate(self.config.processor_dts):
            processors[str(dt)] = SlideLayers3D(
                dim=self.config.latent_size, 
                depth=self.config.pr_depth[i], 
                num_heads=self.config.num_heads, 
                window_size=self.config.window_size, 
                checkpoint_type=self.config.checkpoint_type
            )
            
        return processors
    
    def latent_l2(self, x):
        return torch.mean(x**2)

    def encode(self, x, t0s):
        """
        Handles the encoding of the input data at timestep t0s
        """
        if self.config.parallel_encoders:
            num_encs = len(self.encoders)
            assert len(x) == num_encs, f"The number of inputs does not match the number of encoders. {len(x)} vs {num_encs}"
            
            output_x = None
            encoder_weight_sum = sum(self.config.encoder_weights)
            for i,encoder in enumerate(self.encoders):
                encoder_weight = self.config.encoder_weights[i] / encoder_weight_sum
                x_e = encoder(x[i], t0s) * encoder_weight
                
                if output_x is None: output_x = x_e
                else: output_x += x_e # Add the outputs of parallel encoders together
                
                del x_e
                
        else:
            encoder = self.encoders[0]
            output_x = encoder(x[0], t0s)
        
        return output_x
    
    def decode(self, x):
        """
        Handles the decoding of the output data
        """
        outputs = []
        for dec in self.decoders:
            args = [x]
            outputs.append(dec(*args))
        return outputs

    def forward(self, x, todo, send_to_cpu=False, callback=None):
        # Clears matepoint context
        matepoint.Gmatepoint_ctx = []
        
        # Helper function to check if todo is a list of ints (timesteps to process)
        def is_list_of_ints(obj): 
            return isinstance(obj, list) and all(isinstance(obj_i, int) for obj_i in obj)
        
        if is_list_of_ints(todo):
            todo = simple_gen_todo(sorted(todo), self.config.processor_dts)
        
        return self.forward_inner(x, todo, send_to_cpu, callback)
    
    def forward_inner(self, x, todo_dict, send_to_cpu, callback):
        assert isinstance(x, list), f"This is not yet open source"
        t0s = x[-1]
        x = x[0:-1]
        
        # Memory management for todos 
        todos = [
            SimpleNamespace(
                target_dt = dt,
                remaining_steps = todo.split(','),
                completed_steps = [],
                state = x,
                accum_dt = 0,
                ) 
            for dt, todo in todo_dict.items()
        ]
             
        total_l2 = 0
        total_l2_n = 0
        outputs = {}
        while todos:
            current_todo = todos[0]
            current_step = current_todo.remaining_steps[0]
            completed_steps = current_todo.completed_steps.copy()
            x = current_todo.state
            accumulated_dt = 0
            
            # Encoders
            if current_step == 'E':
                prepared_x = [x_iter.clone() for x_iter in x] # Guarantee a new computation graph
                prepared_t0s = t0s + 3600 * current_todo.accum_dt
                
                x = self.encode(prepared_x, prepared_t0s)
                
                total_l2 += self.latent_l2(x)
                total_l2_n += 1

            # Processors
            elif current_step.startswith('P'):
                processor_dt = int(current_step[1:])
                
                x = self.processors[str(processor_dt)](x)
                
                accumulated_dt += processor_dt
                
            # Decoders
            elif current_step == 'D':
                total_l2 += self.latent_l2(x)
                total_l2_n += 1
                
                x = self.decode(x)
                
                if send_to_cpu:
                    x = [x_iter.cpu() for x_iter in x]
            
            else:
                assert False, f"Unknown step type {current_step}"
                
            # Update todos
            for todo in todos:
                # Only update the todos that have the same current step and completed steps as the current todo
                if todo.remaining_steps[0] == current_step and todo.completed_steps == completed_steps:
                    todo.state = x
                    todo.accum_dt += accumulated_dt
                    todo.remaining_steps = todo.remaining_steps[1:]
                    todo.completed_steps.append(current_step)

                    if not todo.remaining_steps:
                        if callback is not None:
                            callback(todo.target_dt, x)
                        else:
                            outputs[todo.target_dt] = x
                        todos.remove(todo)
        
        outputs["latent_l2"] = total_l2 / total_l2_n
        return outputs