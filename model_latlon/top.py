import torch
import torch.nn as nn
from types import SimpleNamespace

from meshes import LatLonGrid
from model_latlon.transformer3d import SlideLayers3D
from utils import SourceCodeLogger, MAGENTA
import matepoint


class ForecastModel(nn.Module,SourceCodeLogger):
    def __init__(self, config, encoders=[], decoders=[], processors=None):
        super(ForecastModel, self).__init__()
        self.code_gen = 'gen3'
        self.config = config
        
        self.embedding_module = config.embedding_module
        self.encoders = nn.ModuleList(encoders)
        if processors is None:
            processors = {}
            for i,dt in enumerate(self.config.processor_dt):
                processors[str(dt)] = SlideLayers3D(dim=self.config.latent_size, depth=config.pr_depth[i], num_heads=config.num_heads, window_size=self.config.window_size, embedding_module=self.embedding_module, checkpoint_type=self.config.checkpoint_type)

        self.processors = nn.ModuleDict(processors)
        self.decoders = nn.ModuleList(decoders)

        if isinstance(self.config.outputs, list): assert isinstance(self.config.outputs[0], LatLonGrid), f"Code only works right now if LatLonGrid is the first mesh type"

        print(MAGENTA(f"Initializing model. model.config.checkpoint_type is 👉{self.config.checkpoint_type}👈 Is this what you want?"))

    def forward_inner(self, xs, todo_dict, send_to_cpu, callback, return_latents=False, **kwargs):
        # TODO: Clean this up to use the stuff in process.py.

        c = self.config

        if type(xs) == list:
            t0s = xs[-1]
            xs = xs[0:-1]
        else:
            t0s = None
        
        def encode(xs,t0s):
            if c.parallel_encoders:
                num_encs = len(self.encoders)
                assert len(xs) == num_encs, f"number of inputs does not match number of encoders. Probably should be calling the rollout reencoder. {len(xs)} vs {num_encs}"
                x = None
                sW = sum(c.encoder_weights)
                for i,encoder in enumerate(self.encoders):
                    print("Encoder: ", encoder, only_rank_0=True)
                    x_e = encoder(xs[i],t0s)*(c.encoder_weights[i]/sW)
                    if x == None: x = x_e
                    else: x = x + x_e
                    del x_e
            else:
                x = self.encoders[0](xs[0],t0s) 
            return x
        
        # Time is in to_unix format (seconds since 1970)
        def decode(x, time, output_idx):
            outputs = []
            for dec in self.decoders:
                args = [x]
                outputs.append(dec(*args))
            return outputs
        
        todos = [SimpleNamespace(
                target_dt=k,
                remaining_steps = v.split(','),
                completed_steps = [],
                state = (xs if v.startswith('E') else xs[0],None), # in case we go straight from latent
                accum_dt=0,
                output_idx=i, # 0 is the first output
            ) for i, (k,v) in enumerate(todo_dict.items())]

        total_l2 = 0
        total_l2_n = 0
        outputs = {}
        def latent_l2(aa):
            return torch.mean(aa**2)
            return call_checkpointed(lambda a: torch.mean(a**2), aa)
        while todos:
            tnow = todos[0]
            #torch.cuda.empty_cache()
            #mem = torch.cuda.mem_get_info()
            #if int(os.environ.get('RANK', 0)) == 0: print("doing", tnow.remaining_steps[0], mem[0]/1e9, mem[1]/1e9)
            #_t0 = time.time()
            step = tnow.remaining_steps[0]
            completed_steps = tnow.completed_steps.copy()
            x,extra = tnow.state
            accumulated_dt = 0
            #for todo in todos:
            #    print(f"{todo.target_dt}: {','.join(todo.remaining_steps)}")
            if step == 'E':
                x = encode([xx.clone() for xx in x],t0s+3600*tnow.accum_dt)
                total_l2 += latent_l2(x); total_l2_n += 1
            elif step == 'rE':
                assert c.rollout_reencoder, "rollout reencoder not enabled"
                x = self.rollout_reencoder(x[0],t0s+3600*tnow.accum_dt)
            elif step.startswith('P'):
                pdt = int(step[1:])
                x = self.processors[str(pdt)](x)
                #pdt = c.processor_dt[0]
                #assert len(c.processor_dt) == 1, "only one processor_dt is supported"
                #x = self.processor(x)
                accumulated_dt += pdt
            elif step == 'D':
                if not return_latents:
                    total_l2 += latent_l2(x); total_l2_n += 1
                    x = decode(x, (t0s + ( tnow.accum_dt * 60 * 60)).item(), tnow.output_idx)
                    if send_to_cpu:
                        x = [xx.cpu() for xx in x]
                else:
                    wpad = self.config.window_size[2]//2
                    assert len(x.shape) == 5, x.shape
                    x = x[:,:,:,wpad:-wpad,:]
            else:
                assert False, f"Unknown step {step}"
            for todo in todos:
                if todo.remaining_steps[0] == step and todo.completed_steps == completed_steps:
                    todo.state = (x,extra)
                    todo.accum_dt += accumulated_dt
                    todo.remaining_steps = todo.remaining_steps[1:]
                    todo.completed_steps.append(step)
                    if not todo.remaining_steps:
                        if callback is not None:
                            callback(todo.target_dt, x)
                        else:
                            outputs[todo.target_dt] = x
                        todos.remove(todo)
            #torch.cuda.synchronize()
            #if int(os.environ.get('RANK', 0)) == 0: print("took", time.time()-_t0)
        outputs["latent_l2"] = total_l2 / total_l2_n
        return outputs
    
    def forward(self, xs, todo, send_to_cpu=False, callback=None, return_latents=False, from_latent=False, clear_matepoint=True, **kwargs):
        if clear_matepoint: matepoint.Gmatepoint_ctx = []
        def is_list_of_ints(obj): return isinstance(obj, list) and all(isinstance(x, int) for x in obj)
        if is_list_of_ints(todo):
            todo = simple_gen_todo(sorted(todo),self.config.processor_dt, from_latent=from_latent)
        y = self.forward_inner(xs, todo, send_to_cpu, callback, return_latents=return_latents, **kwargs)
        return y
    
    def get_matepoint_tensors(self):
        return []

def simple_gen_todo(dts,processor_dts, from_latent=False):
    out = {}
    for dt in dts:
        rem_dt = dt
        todo = "E," if not from_latent else ""
        for pdt in reversed(processor_dts):
            num = rem_dt // pdt
            rem_dt -= pdt*num
            todo+= f"P{pdt},"*num
            if rem_dt == 0:
                break
        assert rem_dt == 0
        todo+="D"
        out[dt] = todo
    return out