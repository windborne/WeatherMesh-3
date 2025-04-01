import inspect
from datetime import datetime, timedelta, timezone
import numpy as np
import pickle

levels_gfs = [10, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 975, 1000]
levels_full = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
levels_medium = [10, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000]
levels_hres = [10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 925, 950, 1000]
levels_tiny = [50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925, 1000] # Open data IFS, also weatherbench HRES

core_pressure_vars = ["129_z", "130_t", "131_u", "132_v", "133_q"]
core_sfc_vars = ["165_10u", "166_10v", "167_2t", "151_msl"]

CONSTS_PATH = '/fast/consts'
PROC_PATH = '/fast/proc'
RUNS_PATH = '/huge/deep'

num2levels = {}
for levels in [levels_gfs, levels_full, levels_medium, levels_hres]:
    if len(levels) in num2levels: continue
    num2levels[len(levels)] = levels

vars_with_nans = ["034_sstk", "tc-maxws", "tc-minp"]

class SourceCodeLogger:
    def get_source(self):
        return inspect.getsource(self.__class__)

def RED(text): return f"\033[91m{text}\033[0m"
def GREEN(text): return f"\033[92m{text}\033[0m"
def YELLOW(text): return f"\033[93m{text}\033[0m"
def BLUE(text): return f"\033[94m{text}\033[0m"
def MAGENTA(text): return f"\033[95m{text}\033[0m"
def CYAN(text): return f"\033[96m{text}\033[0m"
def WHITE(text): return f"\033[97m{text}\033[0m"
def ORANGE(text): return f"\033[38;5;214m{text}\033[0m"

def get_date(date):
    if type(date) == datetime:
        assert date.tzinfo is None or date.tzinfo == timezone.utc, "this is not designed to be used with non-utc dates"
        return date.replace(tzinfo=timezone.utc) 
    elif np.issubdtype(type(date), np.number):
        nix = int(date)
        return datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(seconds=nix)
    assert False, f"brother I don't know how to work with what you gave me: {date} {type(date)}"

def load_state_norm(wh_lev, config, with_means=False):
    norms = pickle.load(open(f'{CONSTS_PATH}/normalization.pickle', 'rb'))
    for k,v in norms.items():
        mean, var = v
        norms[k] = (mean, np.sqrt(var))
    
    state_norm_matrix = []
    state_norm_matrix2 = []
    for i, v in enumerate(config.pressure_vars):
        # note: this doesn't include full levels!!
        state_norm_matrix2.append(norms[v][0][wh_lev])
        state_norm_matrix.append(norms[v][1][wh_lev])
    for i, s in enumerate(config.sfc_vars):
        if s == 'zeropad':
            state_norm_matrix2.append(np.array([0.]))
            state_norm_matrix.append(np.array([1.]))
            continue
        state_norm_matrix2.append(norms[s][0])
        state_norm_matrix.append(norms[s][1])

    state_norm_matrix = np.concatenate(state_norm_matrix).astype(np.float32)
    state_norm_matrix2 = np.concatenate(state_norm_matrix2).astype(np.float32)
    if with_means:
        return norms,state_norm_matrix,state_norm_matrix2
    return norms,state_norm_matrix