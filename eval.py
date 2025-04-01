import torch
import torch.distributed as dist
from utils import vars_with_nans

def eval_rms_train(preds, actuals, norm_matrix, weight, keys=None, by_level=False, stdout=True, ddp_reduce=True, mesh=None):
    # this is the vesion called by train.py
    # it's slightly different as it has deal with some stuff being on GPU
    with torch.no_grad():
        B,N1,N2,D = preds.shape[:4]
        if keys is None:
            keys = ['var'+str(i) for i in range(D)]
        assert len(preds.shape) <= 5, f"preds must have shape (B,N1,N2,D) or (B,N1,N2,D,O)"
        assert preds.shape == actuals.shape, f"{preds.shape} != {actuals.shape}"
        assert D == norm_matrix.shape[0], f"preds and norm must have same number of variables. preds: {preds.shape}, norm: {norm_matrix.shape}"
        assert preds.shape[1:3] == weight.shape, f"preds and weights must have same spacial dims. preds: {preds.shape}, weight: {weight.shape}"

        error = preds - actuals

        O = 1 if len(preds.shape) == 4 else preds.shape[4]
        levs = mesh.levels

        ws = torch.sum(weight)*B*O
        rms_ = []
        rms_dict = {}
        extra = {}
        for i,k in enumerate(keys):
            f = 1e6 if k == '133_q' else 1 # use mg/kg instead of kg/kg for Q
            e = error[:,:,:,i] * norm_matrix[i]
            if k in vars_with_nans:
                # Create a mask for non-NaN values
                mask = ~torch.isnan(e)
                # Replace NaN with 0 in e for the einsum operation
                e = torch.nan_to_num(e, nan=0.0)
                # Compute the sum of weights for non-NaN values
                ws = torch.einsum('bnm...,nm->',mask.float(), weight)
                if ws == 0: ws = 1
            #print("uhhhh", e.dtype, "sq", torch.square(e).dtype, "weight", weight.dtype, "preds", preds.dtype, "error", error.dtype)
            msall = torch.einsum('bnm...,nm->',torch.square(e),weight)/ws * f**2
            if ddp_reduce:
                dist.all_reduce(msall, op=dist.ReduceOp.SUM)
                rmsall = torch.sqrt(msall / dist.get_world_size())
            else:
                rmsall = torch.sqrt(msall)
            rmsall = rmsall.to('cpu').numpy()
            if stdout: print(f"{k}: {rmsall:.2f}")
            if by_level and O > 1:
                #print(e.shape,weight.shape)
                mslev = torch.einsum('bnml,nm->l',torch.square(e),weight)/ws*O * f**2
                if ddp_reduce:
                    dist.all_reduce(mslev, op=dist.ReduceOp.SUM)
                    rmslev = torch.sqrt(mslev / dist.get_world_size())
                else:
                    rmslev = torch.sqrt(mslev)
                rmslev = rmslev.to('cpu').numpy()
                for j in range(O):
                    extra[k+"_"+str(levs[j])] = float(rmslev[j])
                    if stdout: print(f"  {k} {levs[j]}: {rmslev[j]:.2f}")
            rms_.append(rmsall)
            rms_dict[k] = float(rmsall)
        if by_level: rms_dict.update(extra)
        return rms_dict