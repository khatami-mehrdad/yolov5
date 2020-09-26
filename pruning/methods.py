
import math
import torch
from .modules import PrunableModule, PrunableLinear, PrunableConv2d

class ImportanceHook():
    def __init__(self, module: PrunableModule):
        self.module = module
        self.reset_importance()

    def apply_importance_thr(self, thr_val : float):
        self.module.set_mask( self.compute_mask(thr_val) )

    def get_importance_flat(self):
        return torch.flatten( self.get_importance() ) 

    def compute_mask(self, thr_val : float):
        return torch.ge(self.get_importance(), thr_val).to(torch.int8)       

    def apply_sparsity(self, sparsity : float):
        thr = self.compute_importance_thr(sparsity)
        self.apply_importance_thr(thr)

    def compute_importance_thr(self, sparsity : float) :
        imp_flat = self.get_importance_flat()
        sorted_index = torch.argsort(imp_flat)
        percentile_index = math.floor(torch.numel(sorted_index) * sparsity)
        return imp_flat[ sorted_index[percentile_index] ].item()

    def compute_avg_importance_from_thr(self, thr_val : float):
        mask = self.compute_mask(thr_val)
        return torch.sum(self.get_importance() * mask).item() / torch.sum( mask ).item()

    def compute_avg_importance_from_sprasity(self, sparsity : float) :
        thr = self.compute_importance_thr(sparsity)
        return self.compute_avg_importance_from_thr(thr)    

    def compute_thr_from_avg_importance(self, avg_imp : float):
        imp_flat = self.get_importance_flat()
        sorted_index = torch.argsort(imp_flat, descending=True)
        accum_val = 0
        for i in range(imp_flat.numel()):
            accum_val += imp_flat[sorted_index[i]].item()
            imp = accum_val / (i + 1)
            if imp <= avg_imp:
                return imp_flat[sorted_index[i]].item(), imp, (i + 1) / imp_flat.numel()
        return imp_flat[sorted_index[-1]].item(), imp, 1

    def reset_importance(self):
        pass
    
    def get_importance(self):
        pass
    

class TaylorImportance(ImportanceHook):
    def __init__(self, module: PrunableModule):
        super().__init__(module)
        if isinstance(module, PrunableLinear):
            self.hook = module.register_backward_hook(self.back_hook_fn_linear)
        elif isinstance(module, PrunableConv2d):
            self.hook = module.register_backward_hook(self.back_hook_fn_conv)
          
    def back_hook_fn_linear(self, module, grad_input, grad_output):
        new_imp = torch.abs( module.org_module.weight * module.mask * torch.transpose(grad_input[2], 0, 1) )
        new_imp[new_imp == float("Inf")] = 0
        new_imp[new_imp == float("NaN")] = 0
        self.importance += new_imp
        self.count += 1
    
    def back_hook_fn_conv(self, module, grad_input, grad_output):
        new_imp = torch.abs( module.org_module.weight * module.mask * grad_input[1] )
        new_imp[new_imp == float("Inf")] = 0
        new_imp[new_imp == float("NaN")] = 0
        self.importance += new_imp
        self.count += 1
    
    def reset_importance(self):
        self.importance = torch.zeros_like(self.module.org_module.weight) 
        self.count = 0
    
    def get_importance(self):
        return self.importance if (self.count == 0) else self.importance / self.count

    def close(self):
        self.hook.remove()


class MagnitudeImportance(ImportanceHook):
    def __init__(self, module: PrunableModule):
        super().__init__(module)
             
    def get_importance(self):
        return torch.abs( self.module.org_module.weight * self.module.mask )