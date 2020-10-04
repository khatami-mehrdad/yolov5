
import math
import torch
from .modules import PrunableModule, PrunableLinear, PrunableConv2d

class ImportanceHook():
    def __init__(self, module: PrunableModule):
        self.module = module
        self.reset_importance()
        self.reset_growth()

    def set_mask(self, mask):
        self.module.set_mask( mask )

    def apply_importance_thr(self, thr_val : float):
        self.set_mask( self.compute_imp_mask_thr(thr_val) )

    def apply_growth_thr(self, thr_val : float): # new mask is added to the old : or_mask
        growth_mask = self.compute_growth_mask_thr(thr_val)
        self.module.or_mask( growth_mask )
        with torch.no_grad():   # setting the weights of new growth to 0
            self.module.org_module.weight[growth_mask == 1] = 0 

    def get_importance_flat(self):
        return torch.flatten( self.get_importance() ) 

    def get_growth_flat(self):
        return torch.flatten( self.get_growth() ) 

    def compute_imp_mask_thr(self, thr_val : float):
        return torch.ge(self.get_importance(), thr_val).to(torch.int8)       

    def compute_growth_mask_thr(self, thr_val : float):
        return torch.ge(self.get_growth(), thr_val).to(torch.int8)    

    def compute_imp_mask(self, sparsity : float):
        thr = self.compute_importance_thr(sparsity)
        return self.compute_imp_mask_thr(thr)

    def compute_growth_mask(self, growth : float):
        thr = self.compute_growth_thr(growth)
        return self.compute_growth_mask_thr(thr)

    def apply_sparsity(self, sparsity : float):
        thr = self.compute_importance_thr(sparsity)
        self.apply_importance_thr(thr)

    def apply_growth(self, growth : float):
        thr = self.compute_growth_thr(growth)
        self.apply_growth_thr(thr)

    def compute_importance_thr(self, sparsity : float) :
        imp_flat = self.get_importance_flat()
        sorted_index = torch.argsort(imp_flat, descending=False)
        percentile_index = math.floor(torch.numel(sorted_index) * sparsity)
        return imp_flat[ sorted_index[percentile_index] ].item()

    def compute_growth_thr(self, growth_perc : float) :
        grow_flat = self.get_growth_flat()
        sorted_index = torch.argsort(grow_flat, descending=True)
        percentile_index = math.ceil(torch.numel(sorted_index) * growth_perc)
        return grow_flat[ sorted_index[percentile_index] ].item()

    def reset_importance(self):
        pass

    def reset_growth(self):
        pass   

    def get_importance(self):
        pass

    def get_growth(self):
        pass
 
    # avg: all layer functions
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


class TaylorImportance(ImportanceHook):
    def __init__(self, module: PrunableModule):
        super().__init__(module)
        self.hook = module.org_module.weight.register_hook(self.back_hook)
          
    def back_hook(self, grad):
        new_imp = torch.abs( self.module.org_module.weight * self.module.mask * grad )
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

class RigLImportance(ImportanceHook):
    def __init__(self, module: PrunableModule):
        super().__init__(module)
        self.hook = module.org_module.weight.register_hook(self.back_hook)
        self.ema_alpha = 0.9
          
    def back_hook(self, grad):
        # new_growth = grad
        # new_growth[new_growth == float("Inf")] = 0
        # new_growth[new_growth == float("NaN")] = 0
        # self.growth += new_growth
        self.growth = grad if (self.count == 0) else ( self.growth * (1 - self.ema_alpha) + grad * self.ema_alpha )
        self.count += 1

    
    def reset_growth(self):
        self.growth = torch.zeros_like(self.module.org_module.weight) 
        self.count = 0

    def get_importance(self):
        return torch.abs( self.module.org_module.weight * self.module.mask )

    def get_growth(self):
        # return torch.abs( self.module.org_module.weight if (self.count == 0) else (self.growth / self.count) ) * (self.module.mask == 0) 
        return torch.abs( self.module.org_module.weight if (self.count == 0) else self.growth ) * (self.module.mask == 0) 

    def get_growth_no_mask(self):
        return torch.abs( self.module.org_module.weight if (self.count == 0) else self.growth )

    def close(self):
        self.hook.remove()