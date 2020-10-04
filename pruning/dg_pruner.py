
import pruning as prn
import torch.nn as nn

class DG_Pruner():
    def __init__(self):
        self.hooks = {}
        self.pruners = {}

    @staticmethod
    def swap_prunable_modules(model : nn.Module):
        return prn.swap_prunable_modules(model)
    
    def add_custom_pruning(self, model : nn.Module, custom_class, parent_name : str = ''):
        self.hooks = prn.add_custom_pruning(model, custom_class, parent_name)
        return self.hooks

    def pruners_from_file(self, file_path : str):
        self.pruners = prn.pruners_from_file(file_path)
        return self.pruners

    def reset_importance(self):
        for h in self.hooks.values():
            h.reset_importance()

    def reset_growth(self):
        for h in self.hooks.values():
            h.reset_growth()

    def apply_pruning_step(self, epoch: float):
        prn.apply_pruning_step(epoch, self.pruners, self.hooks)

    def dump_importance_stat(self, output_dir : str = '', epoch : int = 0):
        prn.dump_importance_stat(self.hooks, output_dir, epoch)

    def dump_growth_stat(self, output_dir : str = '', epoch : int = 0):
        prn.dump_growth_stat(self.hooks, output_dir, epoch)

    @staticmethod
    def dump_sparsity_stat(model : nn.Module, output_dir : str = '', epoch : int = 0):
        prn.dump_sparsity_stat(model, output_dir, epoch)

    @staticmethod
    def dump_json(data: dict, file_name : str, output_dir : str = ''):
        prn.dump_json(data, file_name, output_dirs)

    def compute_sparsity_table_from_layer(self, layer_name : str, sparsity : float):
        return prn.compute_sparsity_table_from_layer(self.hooks, layer_name, sparsity)

    def prune_n_reset(self, epoch : float):
        self.apply_pruning_step(epoch)
        self.reset_importance()        
        self.reset_growth()

    def num_iter_per_update(self, num_batch_per_epoch : int) -> int:
        return round( self.pruners[list(self.pruners)[0]].opt['frequency'] * num_batch_per_epoch )