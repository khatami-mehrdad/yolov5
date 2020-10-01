

import torch
import torch.nn as nn
import numpy as np
import math
import json
import os

from .modules import PrunableConv2d, PrunableLinear, PrunableModule
from .methods import ImportanceHook
from .pruners import pruner_factory

def get_prefix(prefix):
    return prefix if prefix == "" else prefix + '.'

def reset_importance(hooks : dict):
    for h in hooks.values():
        h.reset_importance()

def swap_prunable_modules(model : nn.Module):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Conv2d) and child.groups == 1:
            setattr(model, child_name, PrunableConv2d(child))
        elif isinstance(child, nn.Linear):
            setattr(model, child_name, PrunableLinear(child))
        else:
            swap_prunable_modules(child)

    return model

def add_custom_pruning(model : nn.Module, custom_class, parent_name : str = ''):
    hook_dict = {}
    for child_name, child in model.named_children():
        module_name = get_prefix(parent_name) + child_name
        if isinstance(child, PrunableModule):
            hook_dict[module_name] = custom_class(child)
        else:
            hook_dict.update( add_custom_pruning(child, custom_class, module_name) )
    return hook_dict

#############################################
## Creating Pruners from File

def pruners_from_file(file_path : str):
    pruner_obj_dict = {}
    pruner_dict = read_json_from_file(file_path)
    for name, pruner_property in pruner_dict.items():
        pruner_obj_dict[name] = create_pruner(pruner_property)
    return pruner_obj_dict
    
def read_json_from_file(file_path : str):
    with open(file_path) as f:
        data = json.load(f)
    return data

def create_pruner(pruner_property):
    cls_ = pruner_factory(pruner_property["class"])
    return cls_(pruner_property)
#############################################
## Applying Sparsity
def apply_pruning_step(epoch: float, pruners: dict, hooks: dict):
    for pruner in pruners.values():
        curr_sparsity, curr_grow = pruner.step_all(epoch)
        if ( (pruner.stage_cnt >= 0) and (pruner.stage_cnt <= pruner.num_stages) ):
            for name, sparsity in curr_sparsity.items():
                if (sparsity > 0):
                    hooks[name].apply_sparsity( sparsity )
                if (curr_grow[name] > 0):
                    hooks[name].apply_growth( curr_grow[name] )


#############################################
## Computing Imortance & Applying Sparsity mask

def get_global_importance_flat(hooks : dict):
    imp_tot = None
    for h in hooks.values():
        imp_flat = h.get_importance_flat() 
        imp_tot = imp_flat if (imp_tot == None) else torch.cat( (imp_tot, imp_flat), dim=0)
    return imp_tot

def apply_global_importance_thr(hooks : dict, thr_val : float):
    for h in hooks.values():
        h.apply_importance_thr(thr_val) 

def compute_global_importance_thr(hooks : dict, sparsity : float):
    imp_flat = get_global_importance_flat(hooks)
    sorted_index = torch.argsort(imp_flat)
    percentile_index = math.floor(torch.numel(sorted_index) * sparsity)
    return imp_flat[ sorted_index[percentile_index] ].item()

def apply_global_sparsity(hooks : dict, sparsity : float):
    thr = compute_global_importance_thr(hooks, sparsity)
    apply_global_importance_thr(hooks, thr)

def compute_sparsity_table_from_layer(hooks : dict, layer_name : str, sparsity : float):
    avg_imp = compute_avg_importance_from_sprasity(hooks[layer_name], sparsity)
    thr_dict = {}
    imp_dict = {}
    sparsity_dict = {}
    for name, h in hooks.items():
        thr_dict[name], imp_dict[name], sparsity_dict[name] = h.compute_thr_from_avg_importance(avg_imp)
    return thr_dict, imp_dict, sparsity_dict
################################################################
## Stat: Get & Dump

def get_importance_stat(hooks : dict):
    imp_dict = {}
    for name, h in hooks.items():
        imp_dict[name] = torch.sum(h.get_importance() * h.module.mask).item() / torch.sum(h.module.mask).item()
    return imp_dict 

def get_sparsity_stat(model : nn.Module, parent_name : str = ''):
    sparsity_dict = {}
    for child_name, child in model.named_children():
        module_name = get_prefix(parent_name) + child_name
        if isinstance(child, PrunableModule):
            sparsity_dict[module_name] = 1 - ( torch.sum(child.mask).item() / child.mask.numel() )
        else:
            sparsity_dict.update( get_sparsity_stat(child, module_name) )
    return sparsity_dict

def dump_importance_stat(hooks : dict, output_dir : str = '', epoch : int = 0):
    imp_dict = get_importance_stat(hooks)
    dump_json(imp_dict, 'importance_report_epoch{}.json'.format(epoch), output_dir)

def dump_sparsity_stat(model : nn.Module, output_dir : str = '', epoch : int = 0):
    sparsity_dict = get_sparsity_stat(model)
    dump_json(sparsity_dict, 'sparsity_report_epoch{}.json'.format(epoch), output_dir)
    
def dump_json(data: dict, file_name : str, output_dir : str = ''):
    with open(os.path.join(output_dir, file_name), 'w') as fp:
        fp.write(json.dumps(data, indent=1))