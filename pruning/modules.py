
import torch
import torch.nn as nn
import torch.nn.functional as F

class FakePrune(torch.autograd.Function):
    r""" Simulate the prune function in training time.
    The output of this function is given by

    x_out = mask * x_in

    """
    @staticmethod
    def forward(ctx, weight, mask):
        ctx.save_for_backward(weight, mask)
        return mask * weight
    
    @staticmethod
    def backward(ctx, grad_output):
        weight, mask = ctx.saved_tensors
        grad_weight = grad_mask = None
        
        print(grad_output.shape)
        print(mask.shape)

        if ctx.needs_input_grad[0]:
            grad_weight = grad_output * mask
        if ctx.needs_input_grad[1]:
            grad_mask = grad_output * weight

        return grad_weight, grad_mask

class PrunableModule(nn.Module):
    r"""
    A module attached with FakePrune modules for prune aware training
    """
    def __init__(self, org_module):
        super(PrunableModule, self).__init__()

        if not ( (isinstance(org_module, nn.Conv2d) and org_module.groups == 1) or isinstance(org_module, nn.Linear) ):
            raise ValueError('Prunable Module of {} is not allowed'.type(org_module))

        self.org_module = org_module
        self.register_buffer('mask', torch.ones_like(self.org_module.weight, dtype=torch.int8))

    def set_mask(self, new_mask):
        self.mask = new_mask

    def forward(self, input):
        pass

class PrunableConv2d(PrunableModule):
    r"""
    A conv2d module attached with FakePrune modules for prune aware training
    """
    def __init__(self, org_module: nn.Conv2d):
        super().__init__(org_module)

    def forward(self, input):
        return self.org_module._conv_forward( input, FakePrune.apply(self.org_module.weight, self.mask) )

class PrunableLinear(PrunableModule):
    r"""
    A Linear module attached with FakePrune modules for prune aware training
    """
    def __init__(self, org_module: nn.Linear):
        super().__init__(org_module)

    def forward(self, input):
        return F.linear(input, FakePrune.apply(self.org_module.weight, self.mask), self.org_module.bias)