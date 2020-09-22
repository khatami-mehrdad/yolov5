
import math

class PrunerBase():
    def __init__(self, vars : dict):
        self.init(vars['starting_epoch'], vars['frequency'], vars['ending_epoch'], vars['weights'])

    def init(self, starting_epoch: int, frequency: int, ending_epoch: int, weight_dict: dict):
        self.starting_epoch = starting_epoch
        self.ending_epoch = ending_epoch
        self.frequency = frequency
        self.num_stages = (self.ending_epoch - self.starting_epoch) // self.frequency + 1        
        self.weight_dict = weight_dict
        self.curr_sparsity = {key:0.0 for key in self.weight_dict.keys()}
        self.stage_cnt = 0

    def compute_stage_cnt(self, epoch):
        self.stage_cnt = 1 + (epoch - self.starting_epoch) // self.frequency
        if (self.stage_cnt < 0):
            self.stage_cnt = 0
        elif (self.stage_cnt > self.num_stages):
            self.stage_cnt = self.num_stages

    def step(self, final_sparsity: float):
        pass

    def get_curr_spasity(self):
        return self.curr_sparsity

    def step_all(self, epoch: int):
        self.compute_stage_cnt(epoch)
        for layer_name, final_sparsity in self.weight_dict.items():
            self.curr_sparsity[layer_name] = self.step(final_sparsity)
        return self.curr_sparsity

class AGP(PrunerBase):
    r"""
    sparsity_val = end - (end - start) * ( 1 - (n / num_stages) )^3 )
    """
    def __init__(self, vars : dict):
        super().__init__(vars)
        self.T = vars['T'] if 'T' in vars.keys() else 3

    def step(self, final_sparsity: float):
        val =  final_sparsity - (final_sparsity - 1.0) * ( (1.0 - (self.stage_cnt / self.num_stages)) ** self.T )
        return val

def pruner_factory(classname):
    cls = globals()[classname]
    return cls