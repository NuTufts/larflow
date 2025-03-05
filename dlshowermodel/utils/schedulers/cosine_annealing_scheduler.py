import numpy as np
import math
from ..lr_scheduler import scheduler_registry

@scheduler_registry.register()
class CosineAnnealingWithWarmup:
    NAME="CosineAnnealingWithWarmup"

    @classmethod
    def from_config(cls,scheduler_config):
        # so lazy
        return CosineAnnealingWithWarmup(**scheduler_config)

    def __init__( self, warmup_epochs=1, lr_warmup=1.0e-6, 
                lr_min=1.0e-4, lr_max=5.0e-3, 
                epoch_period=10, iters_per_epoch=100,
                epoch_offset=0, iter_offset=0,
                linear_ramp_epochs=0 ):
        self.warmup_epochs = warmup_epochs
        self.lr_warmup = lr_warmup
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.epoch_period = epoch_period
        self.iters_per_epoch = iters_per_epoch
        self.epoch_offset = epoch_offset
        self.iter_offset = iter_offset
        self.linear_ramp_epochs = linear_ramp_epochs

    def get_lr( self, iiter  ):

        # Adjust for offsets
        x = iiter - self.iter_offset
        epoch_equiv = max( float(x) / float(self.iters_per_epoch) - float(self.epoch_offset), 0.0 )
        
        
        # Are we in the warm-up phase?
        if epoch_equiv < self.warmup_epochs:
            return self.lr_warmup

        # Remove offset from warmup epochs
        epoch_equiv -= self.warmup_epochs
        epoch_equiv = max(0,epoch_equiv)

        # Are we in the linear ramp phase?
        if epoch_equiv < self.linear_ramp_epochs:
            # Calculate linear interpolation between lr_warmup and lr_max
            fraction = epoch_equiv / self.linear_ramp_epochs
            return self.lr_warmup + (self.lr_max - self.lr_warmup) * fraction

        # Remove linear ramp offset
        epoch_equiv -= self.linear_ramp_epochs
        epoch_equiv = max(0,epoch_equiv)

        # Get current phase within the cosine cycle
        icycle_epoch = math.floor( epoch_equiv / float(self.epoch_period) )
        fcycle_epoch = epoch_equiv - icycle_epoch*float(self.epoch_period)
        phi = np.pi * fcycle_epoch / float(self.epoch_period)
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(phi))

        return lr

