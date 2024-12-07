import os,sys
import numpy as np

def get_lr_cosine_annealing_with_warmup( epoch, warmup_epochs, lr_warmup, lr_min, lr_max, epoch_period ):

    if epoch < warmup_epochs:
        return lr_warmup

    if epoch >= warmup_epochs and epoch < 2.0*warmup_epochs:
        # linear
        return lr_warmup + (lr_max-lr_warmup)*(epoch-warmup_epochs)/warmup_epochs

    depoch = epoch-2.0*warmup_epochs
    nepochs = int(depoch/epoch_period)
    x = (depoch - float(nepochs*epoch_period))/epoch_period

    lr = lr_min + 0.5*(lr_max-lr_min)*(1 + np.cos( np.pi*x ) )
    return lr
    
