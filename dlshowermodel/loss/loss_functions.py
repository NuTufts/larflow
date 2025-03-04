import os,sys

from .bce import WeightedBCELoss
from .focal import WeightedFocalLoss

def get_loss_function( train_cfg ):
    cfg = train_cfg
    if "Loss" in cfg:
        cfg = cfg['Loss']

    name = cfg['name']
    loss_cfg = cfg['params']

    print("-------------------------------------")
    print("Loading Loss function: ",name)
    print("params:")
    for k,v in loss_cfg.items():
        print("  ",k,": ",v)

    if name=="WeightedBCELoss":
        return WeightedBCELoss.from_config( loss_cfg )
    elif name=="WeightedFocalLoss":
        return WeightedFocalLoss.from_config( loss_cfg )
    else:
        raise ValueError(f"No weight function found matching name ({name}) in config.",)

    print("Never gets here.")
    return None

