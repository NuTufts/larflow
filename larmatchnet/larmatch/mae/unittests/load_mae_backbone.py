import os,sys
import torch
import yaml
import larmatch
from larmatch.mae.models import SpacepointMAE

ub_production_larmatch = "larmatch_ubprod_ckpt78k_slimmed.pt"
config_path = "config_test_mae_backbone.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model = SpacepointMAE(config)

print(model)
    
