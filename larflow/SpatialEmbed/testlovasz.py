import sys
import torch

sys.path.insert(1, "/home/jhwang/ubdl/larflow/larflow/SpatialEmbed/LovaszSoftmax/pytorch")
import lovasz_losses

a = torch.Tensor([0,0,1,1,1,0,0,0.5,1,1,1,0,0,0,1,1,1,1,1,0,0])
b = torch.Tensor([0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,1,1,1,0,0])

print lovasz_losses.lovasz_hinge_flat(a, b)