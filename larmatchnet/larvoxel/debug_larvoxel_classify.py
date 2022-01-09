import os,sys
import torch
import numpy as np

# debug the setup of larmatch minkowski model
from model.larvoxelclassifier import LArVoxelClassifier

# use CPU or GPU
DEVICE = torch.device("cuda")
#DEVICE = torch.device("cpu")

# Just create model
model = LArVoxelClassifier()
#print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters: ",pytorch_total_params/1.0e6," M")
model = model.to(DEVICE)

# Load some test data and push it through
from larvoxelclass_dataset import larvoxelClassDataset
import MinkowskiEngine as ME

niter = 1
batch_size = 1
test = larvoxelClassDataset( filelist=["prepdata/testdata/testdata.root"])
print("NENTRIES: ",len(test))
loader = torch.utils.data.DataLoader(test,batch_size=batch_size,collate_fn=larvoxelClassDataset.collate_fn)
batch = next(iter(loader))

data = batch[0]
print("data blob keys: ",data.keys())
print("coord: ",data["coord"].shape,"  feat: ",data["feat"].shape)

coord_v = [ torch.from_numpy( data["coord"] ).to(DEVICE) ]
feat_v  = [ torch.from_numpy( data["feat"] ).to(DEVICE) ]
coords, feats = ME.utils.sparse_collate(coord_v,feat_v)
sp = ME.TensorField(features=feats,coordinates=coords)
print("input made")
#print(sp)

with torch.no_grad():
    out = model(sp)
print("output returned")
print(out)
print(out.F)

