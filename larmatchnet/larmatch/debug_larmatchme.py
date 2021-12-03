import os,sys
import torch
import numpy as np

# debug the setup of larmatch minkowski model
from model.larmatchminkowski import LArMatchMinkowski

# use CPU or GPU
DEVICE = torch.device("cuda")
#DEVICE = torch.device("cpu")

# Just create model
inputshape=(1024,3584)
model = LArMatchMinkowski(ndimensions=2,inputshape=inputshape,input_nfeatures=1,input_nplanes=3).to(DEVICE)
print(model)


# Load some test data and push it through
from larmatch_dataset import larmatchDataset
import MinkowskiEngine as ME

niter = 1
batch_size = 2
test = larmatchDataset( filelist=["temp.root"])
print("NENTRIES: ",len(test))
loader = torch.utils.data.DataLoader(test,batch_size=batch_size,collate_fn=larmatchDataset.collate_fn)


batch = next(iter(loader))
wireplane_sparsetensors = []

for p in range(3):
    print("plane ",p)    
    coord_v = [ torch.from_numpy(data["coord_%d"%(p)]).to(DEVICE) for data in batch]
    feat_v  = [ torch.from_numpy(data["feat_%d"%(p)]).to(DEVICE) for data in batch ]
    print(" len(coord_v): ",len(coord_v))
    
    coords, feats = ME.utils.sparse_collate(coord_v, feat_v,device=DEVICE)
    print(" coords: ",coords.shape)
    print(" feats: ",feats.shape)
    wireplane_sparsetensors.append( ME.SparseTensor(features=feats, coordinates=coords) )

matchtriplet_v = []
for b,data in enumerate(batch):
    matchtriplet_v.append( torch.from_numpy(data["matchtriplet_v"]).to(DEVICE) )
    print("batch ",b," matchtriplets: ",matchtriplet_v[b].shape)
input()

out = model.forward( wireplane_sparsetensors, matchtriplet_v, batch_size )

print("model run: batchsize=",len(out))
for k,arr_v in out.items():
    print("output: ",k)
    for b,arr in enumerate(arr_v):
        print("  batch[%d]"%(b),": ",arr.shape)
input()
