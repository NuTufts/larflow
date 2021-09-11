import os,sys,time
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace
import matplotlib.pyplot as plt
from larennet_dataset import larennetDataset
from larennet import LArEnnet

device = torch.device("cuda")
#device = torch.device("cpu")

# model
model = LArEnnet(device).to(device)
print(model)

batch_size = 1
test = larennetDataset( filelist=["larmatchtriplet_ana_trainingdata_testfile.root"])
print("NENTRIES: ",len(test))

loader = torch.utils.data.DataLoader(test,batch_size=batch_size,collate_fn=larennetDataset.collate_fn)
data = next(iter(loader))
print("spacepoints: ",data[0]["spacepoint_t"].shape)
spt = data[0]["spacepoint_t"]
# reduce
spt = spt[ spt[:,2]<256 ]

spt = spt[ spt[:,0]>0 ]
spt = spt[ spt[:,0]<256 ]
pos = torch.from_numpy( spt[:,0:3] )
pix = torch.from_numpy( spt[:,3:] )
print("pos: ",pos.shape)
print("pix: ",pix.shape)
print("[enter] to continue")
input()

print("here goes nothing!")
out = model(pos,pix)
print("out: ",out.shape)
print("[enter] to quit.")
input()
