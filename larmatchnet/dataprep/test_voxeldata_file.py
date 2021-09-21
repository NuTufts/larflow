import os,sys
sys.path.append("../")

import torch
import numpy as np

# SETUP DATA LOADERS
from larvoxel_dataset import larvoxelDataset

filepaths = ["test.root"]
batch_size = 1
niter = 100

dataset = larvoxelDataset(filelist=filepaths,is_voxeldata=True)
nentries = len(dataset)
print("Num Entries: ",nentries)

loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,collate_fn=larvoxelDataset.collate_fn)

for iiter in range(niter):
    print("====================================")
    batch = next(iter(loader))
    for ib,data in enumerate(batch):
        print("ITER[%d]:BATCH[%d]"%(iiter,ib))
        print(" keys: ",data.keys())
        for name,d in data.items():
            if type(d) is np.ndarray:
                print("  ",name,": ",d.shape)
            else:
                print("  ",name,": ",type(d))
