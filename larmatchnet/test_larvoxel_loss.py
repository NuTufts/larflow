import os,sys

import numpy as np
import torch
from larvoxel_dataset import larvoxelDataset
from loss_larvoxel import LArVoxelLoss
import MinkowskiEngine as ME

device=torch.device("cpu")

criterion = LArVoxelLoss( eval_ssnet=True,
                          eval_keypoint_label=True )

filelist = ["larmatchtriplet_ana_trainingdata_testfile.root"]

train_dataset = larvoxelDataset( filelist=filelist, random_access=True )
TRAIN_NENTRIES = len(train_dataset)
print("TRAIN DATASET NENTRIES: ",TRAIN_NENTRIES," = 1 epoch")
dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=1,collate_fn=larvoxelDataset.collate_fn)

data = next(iter(dataloader))[0]

coordshape = data["voxcoord"].shape
coord   = torch.from_numpy( data["voxcoord"] ).int().to(device)
feat    = torch.from_numpy( np.clip( data["voxfeat"]/40.0, 0, 10.0 ) ).to(device)
truth   = torch.from_numpy( data["truetriplet_t"] ).to(device)
ssnet   = torch.from_numpy( data["ssnet_labels"] ).to(device)
kplabel = torch.from_numpy( data["kplabel"] ).to(device)    
    
match_label_t  = torch.from_numpy( data["voxlabel"] ).to(device)
ssnet_label_t  = torch.from_numpy( data["ssnet_labels"] ).to(device).squeeze().unsqueeze(0)
kp_label_t     = torch.from_numpy( np.transpose(data["kplabel"],(1,0)) ).to(device).unsqueeze(0)
match_weight_t = torch.from_numpy( data["voxlmweight"] ).to(device)
kp_weight_t    = None
kpshift_t      = None
paf_label_t    = None
truematch_idx_t = None
print("match label: ",match_label_t.shape)
print("ssnet label: ",ssnet_label_t.shape)
print("kp label: ",kp_label_t.shape)
print("voxlmweight: ",match_weight_t.shape)

# turn voxel weighting down:
match_weight_t = None

# make perfect prediction
perfect = torch.zeros( (match_label_t.shape[0],2) )
perfect_ssnet = torch.zeros( (1,7,match_label_t.shape[0]) )
perfect_kplabel = torch.zeros( (1,6,match_label_t.shape[0]) )
# PERFECT
if False:
    perfect[:,0][ match_label_t.eq(1) ] = -100.0
    perfect[:,1][ match_label_t.eq(1) ] = 100.0
    perfect[:,0][ match_label_t.eq(0) ] = 100.0
    perfect[:,1][ match_label_t.eq(0) ] = -100.0

    for c in range(7):
        perfect_ssnet[0,c,:][ ssnet_label_t[0,:]==c ] = 1.0
        perfect_ssnet[0,c,:][ ssnet_label_t[0,:]!=c ] = -1.0

    perfect_kplabel = kp_label_t
    
# ABSOLUTELY WRONG
if False:
    perfect[:,0][ match_label_t.eq(1) ] = 100.0
    perfect[:,1][ match_label_t.eq(1) ] = -100.0
    perfect[:,0][ match_label_t.eq(0) ] = -100.0
    perfect[:,1][ match_label_t.eq(0) ] = 100.0

    for c in range(7):
        perfect_ssnet[0,c,:][ ssnet_label_t[0,:]==c ] = -100.0
        perfect_ssnet[0,c,:][ ssnet_label_t[0,:]!=c ] = (c+1)*10.0

    perfect_kplabel = 1-kp_label_t
    
# UNSURE
if True:
    perfect[:,0][ match_label_t.eq(1) ] = 0.0
    perfect[:,1][ match_label_t.eq(1) ] = 0.0
    perfect[:,0][ match_label_t.eq(0) ] = 0.0
    perfect[:,1][ match_label_t.eq(0) ] = 0.0
    
# ALL YES
if False:
    perfect[:,0] = -100
    perfect[:,1] = 100.0


print()
print(perfect[0,-50:])
print(match_label_t[-50:],match_label_t.sum())
match_pred_t = ME.SparseTensor( features=perfect, coordinates=coord )    


totloss,larmatch_loss,ssnet_loss,kp_loss,paf_loss = criterion( match_pred_t,   perfect_ssnet, perfect_kplabel, None, None,
                                                               match_label_t,  ssnet_label_t, kp_label_t, kpshift_t, paf_label_t,
                                                               match_weight_t, None, None, None,
                                                               verbose=True )
print(totloss)
print(larmatch_loss)
print(ssnet_loss)
print(kp_loss)
print(paf_loss)

