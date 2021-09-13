import os,sys,time
import numpy as np
import torch
import torch.optim
import matplotlib.pyplot as plt
from larennet_dataset import larennetDataset
from larmatchvoxel import LArMatchVoxel
import MinkowskiEngine as ME

device = torch.device("cuda")
#device = torch.device("cpu")
LR = 1.0e-3
NITERS = 100
batch_size = 1
focal_loss_gamma = 2

# model
model = LArMatchVoxel().to(device)
smax = torch.nn.Softmax( dim=1 )
print(model)
if False:
    sys.exit(0)
    
test = larennetDataset( filelist=["larmatchtriplet_ana_trainingdata_testfile.root"], voxelize=True, voxelsize_cm=0.5 )
print("NENTRIES: ",len(test))

loader = torch.utils.data.DataLoader(test,batch_size=batch_size,collate_fn=larennetDataset.collate_fn)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=LR,
                             weight_decay=1.0e-4)

#model.init_weights()

if True:
    # For single entry testing
    dt_io = time.time()
    data = next(iter(loader))
    print(data[0].keys())

    coordshape = data[0]["voxcoord"].shape
    coord = torch.from_numpy( data[0]["voxcoord"] ).int().to(device)
    feat  = torch.from_numpy( np.clip( data[0]["voxfeat"]/40.0, 0, 10.0 ) ).to(device)
    truth = torch.from_numpy( data[0]["truetriplet_t"] ).to(device)
    print("coord: ",coord.shape," ",coord[:10])
    print("feat: ",feat.shape," ",feat[:10])
    print("truth: ",truth.shape)
    dt_io = time.time() - dt_io
    print("IO: %.3f secs"%(dt_io))

for iiter in range(NITERS):
    optimizer.zero_grad()    

    if False:
        # For multi-entry sample testing
        dt_io = time.time()
        data = next(iter(loader))
        #print(data[0].keys())
        
        coord = torch.from_numpy( data[0]["voxcoord"] ).int().to(device)
        feat  = torch.from_numpy( data[0]["voxfeat"] ).to(device)
        truth = torch.from_numpy( data[0]["truetriplet_t"] ).to(device)
        print("coord: ",coord.shape," ",coord[:10])
        print("feat: ",feat.shape," ",feat[:10])
        print("truth: ",truth.shape)
        print("[enter] to continue")
        dt_io = time.time() - dt_io
        print("IO: %.3f secs"%(dt_io))
    if False:
        input()

    coords, feats = ME.utils.sparse_collate(coords=[coord], feats=[feat])
    xinput = ME.SparseTensor( features=feats, coordinates=coords.to(device) )    
    out = model(xinput)
    print("out: ",out.shape)
    print("out.F: ",out.F[:10])

    #print("out: ",out.shape,out.requires_grad)
    #print("out raw: ",out[:10,:])
    out = smax(out.F)
    print("softmax(out): ",out[:10,:])

    # focal loss
    fmatchlabel = truth.type(torch.float).requires_grad_(False)
    #print("tru: ",tru[:10])

    p_t = fmatchlabel*(out[:,1]+1.0e-6) + (1-fmatchlabel)*(out[:,0]+1.0e-6) # p if y==1; 1-p if y==0            
    print("p_t: ",p_t[:10]," ",p_t.requires_grad)

    loss = (-torch.log( p_t )*torch.pow( 1-p_t, focal_loss_gamma )).mean()
    #loss = (-torch.log( p_t )).mean()    
    print("loss: ",loss," ",loss.requires_grad)

    loss.backward()
    optimizer.step()
    
    #for name,p in model.named_parameters():
    #    print(name,": ",p.shape,p.grad)

    with torch.no_grad():
        acc = ((out[:,1]>0.5).type(torch.long).eq( truth )).type(torch.float).mean()
        print("acc: ",acc.item())

    if False:
        print("[enter] to quit.")
        input()

print("DONE")
