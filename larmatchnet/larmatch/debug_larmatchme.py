import os,sys
import torch
import numpy as np

# debug the setup of larmatch minkowski model
# this tests the individual elements of loading the model, data, and calculating the loss

# model defined here
from larmatch.model.larmatchminkowski import LArMatchMinkowski
from larmatch.utils.larmatchme_engine import get_model, load_config_file
from larmatch.utils.common import prepare_me_sparsetensor

class argstest:
    def __init__(self):
        self.config_file = sys.argv[1]

args = argstest()
config = load_config_file( args, dump_to_stdout=True )
config["DEVICE"] = "cuda:0"
DEVICE = torch.device(config["DEVICE"])
model = get_model( config, dump_model=False ).to(DEVICE)

# create model
inputshape=(1024,3584)

# Load some test data and push it through
from larmatch_dataset import larmatchDataset
from larmatch_mp_dataloader import larmatchMultiProcessDataloader
import MinkowskiEngine as ME

niter = 1
batch_size = 4

dataloader_config = config["TRAIN_DATALOADER_CONFIG"]
dataloader_config["BATCHSIZE"] = batch_size
loader = larmatchMultiProcessDataloader(dataloader_config,
                                        batch_size,
                                        num_workers=1,
                                        prefetch_batches=1,
                                        collate_fn=larmatchDataset.collate_fn)

# get one batch
batch = next(iter(loader))

# data comes back as torch arrays
batch = next(iter(loader))
print(batch.keys())
wireplane_sparsetensors, matchtriplet_v, batch_truth, batch_weight \
            = prepare_me_sparsetensor( batch, DEVICE, verbose=True )
# running the model requires the wire plane image (in sparsetensor format)
# and the 3D-to-2D correspondence
with torch.autograd.detect_anomaly():
    out = model.forward( wireplane_sparsetensors, matchtriplet_v, batch_size )

# what comes out is a class score for each possible 3D spacepoint
print("model run")
for b in range(batch_size):
    print("batch[%d]"%(b))    
    for k in out.keys():
        print("  ",k,": ",out[k][b].shape," ",type(out[k]))
if True:
    print("Ran one forward pass. [enter] to continue.")
    input()

# test the loss
from loss.loss_larmatch_kps import SparseLArMatchKPSLoss
loss_fn = SparseLArMatchKPSLoss( eval_lm=config["RUN_LARMATCH"],
                                 lm_loss_type='focal-soft-bse',
                                 eval_ssnet=config["RUN_SSNET"],
                                 eval_keypoint_label=config["RUN_KPLABEL"],
                                 eval_keypoint_shift=False,
                                 eval_affinity_field=False ).to(DEVICE)

print("=========================")
print("RUN LOSS FUNCTION")
print("=========================")
with torch.autograd.detect_anomaly():    
    loss = loss_fn( out, batch_truth, batch_weight, batch_size, DEVICE, verbose=True )

print("LOSS: ",loss)
print("[ENTER] to continue to backward pass")
input()

print("===============================")
print("CALL BACKWARD")
print("===============================")

# Backward pass
with torch.autograd.detect_anomaly():    
    loss["tot"].backward()
print("completed. [ENTER] to continue to perfect loss tests")
input()

print("================================")
print("PERFECT LOSS TEST")
print("================================")

# make perfect prediction tensors to test loss
batch_perfect = {}
perfect_lm = torch.zeros( out["lm"].shape ).to(DEVICE)
print("perfect_lm: ",perfect_lm.shape)

true_lm  = batch_truth["lm"].gt(0.5)
false_lm = batch_truth["lm"].lt(0.5)
#print(true_lm.shape)
#print(out[ib]["lm"].shape)

# for ib in range(batch_size):
#     # hard label solution
#     perfect_lm[ib,0,true_lm[ib,0,:]] = -9.0
#     perfect_lm[ib,1,true_lm[ib,0,:]] =  9.0    
#     perfect_lm[ib,0,false_lm[ib,0,:]] =  9.0
#     perfect_lm[ib,1,false_lm[ib,0,:]] = -9.0

#     #print(perfect_lm[0,:,:10])
#     #print(batch_truth[ib]["lm"][:10])
#     #print(out[ib]["lm"][0,:,:10])

#     #perfect_lm += 0.5

#     #perfect_ssnet = torch.ones( out["ssnet"][ib].shape ).to(DEVICE)*(-9.0)
#     #for i,x in enumerate(batch_truth[ib]["ssnet"]):
#     #    perfect_ssnet[0,x,i] = 9.0
#     #perfect_ssnet[0, batch_truth[ib]["ssnet"][:], seq[:] ] = 9.0
#     #print("indexed: ",perfect_ssnet.shape)
#     #print(perfect_ssnet[0,:,:10])
#     #print(batch_truth[ib]["ssnet"][:10])
#     #print(seq[:10])

#     #perfect_kp = torch.clone( batch_truth[ib]["kp"] ).unsqueeze(0)
#     #print("perfect keypoint: ",perfect_kp.shape)
    
#     #batch_perfect.append({"lm":perfect_lm,"ssnet":perfect_ssnet,"kp":perfect_kp})
#     #batch_perfect_truth.append({"lm":perfect_lm})

# soft label solution
p = torch.clamp(batch_truth["lm"], 0.0001,0.99)
print("p: ",p.shape)
print(p[0,0,0:10])

x = 0.5*(torch.log(p)-torch.log(1-p))
print("x: ",x.shape)
print(x[0,0,0:10])

perfect_lm[:,1,:] = x[:,0,:]
perfect_lm[:,0,:] = -x[:,0,:]

sm_perfect = torch.softmax(perfect_lm, dim=1)
print("sm_perfect: ",sm_perfect.shape)
print(sm_perfect[0,1,:10])
sm_diff = torch.abs(sm_perfect[:,1,:]-p[:,0,:]).sum()
print("sm_diff: ",sm_diff)
    
batch_perfect = {"lm":perfect_lm}

with torch.autograd.detect_anomaly():        
    perfect_loss = loss_fn( batch_perfect, batch_truth, batch_weight, batch_size, DEVICE, verbose=True )
print("PERFECT LOSS: ",perfect_loss)
#perfect_loss["tot"].backward()
