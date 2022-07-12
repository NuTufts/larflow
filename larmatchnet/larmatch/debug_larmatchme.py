import os,sys
import torch
import numpy as np

# debug the setup of larmatch minkowski model
# this tests the individual elements of loading the model, data, and calculating the loss

# model defined here
from larmatch.model.larmatchminkowski import LArMatchMinkowski
from larmatch.utils.larmatchme_engine import get_model, load_config_file
from larmatch.utils.common import prepare_me_sparsetensor
from larmatch.utils.perfect_pred_maker import make_perfect_ssnet,make_perfect_larmatch,make_perfect_kplabel

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
batch_size = 1

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
            = prepare_me_sparsetensor( batch, DEVICE, make_batch_tensor=False, verbose=True )
print("made batch truth:")
for k in batch_truth:
    print(k,": ",type(batch_truth[k]))
    if type(batch_truth[k]) is list:
        print(type(batch_truth[k][0]))
        if type(batch_truth[k][0]) is torch.Tensor:
            for x in batch_truth[k]:
                print("  ",x.shape)
print("[ENTER] to continue")
input()

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
    loss = loss_fn( out, batch_truth, batch_weight, batch_size, DEVICE, verbose=True, whole_batch=False )

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
if config["RUN_LARMATCH"]:
    perfect_lm = make_perfect_larmatch( batch_truth["lm"], DEVICE )
    batch_perfect["lm"] = perfect_lm

if config["RUN_SSNET"]:
    ssnet_perfect = make_perfect_ssnet( batch_truth["ssnet"], DEVICE, logitval=3.0 )
    batch_perfect["ssnet"] = ssnet_perfect
    print("ssnet weight sum: ",batch_weight["ssnet"].sum())

if config["RUN_KPLABEL"]:
    kp_perfect = make_perfect_kplabel( batch_truth["kp"], DEVICE )
    batch_perfect["kp"] = kp_perfect

with torch.autograd.detect_anomaly():        
    perfect_loss = loss_fn( batch_perfect, batch_truth, batch_weight, batch_size, DEVICE, verbose=True )
print("PERFECT LOSS: ",perfect_loss)
#perfect_loss["tot"].backward()
