import os,sys
import torch
import numpy as np

# debug the setup of larmatch minkowski model
# this tests the individual elements of loading the model, data, and calculating the loss

niter = 10
batch_size = 1

# model defined here
from model.larmatchscn import LArMatch

# utils
import utils.larmatchscn_engine as engine

config = engine.load_config_file("config/config_larmatchscn.test.yaml")

DEVICE = torch.device(config["DEVICE"])

model = engine.get_gen2_larmatch_model(dump_model=config["PRINT_MODEL"])

engine.load_gen2_scn_model_weights( config["WEIGHT_FILE"], model )

model.to(DEVICE)

# Load some test data and push it through
from larmatch_dataset import larmatchDataset

test = larmatchDataset( filelist=["testdata/temp.root"], load_truth=True, normalize_inputs=False )
print("NENTRIES: ",len(test))
loader = torch.utils.data.DataLoader(test,batch_size=batch_size,collate_fn=larmatchDataset.collate_fn)

# get one batch
batch = next(iter(loader))

# data comes back as numpy arrays.
batch_data = engine.prepare_gen2_sparse_tensor( batch, device=DEVICE )
for ib in range(len(batch_data)):
    print("batch[%d]"%(ib))
    for p in range(3):
        print("  plane[%d]"%(p)," coord: ",batch_data[ib][0][p].shape,"  feat: ",batch_data[ib][1][p].shape)

# we also need the metadata associating possible 3d spacepoints
# to the wire image location they project to
matchtriplet_v = []
for b,data in enumerate(batch):
    matchtriplet_v.append( torch.from_numpy(data["matchtriplet_v"]).to(DEVICE) )
    print("batch ",b," matchtriplets: ",matchtriplet_v[b].shape)
if True:
    print("Made batch. [enter] to run.")
    input()

# running the model requires the wire plane image (in sparsetensor format)
# and the 3D-to-2D correspondence
for b,data in enumerate(batch_data):
    print("FORWARD BATCH[%d]"%(b),"===============================")
    coord_v = data[0]
    feat_v  = data[1]
    triplets = matchtriplet_v[b]
    with torch.autograd.detect_anomaly():
        pred = model.forward( coord_v, feat_v, triplets, triplets.shape[0], DEVICE, verbose=True )

    sys.exit(0)

# what comes out is a class score for each possible 3D spacepoint
print("model run: batchsize=",len(out))
for b,arr_v in enumerate(out):
    print("batch[%d]"%(b))
    for k,arr in arr_v.items():
        print("  ",k,": ",arr.shape," ",type(arr))
if True:
    print("Ran one forward pass. [enter] to continue.")
    input()

# test the loss
from loss.loss_larmatch_kps import SparseLArMatchKPSLoss
loss_fn = SparseLArMatchKPSLoss( eval_ssnet=True,
                                 eval_keypoint_label=True,
                                 eval_keypoint_shift=False,
                                 eval_affinity_field=False )

# load the truth data
print("== TRUTH DATA ===")
batch_truth = []
batch_weight = []
# larmatch
for b,data in enumerate(batch):
    #print("batch [",b,"]")
    
    lm_truth_t = torch.from_numpy(data["larmatch_truth"]).to(DEVICE)
    lm_weight_t = torch.from_numpy(data["larmatch_weight"]).to(DEVICE)
    #print("  truth: ",lm_truth_t.shape)
    #print("  weight: ",lm_weight_t.shape)
 
    ssnet_truth_t  = torch.from_numpy(data["ssnet_truth"]).to(DEVICE)
    ssnet_weight_t = torch.from_numpy(data["ssnet_weight"]).to(DEVICE)
    ssnet_max = ssnet_truth_t.max()
    for i in range(ssnet_max):
        print("  ssnet class[",i,"]: ",ssnet_truth_t.eq(i).sum())

    kp_truth_t  = torch.from_numpy(data["keypoint_truth"]).to(DEVICE)
    kp_weight_t = torch.from_numpy(data["keypoint_weight"]).to(DEVICE)
        
    truth_data = {"lm":lm_truth_t,"ssnet":ssnet_truth_t,"kp":kp_truth_t}
    weight_data = {"lm":lm_weight_t,"ssnet":ssnet_weight_t,"kp":kp_weight_t}
    
    batch_truth.append( truth_data )
    batch_weight.append( weight_data )

with torch.autograd.detect_anomaly():    
    loss = loss_fn( out, batch_truth, batch_weight, batch_size, DEVICE, verbose=True )

print("LOSS: ",loss)

# Backward pass
loss["tot"].backward()
#with torch.autograd.detect_anomaly():
#    out[0]["lm"].sum().backward()

# make perfect prediction tensors to test loss
batch_perfect = []
for ib, data in enumerate(batch):
    perfect_lm = torch.zeros( out[ib]["lm"].shape ).to(DEVICE)

    seq = torch.arange( perfect_lm.shape[2] )

    true_lm  = batch_truth[ib]["lm"].eq(1)
    false_lm = batch_truth[ib]["lm"].eq(0)
    #print(true_lm.shape)
    #print(out[ib]["lm"].shape)
    
    perfect_lm[0,0,true_lm] = -9.0
    perfect_lm[0,1,true_lm] =  9.0    
    perfect_lm[0,0,false_lm] =  9.0
    perfect_lm[0,1,false_lm] = -9.0

    #print(perfect_lm[0,:,:10])
    #print(batch_truth[ib]["lm"][:10])
    #print(out[ib]["lm"][0,:,:10])

    perfect_lm += 0.5

    perfect_ssnet = torch.ones( out[ib]["ssnet"].shape ).to(DEVICE)*(-9.0)
    #for i,x in enumerate(batch_truth[ib]["ssnet"]):
    #    perfect_ssnet[0,x,i] = 9.0
    perfect_ssnet[0, batch_truth[ib]["ssnet"][:], seq[:] ] = 9.0
    print("indexed: ",perfect_ssnet.shape)
    print(perfect_ssnet[0,:,:10])
    print(batch_truth[ib]["ssnet"][:10])
    print(seq[:10])

    perfect_kp = torch.clone( batch_truth[ib]["kp"] ).unsqueeze(0)
    print("perfect keypoint: ",perfect_kp.shape)
    
    batch_perfect.append({"lm":perfect_lm,"ssnet":perfect_ssnet,"kp":perfect_kp})
    
perfect_loss = loss_fn( batch_perfect, batch_truth, batch_weight, batch_size, DEVICE, verbose=True )
print("PERFECT LOSS: ",perfect_loss)
#perfect_loss["tot"].backward()
