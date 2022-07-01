import os,sys
import torch
import numpy as np

# debug the setup of larmatch minkowski model
# this tests the individual elements of loading the model, data, and calculating the loss

# model defined here
from larmatch.model.larmatchminkowski import LArMatchMinkowski
from larmatch.utils.larmatchme_engine import get_model, load_config_file

class argstest:
    def __init__(self):
        self.config_file = sys.argv[1]

args = argstest()
config = load_config_file( args, dump_to_stdout=True )
config["DEVICE"] = "cuda:0"
DEVICE = torch.device(config["DEVICE"])
model = get_model( config, dump_model=True ).to(DEVICE)

# create model
inputshape=(1024,3584)

# Load some test data and push it through
from larmatch_dataset import larmatchDataset
import MinkowskiEngine as ME

niter = 1
batch_size = 4
test = larmatchDataset( txtfile=config["TRAIN_DATASET_INPUT_TXTFILE"],
                        load_truth=True,
                        random_access=True,
                        num_triplet_samples=config["NUM_TRIPLET_SAMPLES"])
print("NENTRIES: ",len(test))
loader = torch.utils.data.DataLoader(test,batch_size=batch_size,collate_fn=larmatchDataset.collate_fn)

# get one batch
batch = next(iter(loader))

# data comes back as numpy arrays.
# we need to move it to DEVICE and then form MinkowskiEngine SparseTensors
# needs to be done three times: one for each wire plane of the detector
wireplane_sparsetensors = []
original_coord_batch = []

for p in range(3):
    print("plane ",p)
    for b,data in enumerate(batch):
        print(" coord plane[%d] batch[%d]"%(p,b),": ",data["coord_%d"%(p)].shape)
    coord_v = [ torch.from_numpy(data["coord_%d"%(p)]).to(DEVICE) for data in batch]
    feat_v  = [ torch.from_numpy(data["feat_%d"%(p)]).to(DEVICE) for data in batch ]
    print(" len(coord_v): ",len(coord_v))
    
    coords, feats = ME.utils.sparse_collate(coord_v, feat_v,device=DEVICE)
    print(" coords: ",coords.shape)
    print(" feats: ",feats.shape)
    wireplane_sparsetensors.append( ME.SparseTensor(features=feats, coordinates=coords) )
    original_coord_batch.append( coord_v )

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
                                 eval_ssnet=config["RUN_SSNET"],
                                 eval_keypoint_label=config["RUN_KPLABEL"],
                                 eval_keypoint_shift=False,
                                 eval_affinity_field=False ).to(DEVICE)

# load the truth data
print("== TRUTH DATA ===")
batch_truth = []
batch_weight = []
# larmatch
for b,data in enumerate(batch):
    #print("batch [",b,"]")
    
    lm_truth_t = torch.from_numpy(data["larmatch_label"]).to(DEVICE)
    lm_weight_t = torch.from_numpy(data["larmatch_weight"]).to(DEVICE)
    #print("  truth: ",lm_truth_t.shape)
    #print("  weight: ",lm_weight_t.shape)
 
    ssnet_truth_t  = torch.from_numpy(data["ssnet_truth"]).to(DEVICE)
    ssnet_weight_t = torch.from_numpy(data["ssnet_class_weight"]).to(DEVICE)
    for i in range(7):
        print("  ssnet class[",i,"]: ",ssnet_truth_t.eq(i).sum())

    kp_truth_t  = torch.from_numpy(data["keypoint_truth"]).to(DEVICE)
    kp_weight_t = torch.from_numpy(data["keypoint_weight"]).to(DEVICE)
        
    truth_data = {"lm":lm_truth_t,"ssnet":ssnet_truth_t,"kp":kp_truth_t}
    weight_data = {"lm":lm_weight_t,"ssnet":ssnet_weight_t,"kp":kp_weight_t}

    for k in ["lm","ssnet","kp"]:
        print("[",b,"] ",k,": truth=",truth_data[k].shape," weight=",weight_data[k].shape)
    
    batch_truth.append( truth_data )
    batch_weight.append( weight_data )

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
for ib, data in enumerate(batch):
    perfect_lm = torch.zeros( out["lm"][ib].shape ).to(DEVICE)
    print("perfect_lm: ",perfect_lm.shape)

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

    perfect_ssnet = torch.ones( out["ssnet"][ib].shape ).to(DEVICE)*(-9.0)
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
