#!/bin/env python

"""
TRAINING SCRIPT FOR LARMATCH+KEYPOINT+SSNET NETWORKS
"""
## IMPORT

# python,numpy
from __future__ import print_function
import os,sys
import shutil
import time
import traceback
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

# tensorboardX
from torch.utils.tensorboard import SummaryWriter

# larmatch imports
from larmatch import LArMatch
from larmatch_ssnet_classifier import LArMatchSSNetClassifier
from larmatch_keypoint_classifier import LArMatchKeypointClassifier
from larmatch_kpshift_regressor   import LArMatchKPShiftRegressor
from larmatch_affinityfield_regressor import LArMatchAffinityFieldRegressor
from load_larmatch_kps import load_larmatch_kps
from loss_larmatch_kps import SparseLArMatchKPSLoss

# ROOT, larcv
import ROOT as rt
from ROOT import std
from larcv import larcv
from larflow import larflow

# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT=False
RESUME_OPTIM_FROM_CHECKPOINT=False
RUNPROFILER=False
CHECKPOINT_FILE="train_kps_no_ssnet/checkpoint.1262000th.tar"
EXCLUDE_NEG_EXAMPLES = False
TRAIN_SSNET=True
TRAIN_KP=True
TRAIN_KPSHIFT=False
TRAIN_PAF=True
TRAIN_VERBOSE=True
FREEZE_LAYERS=False
SSNET_CLASS_NAMES=["bg","electron","gamma","muon","pion","proton","other"]

# Hard example training parameters (not yet implemented)
# =======================================================
HARD_EXAMPLE_TRAINING=False
HARDEX_GPU="cuda:0"
HARDEX_CHECKPOINT_FILE="train_kps_nossnet/checkpoint.260000th.tar"

# TRAINING+VALIDATION DATA PATHS
# ================================
#TRAIN_DATA_FOLDER="/home/twongj01/data/larmatch_kps_training_data/"
TRAIN_DATA_FOLDER="/home/twongjirad/working/larbys/ubdl/larflow/larmatchnet/"
INPUTFILE_TRAIN=["traininglabels_mcc9_v13_bnbnue_corsika_run00001_subrun00001.root"]
INPUTFILE_VALID=["traininglabels_mcc9_v13_bnbnue_corsika_run00001_subrun00001.root"]

# TRAINING PARAMETERS
# =======================
START_ITER  = 0
NUM_ITERS   = 2000000
TEST_NUM_MATCH_PAIRS = 50000
ADC_MAX = 400.0

BATCHSIZE_TRAIN=1  # batches per training iteration
BATCHSIZE_VALID=1  # batches per validation iteration
NWORKERS_TRAIN=2   # number of threads data loader will use for training set
NWORKERS_VALID=2   # number of threads data loader will use for validation set

NBATCHES_per_itertrain = 1
NBATCHES_per_step      = 1 # if >1 we use gradient accumulation
trainbatches_per_print = -1

NBATCHES_per_itervalid = 1
validbatches_per_print = -1

ITER_PER_VALID = 10

# CHECKPOINT PARAMETERS
# =======================
DEVICE_IDS=[0]
# map multi-training weights 
CHECKPOINT_MAP_LOCATIONS={"cuda:0":"cuda:0",
                          "cuda:1":"cuda:1"}
HARDEX_MAP_LOCATIONS={"cuda:0":"cuda:0",
                      "cuda:1":"cuda:1"}
CHECKPOINT_MAP_LOCATIONS=None
CHECKPOINT_FROM_DATA_PARALLEL=False
ITER_PER_CHECKPOINT=1000
PREDICT_CLASSVEC=True
# ===================================================

# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights
writer = SummaryWriter()
train_entry = 0
valid_entry = 0
TRAIN_NENTRIES = 0
VALID_NENTRIES = 0

def main():

    # load global variables we plan to modify
    global best_prec1
    global writer
    global num_iters
    global TRAIN_NENTRIES
    global VALID_NENTRIES

    # set device where we'll do the computations
    if GPUMODE:
        DEVICE = torch.device("cuda:%d"%(DEVICE_IDS[0]))
    else:
        DEVICE = torch.device("cpu")
    
    # create model, mark it to run on the device
    model = LArMatch(use_unet=True).to(DEVICE)
    ssnet_head    = LArMatchSSNetClassifier().to(DEVICE)
    kplabel_head  = LArMatchKeypointClassifier().to(DEVICE)
    kpshift_head  = LArMatchKPShiftRegressor().to(DEVICE)
    affinity_head = LArMatchAffinityFieldRegressor(layer_nfeatures=[64,64,64]).to(DEVICE)
    # the model is multi-tasked, so we group the different tasks into a map
    model_dict = {"larmatch":model,
                  "ssnet":ssnet_head,
                  "kplabel":kplabel_head,
                  "kpshift":kpshift_head,
                  "paf":affinity_head}
    parameters = []
    for n,model in model_dict.items():
        for p in model.parameters():
            parameters.append( p )
    
    if True:
        # DUMP MODEL (for debugging)
        print(model)
        print(ssnet_head)
        print(kplabel_head)
        print(kpshift_head)

        # uncomment to dump model parameters
        if False:
            print(model.module.parameters)
            return
        
        if False: sys.exit(-1)
    
    # Resume training option
    if RESUME_FROM_CHECKPOINT:
        print("RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE)
        checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS ) # load weights to gpuid

        # hack to be able to load sparseconvnet<1.3
        for name,arr in checkpoint["state_larmatch"].items():
            if ( ("resnet" in name and "weight" in name and len(arr.shape)==3) or
                 ("stem" in name and "weight" in name and len(arr.shape)==3) or
                 ("unet_layers" in name and "weight" in name and len(arr.shape)==3) or         
                 ("feature_layer.weight" == name and len(arr.shape)==3 ) ):
                print("reshaping ",name)
                checkpoint["state_larmatch"][name] = arr.reshape( (arr.shape[0], 1, arr.shape[1], arr.shape[2]) )
        
        best_prec1 = checkpoint["best_prec1"]
        if CHECKPOINT_FROM_DATA_PARALLEL:
            model = nn.DataParallel( model, device_ids=DEVICE_IDS ) # distribute across device_ids
        for n,m in model_dict.items():
            if "state_"+n in checkpoint:
                if n in ["kplabel"]:
                    print("skip re-loading keypoint")
                    continue
                m.load_state_dict(checkpoint["state_"+n])

    # data parallel training -- does not work
    if GPUMODE and not CHECKPOINT_FROM_DATA_PARALLEL and len(DEVICE_IDS)>1:
        model = nn.DataParallel( model, device_ids=DEVICE_IDS ) # distribute across device_ids

    # if hard example training, we load a copy of the network we will not modify
    # that generates scores for each triplet in the event
    # we will use those scores to sample the space points for training in a biased manner.
    # we will bias towards examples the network gets incorrect
    if HARD_EXAMPLE_TRAINING:
        hardex_larmatch_model  = LArMatch(neval=TEST_NUM_MATCH_PAIRS,use_unet=True).to( torch.device(HARDEX_GPU) )
        hardex_checkpoint = torch.load( HARDEX_CHECKPOINT_FILE, map_location=HARDEX_MAP_LOCATIONS )
        hardex_model = {"larmatch":hardex_larmatch_model}
        for n,m in hardex_model.items():
            m.load_state_dict(hardex_checkpoint["state_"+n])
    else:
        hardex_model=None
            

    # FIX CERTAIN PARAMETERS
    if FREEZE_LAYERS:
        for fixed_model in ["larmatch","ssnet","kpshift"]:
            for param in model_dict[fixed_model].parameters():
                param.requires_grad = False
        
    # define loss function (criterion) and optimizer
    criterion = SparseLArMatchKPSLoss( eval_ssnet=TRAIN_SSNET,
                                       eval_keypoint_label=TRAIN_KP,
                                       eval_keypoint_shift=TRAIN_KPSHIFT,
                                       eval_affinity_field=TRAIN_PAF )

    # training parameters
    lr = 1e-3
    momentum = 0.9
    weight_decay = 1.0e-4

    # training variables
    itersize_train         = BATCHSIZE_TRAIN*NBATCHES_per_itertrain # number of images per iteration
    itersize_valid         = BATCHSIZE_VALID*NBATCHES_per_itervalid # number of images per validation    
    
    # SETUP OPTIMIZER
    # ADAM
    # betas default: (0.9, 0.999) for (grad, grad^2). smoothing coefficient for grad. magnitude calc.
    optimizer = torch.optim.Adam(parameters, 
                                 lr=lr, 
                                 weight_decay=weight_decay)
    if RESUME_OPTIM_FROM_CHECKPOINT:
        optimizer.load_state_dict( checkpoint["optimizer"] )

    # optimize algorithms based on input size (good if input size is constant)
    cudnn.benchmark = False

    # LOAD THE DATASET HELPERS
    traindata_v = std.vector("std::string")()
    for x in INPUTFILE_TRAIN:
        traindata_v.push_back( TRAIN_DATA_FOLDER+"/"+x )
    iotrain = {"kps":larflow.keypoints.LoaderKeypointData(traindata_v),
               "affinity":larflow.keypoints.LoaderAffinityField(traindata_v)}

    validdata_v = std.vector("std::string")()
    for x in INPUTFILE_VALID:
        validdata_v.push_back( TRAIN_DATA_FOLDER+"/"+x )
    iovalid = {"kps":larflow.keypoints.LoaderKeypointData(validdata_v),
               "affinity":larflow.keypoints.LoaderAffinityField(validdata_v)}

    if not EXCLUDE_NEG_EXAMPLES:
        for name,loader in iotrain.items():
            loader.exclude_false_triplets( EXCLUDE_NEG_EXAMPLES )
        for name,loader in iovalid.items():
            loader.exclude_false_triplets( EXCLUDE_NEG_EXAMPLES )

    TRAIN_NENTRIES = iotrain["kps"].GetEntries()
    iter_per_epoch = TRAIN_NENTRIES/(itersize_train)
    epochs = float(NUM_ITERS)/float(TRAIN_NENTRIES)
    VALID_NENTRIES = iovalid["kps"].GetEntries()

    print("Number of iterations to run: ",NUM_ITERS)
    print("Entries in the training set: ",TRAIN_NENTRIES)
    print("Entries in the validation set: ",VALID_NENTRIES)
    print("Entries per iter (train): ",itersize_train)
    print("Entries per iter (valid): ",itersize_valid)
    print("Number of (training) Epochs to run: ",epochs)
    print("Iterations per epoch: ",iter_per_epoch)

    if False:
        print("passed setup successfully")
        sys.exit(0)

    # TRAINING LOOP
    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:

        # LOOP OVER FIXED NUMBER OF INTERATIONS
        for ii in range(START_ITER, NUM_ITERS):

            # modify learning rate based on interation number (not used)
            adjust_learning_rate(optimizer, ii, lr)
            print("MainLoop Iter:%d Epoch:%d.%d "%(ii,ii/iter_per_epoch,ii%iter_per_epoch),)
            for param_group in optimizer.param_groups:
                print("lr=%.3e"%(param_group['lr']),)
                print()

            # train for one iteration
            try:
                _ = train(iotrain, DEVICE, BATCHSIZE_TRAIN,
                          model_dict, criterion, optimizer,
                          NBATCHES_per_itertrain, NBATCHES_per_step,
                          ii, trainbatches_per_print, hardex_model=hardex_model)
                
            except Exception as e:
                print("Error in training routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break

            # evaluate on validation set at a given interval (every ITER_PER_VALID training iterations)
            if ii%ITER_PER_VALID==0 and ii>0:
                try:
                    totloss, totacc = validate(iovalid, DEVICE, BATCHSIZE_VALID,
                                               model_dict, criterion,
                                               NBATCHES_per_itervalid, ii,
                                               validbatches_per_print)
                except Exception as e:
                    print("Error in validation routine!")
                    print(e.message)
                    print(e.__class__.__name__)
                    traceback.print_exc(e)
                    break

                # remember best prec@1 and save checkpoint
                prec1   = totacc
                is_best =  prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)

                # check point for best model
                if is_best:
                    print("Saving best model")
                    save_checkpoint({
                        'iter':ii,
                        'epoch': ii/iter_per_epoch,
                        'state_larmatch': model_dict["larmatch"].state_dict(),
                        'state_ssnet': model_dict["ssnet"].state_dict(),
                        'state_kplabel': model_dict["kplabel"].state_dict(),
                        'state_kpshift': model_dict["kpshift"].state_dict(),
                        'state_paf': model_dict["paf"].state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, -1)

            # periodic checkpoint
            if ii>0 and ii%ITER_PER_CHECKPOINT==0:
                print("saving periodic checkpoint")
                save_checkpoint({
                    'iter':ii,
                    'epoch': ii/iter_per_epoch,
                    'state_larmatch': model_dict["larmatch"].state_dict(),
                    'state_ssnet': model_dict["ssnet"].state_dict(),
                    'state_kplabel': model_dict["kplabel"].state_dict(),
                    'state_kpshift': model_dict["kpshift"].state_dict(),
                    'state_paf': model_dict["paf"].state_dict(),                    
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, False, ii)
            # flush the print buffer after iteration
            sys.stdout.flush()
                
        # end of profiler context
        print("saving last state")
        save_checkpoint({
            'iter':NUM_ITERS,
            'epoch': float(NUM_ITERS)/iter_per_epoch,
            'state_larmatch': model_dict["larmatch"].state_dict(),
            'state_ssnet': model_dict["ssnet"].state_dict(),
            'state_kplabel': model_dict["kplabel"].state_dict(),
            'state_kpshift': model_dict["kpshift"].state_dict(),
            'state_paf': model_dict["paf"].state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, False, NUM_ITERS)


    print("FIN")
    print("PROFILER")
    if RUNPROFILER:
        print(prof)
    writer.close()


def train(train_loader, device, batchsize,
          model, criterion, optimizer,
          nbatches, nbatches_per_step,
          iiter, print_freq, hardex_model=None):
    """
    train_loader: helper module to load event data
    device: device we are training on
    batchsize: number of images per batch
    model: network are training
    criterion: the loss calculation module we are using
    optimizer: optimizer
    nbatches: number of batches to run in this iteraction
    nbatches_per_step: allows for gradient accumulation if >1
    iiter: current iteration
    print_freq: number of batches we run before printing out some statistics
    hardex_model: if not None, the hardex_model we'll use to provide scores and bias sampling of events
                  to spacepoints the network got incorrect.
    """
    # global variables we will modify
    global writer
    global train_entry
    

    # timers for profiling
    batch_time    = AverageMeter() # total for batch
    data_time     = AverageMeter()
    forward_time  = AverageMeter()
    backward_time = AverageMeter()
    acc_time      = AverageMeter()

    # accruacy and loss meters
    lossnames    = ("total","lm","ssnet","kp","paf")
    flowaccnames = ["lm_pos","lm_neg","lm_all","kp_nu","kp_trk","kp_shr","paf"]+SSNET_CLASS_NAMES+["ssnet-all"]

    acc_meters  = {}
    for n in flowaccnames:
        acc_meters[n] = AverageMeter()

    loss_meters = {}
    for n in lossnames:
        loss_meters[n] = AverageMeter()

    time_meters = {}
    for l in ["batch","data","forward","backward","accuracy"]:
        time_meters[l] = AverageMeter()
    
    # switch to train mode
    for name,m in model.items():
        m.train()

    # clear gradients
    optimizer.zero_grad()
    nnone = 0

    #print "ZERO GRAD BY OPTIMIZER"
    #for n,p in model["kplabel"].named_parameters():
    #    if "out" in n:
    #        print n,": grad: ",p.grad
    #        print n,": ",p
    

    # run predictions over nbatches before calculating gradients
    # if nbatches>1, this is so-called "gradient accumulation".
    # not typically used
    for i in range(0,nbatches):
        #print "iiter ",iiter," batch ",i," of ",nbatches
        batchstart = time.time()

        # GET THE DATA
        end = time.time()
            
        flowdata = load_larmatch_kps( train_loader, train_entry, 1,
                                      npairs=TEST_NUM_MATCH_PAIRS,
                                      exclude_neg_examples=False,
                                      verbose=True, single_batch_mode=True )

        coord_t = [ torch.from_numpy( flowdata['coord_%s'%(p)] ).to(device) for p in [0,1,2] ]
        feat_t  = [ torch.from_numpy( flowdata['feat_%s'%(p)] ).to(device) for p in [0,1,2] ]

        npairs          = flowdata['npairs']        
        match_t         = torch.from_numpy( flowdata['matchpairs'] ).to(device).requires_grad_(False)
        match_label_t   = torch.from_numpy( flowdata['larmatchlabels'] ).to(device).requires_grad_(False)
        match_weight_t  = torch.from_numpy( flowdata['match_weight'] ).to(device).requires_grad_(False)
        truematch_idx_t = torch.from_numpy( flowdata['positive_indices'] ).to(device).requires_grad_(False)
        
        ssnet_label_t  = torch.from_numpy( flowdata['ssnet_label'] ).to(device).requires_grad_(False)
        ssnet_cls_weight_t = torch.from_numpy( flowdata['ssnet_class_weight'] ).to(device).requires_grad_(False)
        ssnet_top_weight_t = torch.from_numpy( flowdata['ssnet_top_weight'] ).to(device).requires_grad_(False)
        
        kp_label_t    = torch.from_numpy( flowdata['kplabel'] ).to(device).requires_grad_(False)
        kp_weight_t   = torch.from_numpy( flowdata['kplabel_weight'] ).to(device).requires_grad_(False)        
        kpshift_t     = torch.from_numpy( flowdata['kpshift'] ).to(device)
        
        paf_label_t   = torch.from_numpy( flowdata['paf_label'] ).to(device).requires_grad_(False)
        paf_weight_t  = torch.from_numpy( flowdata['paf_weight'] ).to(device).requires_grad_(False)

        for p in range(3):
            feat_t[p] = torch.clamp( feat_t[p], 0, ADC_MAX )

        print("loaded train entry: ",train_entry," ",flowdata["entry"]," ",flowdata["tree_entry"],"npairs=",npairs)
        if train_entry+1<TRAIN_NENTRIES:
            train_entry += 1
        else:
            train_entry = 0
        
        # compute output
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()
        

        # first get feature vectors
        feat_u_t, feat_v_t, feat_y_t = model['larmatch'].forward_features( coord_t[0], feat_t[0],
                                                                           coord_t[1], feat_t[1],
                                                                           coord_t[2], feat_t[2], 1,
                                                                           verbose=TRAIN_VERBOSE )
        if HARD_EXAMPLE_TRAINING and hardex_model is not None:
            # we use the fixed network to calculate score for all triplets
            raise RuntimeError("HARD_EXAMPLE_TRAINING not implemented yet. This is just a stub.")
            fixednet_ntriplets = preplarmatch._triplet_v.size()
            fixednet_startidx  = 0
            fixednet_scores_np = np.zeros( ntriplets )
            while startidx<ntriplets:
                print("create matchpairs: startidx=",startidx," of ",ntriplets)
                t_chunk = time.time()
                matchpair_np = preplarmatch.get_chunk_triplet_matches( startidx,
                                                                       NUM_PAIRS,
                                                                       last_index,
                                                                       npairs,
                                                                       with_truth )
                t_chunk = time.time()-t_chunk
                print("  made matchpairs: ",matchpair_np.shape," npairs_filled=",npairs.value,"; time to make chunk=",t_chunk," secs") 
                dt_chunk += t_chunk            
                startidx = int(last_index.value)

            # make torch tensor or array providing index of pixels in each plane we should group
            # to form a 3D spacepoint proposal
            matchpair_t = torch.from_numpy( matchpair_np.astype(np.long) ).to(DEVICE)
                
            if with_truth:
                truthvec = torch.from_numpy( matchpair_np[:,3].astype(np.long) ).to(DEVICE)

            with torch.no_grad():
                feat_triplet_t = model_dict['larmatch'].extract_features( outfeat_u, outfeat_v, outfeat_y,
                                                                          matchpair_t, npairs.value,
                                                                          DEVICE, verbose=True )
            tstart = time.time()
            with torch.no_grad():
                pred_t = model_dict['larmatch'].classify_triplet( feat_triplet_t )
            dt_net_classify = time.time()-tstart
            dt_net  += dt_net_classify
            prob_t = sigmoid(pred_t) # should probably move inside classify_triplet method
            print("  prob_t=",prob_t.shape," time-elapsed=",dt_net_classify,"secs")
            

        # TRAINING MODEL: EVALUATE LARMATCH SCORES
        # ==========================================
        # extract features according to sampled match indices
        feat_triplet_t = model['larmatch'].extract_features( feat_u_t, feat_v_t, feat_y_t,
                                                             match_t, flowdata['npairs'],
                                                             device, verbose=TRAIN_VERBOSE )
        print("[larmatch train] feat_triplet_t=",feat_triplet_t.shape)

        # evaluate larmatch match classifier
        match_pred_t = model['larmatch'].classify_triplet( feat_triplet_t )
        match_pred_t = match_pred_t.reshape( (match_pred_t.shape[-1]) )
        print("[larmatch train] match-pred=",match_pred_t.shape)

        # evaluate ssnet classifier
        if TRAIN_SSNET:
            ssnet_pred_t = model['ssnet'].forward( feat_triplet_t )
            ssnet_pred_t = ssnet_pred_t.reshape( (ssnet_pred_t.shape[1],ssnet_pred_t.shape[2]) )
            ssnet_pred_t = torch.transpose( ssnet_pred_t, 1, 0 )
            print("[larmatch train] ssnet-pred=",ssnet_pred_t.shape)
        else:
            ssnet_pred_t = None
        
        # next evaluate keypoint classifier
        if TRAIN_KP:
            kplabel_pred_t = model['kplabel'].forward( feat_triplet_t )
            print("[larmatch train] kplabel-pred=",kplabel_pred_t.shape)
            kplabel_pred_t = kplabel_pred_t.reshape( (kplabel_pred_t.shape[1], kplabel_pred_t.shape[2]) )
            kplabel_pred_t = torch.transpose( kplabel_pred_t, 1, 0 )
            print("[larmatch train] kplabel-pred=",kplabel_pred_t.shape)
        else:
            kplabel_pred_t = None
        
        # next evaluate keypoint shift predictor
        if TRAIN_KPSHIFT:
            kpshift_pred_t = model['kpshift'].forward( feat_triplet_t )
            kpshift_pred_t = kpshift_pred_t.reshape( (kpshift_pred_t.shape[1],kpshift_pred_t.shape[2]) )
            kpshift_pred_t = torch.transpose( kpshift_pred_t, 1, 0 )
            print("[larmatch train] kpshift-pred=",kpshift_pred_t.shape)
        else:
            kpshift_pred_t = None

        # next evaluate affinity field predictor
        if TRAIN_PAF:
            paf_pred_t = model["paf"].forward( feat_triplet_t )
            print("[larmatch train]: paf pred=",paf_pred_t.shape)
            paf_pred_t = paf_pred_t.reshape( (paf_pred_t.shape[1],paf_pred_t.shape[2]) )
            paf_pred_t = torch.transpose( paf_pred_t, 1, 0 )
            print("[larmatch train]: paf pred=",paf_pred_t.shape)
        else:
            paf_pred_t = None

        # Calculate the loss
        totloss,larmatch_loss,ssnet_loss,kp_loss,kpshift_loss, paf_loss = criterion( match_pred_t,   ssnet_pred_t,  kplabel_pred_t, kpshift_pred_t, paf_pred_t,
                                                                                     match_label_t,  ssnet_label_t, kp_label_t, kpshift_t, paf_label_t,
                                                                                     truematch_idx_t,
                                                                                     match_weight_t, ssnet_cls_weight_t*ssnet_top_weight_t, kp_weight_t, paf_weight_t,
                                                                                     verbose=TRAIN_VERBOSE )

        if RUNPROFILER:
            torch.cuda.synchronize()
        time_meters["forward"].update(time.time()-end)

        # compute gradient and do SGD step
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()

        # allow for gradient accumulation
        if nbatches_per_step>1:
            # if we apply gradient accumulation, we average over accumulation steps
            print("average over batches per step when gradient accumulating.")
            totloss /= float(nbatches_per_step)
        
        # of course, we calculate gradients for this batch
        totloss.backward()

        # clip the gradients
        for n,p in model["kplabel"].named_parameters():
            torch.nn.utils.clip_grad_value_( p, 0.5 )
        
        # only step, i.e. adjust weights every nbatches_per_step or if last batch
        if (i>0 and (i+1)%nbatches_per_step==0) or i+1==nbatches:
            print("batch %d of %d. making step, then clearing gradients. nbatches_per_step=%d"%(i,nbatches,nbatches_per_step))
            optimizer.step()

            # inspect gradients
            #inspect_model_grad = "larmatch"
            #for n,p in model[inspect_model_grad].named_parameters():
            #    if "out" in n:
            #        print(n,": grad: ",p.grad)
            #        print(n,": ",p)
            
            optimizer.zero_grad()
            
        if RUNPROFILER:        
            torch.cuda.synchronize()                
        time_meters["backward"].update(time.time()-end)

        # measure accuracy and record loss
        end = time.time()

        # update loss meters
        loss_meters["total"].update( totloss.detach().item(),    nbatches_per_step )
        loss_meters["lm"].update( larmatch_loss, nbatches_per_step )
        loss_meters["ssnet"].update( ssnet_loss, nbatches_per_step )
        loss_meters["kp"].update( kp_loss,       nbatches_per_step )
        loss_meters["paf"].update( paf_loss,     nbatches_per_step )
        
        
        # measure accuracy and update meters
        acc = accuracy(match_pred_t, match_label_t,
                       ssnet_pred_t, ssnet_label_t,
                       kplabel_pred_t, kp_label_t,
                       paf_pred_t, paf_label_t,
                       truematch_idx_t,
                       acc_meters)
            
        # update time meter
        time_meters["accuracy"].update(time.time()-end)            

        # measure elapsed time for batch
        time_meters["batch"].update(time.time()-batchstart)

        # print status
        if print_freq>0 and i%print_freq == 0:
            prep_status_message( "train-batch", i, acc_meters, loss_meters, time_meters,True )

    prep_status_message( "Train-Iteration", iiter, acc_meters, loss_meters, time_meters, True )

    # write to tensorboard
    loss_scalars = { x:y.avg for x,y in loss_meters.items() }
    writer.add_scalars('data/train_loss', loss_scalars, iiter )

    acc_scalars = { x:y.avg for x,y in acc_meters.items() }
    writer.add_scalars('data/train_accuracy', acc_scalars, iiter )
    
    return loss_meters['total'].avg,acc_meters['lm_all'].avg


def validate(val_loader, device, batchsize, model, criterion, nbatches, iiter, print_freq):
    """
    inputs
    ------
    val_loader: instance of LArCVDataSet for loading data
    batchsize (int): image (sets) per batch
    model (pytorch model): network
    criterion (pytorch module): loss function
    nbatches (int): number of batches to process
    print_freq (int): number of batches before printing output
    iiter (int): current iteration number of main loop
    
    outputs
    -------
    average percent of predictions within 5 pixels of truth
    """
    global writer
    global valid_entry

    # accruacy and loss meters
    lossnames    = ("total","lm","ssnet","kp","paf")
    flowaccnames = ["lm_pos","lm_neg","lm_all","kp_nu","kp_trk","kp_shr","paf"]+SSNET_CLASS_NAMES+["ssnet-all"]    

    acc_meters  = {}
    for n in flowaccnames:
        acc_meters[n] = AverageMeter()

    loss_meters = {}
    for n in lossnames:
        loss_meters[n] = AverageMeter()

    time_meters = {}
    for l in ["batch","data","forward","backward","accuracy"]:
        time_meters[l] = AverageMeter()
    
    # switch to evaluate mode
    #for name,m in model.items():
    #    m.eval()
    
    iterstart = time.time()
    nnone = 0
    for i in range(0,nbatches):
        batchstart = time.time()
        
        tdata_start = time.time()

        flowdata = load_larmatch_kps( val_loader, valid_entry, 1,
                                      npairs=TEST_NUM_MATCH_PAIRS,
                                      verbose=True, single_batch_mode=True )
        if valid_entry+1<VALID_NENTRIES:
            valid_entry += 1
        else:
            valid_entry = 0        

        coord_t = [ torch.from_numpy( flowdata['coord_%s'%(p)] ).to(device) for p in [0,1,2] ]
        feat_t  = [ torch.from_numpy( flowdata['feat_%s'%(p)] ).to(device) for p in [0,1,2] ]

        match_t         = torch.from_numpy( flowdata['matchpairs'] ).to(device).requires_grad_(False)
        match_label_t   = torch.from_numpy( flowdata['larmatchlabels'] ).to(device).requires_grad_(False)
        match_weight_t  = torch.from_numpy( flowdata['match_weight'] ).to(device).requires_grad_(False)
        truematch_idx_t = torch.from_numpy( flowdata['positive_indices'] ).to(device).requires_grad_(False)        
        
        ssnet_label_t  = torch.from_numpy( flowdata['ssnet_label'] ).to(device).requires_grad_(False)
        ssnet_cls_weight_t = torch.from_numpy( flowdata['ssnet_class_weight'] ).to(device).requires_grad_(False)
        ssnet_top_weight_t = torch.from_numpy( flowdata['ssnet_top_weight'] ).to(device).requires_grad_(False)
        
        kp_label_t    = torch.from_numpy( flowdata['kplabel'] ).to(device).requires_grad_(False)
        kp_weight_t   = torch.from_numpy( flowdata['kplabel_weight'] ).to(device).requires_grad_(False)        
        kpshift_t     = torch.from_numpy( flowdata['kpshift'] ).to(device)

        paf_label_t   = torch.from_numpy( flowdata['paf_label'] ).to(device).requires_grad_(False)
        paf_weight_t  = torch.from_numpy( flowdata['paf_weight'] ).to(device).requires_grad_(False)        
        
        # CLAMP ADC VALUES
        for p in range(3):
            feat_t[p] = torch.clamp( feat_t[p], 0, ADC_MAX )
        
        print("loaded valid entry: ",flowdata["entry"])
        time_meters["data"].update( time.time()-tdata_start )
        
        # compute model output
        if RUNPROFILER:
            torch.cuda.synchronize()
        tforward = time.time()

        # first get feature vectors
        with torch.no_grad():        
            feat_u_t, feat_v_t, feat_y_t = model['larmatch'].forward_features( coord_t[0], feat_t[0],
                                                                               coord_t[1], feat_t[1],
                                                                               coord_t[2], feat_t[2], 1,
                                                                               verbose=False )


            # extract features according to sampled match indices
            feat_triplet_t = model['larmatch'].extract_features( feat_u_t, feat_v_t, feat_y_t,
                                                                 match_t, flowdata['npairs'],
                                                                 device, verbose=False )

            # next evaluate larmatch match classifier
            match_pred_t = model['larmatch'].classify_triplet( feat_triplet_t )
            match_pred_t = match_pred_t.reshape( (match_pred_t.shape[-1]) )
            print("[larmatch valid] match-pred=",match_pred_t.shape)

            # evaluate ssnet classifier
            if TRAIN_SSNET:
                ssnet_pred_t = model['ssnet'].forward( feat_triplet_t )
                ssnet_pred_t = ssnet_pred_t.reshape( (ssnet_pred_t.shape[1],ssnet_pred_t.shape[2]) )
                ssnet_pred_t = torch.transpose( ssnet_pred_t, 1, 0 )
                #ssnet_pred_t = ssnet_pred_t.reshape( (ssnet_pred_t.shape[-1]) )
                print("[ssnet valid] ssnet-pred=",ssnet_pred_t.shape)
            else:
                ssnet_pred_t = None
        
            # evaluate keypoint regression
            if TRAIN_KP:
                kplabel_pred_t = model['kplabel'].forward( feat_triplet_t )
                kplabel_pred_t = kplabel_pred_t.reshape( (kplabel_pred_t.shape[1], kplabel_pred_t.shape[2]) )
                kplabel_pred_t = torch.transpose( kplabel_pred_t, 1, 0 )
                print("[keypoint valid] kplabel-pred=",kplabel_pred_t.shape)
            else:
                kplabel_pred_t = None
        
            # next evaluate keypoint shift
            if TRAIN_KPSHIFT:
                kpshift_pred_t = model['kpshift'].forward( feat_triplet_t )
                kpshift_pred_t = kpshift_pred_t.reshape( (kpshift_pred_t.shape[1],kpshift_pred_t.shape[2]) )
                kpshift_pred_t = torch.transpose( kpshift_pred_t, 1, 0 )
                print("[keypoint shift valid] kpshift-pred=",kpshift_pred_t.shape)
            else:
                kpshift_pred_t = None

            # next evaluate affinity field predictor
            if TRAIN_PAF:
                paf_pred_t = model["paf"].forward( feat_triplet_t )
                print("[larmatch valid]: paf pred=",paf_pred_t.shape)
                paf_pred_t = paf_pred_t.reshape( (paf_pred_t.shape[1],paf_pred_t.shape[2]) )
                paf_pred_t = torch.transpose( paf_pred_t, 1, 0 )
                print("[larmatch valid]: paf pred=",paf_pred_t.shape)
            else:
                paf_pred_t = None            

            totloss,larmatch_loss,ssnet_loss,kp_loss,kpshift_loss,paf_loss = criterion( match_pred_t,  ssnet_pred_t,  kplabel_pred_t, kpshift_pred_t, paf_pred_t,
                                                                                        match_label_t, ssnet_label_t, kp_label_t,     kpshift_t,      paf_label_t,
                                                                                        truematch_idx_t,
                                                                                        match_weight_t, ssnet_cls_weight_t*ssnet_top_weight_t, kp_weight_t, paf_weight_t,
                                                                                        verbose=False )
            time_meters["forward"].update(time.time()-tforward)

            # update loss meters
            loss_meters["total"].update( totloss.item(),    nbatches )
            loss_meters["lm"].update( larmatch_loss, nbatches )
            loss_meters["ssnet"].update( ssnet_loss, nbatches )
            loss_meters["kp"].update( kp_loss,       nbatches )
            loss_meters["paf"].update( paf_loss,     nbatches )
        
            # measure accuracy and update meters
            end = time.time()
            acc = accuracy(match_pred_t, match_label_t,
                           ssnet_pred_t, ssnet_label_t,
                           kplabel_pred_t, kp_label_t,
                           paf_pred_t, paf_label_t,
                           truematch_idx_t,
                           acc_meters)
            time_meters["accuracy"].update(time.time()-end)        
            
            # measure elapsed time for batch
            time_meters["batch"].update( time.time()-batchstart )
            end = time.time()
            if print_freq>0 and i % print_freq == 0:
                prep_status_message( "valid-batch", i, acc_meters, loss_meters, time_meters, False )

            
    prep_status_message( "Valid-Iter", iiter, acc_meters, loss_meters, time_meters, False )

    # write to tensorboard
    loss_scalars = { x:y.avg for x,y in loss_meters.items() }
    writer.add_scalars('data/valid_loss', loss_scalars, iiter )

    acc_scalars = { x:y.avg for x,y in acc_meters.items() }
    writer.add_scalars('data/valid_accuracy', acc_scalars, iiter )
    
    return loss_meters['total'].avg,acc_meters['lm_all'].avg

def save_checkpoint(state, is_best, p, filename='checkpoint.pth.tar'):
    if p>0:
        filename = "checkpoint.%dth.tar"%(p)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = lr * (0.5 ** (epoch // 300))
    lr = lr
    #lr = lr*0.992
    #print "adjust learning rate to ",lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(match_pred_t, match_label_t,
             ssnet_pred_t, ssnet_label_t,
             kp_pred_t, kp_label_t,
             paf_pred_t, paf_label_t,
             truematch_indices_t,
             acc_meters):
    """Computes the accuracy metrics."""

    # LARMATCH METRICS
    match_pred = match_pred_t.detach()
    npairs = match_pred.shape[0]
    
    pos_correct = (match_pred.gt(0.0)*match_label_t[:npairs].eq(1)).sum().to(torch.device("cpu")).item()
    neg_correct = (match_pred.lt(0.0)*match_label_t[:npairs].eq(0)).sum().to(torch.device("cpu")).item()
    npos = float(match_label_t[:npairs].eq(1).sum().to(torch.device("cpu")).item())
    nneg = float(match_label_t[:npairs].eq(0).sum().to(torch.device("cpu")).item())

    acc_meters["lm_pos"].update( float(pos_correct)/npos )
    acc_meters["lm_neg"].update( float(neg_correct)/nneg )
    acc_meters["lm_all"].update( float(pos_correct+neg_correct)/(npos+nneg) )

    # SSNET METRICS
    if ssnet_pred_t is not None:
        if ssnet_pred_t.shape[0]!=ssnet_label_t.shape[0]:
            ssnet_pred     = torch.index_select( ssnet_pred_t.detach(), 0, truematch_indices_t )
        else:
            ssnet_pred     = ssnet_pred_t.detach()
        ssnet_class    = torch.argmax( ssnet_pred, 1 )
        ssnet_correct  = ssnet_class.eq( ssnet_label_t )
        ssnet_tot_correct = ssnet_correct.sum().item()        
        for iclass,classname in enumerate(SSNET_CLASS_NAMES):
            if ssnet_label_t.eq(iclass).sum().item()>0:                
                ssnet_class_correct = ssnet_correct[ ssnet_label_t==iclass ].sum().item()    
                acc_meters[classname].update( float(ssnet_class_correct)/float(ssnet_label_t.eq(iclass).sum().item()) )
        acc_meters["ssnet-all"].update( ssnet_tot_correct/float(ssnet_label_t.shape[0]) )

    # KP METRIC
    if kp_pred_t is not None:
        if kp_pred_t.shape[0]!=kp_label_t.shape[0]:
            kp_pred  = torch.index_select( kp_pred_t.detach(),  0, truematch_indices_t )
            kp_label = torch.index_select( kp_label_t.detach(), 0, truematch_indices_t )
        else:
            kp_pred  = kp_pred_t.detach()
            kp_label = kp_label_t.detach()[:npairs]
        names = ["nu","trk","shr"]
        for c in range(3):
            kp_n_pos = float(kp_label[:,c].gt(0.5).sum().item())
            kp_pos   = float(kp_pred[:,c].gt(0.5)[ kp_label[:,c].gt(0.5) ].sum().item())
            print("kp[",c,"] n_pos[>0.5]: ",kp_n_pos," pred[>0.5]: ",kp_pos)
            acc_meters["kp_"+names[c]].update( kp_pos/kp_n_pos )

    # PARTICLE AFFINITY FLOW
    if paf_pred_t is not None:
        # we define accuracy with the direction is less than 20 degress
        if paf_pred_t.shape[0]!=paf_label_t.shape[0]:
            paf_pred  = torch.index_select( paf_pred_t.detach(),  0, truematch_indices_t )
            paf_label = torch.index_select( paf_label_t.detach(), 0, truematch_indices_t )
        else:
            paf_pred  = paf_pred_t.detach()
            paf_label = paf_label_t.detach()[:npairs]
        # calculate cosine
        paf_truth_lensum = torch.sum( paf_label*paf_label, 1 )
        paf_pred_lensum  = torch.sum( paf_pred*paf_pred, 1 )
        paf_pred_lensum  = torch.sqrt( paf_pred_lensum )
        paf_posexamples = paf_truth_lensum.gt(0.5)
        #print paf_pred[paf_posexamples,:].shape," ",paf_label[paf_posexamples,:].shape," ",paf_pred_lensum[paf_posexamples].shape
        paf_cos = torch.sum(paf_pred[paf_posexamples,:]*paf_label[paf_posexamples,:],1)/(paf_pred_lensum[paf_posexamples]+0.001)
        paf_npos  = paf_cos.shape[0]
        paf_ncorr = paf_cos.gt(0.94).sum().item()
        paf_acc = float(paf_ncorr)/float(paf_npos)
        print("paf: npos=",paf_npos," acc=",paf_acc)
        if paf_npos>0:
            acc_meters["paf"].update( paf_acc )
    
    
    return True

def dump_lr_schedule( startlr, numepochs ):
    for epoch in range(0,numepochs):
        lr = startlr*(0.5**(epoch//300))
        if epoch%10==0:
            print("Epoch [%d] lr=%.3e"%(epoch,lr))
    print("Epoch [%d] lr=%.3e"%(epoch,lr))
    return

def prep_status_message( descripter, iternum, acc_meters, loss_meters, timers, istrain ):
    print("------------------------------------------------------------------------")
    print(" Iter[",iternum,"] ",descripter)
    print("  Time (secs): iter[%.2f] batch[%.3f] Forward[%.3f/batch] Backward[%.3f/batch] Acc[%.3f/batch] Data[%.3f/batch]"%(timers["batch"].sum,
                                                                                                                             timers["batch"].avg,
                                                                                                                             timers["forward"].avg,
                                                                                                                             timers["backward"].avg,
                                                                                                                             timers["accuracy"].avg,
                                                                                                                             timers["data"].avg))
    print("  Losses: ")
    for name,meter in loss_meters.items():
        print("    ",name,": ",meter.avg)
    print("  Accuracies: ")
    for name,meter in acc_meters.items():
        print("    ",name,": ",meter.avg)
    print("------------------------------------------------------------------------")


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
