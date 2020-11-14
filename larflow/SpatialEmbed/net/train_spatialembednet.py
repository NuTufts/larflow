#!/bin/env python

"""
TRAINING SCRIPT FOR 3D VOXEL SPATIAL EMBED CLUSTERING
"""

## IMPORT

# python,numpy
import os,sys,commands
import shutil
import time
import traceback
import numpy as np

# ROOT, larcv
import ROOT as rt
from ROOT import std
from larcv import larcv
from larflow import larflow

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

# tensorboardX
from torch.utils.tensorboard import SummaryWriter

# network and loss modules
from spatialembednet import SpatialEmbedNet
from loss_spatialembed import SpatialEmbedLoss

# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT=False
RESUME_OPTIM_FROM_CHECKPOINT=False
RUNPROFILER=False
CHECKPOINT_FILE="train_kps_no_ssnet/checkpoint.1262000th.tar"
EXCLUDE_NEG_EXAMPLES = False
TRAIN_NET_VERBOSE=False
TRAIN_LOSS_VERBOSE=False
FREEZE_LAYERS=False

# Hard example training parameters (not yet implemented)
# =======================================================
HARD_EXAMPLE_TRAINING=False
HARDEX_GPU="cuda:0"
HARDEX_CHECKPOINT_FILE="train_kps_nossnet/checkpoint.260000th.tar"

# TRAINING+VALIDATION DATA PATHS
# ================================
#TRAIN_DATA_FOLDER="/home/twongj01/data/larmatch_kps_data/"
#INPUTFILE_TRAIN=["larmatch_kps_train_p06.root",
#                 "larmatch_kps_train_p07.root",
#                 "larmatch_kps_train_p08.root",
#                 "larmatch_kps_train_p09.root",
#                 "larmatch_kps_train_p01.root",
#                 "larmatch_kps_train_p02.root",
#                 "larmatch_kps_train_p03.root",
#                 "larmatch_kps_train_p04.root"]
#INPUTFILE_VALID=["larmatch_kps_train_p05.root"]
TRAIN_DATA_FOLDER="/home/twongj01/working/spatial_embed_net/ubdl/larflow/larflow/SpatialEmbed/net/"
INPUTFILE_TRAIN=["test_s3dembed.root"]
INPUTFILE_VALID=["test_s3dembed.root"]
TICKBACKWARD=False # Is data in tick-backward format (typically no)

# TRAINING PARAMETERS
# =======================
START_ITER  = 0
NUM_ITERS   = 2000

BATCHSIZE_TRAIN=1  # batches per training iteration
BATCHSIZE_VALID=1  # batches per validation iteration
NWORKERS_TRAIN=2   # number of threads data loader will use for training set
NWORKERS_VALID=2   # number of threads data loader will use for validation set

NBATCHES_per_itertrain = 1
NBATCHES_per_step      = 1 # if >1 we use gradient accumulation
trainbatches_per_print = -1

NBATCHES_per_itervalid = 1
validbatches_per_print = -1

ITER_PER_VALID = 10000000

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
ITER_PER_CHECKPOINT=1000000000000
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
    voxel_dims = (2048, 1024, 4096)
    model = SpatialEmbedNet(3, voxel_dims,
                            input_nfeatures=3,
                            nclasses=1,
                            num_unet_layers=5,
                            stem_nfeatures=32).to(DEVICE)
    model.init_embedout()

    # define loss function (criterion) and optimizer
    criterion = SpatialEmbedLoss(dim_nvoxels=voxel_dims)
    
    model_dict = {"spatialembed":model}
    parameters = []
    for n,model in model_dict.items():
        for p in model.parameters():
            parameters.append( p )
    
    if False:
        # DUMP MODEL (for debugging)
        print model

        # uncomment to dump model parameters
        if False:
            print model.module.parameters
            return
        
        if False: sys.exit(-1)
    
    # Resume training option
    if RESUME_FROM_CHECKPOINT:
        print "RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE
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
                    print "skip re-loading keypoint"
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
        

    # training parameters
    lr = 1e-3
    momentum = 0.9
    weight_decay = 1.0e-3

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
    iotrain = {"spatialembed":larflow.spatialembed.Prep3DSpatialEmbed(traindata_v)}

    validdata_v = std.vector("std::string")()
    for x in INPUTFILE_VALID:
        validdata_v.push_back( TRAIN_DATA_FOLDER+"/"+x )
    iovalid = {"spatialembed":larflow.spatialembed.Prep3DSpatialEmbed(validdata_v)}


    TRAIN_NENTRIES = iotrain["spatialembed"].getTree().GetEntries()
    iter_per_epoch = TRAIN_NENTRIES/(itersize_train)
    epochs = float(NUM_ITERS)/float(TRAIN_NENTRIES)
    VALID_NENTRIES = iovalid["spatialembed"].getTree().GetEntries()

    print "Number of iterations to run: ",NUM_ITERS
    print "Entries in the training set: ",TRAIN_NENTRIES
    print "Entries in the validation set: ",VALID_NENTRIES
    print "Entries per iter (train): ",itersize_train
    print "Entries per iter (valid): ",itersize_valid
    print "Number of (training) Epochs to run: ",epochs    
    print "Iterations per epoch: ",iter_per_epoch

    if False:
        print "passed setup successfully"
        sys.exit(0)

    # TRAINING LOOP
    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:

        # LOOP OVER FIXED NUMBER OF INTERATIONS
        for ii in range(START_ITER, NUM_ITERS):

            # modify learning rate based on interation number (not used)
            adjust_learning_rate(optimizer, ii, lr)
            print "MainLoop Iter:%d Epoch:%d.%d "%(ii,ii/iter_per_epoch,ii%iter_per_epoch),
            for param_group in optimizer.param_groups:
                print "lr=%.3e"%(param_group['lr']),
                print

            # train for one iteration
            try:
                _ = train(iotrain, DEVICE, BATCHSIZE_TRAIN,
                          model_dict, criterion, optimizer,
                          NBATCHES_per_itertrain, NBATCHES_per_step,
                          ii, trainbatches_per_print, hardex_model=hardex_model)
                
            except Exception,e:
                print "Error in training routine!"            
                print e.message
                print e.__class__.__name__
                traceback.print_exc(e)
                break

            print "made it to train!"

            # evaluate on validation set at a given interval (every ITER_PER_VALID training iterations)
            if ii%ITER_PER_VALID==0 and ii>0:
                try:
                    totloss, totacc = validate(iovalid, DEVICE, BATCHSIZE_VALID,
                                               model_dict, criterion,
                                               NBATCHES_per_itervalid, ii,
                                               validbatches_per_print)
                except Exception,e:
                    print "Error in validation routine!"            
                    print e.message
                    print e.__class__.__name__
                    traceback.print_exc(e)
                    break

                # remember best prec@1 and save checkpoint
                prec1   = totacc
                is_best =  prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)

                # check point for best model
                if is_best:
                    print "Saving best model"
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
                print "saving periodic checkpoint"
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
        print "saving last state"
        save_checkpoint({
            'iter':NUM_ITERS,
            'epoch': float(NUM_ITERS)/iter_per_epoch,
            'state_embed': model_dict["spatialembed"].state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, False, NUM_ITERS)


    print "FIN"
    print "PROFILER"
    if RUNPROFILER:
        print prof
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
    lossnames    = ["total","instance","seed","var"]
    flowaccnames = ["iou"]

    acc_meters  = {}
    for n in flowaccnames:
        acc_meters[n] = AverageMeter()

    loss_meters = {}
    for n in lossnames:
        loss_meters[n] = AverageMeter()

    time_meters = {}
    for l in ["batch","data","forward","backward","loss"]:
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
            
        #data = train_loader["spatialembed"].getNextTreeEntryDataAsArray()
        data = train_loader["spatialembed"].getTreeEntryDataAsArray(0)
        
        # convert into torch tensors
        coord_t    = torch.from_numpy( data["coord_t"] ).to(device)
        feat_t     = torch.from_numpy( data["feat_t"] ).to(device)
        instance_t = torch.from_numpy( data["instance_t"] ).to(device)
        coord_t.requires_grad = False
        feat_t.requires_grad = False
        instance_t.requires_grad = False        

        #for p in xrange(3):
        #    feat_t[p] = torch.clamp( feat_t[p], 0, ADC_MAX )
        train_entry = train_loader["spatialembed"].getCurrentEntry() 
        print("loaded entry[",train_entry,"] voxel entries: ",data["coord_t"].shape)

        
        # compute output
        if RUNPROFILER:
            torch.cuda.synchronize()
        
        # run network
        start = time.time()    
        embed_t,seed_t = model["spatialembed"]( coord_t, feat_t, device, verbose=TRAIN_NET_VERBOSE )
        dt_forward = time.time()-start

        # Calculate the loss
        start = time.time()
        loss,ninstances,iou_out,_loss = criterion( coord_t, embed_t, seed_t, instance_t, verbose=TRAIN_LOSS_VERBOSE, calc_iou=True )
        dt_loss = time.time()-start

        if RUNPROFILER:
            torch.cuda.synchronize()
        time_meters["forward"].update(dt_forward)
        time_meters["loss"].update(dt_loss)

        # compute gradient and do SGD step
        if RUNPROFILER:
            torch.cuda.synchronize()
        start = time.time()
        loss.backward()
        dt_backward = time.time()-start

        # only step, i.e. adjust weights every nbatches_per_step or if last batch
        if (i>0 and (i+1)%nbatches_per_step==0) or i+1==nbatches:
            print "batch %d of %d. making step, then clearing gradients. nbatches_per_step=%d"%(i,nbatches,nbatches_per_step)
            optimizer.step()
            optimizer.zero_grad()
            
        if RUNPROFILER:        
            torch.cuda.synchronize()                
        time_meters["backward"].update(dt_backward)

        # measure accuracy and record loss
        end = time.time()

        # update loss meters
        loss_meters["total"].update( loss.detach().item(),    nbatches_per_step )
        loss_meters["instance"].update( _loss[0],  ninstances )
        loss_meters["seed"].update( _loss[1],  ninstances )
        loss_meters["var"].update( _loss[2], ninstances )
        
        # update acc meters
        acc_meters["iou"].update( iou_out, ninstances )

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
    
    return loss_meters['total'].avg


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
    flowaccnames = ("lm_pos","lm_neg","lm_all","ss-bg","shower","track","ssnet-all","kp_nu","kp_trk","kp_shr","paf")

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
    for name,m in model.items():
        m.eval()
    
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
        for p in xrange(3):
            feat_t[p] = torch.clamp( feat_t[p], 0, ADC_MAX )
        
        print "loaded valid entry: ",flowdata["entry"]
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
            print "[larmatch valid] match-pred=",match_pred_t.shape

            # evaluate ssnet classifier
            if TRAIN_SSNET:
                ssnet_pred_t = model['ssnet'].forward( feat_triplet_t )
                ssnet_pred_t = ssnet_pred_t.reshape( (ssnet_pred_t.shape[1],ssnet_pred_t.shape[2]) )
                ssnet_pred_t = torch.transpose( ssnet_pred_t, 1, 0 )
                #ssnet_pred_t = ssnet_pred_t.reshape( (ssnet_pred_t.shape[-1]) )
                print "[ssnet valid] ssnet-pred=",ssnet_pred_t.shape
            else:
                ssnet_pred_t = None
        
            # evaluate keypoint regression
            if TRAIN_KP:
                kplabel_pred_t = model['kplabel'].forward( feat_triplet_t )
                kplabel_pred_t = kplabel_pred_t.reshape( (kplabel_pred_t.shape[1], kplabel_pred_t.shape[2]) )
                kplabel_pred_t = torch.transpose( kplabel_pred_t, 1, 0 )
                print "[keypoint valid] kplabel-pred=",kplabel_pred_t.shape
            else:
                kplabel_pred_t = None
        
            # next evaluate keypoint shift
            if TRAIN_KPSHIFT:
                kpshift_pred_t = model['kpshift'].forward( feat_triplet_t )
                kpshift_pred_t = kpshift_pred_t.reshape( (kpshift_pred_t.shape[1],kpshift_pred_t.shape[2]) )
                kpshift_pred_t = torch.transpose( kpshift_pred_t, 1, 0 )
                print "[keypoint shift valid] kpshift-pred=",kpshift_pred_t.shape
            else:
                kpshift_pred_t = None

            # next evaluate affinity field predictor
            if TRAIN_PAF:
                paf_pred_t = model["paf"].forward( feat_triplet_t )
                print "[larmatch valid]: paf pred=",paf_pred_t.shape
                paf_pred_t = paf_pred_t.reshape( (paf_pred_t.shape[1],paf_pred_t.shape[2]) )
                paf_pred_t = torch.transpose( paf_pred_t, 1, 0 )
                print "[larmatch valid]: paf pred=",paf_pred_t.shape
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
        ssbg_correct   = ssnet_correct[ ssnet_label_t==0 ].sum().item()    
        track_correct  = ssnet_correct[ ssnet_label_t==1 ].sum().item()
        shower_correct = ssnet_correct[ ssnet_label_t==2 ].sum().item()    
        ssnet_tot_correct = ssnet_correct.sum().item()
        if ssnet_label_t.eq(2).sum().item()>0:
            acc_meters["shower"].update( float(shower_correct)/float(ssnet_label_t.eq(2).sum().item()) )
        if ssnet_label_t.eq(1).sum().item()>0:
            acc_meters["track"].update(  float(track_correct)/float(ssnet_label_t.eq(1).sum().item())  )
        if ssnet_label_t.eq(0).sum().item()>0:
            acc_meters["ss-bg"].update(  float(ssbg_correct)/float(ssnet_label_t.eq(0).sum().item())  )
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
        for c in xrange(3):
            kp_n_pos = float(kp_label[:,c].gt(0.5).sum().item())
            kp_pos   = float(kp_pred[:,c].gt(0.5)[ kp_label[:,c].gt(0.5) ].sum().item())
            print "kp[",c,"] n_pos[>0.5]: ",kp_n_pos," pred[>0.5]: ",kp_pos
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
        print "paf: npos=",paf_npos," acc=",paf_acc
        if paf_npos>0:
            acc_meters["paf"].update( paf_acc )
    
    
    return True

def dump_lr_schedule( startlr, numepochs ):
    for epoch in range(0,numepochs):
        lr = startlr*(0.5**(epoch//300))
        if epoch%10==0:
            print "Epoch [%d] lr=%.3e"%(epoch,lr)
    print "Epoch [%d] lr=%.3e"%(epoch,lr)
    return

def prep_status_message( descripter, iternum, acc_meters, loss_meters, timers, istrain ):
    print "------------------------------------------------------------------------"
    print " Iter[",iternum,"] ",descripter
    print "  Time (secs): iter[%.2f] batch[%.3f] Forward[%.3f/batch] Backward[%.3f/batch] Data[%.3f/batch]"%(timers["batch"].sum,
                                                                                                             timers["batch"].avg,
                                                                                                             timers["forward"].avg,
                                                                                                             timers["backward"].avg,
                                                                                                             timers["data"].avg)    
    print "  Losses: "
    for name,meter in loss_meters.items():
        print "    ",name,": ",meter.avg
    print "  Accuracies: "
    for name,meter in acc_meters.items():
        print "    ",name,": ",meter.avg
    print "------------------------------------------------------------------------"    


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
