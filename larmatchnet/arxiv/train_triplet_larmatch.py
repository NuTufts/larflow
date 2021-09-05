#!/bin/env python

## IMPORT

# python,numpy
import os,sys,commands
import shutil
import time
import traceback
import numpy as np

# ROOT, larcv
import ROOT as rt
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

# dataset interface
from larcvdataset.larcvserver import LArCVServer

from larmatch import LArMatch
from load_larmatch_triplets import load_larmatch_triplets
from loss_larmatch import SparseLArMatchLoss

# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT=True
RUNPROFILER=False
CHECKPOINT_FILE="checkpoint.2000th.tar"

TRAIN_DATA_FOLDER="/home/twongj01/data/larmatch_triplet_data"
INPUTFILE_TRAIN=["larmatch_train_p00.root",
                 "larmatch_train_p01.root",
                 "larmatch_train_p02.root",
                 "larmatch_train_p03.root",
                 "larmatch_train_p04.root"]
INPUTFILE_VALID=["larmatch_valid_p00.root",
                 "larmatch_valid_p01.root"]
TICKBACKWARD=False

# TRAINING PARAMETERS
# =======================
START_ITER  = 2000
NUM_ITERS   = 150000
TEST_NUM_MATCH_PAIRS = 50000

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

# IMAGE/LOSS PARAMETERS
# =====================
IMAGE_WIDTH=3456
IMAGE_HEIGHT=1024
ADC_THRESH=10.0
VISI_WEIGHT=0.0
CONSISTENCY_WEIGHT=0.1
USE_VISI=False
DEVICE_IDS=[0]
# map multi-training weights 
CHECKPOINT_MAP_LOCATIONS={"cuda:0":"cuda:0",
                          "cuda:1":"cuda:1"}
CHECKPOINT_MAP_LOCATIONS=None
CHECKPOINT_FROM_DATA_PARALLEL=False
ITER_PER_CHECKPOINT=1000
PREDICT_CLASSVEC=True
# ===================================================

# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights
writer = SummaryWriter()
train_entry = 2000
valid_entry = 200
TRAIN_NENTRIES = 0
VALID_NENTRIES = 0

def main():

    global best_prec1
    global writer
    global num_iters
    global TRAIN_NENTRIES
    global VALID_NENTRIES

    if GPUMODE:
        DEVICE = torch.device("cuda:%d"%(DEVICE_IDS[0]))
    else:
        DEVICE = torch.device("cpu")
    
    # create model, mark it to run on the GPU
    model = LArMatch(neval=TEST_NUM_MATCH_PAIRS).to(DEVICE)

    if False:
        # DUMP MODEL
        print model
        sys.exit(-1)
    
    # Resume training option
    if RESUME_FROM_CHECKPOINT:
        print "RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE
        checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS ) # load weights to gpuid
        best_prec1 = checkpoint["best_prec1"]
        if CHECKPOINT_FROM_DATA_PARALLEL:
            model = nn.DataParallel( model, device_ids=DEVICE_IDS ) # distribute across device_ids
        model.load_state_dict(checkpoint["state_dict"])

    if GPUMODE and not CHECKPOINT_FROM_DATA_PARALLEL and len(DEVICE_IDS)>1:
        model = nn.DataParallel( model, device_ids=DEVICE_IDS ) # distribute across device_ids

    # uncomment to dump model
    if False:
        print "Loaded model: ",model
        #print model.module.source_encoder.variable
        print model.module.parameters
        return

    # define loss function (criterion) and optimizer
    criterion = SparseLArMatchLoss()

    # training parameters
    lr = 1.0e-3
    momentum = 0.9
    weight_decay = 1.0e-4

    # training variables
    itersize_train         = BATCHSIZE_TRAIN*NBATCHES_per_itertrain # number of images per iteration
    itersize_valid         = BATCHSIZE_VALID*NBATCHES_per_itervalid # number of images per validation    
    
    # SETUP OPTIMIZER
    # ADAM
    # betas default: (0.9, 0.999) for (grad, grad^2). smoothing coefficient for grad. magnitude calc.
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=lr, 
                                 weight_decay=weight_decay)
    if RESUME_FROM_CHECKPOINT:
        optimizer.load_state_dict( checkpoint["optimizer"] )

    # optimize algorithms based on input size (good if input size is constant)
    cudnn.benchmark = False

    # LOAD THE DATASET
    traindata = [ TRAIN_DATA_FOLDER+"/"+x for x in INPUTFILE_TRAIN ]
    iotrain = rt.TChain("larmatchtriplet")
    for f in traindata:
        iotrain.Add(f)

    validdata = [ TRAIN_DATA_FOLDER+"/"+x for x in INPUTFILE_VALID ]            
    iovalid = rt.TChain("larmatchtriplet")
    for f in validdata:
        iovalid.Add(f)

    TRAIN_NENTRIES = iotrain.GetEntries()
    iter_per_epoch = TRAIN_NENTRIES/(itersize_train)
    epochs = float(NUM_ITERS)/float(TRAIN_NENTRIES)
    VALID_NENTRIES = iovalid.GetEntries()

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

    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:

        for ii in range(START_ITER, NUM_ITERS):

            adjust_learning_rate(optimizer, ii, lr)
            print "MainLoop Iter:%d Epoch:%d.%d "%(ii,ii/iter_per_epoch,ii%iter_per_epoch),
            for param_group in optimizer.param_groups:
                print "lr=%.3e"%(param_group['lr']),
                print

            # train for one iteration
            try:
                _ = train(iotrain, DEVICE, BATCHSIZE_TRAIN,
                          model, criterion, optimizer,
                          NBATCHES_per_itertrain, NBATCHES_per_step,
                          ii, trainbatches_per_print)
                
            except Exception,e:
                print "Error in training routine!"            
                print e.message
                print e.__class__.__name__
                traceback.print_exc(e)
                break

            # evaluate on validation set
            if ii%ITER_PER_VALID==0 and ii>0:
                try:
                    totloss, totacc = validate(iovalid, DEVICE, BATCHSIZE_VALID,
                                               model, criterion,
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
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, -1)

            # periodic checkpoint
            if ii>0 and ii%ITER_PER_CHECKPOINT==0:
                print "saving periodic checkpoint"
                save_checkpoint({
                    'iter':ii,
                    'epoch': ii/iter_per_epoch,
                    'state_dict': model.state_dict(),
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
            'state_dict': model.state_dict(),
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
          iiter, print_freq):
    """
    train_loader: TChain with data
    device: device we are training on
    batchsize: number of images per batch
    model: network are training
    criterion: the loss we are using
    optimizer: optimizer
    nbatches: number of batches to run in this iteraction
    nbatches_per_step: allows for gradient accumulation if >1
    iiter: current iteration
    print_freq: number of batches we run before printing out some statistics
    """
    global writer
    global train_entry

    # timers for profiling
    batch_time = AverageMeter() # total for batch
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    acc_time = AverageMeter()

    # accruacy and loss meters
    lossnames    = ("total")
    flowaccnames = ("pos_correct","neg_correct","pos_wrong","neg_wrong","tot_correct")

    acc_meters  = {}
    for n in flowaccnames:
        acc_meters[n] = AverageMeter()

    loss_meters = {"total":AverageMeter()}    

    time_meters = {}
    for l in ["batch","data","forward","backward","accuracy"]:
        time_meters[l] = AverageMeter()
    
    # switch to train mode
    model.train()

    # clear gradients
    optimizer.zero_grad()
    nnone = 0
    for i in range(0,nbatches):
        #print "iiter ",iiter," batch ",i," of ",nbatches
        batchstart = time.time()

        # GET THE DATA
        end = time.time()
            
        flowdata = load_larmatch_triplets( train_loader, train_entry, TEST_NUM_MATCH_PAIRS, True )
        if train_entry+1<TRAIN_NENTRIES:
            train_entry += 1
        else:
            train_entry = 0
        print "loaded train entry: ",flowdata["entry"]
        
        # compute output
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()

        coord_t = [ torch.from_numpy( flowdata['coord_%s'%(p)] ).to(device) for p in [0,1,2] ]
        feat_t  = [ torch.from_numpy( flowdata['feat_%s'%(p)] ).to(device) for p in [0,1,2] ]
        match_t = torch.from_numpy( flowdata['matchpairs'] ).to(device)
        label_t = torch.from_numpy( flowdata['labels'] ).to(device)

        # first get feature vectors
        feat_u_t, feat_v_t, feat_y_t = model.forward_features( coord_t[0], feat_t[0],
                                                               coord_t[1], feat_t[1],
                                                               coord_t[2], feat_t[2], 1 )

        # next evaluate match classifier
        pred_t = model.classify_triplet( feat_u_t, feat_v_t, feat_y_t, match_t, flowdata['npairs'], device )

        pred_t = pred_t.reshape( (pred_t.shape[-1]) )
                
        totloss = criterion.forward_triplet( pred_t[:flowdata['npairs']], label_t[:flowdata['npairs']] )
            
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
            print "average over batches per step when gradient accumulating."
            totloss /= float(nbatches_per_step)
        
        # of course, we calculate gradients for this batch
        totloss.backward()
        # only step, i.e. adjust weights every nbatches_per_step or if last batch
        if (i>0 and (i+1)%nbatches_per_step==0) or i+1==nbatches:
            print "batch %d of %d. making step, then clearing gradients. nbatches_per_step=%d"%(i,nbatches,nbatches_per_step)
            optimizer.step()
            optimizer.zero_grad()
            
        if RUNPROFILER:        
            torch.cuda.synchronize()                
        time_meters["backward"].update(time.time()-end)

        # measure accuracy and record loss
        end = time.time()

        # update loss meters
        print loss_meters.keys()
        loss_meters["total"].update( totloss.item()*float(nbatches_per_step) )
        
        # measure accuracy and update meters
        acc = accuracy(pred_t[:flowdata['npairs']],label_t[:flowdata['npairs']],acc_meters)
            
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
    
    return loss_meters['total'].avg,acc_meters['tot_correct'].avg


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
    lossnames    = ("total")
    flowaccnames = ("pos_correct","neg_correct","pos_wrong","neg_wrong","tot_correct")

    acc_meters  = {}
    for n in flowaccnames:
        acc_meters[n] = AverageMeter()

    loss_meters = {"total":AverageMeter()}    

    time_meters = {}
    for l in ["batch","data","forward","backward","accuracy"]:
        time_meters[l] = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    iterstart = time.time()
    nnone = 0
    for i in range(0,nbatches):
        batchstart = time.time()
        
        tdata_start = time.time()

        flowdata = load_larmatch_triplets( val_loader, valid_entry, TEST_NUM_MATCH_PAIRS, True )
        if valid_entry+1<VALID_NENTRIES:
            valid_entry += 1
        else:
            valid_entry = 0        

        coord_t = [ torch.from_numpy( flowdata['coord_%s'%(p)] ).to(device) for p in [0,1,2] ]
        feat_t  = [ torch.from_numpy( flowdata['feat_%s'%(p)] ).to(device) for p in [0,1,2] ]
        match_t = torch.from_numpy( flowdata['matchpairs'] ).to(device)
        label_t = torch.from_numpy( flowdata['labels'] ).to(device)

        print "loaded train entry: ",flowdata["entry"]
        time_meters["data"].update( time.time()-tdata_start )
        
        # compute model output
        if RUNPROFILER:
            torch.cuda.synchronize()
        tforward = time.time()
               
        # first get feature vectors
        feat_u_t, feat_v_t, feat_y_t = model.forward_features( coord_t[0], feat_t[0],
                                                               coord_t[1], feat_t[1],
                                                               coord_t[2], feat_t[2], 1 )
        
        # next evaluate match classifier
        pred_t = model.classify_triplet( feat_u_t, feat_v_t, feat_y_t, match_t, flowdata['npairs'], device )

        pred_t = pred_t.reshape( (pred_t.shape[-1]) )        
                
        totloss = criterion.forward_triplet( pred_t[:flowdata['npairs']],
                                             label_t[:flowdata['npairs']] )

                
        time_meters["forward"].update(time.time()-tforward)

        # update loss meters
        print loss_meters.keys()
        loss_meters["total"].update( totloss.item() )
        
        # measure accuracy and update meters
        end = time.time()
        acc = accuracy(pred_t[:flowdata['npairs']],label_t[:flowdata['npairs']],acc_meters)
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
    
    return loss_meters['total'].avg,acc_meters['tot_correct'].avg

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


def accuracy(predict_t, truth_t, acc_meters):
    """Computes the accuracy metrics."""
    pred = predict_t.detach()
    
    pos_correct = (pred.gt(0.0).type(torch.int)*truth_t).sum().to(torch.device("cpu")).item()
    pos_wrong   = (pred.lt(0.0).type(torch.int)*truth_t).sum().to(torch.device("cpu")).item()
    neg_correct = (pred.lt(0.0)*truth_t.eq(0)).sum().to(torch.device("cpu")).item()
    neg_wrong   = (pred.gt(0.0)*truth_t.eq(0)).sum().to(torch.device("cpu")).item()

    npos = float(truth_t.sum().to(torch.device("cpu")).item())
    nneg = float(truth_t.eq(0).sum().to(torch.device("cpu")).item())

    acc_meters["pos_correct"].update( float(pos_correct)/npos )
    acc_meters["pos_wrong"].update(   float(pos_wrong)/npos )
    acc_meters["neg_correct"].update( float(neg_correct)/nneg )
    acc_meters["neg_wrong"].update(   float(neg_wrong)/nneg )
    acc_meters["tot_correct"].update( float(pos_correct+neg_correct)/(npos+nneg) )
    
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
    print "  Time (secs): iter[%.2f] batch[%.3f] Forward[%.3f/batch] Backward[%.3f/batch] Acc[%.3f/batch] Data[%.3f/batch]"%(timers["batch"].sum,
                                                                                                                             timers["batch"].avg,
                                                                                                                             timers["forward"].avg,
                                                                                                                             timers["backward"].avg,
                                                                                                                             timers["accuracy"].avg,
                                                                                                                             timers["data"].avg)    
    print "  Loss: Total[%.3e]"%(loss_meters["total"].avg)
    print "  Acc: pos-correct[%.2f] neg-correct[%.2f] pos-wrong[%.2f] neg-wrong[%.2f] TOT[%.2f]"%(acc_meters["pos_correct"].avg,
                                                                                                  acc_meters["neg_correct"].avg,
                                                                                                  acc_meters["pos_wrong"].avg,
                                                                                                  acc_meters["neg_wrong"].avg,
                                                                                                  acc_meters["tot_correct"].avg)
    print "------------------------------------------------------------------------"    


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
