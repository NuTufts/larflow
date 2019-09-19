#!/bin/env python

## IMPORT

# python,numpy
import os,sys,commands
import shutil
import time
import traceback
import numpy as np

# ROOT, larcv
import ROOT
from larcv import larcv

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
from tensorboardX import SummaryWriter

# dataset interface
from larcvdataset.larcvserver import LArCVServer

from sparselarflow import SparseLArFlow
from sparselarflowdata import load_larflow_larcvdata
from loss_sparse_larflow import SparseLArFlow3DConsistencyLoss

# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT=True
RUNPROFILER=False
#CHECKPOINT_FILE="train_y2u_checkpoints/checkpoint.1700th.tar"
#CHECKPOINT_FILE="train_y2u_checkpoints/checkpoint.10000th.tar"
#CHECKPOINT_FILE="train_y2u_checkpoints/checkpoint.17000th.tar"
#CHECKPOINT_FILE="train_y2u_checkpoints/checkpoint.24500th.tar"
#CHECKPOINT_FILE="train_y2u_checkpoints/checkpoint.25500th.tar"
#CHECKPOINT_FILE="train_y2u_checkpoints/checkpoint.31600th.tar"
#CHECKPOINT_FILE="train_y2u_checkpoints/checkpoint.35000th.tar"
CHECKPOINT_FILE="train_y2u_checkpoints/checkpoint.36100th.tar"
INPUTFILE_TRAIN=["/mnt/hdd1/twongj01/sparse_larflow_data/larflow_sparsify_cropped_train1_v5.root",
                 "/mnt/hdd1/twongj01/sparse_larflow_data/larflow_sparsify_cropped_train2_v5.root",
                 "/mnt/hdd1/twongj01/sparse_larflow_data/larflow_sparsify_cropped_train3_v5.root"]
INPUTFILE_VALID="/mnt/hdd1/twongj01/sparse_larflow_data/larflow_sparsify_cropped_valid_v5.root"
TICKBACKWARD=False
start_iter  = 36101
num_iters   = 45000
IMAGE_WIDTH=832
IMAGE_HEIGHT=512
BATCHSIZE_TRAIN=20
BATCHSIZE_VALID=10
NWORKERS_TRAIN=3
NWORKERS_VALID=1
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
ITER_PER_CHECKPOINT=100
# ===================================================

# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights
writer = SummaryWriter()

def main():

    global best_prec1
    global writer
    global num_iters

    if GPUMODE:
        DEVICE = torch.device("cuda:%d"%(DEVICE_IDS[0]))
    else:
        DEVICE = torch.device("cpu")
    
    # create model, mark it to run on the GPU
    imgdims = 2
    ninput_features  = 16
    noutput_features = 16
    nplanes = 5
    nfeatures_per_layer = [16,16,32,32,64]
    flowdirs = ['y2u']
    
    model = SparseLArFlow( (IMAGE_HEIGHT,IMAGE_WIDTH), imgdims,
                           ninput_features, noutput_features,
                           nplanes, features_per_layer=nfeatures_per_layer,
                           flowdirs=flowdirs).to(DEVICE)
    
    # Resume training option
    if RESUME_FROM_CHECKPOINT:
        print "RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE
        checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS ) # load weights to gpuid
        best_prec1 = checkpoint["best_prec1"]
        if CHECKPOINT_FROM_DATA_PARALLEL:
            model = nn.DataParallel( model, device_ids=DEVICE_IDS ) # distribute across device_ids
        model.load_state_dict(checkpoint["state_dict"])

    if not CHECKPOINT_FROM_DATA_PARALLEL and len(DEVICE_IDS)>1:
        model = nn.DataParallel( model, device_ids=DEVICE_IDS ).to(device=DEVICE) # distribute across device_ids

    # uncomment to dump model
    if False:
        print "Loaded model: ",model
        return

    # define loss function (criterion) and optimizer
    maxdist   = 200.0
    criterion = SparseLArFlow3DConsistencyLoss(IMAGE_HEIGHT, IMAGE_WIDTH,
                                               larcv_version=1,
                                               calc_consistency=False).to(device=DEVICE)

    # training parameters
    lr = 1.0e-4
    momentum = 0.9
    weight_decay = 1.0e-4

    # training length
    batchsize_train = BATCHSIZE_TRAIN
    batchsize_valid = BATCHSIZE_VALID#*len(DEVICE_IDS)
    start_epoch = 0
    epochs      = 10
    iter_per_epoch = None # determined later
    iter_per_valid = 10


    nbatches_per_itertrain = 5
    itersize_train         = batchsize_train*nbatches_per_itertrain
    trainbatches_per_print = -1
    
    nbatches_per_itervalid = 5
    itersize_valid         = batchsize_valid*nbatches_per_itervalid
    validbatches_per_print = -1

    # SETUP OPTIMIZER

    # SGD w/ momentum
    #optimizer = torch.optim.SGD(model.parameters(), lr,
    #                            momentum=momentum,
    #                            weight_decay=weight_decay)
    
    # ADAM
    # betas default: (0.9, 0.999) for (grad, grad^2). smoothing coefficient for grad. magnitude calc.
    #optimizer = torch.optim.Adam(model.parameters(), 
    #                             lr=lr, 
    #                             weight_decay=weight_decay)
    # RMSPROP
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
    
    # optimize algorithms based on input size (good if input size is constant)
    cudnn.benchmark = True

    # LOAD THE DATASET    
    iotrain = load_larflow_larcvdata( "train", INPUTFILE_TRAIN,
                                      BATCHSIZE_TRAIN, NWORKERS_TRAIN,
                                      producer_name="sparsecropdual",
                                      nflows=len(flowdirs),
                                      tickbackward=TICKBACKWARD,
                                      readonly_products=None )
    iovalid = load_larflow_larcvdata( "valid", INPUTFILE_VALID,
                                      BATCHSIZE_VALID, NWORKERS_VALID,
                                      producer_name="sparsecropdual",                                      
                                      nflows=len(flowdirs),
                                      tickbackward=TICKBACKWARD,
                                      readonly_products=None )

    print "pause to give time to feeders"

    NENTRIES = len(iotrain)
    #NENTRIES = 100000
    print "Number of entries in training set: ",NENTRIES

    if NENTRIES>0:
        iter_per_epoch = NENTRIES/(itersize_train)
        if num_iters is None:
            # we set it by the number of request epochs
            num_iters = (epochs-start_epoch)*NENTRIES
        else:
            epochs = num_iters/NENTRIES
    else:
        iter_per_epoch = 1

    print "Number of epochs: ",epochs
    print "Iter per epoch: ",iter_per_epoch

    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:

        # Resume training option
        #if RESUME_FROM_CHECKPOINT:
        #    print "RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE
        #    checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS )
        #    best_prec1 = checkpoint["best_prec1"]
        #    model.load_state_dict(checkpoint["state_dict"])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #if GPUMODE:
        #    optimizer.cuda(GPUID)

        for ii in range(start_iter, num_iters):

            adjust_learning_rate(optimizer, ii, lr)
            print "MainLoop Iter:%d Epoch:%d.%d "%(ii,ii/iter_per_epoch,ii%iter_per_epoch),
            for param_group in optimizer.param_groups:
                print "lr=%.3e"%(param_group['lr']),
                print

            # train for one iteration
            try:
                _ = train(iotrain, DEVICE, BATCHSIZE_TRAIN, model,
                          criterion, optimizer,
                          nbatches_per_itertrain, ii, trainbatches_per_print)
                
            except Exception,e:
                print "Error in training routine!"            
                print e.message
                print e.__class__.__name__
                traceback.print_exc(e)
                break

            # evaluate on validation set
            if ii%iter_per_valid==0 and ii>0:
                try:
                    totloss, flow1acc5, flow2acc5 = validate(iovalid, DEVICE, BATCHSIZE_VALID,
                                                             model, criterion,
                                                             nbatches_per_itervalid, ii,
                                                             validbatches_per_print)
                except Exception,e:
                    print "Error in validation routine!"            
                    print e.message
                    print e.__class__.__name__
                    traceback.print_exc(e)
                    break

                # remember best prec@1 and save checkpoint
                prec1   = 0.5*(flow1acc5+flow2acc5)
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
            'iter':num_iters,
            'epoch': num_iters/iter_per_epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, False, num_iters)


    print "FIN"
    print "PROFILER"
    if RUNPROFILER:
        print prof
    writer.close()


def train(train_loader, device, batchsize, model, criterion, optimizer, nbatches, iiter, print_freq):

    global writer

    # timers for profiling
    batch_time = AverageMeter() # total for batch
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    acc_time = AverageMeter()

    # accruacy and loss meters
    lossnames    = ("total","flow1","flow2","visi1","visi2","consist3d")
    flowaccnames = ("flow%d<5pix","flow%d<10pix","flow%d<20pix","flow%d<50pix")
    consistnames = ("dist2d")

    acc_meters  = {}
    for i in [1,2]:
        for n in flowaccnames:
            acc_meters[n%(i)] = AverageMeter()
    acc_meters["dist2d"] = AverageMeter()
    acc_meters["flow1@vis"] = AverageMeter()
    acc_meters["flow2@vis"] = AverageMeter()

    loss_meters = {}    
    for l in lossnames:
        loss_meters[l] = AverageMeter()

    time_meters = {}
    for l in ["batch","data","forward","backward","accuracy"]:
        time_meters[l] = AverageMeter()
    
    # switch to train mode
    model.train()

    nnone = 0
    for i in range(0,nbatches):
        #print "iiter ",iiter," batch ",i," of ",nbatches
        batchstart = time.time()

        # GET THE DATA
        end = time.time()
        time_meters["data"].update(time.time()-end)
            
        flowdict = train_loader.get_tensor_batch(device)        
        # ['src', 'flow1', 'coord', 'flow2', 'tar2', 'tar1']
        coord_t  = flowdict["coord"]
        srcpix_t = flowdict["src"]
        tarpix_flow1_t = flowdict["tar1"]
        tarpix_flow2_t = flowdict["tar2"]
        truth_flow1_t  = flowdict["flow1"]
        truth_flow2_t  = flowdict["flow2"]
        
        # compute output
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()

        predict1_t,predict2_t = model( coord_t, srcpix_t,
                                       tarpix_flow1_t, tarpix_flow2_t,
                                       batchsize )
                
        totloss,flow1loss,flow2loss = criterion(coord_t, predict1_t, predict2_t,
                                                truth_flow1_t, truth_flow2_t)
            
        if RUNPROFILER:
            torch.cuda.synchronize()
        time_meters["forward"].update(time.time()-end)

        # compute gradient and do SGD step
        if RUNPROFILER:
            torch.cuda.synchronize()                
        end = time.time()
        optimizer.zero_grad()
        totloss.backward()
        optimizer.step()
        if RUNPROFILER:        
            torch.cuda.synchronize()                
        time_meters["backward"].update(time.time()-end)

        # measure accuracy and record loss
        end = time.time()

        # update loss meters
        loss_meters["total"].update( totloss.item() )
        loss_meters["flow1"].update( flow1loss.item() )
        if flow2loss is not None:
            loss_meters["flow2"].update( flow2loss.item() )
        #loss_meters["visi1"].update( v1.item() )
        #loss_meters["visi2"].update( v2.item() )
        #loss_meters["consist3d"].update( closs.item() )
        
        # measure accuracy and update meters
        nvis1 = accuracy(srcpix_t.detach(),
                         predict1_t.features.detach(),
                         truth_flow1_t.detach(),
                         1,acc_meters,True)
        if predict2_t is not None:
            nvis2 = accuracy(srcpix_t.detach(),
                             predict2_t.features.detach(),
                             truth_flow2_t.detach(),
                             2,acc_meters,True)
        else:
            nvis2 = 0

        if nvis1==0 or (predict2_t is not None and nvis2==0):
            nnone += 1

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
    
    return loss_meters['total'].avg,acc_meters['flow1<5pix'],acc_meters['flow2<5pix']


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



    # accruacy and loss meters
    lossnames    = ("total","flow1","flow2","visi1","visi2","consist3d")
    flowaccnames = ("flow%d<5pix","flow%d<10pix","flow%d<20pix","flow%d<50pix")
    consistnames = ("dist2d")

    acc_meters  = {}
    for i in [1,2]:
        for n in flowaccnames:
            acc_meters[n%(i)] = AverageMeter()
    print acc_meters.keys()
    acc_meters["dist2d"] = AverageMeter()
    acc_meters["flow1@vis"] = AverageMeter()
    acc_meters["flow2@vis"] = AverageMeter()

    loss_meters = {}    
    for l in lossnames:
        loss_meters[l] = AverageMeter()

    # timers for profiling        
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
        
        datadict = val_loader.get_tensor_batch(device)
        coord_t  = datadict["coord"]
        srcpix_t = datadict["src"]
        tarpix_flow1_t = datadict["tar1"]
        tarpix_flow2_t = datadict["tar2"]
        truth_flow1_t  = datadict["flow1"]
        truth_flow2_t  = datadict["flow2"]
        
        time_meters["data"].update( time.time()-tdata_start )
        
        # compute output
        tforward = time.time()
        predict1_t,predict2_t = model( coord_t, srcpix_t,
                                       tarpix_flow1_t, tarpix_flow2_t,
                                       batchsize )        
        totloss,flow1loss,flow2loss = criterion(coord_t, predict1_t, predict2_t,
                                                truth_flow1_t, truth_flow2_t)
        
        time_meters["forward"].update(time.time()-tforward)

        # measure accuracy and update meters
        nvis1 = accuracy(srcpix_t.detach(),
                         predict1_t.features.detach(),
                         truth_flow1_t.detach(),
                         1,acc_meters,True)
        if predict2_t is not None:
            nvis2 = accuracy(srcpix_t.detach(),
                             predict2_t.features.detach(),
                             truth_flow2_t.detach(),
                             2,acc_meters,True)
        else:
            nvis2 = 0
        if nvis1==0 or (predict2_t is not None and nvis2==0):
            nnone += 1

        # update loss meters
        loss_meters["total"].update( totloss.item() )
        loss_meters["flow1"].update( flow1loss.item() )
        if flow2loss is not None:
            loss_meters["flow2"].update( flow2loss.item() )
        #loss_meters["visi1"].update( v1.item() )
        #loss_meters["visi2"].update( v2.item() )
        #loss_meters["consist3d"].update( closs.item() )
            
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
    
    return loss_meters['total'].avg,acc_meters['flow1<5pix'].avg,acc_meters['flow2<5pix'].avg

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


def accuracy(srcpix,flow_pred,flow_truth,flowdir,acc_meters,istrain):
    """Computes the accuracy metrics."""
    # inputs:
    #  assuming all pytorch tensors
    # metrics:
    #  (10,5,2) pixel accuracy fraction. flow within X pixels of target. (truth vis.==1 + fraction good ) / (fraction true vis.==1)
    #  visible accuracy: (true vis==1 && pred vis.>0.5) / ( true vis == 1)
    
    if istrain:
        accvals = (5,10,20,50)
    else:
        accvals = (5,10,20,50)
    
    profile = False
    
    # needs to be as gpu as possible!
    if profile:
        start = time.time()

    flow_err =(flow_pred-flow_truth).abs()
    #nvis     = visi_truth.sum()
    #if nvis<=0:
    #    return None

    mask = torch.ones( flow_truth.shape, dtype=torch.float ).to(flow_truth.device)
    # don't count pixels where:
    #  1) flow is missing i.e. equals zero
    #  2) source pixel is below threshold
    mask[ torch.eq(flow_truth,-4000.0) ] = 0.0
    #mask[ torch.gt(srcpix,10.0)  ] = 0.0
    nvis = mask.sum()
    
    for level in accvals:
        name = "flow%d<%dpix"%(flowdir,level)
        acc_meters[name].update( ((( flow_err<float(level)  ).float()*mask).sum() / nvis ).item() )

    #if visi_pred is not None:
    #    _, visi_max = visi_pred.max( 1, keepdim=False)
    #    mask_visi   = visi_max*visi_truth
    #    visi_acc    = (mask_visi==1).sum() / nvis
    #else:
    #visi_acc = 0.0

    #if visi_pred is not None:
    #    name = "flow%d@vis"
    #    acc_meters[name].update(visi_acc)
        
    if profile:
        torch.cuda.synchronize()            
        start = time.time()
        
    return acc_meters["flow%d<5pix"%(flowdir)]

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
    print "  Loss: Total[%.2f] Flow1[%.2f] Flow2[%.2f] Consistency[%.2f]"%(loss_meters["total"].avg,loss_meters["flow1"].avg,
                                                                           loss_meters["flow2"].avg,loss_meters["consist3d"].avg)
    print "  Flow1 accuracy: <5[%.1f] <10[%.1f] <20[%.1f] <50[%.1f]"%(acc_meters["flow1<5pix"].avg*100,acc_meters["flow1<10pix"].avg*100,acc_meters["flow1<20pix"].avg*100,acc_meters["flow1<50pix"].avg*100)
    print "  Flow2 accuracy: <5[%.1f] <10[%.1f] <20[%.1f] <50[%.1f]"%(acc_meters["flow2<5pix"].avg*100,acc_meters["flow2<20pix"].avg*100,acc_meters["flow2<20pix"].avg*100,acc_meters["flow2<50pix"].avg*100)
        
    print "------------------------------------------------------------------------"    


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
