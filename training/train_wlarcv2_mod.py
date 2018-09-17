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
from larcvdataset import LArCVDataset

# larflow
LARFLOW_MODEL_DIR=None
if "LARFLOW_BASEDIR" in os.environ:
    LARFLOW_MODEL_DIR=os.environ["LARFLOW_BASEDIR"]+"/models"
if LARFLOW_MODEL_DIR not in os.environ:
    sys.path.append(LARFLOW_MODEL_DIR)
else:
    sys.path.append("../models")
                    
# Our model definition
from larflow_uresnet_mod import LArFlowUResNet
#from larflow_uresnet import LArFlowUResNet
from larflow_loss import LArFlowLoss


# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT=False #True
RUNPROFILER=False
#CHECKPOINT_FILE="/cluster/kappa/wongjiradlab/twongj01/llf/pytorch-larflow/v0.4_832x512_fullres_y2u/run1_continue/checkpoint.14000th.tar"
#CHECKPOINT_FILE="/cluster/kappa/wongjiradlab/twongj01/llf/pytorch-larflow/v0.4_832x512_fullres_y2u/run2_bigsample/checkpoint.12500th.tar"
CHECKPOINT_FILE="/cluster/kappa/wongjiradlab/twongj01/llf/pytorch-larflow/v0.4_832x512_fullres_y2u/run3_bigsample2_droplr/checkpoint.14500th.tar"
start_iter  = 0 #14500
# on meitner
#TRAIN_LARCV_CONFIG="flowloader_train.cfg"
#VALID_LARCV_CONFIG="flowloader_valid.cfg"
# on tufts grid
TRAIN_LARCV_CONFIG="flowloader_832x512_y2v_train.cfg"
VALID_LARCV_CONFIG="flowloader_832x512_y2v_valid.cfg"
IMAGE_WIDTH=512
IMAGE_HEIGHT=832
ADC_THRESH=10.0
VISI_WEIGHT=0.001
USE_VISI=False
#DEVICE_IDS=[3,4,5]
DEVICE_IDS=[0,1]
GPUID1=DEVICE_IDS[0]
GPUID2=DEVICE_IDS[1]
# map multi-training weights 
#CHECKPOINT_MAP_LOCATIONS={"cuda:2":"cuda:0",
#                          "cuda:3":"cuda:1",
#                          "cuda:4":"cuda:2"}
CHECKPOINT_MAP_LOCATIONS={"cuda:0":"cuda:2",
                          "cuda:1":"cuda:3",
                          "cuda:2":"cuda:4"}
CHECKPOINT_MAP_LOCATIONS=None
CHECKPOINT_FROM_DATA_PARALLEL=False
# ===================================================


def pad(npimg4d):
    imgpad  = np.zeros( (npimg4d.shape[0],1,512,896), dtype=np.float32 )
    for j in range(0,npimg4d.shape[0]):
        imgpad[j,0,0:512,32:832+32] = npimg4d[j,0,:,:]
    return imgpad
                

# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights
writer = SummaryWriter()

def main():

    global best_prec1
    global writer

    # create model, mark it to run on the GPU
    if GPUMODE:
        model = LArFlowUResNet(inplanes=32,input_channels=1,num_classes=2,showsizes=False, use_visi=USE_VISI, gpuid1=GPUID1, gpuid2=GPUID2)
        #model.cuda(GPUID) # put onto gpuid
    else:
        model = LArFlowUResNet(inplanes=16,input_channels=1,num_classes=2)

    # Resume training option
    if RESUME_FROM_CHECKPOINT:
        print "RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE
        checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS ) # load weights to gpuid
        best_prec1 = checkpoint["best_prec1"]
        if CHECKPOINT_FROM_DATA_PARALLEL:
            model = nn.DataParallel( model, device_ids=DEVICE_IDS ) # distribute across device_ids
        model.load_state_dict(checkpoint["state_dict"])
    '''
    if not CHECKPOINT_FROM_DATA_PARALLEL and len(DEVICE_IDS)>1:
        model = nn.DataParallel( model, device_ids=DEVICE_IDS ) # distribute across device_ids
    '''
    # uncomment to dump model
    print "Loaded model: ",model
    # check where model pars are
    #for p in model.parameters():
    #    print p.is_cuda

    # define loss function (criterion) and optimizer
    maxdist = 32.0
    if GPUMODE:
        criterion = LArFlowLoss(maxdist,VISI_WEIGHT)
        criterion.cuda(GPUID1)
    else:
        criterion = LArFlowLoss(maxdist,VISI_WEIGHT)

    # training parameters
    lr = 1.0e-4
    momentum = 0.9
    weight_decay = 1.0e-4

    # training length
    batchsize_train = 2#*len(DEVICE_IDS)
    batchsize_valid = 1#*len(DEVICE_IDS)
    start_epoch = 0
    epochs      = 10
    num_iters   = 30000
    iter_per_epoch = None # determined later
    iter_per_valid = 10
    iter_per_checkpoint = 300

    nbatches_per_itertrain = 20
    itersize_train         = batchsize_train*nbatches_per_itertrain
    trainbatches_per_print = 100
    
    nbatches_per_itervalid = 40
    itersize_valid         = batchsize_valid*nbatches_per_itervalid
    validbatches_per_print = 100

    # SETUP OPTIMIZER

    # SGD w/ momentum
    #optimizer = torch.optim.SGD(model.parameters(), lr,
    #                            momentum=momentum,
    #                            weight_decay=weight_decay)
    
    # ADAM
    # betas default: (0.9, 0.999) for (grad, grad^2). smoothing coefficient for grad. magnitude calc.
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=lr, 
                                 weight_decay=weight_decay)
    
    # optimize algorithms based on input size (good if input size is constant)
    cudnn.benchmark = True

    # LOAD THE DATASET
    
    iotrain = LArCVDataset(TRAIN_LARCV_CONFIG,"ThreadProcessorTrain")
    iovalid = LArCVDataset(VALID_LARCV_CONFIG,"ThreadProcessorValid")
    iotrain.start( batchsize_train )
    iovalid.start( batchsize_valid )

    NENTRIES = len(iotrain)
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

    
    if False:
        # for debugging/testing data
        sample = "train"
        iosample = {"valid":iovalid,
                    "train":iotrain}
        print "TEST BATCH: sample=",sample
        source_var, target_var, flow_var, visi_var, fvisi_var = prep_data( iosample[sample], sample, batchsize_train, 
                                                                           IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH )
        print "source shape: ",source_var.data.size()
        print "target shape: ",target_var.data.size()
        print "flow shape: ",flow_var.data.size()
        print "visi shape: ",visi_var.data.size()
        print "fvisi shape: ",fvisi_var.data.size()

        # load opencv
        import cv2 as cv
        cv.imwrite( "testout_src.png", source_var.cpu().data.numpy()[0,0,:,:] )
        cv.imwrite( "testout_tar.png", target_var.cpu().data.numpy()[0,0,:,:] )
        
        print "STOP FOR DEBUGGING"
        #iotrain.stop()
        #iovalid.stop()
        sys.exit(-1)

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
                train_ave_loss, train_ave_acc = train(iotrain, batchsize_train, model,
                                                      criterion, optimizer,
                                                      nbatches_per_itertrain, ii, trainbatches_per_print)
            except Exception,e:
                print "Error in training routine!"            
                print e.message
                print e.__class__.__name__
                traceback.print_exc(e)
                break
            print "Train Iter:%d Epoch:%d.%d train aveloss=%.3f aveacc=%.3f"%(ii,ii/iter_per_epoch,ii%iter_per_epoch,train_ave_loss,train_ave_acc)

            # evaluate on validation set
            if ii%iter_per_valid==0:
                try:
                    prec1 = validate(iovalid, batchsize_valid, model, criterion, nbatches_per_itervalid, validbatches_per_print, ii)
                except Exception,e:
                    print "Error in validation routine!"            
                    print e.message
                    print e.__class__.__name__
                    traceback.print_exc(e)
                    break

                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
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
            if ii>0 and ii%iter_per_checkpoint==0:
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
    print prof
    writer.close()


def train(train_loader, batchsize, model, criterion, optimizer, nbatches, iiter, print_freq):

    global writer

    # timers for profiling
    batch_time = AverageMeter() # total for batch
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    acc_time = AverageMeter()

    # accruacy and loss meters
    losses = AverageMeter()
    lossesf = AverageMeter()
    lossesv = AverageMeter()
    vis_acc = AverageMeter()
    acc_list = [] # 10, 5, 2
    for i in range(3):
        acc_list.append( AverageMeter() )

    # switch to train mode
    model.train()

    nnone = 0
    for i in range(0,nbatches):
        #print "iiter ",iiter," batch ",i," of ",nbatches
        batchstart = time.time()

        # GET THE DATA
        end = time.time()        
        source_var, target_var, flow_var, visi_var, fvisi_var = prep_data( train_loader, "train", batchsize, IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH )        
        data_time.update( time.time()-end )

        # compute output
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()
        flow_pred,visi_pred = model.forward(source_var,target_var)
        loss, loss_f, loss_v = criterion.calc_loss(flow_pred,visi_pred,flow_var,visi_var,fvisi_var)
        if RUNPROFILER:
            torch.cuda.synchronize()                
        forward_time.update(time.time()-end)

        # compute gradient and do SGD step
        if RUNPROFILER:
            torch.cuda.synchronize()                
        end = time.time()        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if RUNPROFILER:        
            torch.cuda.synchronize()                
        backward_time.update(time.time()-end)

        # measure accuracy and record loss
        end = time.time()

        # measure accuracy and record loss
        acc_values = accuracy(flow_pred, visi_pred, flow_var, visi_var, fvisi_var)
        if acc_values is not None:
            losses.update(loss.data[0])
            lossesf.update(loss_f.data[0])
            if USE_VISI:
                lossesv.update(loss_v.data[0])
            for iacc,acc in enumerate(acc_list):
                acc.update( acc_values[iacc] )
            vis_acc.update( acc_values[-1] )
        else:
            nnone += 1
        
        acc_time.update(time.time()-end)

        # measure elapsed time for batch
        batch_time.update(time.time() - batchstart)


        if i % print_freq == 0:
            status = (iiter,i,nbatches,
                      batch_time.val,batch_time.avg,
                      data_time.val,data_time.avg,
                      forward_time.val,forward_time.avg,
                      backward_time.val,backward_time.avg,
                      acc_time.val,acc_time.avg,                      
                      losses.val,losses.avg,
                      acc_list[1].val,acc_list[1].avg,
                      vis_acc.val,vis_acc.avg)
            print "Train Iter: [%d][%d/%d]  Batch %.3f (%.3f)  Data %.3f (%.3f)  Forw %.3f (%.3f)  Back %.3f (%.3f) Acc %.3f (%.3f)\t || \tLoss %.3f (%.3f)\tAcc@05 %.3f (%.3f)\tVis %.3f (%.3f)"%status


    status = (iiter,
              batch_time.avg,
              data_time.avg,
              forward_time.avg,
              backward_time.avg,
              acc_time.avg,                      
              losses.avg,
              acc_list[0].avg,
              acc_list[1].avg,
              acc_list[2].avg,
              vis_acc.avg,
              nnone)
    print "Train Iter [%d] Ave: Batch %.3f  Data %.3f  Forw %.3f  Back %.3f  Acc %.3f ||  Loss %.3f Acc @10=%.3f @05=%.3f @02=%.3f  Vis %.3f  None=%d"%status
    print losses.avg, lossesf.avg, lossesv.avg
    #writer.add_scalar( 'data/train_loss',  losses.avg,  iiter )        
    writer.add_scalars('data/train_loss', {'tot_loss': losses.avg,
                                           'flow_loss': lossesf.avg,
                                           'visi_loss': lossesv.avg},iiter)

    writer.add_scalars('data/train_accuracy', {'acc10': acc_list[0].avg,
                                               'acc05': acc_list[1].avg,
                                               'acc02': acc_list[2].avg,
                                               'vis':   vis_acc.avg},iiter)
    
    return losses.avg,acc_list[1].avg


def validate(val_loader, batchsize, model, criterion, nbatches, print_freq, iiter):
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
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    lossesf = AverageMeter()
    lossesv = AverageMeter()
    vis_acc = AverageMeter()
    load_data = AverageMeter()
    acc_list = [] # 10, 5, 2
    for i in range(3):
        acc_list.append( AverageMeter() )
    
    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    nnone = 0
    for i in range(0,nbatches):
        batchstart = time.time()
        
        tdata_start = time.time()
        source_var, target_var, flow_var, visi_var, fvisi_var = prep_data( val_loader, "valid", batchsize, IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH )
        load_data.update( time.time()-tdata_start )
        
        # compute output
        flow_pred,visi_pred = model.forward(source_var,target_var)
        loss, loss_f, loss_v = criterion.calc_loss(flow_pred,visi_pred,flow_var,visi_var,fvisi_var)

        # measure accuracy and record loss
        acc_values = accuracy(flow_pred, visi_pred, flow_var, visi_var, fvisi_var)
        if acc_values is not None:
            losses.update(loss.data[0])
            lossesf.update(loss_f.data[0])
            if USE_VISI:
                lossesv.update(loss_v.data[0])
            for iacc,acc in enumerate(acc_list):
                acc.update( acc_values[iacc] )
            vis_acc.update( acc_values[-1] )
        else:
            nnone += 1
                
        # measure elapsed time
        batch_time.update(time.time() - batchstart)
        end = time.time()

        if i % print_freq == 0:
            status = (i,nbatches,batch_time.val,batch_time.avg,losses.val,losses.avg,acc_list[1].val,acc_list[1].avg)
            print "Valid: [%d/%d]\tTime %.3f (%.3f)\tLoss %.3f (%.3f)\tAcc5@1 %.3f (%.3f)"%status

    status = (iiter,batch_time.avg,load_data.avg,losses.avg,acc_list[1].avg, nnone)
    print "Valid Iter %d sum: Batch %.3f\tData %.3f || Loss %.3f\tAcc5@1 %.3f\tNone=%d"%status    

    #writer.add_scalar( 'data/valid_loss', losses.avg, iiter )
    writer.add_scalars('data/valid_loss', {'tot_loss': losses.avg,
                                           'flow_loss': lossesf.avg,
                                           'visi_loss': lossesv.avg},iiter)

    writer.add_scalars('data/valid_accuracy', {'acc10': acc_list[0].avg,
                                               'acc05': acc_list[1].avg,
                                               'acc02': acc_list[2].avg,
                                               'vis':   vis_acc.avg},iiter)

    print "Test:Result* Acc@5 %.3f\tLoss %.3f"%(acc_list[1].avg,losses.avg)

    return float(acc_list[1].avg)


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


def accuracy(flow_pred,visi_pred,flow_truth,visi_truth,fvisi_truth):
    """Computes the accuracy metrics."""
    # inputs:
    #  assuming all pytorch tensors
    # metrics:
    #  (10,5,2) pixel accuracy fraction. flow within X pixels of target. (truth vis.==1 + fraction good ) / (fraction true vis.==1)
    #  visible accuracy: (true vis==1 && pred vis.>0.5) / ( true vis == 1)
    
    profile = False
    
    # needs to be as gpu as possible!
    if profile:
        start = time.time()

    flow_err = (flow_pred.data - flow_truth.data)*fvisi_truth.data
    flow_err = flow_err.abs()
    nvis = fvisi_truth.data.sum()
    if nvis==0:
        return None
    
    acc10 = (( flow_err<10.0 ).float()*fvisi_truth.data).sum() / nvis
    acc5  = (( flow_err<5.0 ).float()*fvisi_truth.data).sum() / nvis
    acc2  = (( flow_err<2.0 ).float()*fvisi_truth.data).sum() / nvis

    #print "accuracy debug"
    #print "  fvisi_truth.sum()=",nvis
    if visi_pred is not None:
        _, visi_max = visi_pred.data.max( 1, keepdim=False)
        mask_visi = visi_max*visi_truth.data
        visi_acc = (mask_visi==1).sum() / fvisi_truth.data.sum()
    else:
        visi_acc = 0.0
        
    if profile:
        torch.cuda.synchronize()            
        start = time.time()
        
    return acc10,acc5,acc2,visi_acc

def dump_lr_schedule( startlr, numepochs ):
    for epoch in range(0,numepochs):
        lr = startlr*(0.5**(epoch//300))
        if epoch%10==0:
            print "Epoch [%d] lr=%.3e"%(epoch,lr)
    print "Epoch [%d] lr=%.3e"%(epoch,lr)
    return

def prep_data( larcvloader, train_or_valid, batchsize, width, height, src_adc_threshold ):
    """
    inputs
    ------
    larcvloader: instance of LArCVDataloader
    train_or_valid (str): "train" or "valid"
    batchsize (int)
    width (int)
    height(int)
    src_adc_threshold (float)

    outputs
    -------
    source_var (Pytorch Variable): source ADC
    targer_var (Pytorch Variable): target ADC
    flow_var (Pytorch Variable): flow from source to target
    visi_var (Pytorch Variable): visibility of source (long)
    fvisi_var(Pytorch Variable): visibility of target (float)
    """

    #print "PREP DATA: ",train_or_valid,"GPUMODE=",GPUMODE,"GPUID=",GPUID    

    # get data
    data = larcvloader[0]

    # make torch tensors from numpy arrays
    source_t = torch.from_numpy( pad( data["source_%s"%(train_or_valid)].reshape( (batchsize,1,width,height) ) ) ) # source image ADC
    target_t = torch.from_numpy( pad( data["target_%s"%(train_or_valid)].reshape( (batchsize,1,width,height) ) ) ) # target image ADC
    flow_t   = torch.from_numpy( pad( data["pixflow_%s"%(train_or_valid)].reshape( (batchsize,1,width,height) ) ) ) # flow from source to target
    fvisi_t  = torch.from_numpy( pad( data["pixvisi_%s"%(train_or_valid)].reshape( (batchsize,1,width,height) ) ) ) # visibility at source (float)

    # send to gpu if in gpumode
    if GPUMODE:
        source_t = source_t.cuda(GPUID1)
        target_t = target_t.cuda(GPUID1)
        flow_t = flow_t.cuda(GPUID1)
        fvisi_t = fvisi_t.cuda(GPUID1)

    # apply threshold to source ADC values. returns a byte mask
    fvisi_t.masked_fill_(fvisi_t>1.0, 1.0)
    fvisi_t.masked_fill_(fvisi_t<0,0.0)

    # make integer visi
    if GPUMODE:
        fvisi_t= fvisi_t.cuda(GPUID1)
        visi_t = fvisi_t.long().resize_(batchsize,width,fvisi_t.size()[3]).cuda(GPUID1)
    else:
        visi_t = fvisi_t.reshape( (batchsize,width,fvisi_t.size()[3]) ).long()

    # make autograd variable
    source_var = torch.autograd.Variable(source_t)
    target_var = torch.autograd.Variable(target_t)
    flow_var   = torch.autograd.Variable(flow_t)
    visi_var   = torch.autograd.Variable(visi_t)
    fvisi_var  = torch.autograd.Variable(fvisi_t)

    return source_var, target_var, flow_var, visi_var, fvisi_var

if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
