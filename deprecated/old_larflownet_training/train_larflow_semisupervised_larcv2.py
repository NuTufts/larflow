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

# larflow
LARFLOW_MODEL_DIR=None
if "LARFLOW_BASEDIR" in os.environ:
    LARFLOW_MODEL_DIR=os.environ["LARFLOW_BASEDIR"]+"/models"
if LARFLOW_MODEL_DIR not in os.environ:
    sys.path.append(LARFLOW_MODEL_DIR)
else:
    sys.path.append("../models")
                    
# Our model definition
from larflow_uresnet_mod3 import LArFlowUResNet
from larflow_combined_loss import LArFlowCombinedLoss


# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT=False
RUNPROFILER=False
CHECKPOINT_FILE="/media/hdd1/rshara01/test/training/checkpoint.10000th.tar"
INPUTFILE="~/working/nutufts/larflow/testdata/smallsample/larcv_dlcosmictag_5482426_95_smallsample082918.root"
start_iter  = 0
TRAIN_LARCV_CONFIG="test_flowloader_832x512_dualflow_train.cfg"
VALID_LARCV_CONFIG="test_flowloader_832x512_dualflow_valid.cfg"
IMAGE_WIDTH=832
IMAGE_HEIGHT=512
BATCHSIZE=4
BATCHSIZE_VALID=1
ADC_THRESH=10.0
VISI_WEIGHT=0.0
CONSISTENCY_WEIGHT=0.1
USE_VISI=False
DEVICE_IDS=[1]
# map multi-training weights 
CHECKPOINT_MAP_LOCATIONS={"cuda:0":"cuda:0",
                          "cuda:1":"cuda:1"}
CHECKPOINT_MAP_LOCATIONS=None
CHECKPOINT_FROM_DATA_PARALLEL=False
ITER_PER_CHECKPOINT=100
# ===================================================

## =========================================================
## DATA PREP 3: LArCVServerWorker
## =========================================================
def load_data( io ):
    from larcv import larcv
    import numpy as np
    
    width  = 832
    height = 512
    src_adc_threshold = 10.0

    index = (1,0)
    products = ["source","targetu","targetv","flowy2u","flowy2v","visiy2u","visiy2v","meta"]
    data = {}
    for k in products:
        if k !="meta":
            data[k] = np.zeros( (1,width,height), dtype=np.float32 )
        else:
            data[k] = np.zeros( (3,width,height), dtype=np.float32 )            
        
    ev_adc = io.get_data("image2d","adc")
    ev_flo = io.get_data("image2d","pixflow")
    ev_vis = io.get_data("image2d","pixvisi")

    data["source"][0,:,:]  = larcv.as_ndarray( ev_adc.as_vector()[2] ).transpose(1,0)
    data["targetu"][0,:,:] = larcv.as_ndarray( ev_adc.as_vector()[0] ).transpose(1,0)
    data["targetv"][0,:,:] = larcv.as_ndarray( ev_adc.as_vector()[1] ).transpose(1,0)

    data["flowy2u"][0,:,:] = larcv.as_ndarray( ev_flo.as_vector()[0] ).transpose(1,0)
    data["flowy2v"][0,:,:] = larcv.as_ndarray( ev_flo.as_vector()[1] ).transpose(1,0)

    data["visiy2u"][0,:,:] = larcv.as_ndarray( ev_vis.as_vector()[0] ).transpose(1,0)
    data["visiy2v"][0,:,:] = larcv.as_ndarray( ev_vis.as_vector()[1] ).transpose(1,0)

    for ip in xrange(0,3):
        data["meta"][ip,0,0] = ev_adc.as_vector()[ip].meta().min_x()
        data["meta"][ip,0,1] = ev_adc.as_vector()[ip].meta().min_y()
        data["meta"][ip,0,2] = ev_adc.as_vector()[ip].meta().max_x()
        data["meta"][ip,0,3] = ev_adc.as_vector()[ip].meta().max_y()
            
    return data

def prep_data( ioserver, batchsize, width, height, src_adc_threshold, device ):
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
    data = ioserver.get_batch_dict()

    # make torch tensors from numpy arrays
    source_t  = torch.from_numpy( data["source"] ).to( device=device )   # source image ADC
    target1_t = torch.from_numpy( data["targetu"] ).to(device=device )   # target image ADC
    target2_t = torch.from_numpy( data["targetv"] ).to( device=device )  # target2 image ADC
    flow1_t   = torch.from_numpy( data["flowy2u"] ).to( device=device )   # flow from source to target
    flow2_t   = torch.from_numpy( data["flowy2v"] ).to( device=device ) # flow from source to target
    fvisi1_t  = torch.from_numpy( data["visiy2u"] ).to( device=device )  # visibility at source (float)
    fvisi2_t  = torch.from_numpy( data["visiy2v"] ).to( device=device ) # visibility at source (float)

    # apply threshold to source ADC values. returns a byte mask
    fvisi1_t  = fvisi1_t.clamp(0.0,1.0)
    fvisi2_t  = fvisi2_t.clamp(0.0,1.0)

    # make integer visi
    visi1_t = fvisi1_t.reshape( (batchsize,fvisi1_t.size()[2],fvisi1_t.size()[3]) ).long()
    visi2_t = fvisi2_t.reshape( (batchsize,fvisi2_t.size()[2],fvisi2_t.size()[3]) ).long()

    # image column origins
    meta_data_np = data["meta"]
    #print meta_data_np
    source_minx  = torch.from_numpy( meta_data_np[:,2,0,0].reshape((batchsize)) ).to(device=device)
    target1_minx = torch.from_numpy( meta_data_np[:,0,0,0].reshape((batchsize)) ).to(device=device)
    target2_minx = torch.from_numpy( meta_data_np[:,1,0,0].reshape((batchsize)) ).to(device=device)
    
    return source_t, target1_t, target2_t, flow1_t, flow2_t, visi1_t, visi2_t, fvisi1_t, fvisi2_t, source_minx, target1_minx, target2_minx

def pad(npimg4d):
    #imgpad  = np.zeros( (npimg4d.shape[0],1,IMAGE_WIDTH,IMAGE_HEIGHT), dtype=np.float32 )
    #for j in range(0,npimg4d.shape[0]):
    #    imgpad[j,0,32:832+32,0:512] = npimg4d[j,0,:,:]
    imgpad = npimg4d
    return imgpad

# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights
writer = SummaryWriter()

def main():

    global best_prec1
    global writer

    if GPUMODE:
        DEVICE = torch.device("cuda:%d"%(DEVICE_IDS[0]))
    else:
        DEVICE = torch.device("cpu")
    
    # create model, mark it to run on the GPU
    model = LArFlowUResNet( num_classes=2, input_channels=1,
                            layer_channels=[16,32,64,128],
                            layer_strides= [ 2, 2, 2,  2],
                            num_final_features=64,
                            use_deconvtranspose=False,
                            onlyone_res=True,
                            showsizes=False,
                            use_visi=False,
                            use_grad_checkpoints=True,
                            gpuid1=DEVICE_IDS[0],
                            gpuid2=DEVICE_IDS[0] )
    
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
    else:
        model = model.to(device=DEVICE)

    # uncomment to dump model
    if False:
        print "Loaded model: ",model
        return

    # define loss function (criterion) and optimizer
    maxdist = 200.0
    criterion = LArFlowCombinedLoss(IMAGE_WIDTH,IMAGE_HEIGHT,BATCHSIZE,maxdist,
                                    VISI_WEIGHT,CONSISTENCY_WEIGHT).to(device=DEVICE)

    # training parameters
    lr = 1.0e-3
    momentum = 0.9
    weight_decay = 1.0e-4

    # training length
    batchsize_train = BATCHSIZE
    batchsize_valid = BATCHSIZE_VALID#*len(DEVICE_IDS)
    start_epoch = 0
    epochs      = 10
    num_iters   = 10000
    iter_per_epoch = None # determined later
    iter_per_valid = 10


    nbatches_per_itertrain = 20
    itersize_train         = batchsize_train*nbatches_per_itertrain
    trainbatches_per_print = -1
    
    nbatches_per_itervalid = 40
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
    iotrain = LArCVServer(batchsize_train,"train",load_data,INPUTFILE,6)
    iovalid = LArCVServer(batchsize_valid,"valid",load_data,INPUTFILE,2)

    print "pause to give time to feeders"

    #NENTRIES = len(iotrain)
    NENTRIES = 100000
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
        source,target1,target2,flow1,flow2,visi1,visi2,fvisi1,fvisi2,Wminx,Uminx,Vminx = prep_data( iosample[sample], batchsize_train, 
                                                                                                    IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH, DEVICE )
        # load opencv
        print "Print using OpenCV"
        import cv2 as cv
        cv.imwrite( "testout_source.png",  source.cpu().numpy()[0,0,:,:] )
        cv.imwrite( "testout_target1.png", target1.cpu().numpy()[0,0,:,:] )
        cv.imwrite( "testout_target2.png", target2.cpu().numpy()[0,0,:,:] )
        print "source shape: ",source.cpu().numpy().shape
        print "minX-src: ",Wminx
        print "minX-U: ",Uminx
        print "minX-V: ",Vminx

        sample = "valid"
        print "TEST BATCH: sample=",sample
        source,target1,target2,flow1,flow2,visi1,visi2,fvisi1,fvisi2,Wminx,Uminx,Vminx = prep_data( iosample[sample], batchsize_valid, 
                                                                                                    IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH, DEVICE )       
        
        print "STOP FOR DEBUGGING"
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
                _ = train(iotrain, DEVICE, batchsize_train, model,
                          criterion, optimizer,
                          nbatches_per_itertrain, ii, trainbatches_per_print)
                
            except Exception,e:
                print "Error in training routine!"            
                print e.message
                print e.__class__.__name__
                traceback.print_exc(e)
                break

            # evaluate on validation set
            if ii%iter_per_valid==0:
                try:
                    totloss, flow1acc5, flow2acc5 = validate(iovalid, DEVICE, batchsize_valid, model, criterion, nbatches_per_itervalid, ii, validbatches_per_print)
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
    flowaccnames = ("flow%d@5","flow%d@10","flow%d@20","flow%d@50")
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
        source,target1,target2,flow1,flow2,visi1,visi2,fvisi1,fvisi2,src_origin,tar1_origin,tar2_origin = prep_data( train_loader, batchsize, IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH, device )        
        time_meters["data"].update(time.time()-end)

        # compute output
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()
        source.requires_grad = True
        target1.requires_grad = True
        target2.requires_grad = True
        flow1_pred,flow2_pred = model.forward(source,target1,target2)
        totloss,f1,f2,v1,v2,closs = criterion(flow1_pred,flow2_pred,None,None,flow1,flow2,visi1,visi2,src_origin,tar1_origin,tar2_origin)
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
        loss_meters["flow1"].update( f1.item() )
        loss_meters["flow2"].update( f2.item() )        
        loss_meters["visi1"].update( v1.item() )
        loss_meters["visi2"].update( v2.item() )
        loss_meters["consist3d"].update( closs.item() )
        
        # measure accuracy and update meters
        nvis1 = accuracy(flow1_pred.detach(),None,flow1.detach(),fvisi1.detach(),1,acc_meters,True)
        nvis2 = accuracy(flow2_pred.detach(),None,flow2.detach(),fvisi2.detach(),2,acc_meters,True)
        if nvis1==0 or nvis2==0:
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
    
    return loss_meters['total'].avg,acc_meters['flow1@5'],acc_meters['flow2@5']


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
    flowaccnames = ("flow%d@5","flow%d@10","flow%d@20","flow%d@50")
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
        source,target1,target2,flow1,flow2,visi1,visi2,fvisi1,fvisi2,src_origin,tar1_origin,tar2_origin = prep_data( val_loader, batchsize, IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH, device )
        time_meters["data"].update( time.time()-tdata_start )
        
        # compute output
        tforward = time.time()
        flow1_pred,flow2_pred = model.forward(source,target1,target2)
        totloss,f1,f2,v1,v2,closs = criterion(flow1_pred,flow2_pred,None,None,flow1,flow2,visi1,visi2,src_origin,tar1_origin,tar2_origin)
        time_meters["forward"].update(time.time()-tforward)

        # measure accuracy and record loss
        # measure accuracy and update meters
        nvis1 = accuracy(flow1_pred.detach(),None,flow1.detach(),fvisi1.detach(),1,acc_meters,False)
        nvis2 = accuracy(flow2_pred.detach(),None,flow2.detach(),fvisi2.detach(),2,acc_meters,False)
        if nvis1==0 or nvis2==0:
            nnone += 1

        # update loss meters
        loss_meters["total"].update( totloss.item() )
        loss_meters["flow1"].update( f1.item() )
        loss_meters["flow2"].update( f2.item() )        
        loss_meters["visi1"].update( v1.item() )
        loss_meters["visi2"].update( v2.item() )
        loss_meters["consist3d"].update( closs.item() )
            
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
    
    return loss_meters['total'].avg,acc_meters['flow1@5'].avg,acc_meters['flow2@5'].avg

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


def accuracy(flow_pred,visi_pred,flow_truth,visi_truth,flowdir,acc_meters,istrain):
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
    nvis     = visi_truth.sum()
    if nvis<=0:
        return None

    
    for level in accvals:
        name = "flow%d@%d"%(flowdir,level)        
        acc_meters[name].update( ((( flow_err<float(level)  ).float()*visi_truth).sum() / nvis ).item() )

    if visi_pred is not None:
        _, visi_max = visi_pred.max( 1, keepdim=False)
        mask_visi   = visi_max*visi_truth
        visi_acc    = (mask_visi==1).sum() / nvis
    else:
        visi_acc = 0.0

    if visi_pred is not None:
        name = "flow%d@vis"
        acc_meters[name].update(visi_acc)
        
    if profile:
        torch.cuda.synchronize()            
        start = time.time()
        
    return acc_meters["flow%d@5"%(flowdir)]

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
    print "  Loss: Total[%.2f] Flow1[%.2f] Flow2[%.2f] Consistency[%.2f]"%(loss_meters["total"].avg,loss_meters["flow1"].avg,loss_meters["flow2"].avg,loss_meters["consist3d"].avg)
    print "  Flow1 accuracy: @5[%.1f] @10[%.1f] @20[%.1f] @50[%.1f]"%(acc_meters["flow1@5"].avg*100,acc_meters["flow1@10"].avg*100,acc_meters["flow1@20"].avg*100,acc_meters["flow1@50"].avg*100)
    print "  Flow2 accuracy: @5[%.1f] @10[%.1f] @20[%.1f] @50[%.1f]"%(acc_meters["flow2@5"].avg*100,acc_meters["flow2@20"].avg*100,acc_meters["flow2@20"].avg*100,acc_meters["flow2@50"].avg*100)
        
    print "------------------------------------------------------------------------"    


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
