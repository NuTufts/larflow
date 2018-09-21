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
from larflow_uresnet_mod3 import LArFlowUResNet
from larflow_combined_loss import LArFlowTotalLoss


# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT=False
RUNPROFILER=False
CHECKPOINT_FILE="/media/hdd1/rshara01/test/training/checkpoint.10000th.tar"
start_iter  = 0 #14500
TRAIN_LARCV_CONFIG="meitner_flowloader_832x512_dualflow_train.cfg"
VALID_LARCV_CONFIG="meitner_flowloader_832x512_dualflow_valid.cfg"
IMAGE_WIDTH=832
IMAGE_HEIGHT=512
BATCHSIZE=2
ADC_THRESH=10.0
VISI_WEIGHT=0.01
USE_VISI=False
DEVICE_IDS=[0,1]
GPUID1=DEVICE_IDS[0]
GPUID2=DEVICE_IDS[1]
# map multi-training weights 
CHECKPOINT_MAP_LOCATIONS={"cuda:0":"cuda:0",
                          "cuda:1":"cuda:1"}
CHECKPOINT_MAP_LOCATIONS=None
CHECKPOINT_FROM_DATA_PARALLEL=False
# ===================================================


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
                            layer_channels=[16,32,64,128,1024],
                            layer_strides= [ 2, 2, 2,  2,   2],
                            num_final_features=128,
                            use_deconvtranspose=False,
                            onlyone_res=True,
                            showsizes=False,
                            use_visi=False,
                            gpuid1=0,
                            gpuid2=0 ).to(device=DEVICE)
    
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
    #print "Loaded model: ",model

    # define loss function (criterion) and optimizer
    maxdist = 500.0
    criterion = LArFlowTotalLoss(IMAGE_WIDTH,IMAGE_HEIGHT,BATCHSIZE,maxdist,0.0,1.0).to(device=DEVICE)

    # training parameters
    lr = 1.0e-4
    momentum = 0.9
    weight_decay = 1.0e-4

    # training length
    batchsize_train = BATCHSIZE #*len(DEVICE_IDS)
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
        source,target1,target2,flow1,flow2,visi1,visi2,fvisi1,fvisi2,Wminx,Uminx,Vminx = prep_data( iosample[sample], sample, batchsize_train, 
                                                                                                    IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH, DEVICE )
        # load opencv
        import cv2 as cv
        cv.imwrite( "testout_source.png",  source.cpu().numpy()[0,0,:,:] )
        cv.imwrite( "testout_target1.png", target1.cpu().numpy()[0,0,:,:] )
        cv.imwrite( "testout_target2.png", target2.cpu().numpy()[0,0,:,:] )        
        
        print "STOP FOR DEBUGGING"
        iotrain.stop()
        iovalid.stop()
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
                    prec1 = validate(iovalid, DEVICE, batchsize_valid, model, criterion, nbatches_per_itervalid, validbatches_per_print, ii)
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
        source,target1,target2,flow1,flow2,visi1,visi2,fvisi1,fvisi2,src_origin,tar1_origin,tar2_origin = prep_data( train_loader, "train", batchsize, IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH, device )
        time_meters["data"].update(time.time()-end)

        # compute output
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()
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
        nvis1 = accuracy(flow1_pred.detach(),None,flow1.detach(),fvisi1.detach(),1,acc_meters)
        nvis2 = accuracy(flow2_pred.detach(),None,flow2.detach(),fvisi2.detach(),2,acc_meters)
        if nvis1==0 or nvis2==0:
            nnone += 1

        # update time meter
        time_meters["accuracy"].update(time.time()-end)            

        # measure elapsed time for batch
        time_meters["batch"].update(time.time()-batchstart)

        # print status
        prep_status_message( "train-batch", i, acc_meters, loss_meters, time_meters )

    prep_status_message( "Train-Iteration", iiter, acc_meters, loss_meters, time_meters )

    # write to tensorboard
    loss_scalars = { x:y.avg for x,y in loss_meters.items() }
    writer.add_scalars('data/train_loss', loss_scalars, iiter )

    acc_scalars = { x:y.avg for x,y in acc_meters.items() }
    writer.add_scalars('data/train_accuracy', acc_scalars, iiter )
    
    return loss_meters['total'].avg,acc_meters['flow1@5'],acc_meters['flow2@5']


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
    lossesf2 = AverageMeter()
    lossesv = AverageMeter()
    vis_acc = AverageMeter()
    load_data = AverageMeter()
    acc_list = [] # 10, 5, 2
    for i in range(6):
        acc_list.append( AverageMeter() )
    
    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    nnone = 0
    for i in range(0,nbatches):
        batchstart = time.time()
        
        tdata_start = time.time()
        source_var, target_var, target2_var, flow_var, flow2_var, visi_var, visi2_var, fvisi_var, fvisi2_var = prep_data( val_loader, "valid", batchsize, IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH, DEVICE )
        load_data.update( time.time()-tdata_start )
        
        # compute output
        flow_pred,flow2_pred,visi_pred,visi2_pred = model.forward(source_var,target_var,target2_var)
        loss, loss_f, loss_f2, loss_v = criterion.calc_loss(flow_pred,flow2_pred,visi_pred,flow_var,flow2_var,visi_var,fvisi_var,fvisi2_var)

        # measure accuracy and record loss
        acc_values = accuracy(flow_pred, flow2_pred, visi_pred, visi2_pred, flow_var, flow2_var, visi_var, visi2_var, fvisi_var, fvisi2_var)
        if acc_values is not None:
            losses.update(loss.data[0])
            lossesf.update(loss_f.data[0])
            lossesf2.update(loss_f2.data[0])
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
                                           'flow2_loss': lossesf2.avg,
                                           'visi_loss': lossesv.avg},iiter)

    writer.add_scalars('data/valid_accuracy', {'acc10': acc_list[0].avg,
                                               'acc05': acc_list[1].avg,
                                               'acc02': acc_list[2].avg,
                                               'acc10_2': acc_list[3].avg,
                                               'acc05_2': acc_list[4].avg,
                                               'acc02_2': acc_list[5].avg,
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


def accuracy(flow_pred,visi_pred,flow_truth,visi_truth,flowdir,acc_meters):
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

    flow_err =(flow_pred-flow_truth).abs()
    nvis     = visi_truth.sum()
    if nvis<=0:
        return None

    #acc100 = (( flow_err<100.0 ).float()*visi_truth).sum() / nvis        
    acc50  = (( flow_err<50.0  ).float()*visi_truth).sum() / nvis    
    acc20  = (( flow_err<20.0  ).float()*visi_truth).sum() / nvis    
    acc10  = (( flow_err<10.0  ).float()*visi_truth).sum() / nvis
    acc5   = (( flow_err<5.0   ).float()*visi_truth).sum() / nvis
    acc_measures = (acc5,acc10,acc20,acc50)

    if visi_pred is not None:
        _, visi_max = visi_pred.max( 1, keepdim=False)
        mask_visi   = visi_max*visi_truth
        visi_acc    = (mask_visi==1).sum() / nvis
    else:
        visi_acc = 0.0

    ## update the meters
    for level,measure in zip((5,10,20,50),acc_measures):
        name = "flow%d@%d"%(flowdir,level)
        acc_meters[name].update(measure)

    if visi_pred is not None:
        name = "flow%d@vis"
        acc_meters[name].update(visi_acc)
        
    if profile:
        torch.cuda.synchronize()            
        start = time.time()
        
    return acc10

def dump_lr_schedule( startlr, numepochs ):
    for epoch in range(0,numepochs):
        lr = startlr*(0.5**(epoch//300))
        if epoch%10==0:
            print "Epoch [%d] lr=%.3e"%(epoch,lr)
    print "Epoch [%d] lr=%.3e"%(epoch,lr)
    return

def prep_status_message( descripter, iternum, acc_meters, loss_meters, timers ):
    print "------------------------------------------------------------------------"
    print " Iter[",iternum,"] ",descripter
    print "  Loss: Total[%.2f] Flow1[%.2f] Flow2[%.2f] Consistency[%.2f]"%(loss_meters["total"].avg,loss_meters["flow1"].avg,loss_meters["flow2"].avg,loss_meters["consist3d"].avg)
    print "  Flow1 accuracy: @5[%.1f] @10[%.1f] @20[%.1f] @50[%.1f]"%(acc_meters["flow1@5"].avg*100,acc_meters["flow1@10"].avg*100,acc_meters["flow1@20"].avg*100,acc_meters["flow1@50"].avg*100)
    print "  Flow2 accuracy: @5[%.1f] @10[%.1f] @20[%.1f] @50[%.1f]"%(acc_meters["flow2@5"].avg*100,acc_meters["flow2@20"].avg*100,acc_meters["flow2@20"].avg*100,acc_meters["flow2@50"].avg*100)
    print "  Timing: batch[%.1f] Forward[%.1f] Backward[%.1f] Data[%.1f]"%(timers["batch"].avg,timers["forward"].avg,timers["backward"].avg,timers["data"].avg)
    print "------------------------------------------------------------------------"    

def prep_data( larcvloader, train_or_valid, batchsize, width, height, src_adc_threshold, device ):
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
    index = (0,1,2,3)
    source_t  = torch.from_numpy( pad( data["source_%s"%(train_or_valid)].reshape( (batchsize,1,width,height) ).transpose(index) )).to( device=device )   # source image ADC
    target1_t = torch.from_numpy( pad( data["target1_%s"%(train_or_valid)].reshape( (batchsize,1,width,height) ).transpose(index) )).to(device=device )   # target image ADC
    target2_t = torch.from_numpy( pad( data["target2_%s"%(train_or_valid)].reshape( (batchsize,1,width,height)).transpose(index) )).to( device=device )  # target2 image ADC
    flow1_t   = torch.from_numpy( pad( data["pixflow1_%s"%(train_or_valid)].reshape( (batchsize,1,width,height)).transpose(index) )).to( device=device )   # flow from source to target
    flow2_t   = torch.from_numpy( pad( data["pixflow2_%s"%(train_or_valid)].reshape( (batchsize,1,width,height)).transpose(index) )).to( device=device ) # flow from source to target
    fvisi1_t  = torch.from_numpy( pad( data["pixvisi1_%s"%(train_or_valid)].reshape( (batchsize,1,width,height) ).transpose(index) )).to( device=device )  # visibility at source (float)
    fvisi2_t  = torch.from_numpy( pad( data["pixvisi2_%s"%(train_or_valid)].reshape( (batchsize,1,width,height)).transpose(index) )).to( device=device ) # visibility at source (float)

    # apply threshold to source ADC values. returns a byte mask
    fvisi1_t  = fvisi1_t.clamp(0.0,1.0)
    fvisi2_t  = fvisi2_t.clamp(0.0,1.0)

    # make integer visi
    visi1_t = fvisi1_t.reshape( (batchsize,fvisi1_t.size()[2],fvisi1_t.size()[3]) ).long()
    visi2_t = fvisi2_t.reshape( (batchsize,fvisi2_t.size()[2],fvisi2_t.size()[3]) ).long()

    # image column origins
    #print data["meta_%s"%(train_or_valid)]
    source_minx  = torch.from_numpy( data["meta_%s"%(train_or_valid)].reshape((batchsize,3,1,4))[:,0,:,0].reshape((batchsize)) ).to(device=device)
    target1_minx = torch.from_numpy( data["meta_%s"%(train_or_valid)].reshape((batchsize,3,1,4))[:,1,:,0].reshape((batchsize)) ).to(device=device)
    target2_minx = torch.from_numpy( data["meta_%s"%(train_or_valid)].reshape((batchsize,3,1,4))[:,2,:,0].reshape((batchsize)) ).to(device=device)
    
    return source_t, target1_t, target2_t, flow1_t, flow2_t, visi1_t, visi2_t, fvisi1_t, fvisi2_t, source_minx, target1_minx, target2_minx

if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
