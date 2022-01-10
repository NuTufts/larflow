#!/bin/env python

"""
TRAINING SCRIPT FOR LARMATCH+KEYPOINT+SSNET NETWORKS
"""
## IMPORT

# python,numpy
from __future__ import print_function
import os,sys,argparse

import shutil
import time,datetime
import traceback
import numpy as np
import yaml

# torch
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import torch.distributed as dist

# tensorboardX
from torch.utils.tensorboard import SummaryWriter

# larmatch imports
import larvoxel.utils.larvoxel_classify_engine as engine
from larvoxel.larvoxelclass_dataset import larvoxelClassDataset
from larvoxel.loss.larvoxel_class_loss import LArVoxelClassLoss

# ROOT, larcv
import ROOT as rt
from ROOT import std
from larcv import larcv
from larflow import larflow

def cleanup():
    dist.destroy_process_group()

def run(gpu, args ):
    #========================================================
    # CREATE PROCESS
    rank = args.nr * args.gpus + gpu
    print("START run() PROCESS: rank=%d gpu=%d"%(rank,gpu))    
    dist.init_process_group(                                   
    	#backend='nccl',
        backend='gloo',        
   	init_method='env://',
        #init_method='file:///tmp/sharedfile',
    	world_size=args.world_size,                              
    	rank=rank,
        timeout=datetime.timedelta(0, 1800)
    )
    #========================================================
    torch.manual_seed(rank)

    # Get Configuration File
    config = engine.load_config_file(args.config_file)

    # get device
    DEVICE = torch.device(config["DEVICE"])
    
    # Get Model
    single_model = engine.get_model(config)

    # Load Weights if resuming from checkpoint
    if config["RESUME_FROM_CHECKPOINT"]:
        if not os.path.exists(config["CHECKPOINT_FILE"]):
            raise ValueError("Could not find checkpoint to load: ",config["CHECKPOINT_FILE"])

        checkpoint_data = engine.load_model_weights( single_model, config["CHECKPOINT_FILE"] )
    
    if rank==0:
        tb_writer = SummaryWriter(comment="larvoxel_classify")

    single_model.to(DEVICE)
    criterion = engine.get_loss(config).to(DEVICE)

    # Wrap the model for distributed training
    model = nn.parallel.DistributedDataParallel(single_model, device_ids=[gpu],find_unused_parameters=False)
    if rank==0: print("RANK-%d Model"%(rank),model)
    torch.distributed.barrier()

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(config["LEARNING_RATE"]), 
                                  weight_decay=config["WEIGHT_DECAY"])
    
    if config["RESUME_FROM_CHECKPOINT"] and config["RESUME_OPTIM_FROM_CHECKPOINT"]:
        optimizer.load_state_dict( checkpoint_data["optimizer"] )
        
    train_dataset = larvoxelClassDataset( txtfile=config["INPUT_TXTFILE_TRAIN"],
                                          random_access=True, load_truth=True )
    if args.world_size>0:
        train_dataset.set_partition( rank, args.world_size )
    TRAIN_NENTRIES = len(train_dataset)
    if rank==0: print("RANK-%d TRAIN DATASET NENTRIES: "%(rank),TRAIN_NENTRIES," = 1 epoch")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["BATCHSIZE_TRAIN"],
                                               collate_fn=larvoxelClassDataset.collate_fn)
    sys.stdout.flush()

    if rank==0:
        valid_dataset = larvoxelClassDataset( txtfile=config["INPUT_TXTFILE_VALID"],
                                              random_access=True, load_truth=True )
        VALID_NENTRIES = len(valid_dataset)
        print("RANK-%d: LOAD VALID DATASET NENTRIES: "%(rank),VALID_NENTRIES," = 1 epoch")
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=config["BATCHSIZE_VALID"],
                                                   collate_fn=larvoxelClassDataset.collate_fn)
    
    loss_meters,acc_meters,time_meters = engine.make_meters(config)

    if config["DETECT_ANOMALY"]:
        torch.autograd.set_detect_anomaly(True)
    
    with torch.autograd.profiler.profile(enabled=config["RUNPROFILER"]) as prof:    
        for iiter in range(config["NUM_ITERATIONS"]):
            train_iteration = config["START_ITER"] + iiter
            #print("RANK-%d iteration=%d"%(rank,train_iteration))
            engine.do_one_iteration(config, model,train_loader, criterion, optimizer,
                                             acc_meters, loss_meters, time_meters, True,
                                             verbose=config["LOSS_VERBOSE"],rank=rank)

            # periodic checkpoint
            if iiter>0 and train_iteration%config["ITER_PER_CHECKPOINT"]==0:
                if rank==0:
                    print("RANK-0: saving periodic checkpoint")
                    engine.save_checkpoint({
                        'iter':train_iteration,
                        'epoch': train_iteration/float(TRAIN_NENTRIES),
                        'state_larmatch': model.module.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, False, train_iteration, filename="lv.classify.checkpoint.%dth.tar")
                else:
                    print("RANK-%d: waiting for RANK-0 to save checkpoint"%(rank))                

            if train_dataset._current_entry%1000==0:
                print("RANK-%d: current tree entry=%d"%(rank,train_dataset._current_entry))
            if iiter%int(config["TRAIN_ITER_PER_RECORD"])==0 and rank==0:
                # make averages and save to tensorboard, only if rank-0 process
                engine.prep_status_message( "Train-Iteration", train_iteration, acc_meters, loss_meters, time_meters )

                # write to tensorboard
                # --------------------
                # losses go into same plot
                loss_scalars = { x:y.avg for x,y in loss_meters.items() }
                tb_writer.add_scalars('data/train_loss', loss_scalars, train_iteration )
                
                # split acc into different types
                # larmatch
                acc_scalars = {}
                for accname in larvoxelClassDataset.pdg_name:
                    if acc_meters[accname].count>0:
                        acc_scalars[accname] = acc_meters[accname].avg
                tb_writer.add_scalars('data/train_larmatch_accuracy', acc_scalars, train_iteration )

                # reset after storing values
                for meters in [loss_meters,acc_meters,time_meters]:
                    for i,m in meters.items():
                        m.reset()

                # monitor gradients
                # one day

            if config["TRAIN_ITER_PER_VALIDPT"]>0 and iiter%int(config["TRAIN_ITER_PER_VALIDPT"])==0:
                if rank==0:
                    valid_loss_meters,valid_acc_meters,valid_time_meters = engine.make_meters(config)                    
                    for viter in range(int(config["NUM_VALID_ITERS"])):
                        with torch.no_grad():
                            engine.do_one_iteration(config,single_model,
                                                             valid_loader,criterion,optimizer,
                                                             valid_acc_meters,valid_loss_meters,valid_time_meters,
                                                             False,verbose=config["LOSS_VERBOSE"],rank=rank)
                    engine.prep_status_message( "Valid-Iteration", train_iteration,
                                                         valid_acc_meters,
                                                         valid_loss_meters,
                                                         valid_time_meters )
                    # write to tensorboard
                    # --------------------
                    # losses go into same plot
                    loss_scalars = { x:y.avg for x,y in valid_loss_meters.items() }
                    tb_writer.add_scalars('data/valid_loss', loss_scalars, train_iteration )
                
                    # split acc into different types
                    # larmatch
                    val_acc_scalars = {}
                    for accname in larvoxelClassDataset.pdg_name:
                        if valid_acc_meters[accname].count>0:
                            val_acc_scalars[accname] = valid_acc_meters[accname].avg
                    tb_writer.add_scalars('data/valid_accuracy', val_acc_scalars, train_iteration )

                else:
                    print("RANK-%d process waiting for RANK-0 validation run"%(rank))

                # wait for rank-0 to finish reporting and plotting
                #torch.distributed.barrier()

        print("RANK-%d process finished. Waiting to sync."%(rank))
        if rank==0:
            print("RANK-0: saving last checkpoint")
            engine.save_checkpoint({
                'iter':train_iteration,
                'epoch': train_iteration/float(TRAIN_NENTRIES),
                'state_larmatch': model.module.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, False, train_iteration,filename="lv.classify.checkpoint.%dth.tar")
        
        torch.distributed.barrier()


    cleanup()
    return

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', default="config_voxelmultidecoder.yaml", type=str,
                        help='configuration file [default: config.yaml]')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8889'
    mp.spawn(run, nprocs=args.gpus, args=(args,), join=True)
    
    print("DISTRIBUTED MAIN DONE")
    
if __name__ == '__main__':

    main()
