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
import larvoxel_engine
from larvoxel_dataset import larvoxelDataset
from loss_larvoxel import LArVoxelLoss
from larvoxelnet import LArVoxelMultiDecoder

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

    stream = open(args.config_file, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()

    if rank==0:
        tb_writer = SummaryWriter()

    single_model = LArVoxelMultiDecoder(run_ssnet=config["RUN_SSNET"],run_kplabel=config["RUN_KPLABEL"])

    if config["RESUME_FROM_CHECKPOINT"]:
        if not os.path.exists(config["CHECKPOINT_FILE"]):
            raise ValueError("Could not find checkpoint to load: ",config["CHECKPOINT_FILE"])

        checkpoint_data = larvoxel_engine.load_model_weights( single_model, config["CHECKPOINT_FILE"] )
            
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:%d"%(gpu) if torch.cuda.is_available() else "cpu")
    single_model.to(device)

    criterion = LArVoxelLoss( eval_ssnet=config["RUN_SSNET"],
                              eval_keypoint_label=config["RUN_KPLABEL"] ).to(device)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(single_model, device_ids=[gpu],find_unused_parameters=False)
    if rank==0: print("RANK-%d Model"%(rank),model)
    torch.distributed.barrier()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(config["LEARNING_RATE"]), 
                                 weight_decay=config["WEIGHT_DECAY"])
    
    if config["RESUME_FROM_CHECKPOINT"] and config["RESUME_OPTIM_FROM_CHECKPOINT"]:
        optimizer.load_state_dict( checkpoint_data["optimizer"] )
    
    
    # re-specify the dictionary
    model_dict = {"larmatch":model}

    train_dataset = larvoxelDataset( txtfile=config["INPUT_TXTFILE_TRAIN"], random_access=True, is_voxeldata=True )
    if args.world_size>0:
        train_dataset.set_partition( rank, args.world_size )
    TRAIN_NENTRIES = len(train_dataset)
    if rank==0: print("RANK-%d TRAIN DATASET NENTRIES: "%(rank),TRAIN_NENTRIES," = 1 epoch")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["BATCHSIZE_TRAIN"],
                                               collate_fn=larvoxelDataset.collate_fn)
    sys.stdout.flush()

    if rank==0:
        valid_dataset = larvoxelDataset( txtfile=config["INPUT_TXTFILE_VALID"], random_access=True, is_voxeldata=True )
        VALID_NENTRIES = len(valid_dataset)
        print("RANK-%d: LOAD VALID DATASET NENTRIES: "%(rank),VALID_NENTRIES," = 1 epoch")
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=config["BATCHSIZE_VALID"],
                                                   collate_fn=larvoxelDataset.collate_fn)
    
    loss_meters,acc_meters,time_meters = larvoxel_engine.make_meters(config)

    torch.autograd.set_detect_anomaly(True)
    
    with torch.autograd.profiler.profile(enabled=config["RUNPROFILER"]) as prof:    
        for iiter in range(config["NUM_ITERATIONS"]):
            train_iteration = config["START_ITER"] + iiter
            #print("RANK-%d iteration=%d"%(rank,train_iteration))
            larvoxel_engine.do_one_iteration(config,model_dict,train_loader,criterion,optimizer,
                                             acc_meters,loss_meters,time_meters,True,device,
                                             verbose=config["LOSS_VERBOSE"],rank=rank)

            # periodic checkpoint
            if iiter>0 and iiter%config["ITER_PER_CHECKPOINT"]==0:
                if rank==0:
                    print("RANK-0: saving periodic checkpoint")
                    larvoxel_engine.save_checkpoint({
                        'iter':train_iteration,
                        'epoch': train_iteration/float(TRAIN_NENTRIES),
                        'state_larmatch': model_dict["larmatch"].module.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, False, train_iteration)
                else:
                    print("RANK-%d: waiting for RANK-0 to save checkpoint"%(rank))                

            if train_dataset._current_entry%1000==0:
                print("RANK-%d: current tree entry=%d"%(rank,train_dataset._current_entry))
            if iiter%int(config["TRAIN_ITER_PER_RECORD"])==0 and rank==0:
                # make averages and save to tensorboard, only if rank-0 process
                larvoxel_engine.prep_status_message( "Train-Iteration", train_iteration, acc_meters, loss_meters, time_meters )

                # write to tensorboard
                # --------------------
                # losses go into same plot
                loss_scalars = { x:y.avg for x,y in loss_meters.items() }
                tb_writer.add_scalars('data/train_loss', loss_scalars, train_iteration )
                
                # split acc into different types
                # larmatch
                acc_scalars = {}
                for accname in larvoxel_engine.LM_CLASS_NAMES:
                    if acc_meters[accname].count>0:
                        acc_scalars[accname] = acc_meters[accname].avg
                tb_writer.add_scalars('data/train_larmatch_accuracy', acc_scalars, train_iteration )

                # ssnet
                ssnet_scalars = {}
                for accname in larvoxel_engine.SSNET_CLASS_NAMES+["ssnet-all"]:
                    if acc_meters[accname].count>0:                    
                        ssnet_scalars[accname] = acc_meters[accname].avg
                tb_writer.add_scalars('data/train_ssnet_accuracy', ssnet_scalars, train_iteration )
                
                # keypoint
                kp_scalars = {}
                for accname in larvoxel_engine.KP_CLASS_NAMES:
                    if acc_meters[accname].count>0:                                        
                        kp_scalars[accname] = acc_meters[accname].avg
                tb_writer.add_scalars('data/train_kp_accuracy', kp_scalars, train_iteration )
                
                # paf
                paf_acc_scalars = { "paf":acc_meters["paf"].avg  }
                tb_writer.add_scalars("data/train_paf_accuracy", paf_acc_scalars, train_iteration )

                # reset after storing values
                for meters in [loss_meters,acc_meters,time_meters]:
                    for i,m in meters.items():
                        m.reset()

                # monitor gradients
                # one day

            if config["TRAIN_ITER_PER_VALIDPT"]>0 and iiter%int(config["TRAIN_ITER_PER_VALIDPT"])==0:
                if rank==0:
                    valid_loss_meters,valid_acc_meters,valid_time_meters = larvoxel_engine.make_meters(config)                    
                    for viter in range(int(config["NUM_VALID_ITERS"])):
                        with torch.no_grad():
                            larvoxel_engine.do_one_iteration(config,{"larmatch":single_model},
                                                             valid_loader,criterion,optimizer,
                                                             valid_acc_meters,valid_loss_meters,valid_time_meters,
                                                             False,device,verbose=config["LOSS_VERBOSE"],rank=rank)
                    larvoxel_engine.prep_status_message( "Valid-Iteration", train_iteration,
                                                         valid_acc_meters,
                                                         valid_loss_meters,
                                                         valid_time_meters )
                    # write to tensorboard
                    # --------------------
                    # losses go into same plot
                    loss_scalars = { x:y.avg for x,y in loss_meters.items() }
                    tb_writer.add_scalars('data/valid_loss', loss_scalars, train_iteration )
                
                    # split acc into different types
                    # larmatch
                    val_acc_scalars = {}
                    for accname in larvoxel_engine.LM_CLASS_NAMES:
                        val_acc_scalars[accname] = valid_acc_meters[accname].avg
                    tb_writer.add_scalars('data/valid_larmatch_accuracy', val_acc_scalars, train_iteration )

                    # ssnet
                    val_ssnet_scalars = {}
                    for accname in larvoxel_engine.SSNET_CLASS_NAMES+["ssnet-all"]:
                        val_ssnet_scalars[accname] = valid_acc_meters[accname].avg
                    tb_writer.add_scalars('data/valid_ssnet_accuracy', val_ssnet_scalars, train_iteration )
                
                    # keypoint
                    val_kp_scalars = {}
                    for accname in larvoxel_engine.KP_CLASS_NAMES:
                        val_kp_scalars[accname] = valid_acc_meters[accname].avg
                    tb_writer.add_scalars('data/valid_kp_accuracy', val_kp_scalars, train_iteration )
                
                    # paf
                    val_paf_acc_scalars = { "paf":valid_acc_meters["paf"].avg  }
                    tb_writer.add_scalars("data/valid_paf_accuracy", val_paf_acc_scalars, train_iteration )

                else:
                    print("RANK-%d process waiting for RANK-0 validation run"%(rank))

                # wait for rank-0 to finish reporting and plotting
                #torch.distributed.barrier()

        print("RANK-%d process finished. Waiting to sync."%(rank))
        if rank==0:
            print("RANK-0: saving last checkpoint")
            larvoxel_engine.save_checkpoint({
                'iter':train_iteration,
                'epoch': train_iteration/float(TRAIN_NENTRIES),
                'state_larmatch': model_dict["larmatch"].module.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, False, train_iteration)
        
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
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(run, nprocs=args.gpus, args=(args,), join=True)
    
    print("DISTRIBUTED MAIN DONE")
    
if __name__ == '__main__':

    main()
