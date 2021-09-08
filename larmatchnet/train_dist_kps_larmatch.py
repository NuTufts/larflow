#!/bin/env python

"""
TRAINING SCRIPT FOR LARMATCH+KEYPOINT+SSNET NETWORKS
"""
## IMPORT

# python,numpy
from __future__ import print_function
import os,sys,argparse

import shutil
import time
import traceback
import numpy as np

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
import larmatch_engine
from larmatch_dataset import larmatchDataset
from load_larmatch_kps import load_larmatch_kps
from loss_larmatch_kps import SparseLArMatchKPSLoss

# ROOT, larcv
import ROOT as rt
from ROOT import std
from larcv import larcv
from larflow import larflow

def run(gpu, args, config ):
    #========================================================
    # CREATE PROCESS
    rank = args.nr * args.gpus + gpu
    print("START run() PROCESS: rank=%d gpu=%d"%(rank,gpu))    
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )
    #========================================================
    torch.manual_seed(0)

    if rank==0:
        tb_writer = SummaryWriter()

    model, model_dict = larmatch_engine.get_larmatch_model( config )
    gpu=0
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:%d"%(gpu) if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = SparseLArMatchKPSLoss( eval_ssnet=config["TRAIN_SSNET"],
                                       eval_keypoint_label=config["TRAIN_KP"],
                                       eval_keypoint_shift=config["TRAIN_KPSHIFT"],
                                       eval_affinity_field=config["TRAIN_PAF"] ).cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["LEARNING_RATE"], 
                                 weight_decay=config["WEIGHT_DECAY"])

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    train_dataset = larmatchDataset( filefolder="/home/twongjirad/working/data/larmatch_training_data/", random_access=True )
    #train_dataset = larmatchDataset( filelist=["larmatchtriplet_ana_trainingdata_testfile.root"])
    TRAIN_NENTRIES = len(train_dataset)
    print("TRAIN DATASET NENTRIES: ",TRAIN_NENTRIES," = 1 epoch")
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1,collate_fn=larmatchDataset.collate_fn)

    valid_dataset = larmatchDataset( filefolder="/home/twongjirad/working/data/larmatch_training_data/", random_access=True )
    #valid_dataset = larmatchDataset( filelist=["larmatchtriplet_ana_trainingdata_testfile.root"])
    VALID_NENTRIES = len(valid_dataset)
    print("VALID DATASET NENTRIES: ",VALID_NENTRIES," = 1 epoch")
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=1,collate_fn=larmatchDataset.collate_fn)

    loss_meters,acc_meters,time_meters = larmatch_engine.make_meters(config)
    
    with torch.autograd.profiler.profile(enabled=config["RUNPROFILER"]) as prof:    
        for iiter in range(config["NUM_ITERATIONS"]):
            train_iteration = config["START_ITER"] + iiter

            larmatch_engine.do_one_iteration(config,model_dict,train_loader,criterion,optimizer,
                                             acc_meters,loss_meters,time_meters,True,device,verbose=None)

            if iiter%int(config["TRAIN_ITER_PER_RECORD"])==0 and rank==0:
                # make averages and save to tensorboard, only if rank-0 process
                larmatch_engine.prep_status_message( "Train-Iteration", train_iteration, acc_meters, loss_meters, time_meters )

                # write to tensorboard
                # --------------------
                # losses go into same plot
                loss_scalars = { x:y.avg for x,y in loss_meters.items() }
                tb_writer.add_scalars('data/train_loss', loss_scalars, train_iteration )
                
                # split acc into different types
                # larmatch
                acc_scalars = {}
                for accname in larmatch_engine.LM_CLASS_NAMES:
                    acc_scalars[accname] = acc_meters[accname].avg
                tb_writer.add_scalars('data/train_larmatch_accuracy', acc_scalars, train_iteration )

                # ssnet
                ssnet_scalars = {}
                for accname in larmatch_engine.SSNET_CLASS_NAMES+["ssnet-all"]:
                    acc_scalars[accname] = acc_meters[accname].avg
                tb_writer.add_scalars('data/train_ssnet_accuracy', ssnet_scalars, train_iteration )
                
                # keypoint
                kp_scalars = {}
                for accname in larmatch_engine.KP_CLASS_NAMES:
                    acc_scalars[accname] = acc_meters[accname].avg
                tb_writer.add_scalars('data/train_kp_accuracy', kp_scalars, train_iteration )
                
                # paf
                paf_acc_scalars = { "paf":acc_meters["paf"].avg  }
                tb_writer.add_scalars("data/train_paf_accuracy", paf_acc_scalars, iiter )


            
        

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', default="config.yaml", type=str,
                        help='configuration file [default: config.yaml]')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    
    config = larmatch_engine.load_config_file( args )

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(run, nprocs=args.gpus, args=(args,config,))
    
    print("DISTRIBUTED MAIN DONE")
    
if __name__ == '__main__':

    main()
