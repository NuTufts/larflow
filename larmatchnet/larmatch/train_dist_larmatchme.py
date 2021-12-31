#!/bin/env python
"""
TRAINING SCRIPT FOR LARMATCH+KEYPOINT+SSNET NETWORKS
"""
## IMPORT

# python,numpy
from __future__ import print_function
import os,sys,argparse,gc
import shutil
import time,datetime
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

import MinkowskiEngine as ME

def cleanup():
    dist.destroy_process_group()

def run(gpu, args ):
    """
    This is the function run by each worker process
    """
    ME.set_gpu_allocator(ME.GPUMemoryAllocatorType.CUDA)
    
    # tensorboardX
    from torch.utils.tensorboard import SummaryWriter
    
    # larmatch imports
    import larmatch
    import larmatch.utils.larmatchme_engine as engine
    from larmatch_dataset import larmatchDataset

    # ROOT, larcv
    import ROOT as rt
    from ROOT import std
    from larcv import larcv
    from larflow import larflow

    #========================================================
    # CREATE PROCESS
    rank = args.nr * args.gpus + gpu
    print("START run() PROCESS: rank=%d gpu=%d"%(rank,gpu))
    if not args.no_parallel:
        dist.init_process_group(                                   
	    #backend='nccl',
            backend='gloo',        
            #init_method='env://',
            init_method='file:///tmp/sharedfile',
	    world_size=args.world_size,                              
	    rank=rank,
            timeout=datetime.timedelta(0, 1800)
        )
    #========================================================
    torch.manual_seed(gpu)

    config = engine.load_config_file( args )
    verbose = config["VERBOSE_MAIN_LOOP"]

    if rank==0:
        tb_writer = SummaryWriter()

    single_model = engine.get_model( config )

    if config["RESUME_FROM_CHECKPOINT"]:
        if not os.path.exists(config["CHECKPOINT_FILE"]):
            raise ValueError("Could not find checkpoint to load: ",config["CHECKPOINT_FILE"])
        print("RESUME MODEL CHECKPOINT")
        checkpoint_data = engine.load_model_weights( single_model, config["CHECKPOINT_FILE"] )
            
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:%d"%(gpu) if torch.cuda.is_available() else "cpu")
    single_model.to(device)

    criterion = engine.make_loss_fn( config )

    # Wrap the model
    if args.no_parallel:
        model = single_model
    else:
        model = nn.parallel.DistributedDataParallel(single_model, device_ids=[gpu],find_unused_parameters=False)

    print("RANK-%d Model"%(rank),model)
    if not args.no_parallel:
        torch.distributed.barrier()

    print("model.parameters() type: ",type(model.parameters()))
    param_list = list(model.parameters())
    if config["USE_LEARNABLE_LOSS_WEIGHTS"]:
        param_list += list(criterion.parameters())
    optimizer = torch.optim.AdamW(param_list,
                                  lr=float(config["LEARNING_RATE"]), 
                                  weight_decay=config["WEIGHT_DECAY"])
    #if config["USE_LEARNABLE_LOSS_WEIGHTS"]:
    #    print("optimizer params", optimizer.param_groups)
    
    if config["RESUME_FROM_CHECKPOINT"] and config["RESUME_OPTIM_FROM_CHECKPOINT"]:
        print("RESUME OPTIM CHECKPOINT")
        optimizer.load_state_dict( checkpoint_data["optimizer"] )
    
    
    train_dataset = larmatchDataset( txtfile=config["TRAIN_DATASET_INPUT_TXTFILE"],
                                     random_access=True,
                                     verbose=config["TRAIN_DATASET_VERBOSE"],
                                     load_truth=True )
    if args.world_size>0:
        train_dataset.set_partition( rank, args.world_size )
    TRAIN_NENTRIES = len(train_dataset)
    print("RANK-%d TRAIN DATASET NENTRIES: "%(rank),TRAIN_NENTRIES," = 1 epoch")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["BATCH_SIZE"],
                                               collate_fn=larmatchDataset.collate_fn)
    sys.stdout.flush()

    if rank==0:
        valid_dataset = larmatchDataset( txtfile=config["VALID_DATASET_INPUT_TXTFILE"],
                                         random_access=True,
                                         load_truth=True,
                                         verbose=config["VALID_DATASET_VERBOSE"],
                                         npairs=None )
        VALID_NENTRIES = len(valid_dataset)
        print("RANK-%d: LOAD VALID DATASET NENTRIES: "%(rank),VALID_NENTRIES," = 1 epoch")
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=config["BATCH_SIZE"],
                                                   collate_fn=larmatchDataset.collate_fn)
    
    
    with torch.autograd.profiler.profile(enabled=config["RUN_PROFILER"]) as prof:    
        for iiter in range(config["NUM_ITERATIONS"]):
            train_iteration = config["START_ITER"] + iiter
            if verbose: print("RANK-%d iteration=%d"%(rank,train_iteration))

            # should be config parameter
            #if iiter%int(config["ITER_PER_CACHECLEAR"])==0:
            #    print("cache clear")
            torch.cuda.empty_cache() # clear cache and avoid fragmentation + memory overflow issues
            gc.collect()
            loss_meters,acc_meters,time_meters = engine.make_meters(config)
            engine.do_one_iteration(config,model,train_loader,criterion,optimizer,
                                    acc_meters,loss_meters,time_meters,True,device,
                                    verbose=config["VERBOSE_ITER_LOOP"])

            # periodic checkpoint
            if iiter>0 and train_iteration%config["ITER_PER_CHECKPOINT"]==0:
                if rank==0:
                    print("RANK-0: saving periodic checkpoint")
                    engine.save_checkpoint({
                        'iter':train_iteration,
                        'epoch': train_iteration/float(TRAIN_NENTRIES),
                        'state_larmatch': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, False, train_iteration)
                else:
                    print("RANK-%d: waiting for RANK-0 to save checkpoint"%(rank))                
            
            if verbose: print("RANK-%d: current tree entry=%d"%(rank,train_dataset._current_entry))
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
                for accname in engine.LM_CLASS_NAMES:
                    acc_scalars[accname] = acc_meters[accname].avg
                tb_writer.add_scalars('data/train_larmatch_accuracy', acc_scalars, train_iteration )

                # ssnet
                ssnet_scalars = {}
                for accname in engine.SSNET_CLASS_NAMES+["ssnet-all"]:
                    ssnet_scalars[accname] = acc_meters[accname].avg
                tb_writer.add_scalars('data/train_ssnet_accuracy', ssnet_scalars, train_iteration )
                
                # keypoint
                kp_scalars = {}
                for accname in engine.KP_CLASS_NAMES:
                    kp_scalars[accname] = acc_meters[accname].avg
                tb_writer.add_scalars('data/train_kp_accuracy', kp_scalars, train_iteration )
                
                # paf
                paf_acc_scalars = { "paf":acc_meters["paf"].avg  }
                tb_writer.add_scalars("data/train_paf_accuracy", paf_acc_scalars, train_iteration )

                # loss params
                loss_weight_scalars = {}
                for k,par in criterion.named_parameters():
                    loss_weight_scalars[k] = torch.exp( -par.detach() ).item()
                tb_writer.add_scalars("data/loss_weights", loss_weight_scalars, train_iteration )

            if config["TRAIN_ITER_PER_VALIDPT"]>0 and iiter%int(config["TRAIN_ITER_PER_VALIDPT"])==0:
                if rank==0:
                    valid_loss_meters,valid_acc_meters,valid_time_meters = engine.make_meters(config)                    
                    for viter in range(int(config["NUM_VALID_ITERS"])):
                        with torch.no_grad():
                            engine.do_one_iteration(config,single_model,
                                                    valid_loader,criterion,optimizer,
                                                    valid_acc_meters,valid_loss_meters,valid_time_meters,
                                                    False,device,verbose=False)
                    engine.prep_status_message( "Valid-Iteration", train_iteration,
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
                    for accname in engine.LM_CLASS_NAMES:
                        val_acc_scalars[accname] = valid_acc_meters[accname].avg
                    tb_writer.add_scalars('data/valid_larmatch_accuracy', val_acc_scalars, train_iteration )

                    # ssnet
                    val_ssnet_scalars = {}
                    for accname in engine.SSNET_CLASS_NAMES+["ssnet-all"]:
                        val_ssnet_scalars[accname] = valid_acc_meters[accname].avg
                    tb_writer.add_scalars('data/valid_ssnet_accuracy', val_ssnet_scalars, train_iteration )
                
                    # keypoint
                    val_kp_scalars = {}
                    for accname in engine.KP_CLASS_NAMES:
                        val_kp_scalars[accname] = valid_acc_meters[accname].avg
                    tb_writer.add_scalars('data/valid_kp_accuracy', val_kp_scalars, train_iteration )
                
                    # paf
                    val_paf_acc_scalars = { "paf":valid_acc_meters["paf"].avg  }
                    tb_writer.add_scalars("data/valid_paf_accuracy", val_paf_acc_scalars, train_iteration )

                else:
                    if verbose: print("RANK-%d process waiting for RANK-0 validation run"%(rank))

                # wait for rank-0 to finish reporting and plotting
                #if not args.no_parallel:
                #    torch.distributed.barrier()

        if verbose: print("RANK-%d process finished. Waiting to sync."%(rank))
        if rank==0:
            print("RANK-0: saving last checkpoint")
            engine.save_checkpoint({
                'iter':train_iteration,
                'epoch': train_iteration/float(TRAIN_NENTRIES),
                'state_larmatch': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, False, train_iteration)
        
        if not args.no_parallel:
            torch.distributed.barrier()


    if not args.no_parallel:
        cleanup()
    return

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', required=True, default="config.yaml", type=str,
                        help='configuration file [default: config.yaml]')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--no-parallel',default=False,action='store_true',help='if provided, will run without distributed training')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    ME.set_gpu_allocator(ME.GPUMemoryAllocatorType.CUDA)
    if args.no_parallel:
        print("RUNNING WITHOUT USING TORCH DDP")
        run( 0, args )
    else:
        mp.spawn(run, nprocs=args.gpus, args=(args,), join=True)

    
    print("DISTRIBUTED MAIN DONE")
    
if __name__ == '__main__':

    main()
