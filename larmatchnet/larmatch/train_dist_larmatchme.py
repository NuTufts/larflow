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
import wandb

import MinkowskiEngine as ME

def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size, backend, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'
    os.environ['MASTER_PORT'] = port
    # initialize the process group
    #dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    # this function is responsible for synchronizing and successfully communicate across multiple process
    # involving multiple GPUs.

def run(gpu, args ):
    """
    This is the function run by each worker process
    """
    #ME.set_gpu_allocator(ME.GPUMemoryAllocatorType.CUDA)

    # larmatch imports
    import larmatch
    import larmatch.utils.larmatchme_engine as engine
    from larmatch_dataset import larmatchDataset
    from larmatch_mp_dataloader import larmatchMultiProcessDataloader

    # load config
    config = engine.load_config_file( args )
    print("LOAD CONFIG")
    sys.stdout.flush()
    
    #========================================================
    # CREATE PROCESS
    rank = gpu
    print("START run() PROCESS: rank=%d gpu=%d"%(rank,gpu))
    if not args.no_parallel:
        setup( rank, args.gpus, config["MULTIPROCESS_BACKEND"], config["MASTER_PORT"] )
    #========================================================
    torch.manual_seed(0)
    
    # ROOT, larcv
    import ROOT as rt
    from ROOT import std
    from larcv import larcv
    from larflow import larflow
    
    verbose = config["VERBOSE_MAIN_LOOP"]
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:%d"%(gpu) if torch.cuda.is_available() else "cpu")

    if rank==0 and str(config["WANDB_PROJECT"])!="NONE":
        # log into wandb
        print("RANK-0 THREAD: LOAD WANDB")
        sys.stdout.flush()    
        wandb.init(project=config["WANDB_PROJECT"],config=config)

    single_model = engine.get_model( config, dump_model=False )

    criterion = engine.make_loss_fn( config )
    num_loss_pars = len(list(criterion.parameters()))
    print("Num loss parameters: ",num_loss_pars)
    print("lm loss weight: ",criterion.lm)
    sys.stdout.flush()    
    
    if config["RESUME_FROM_CHECKPOINT"]:
        if not os.path.exists(config["CHECKPOINT_FILE"]):
            raise ValueError("Could not find checkpoint to load: ",config["CHECKPOINT_FILE"])
        print("RESUME MODEL CHECKPOINT")
        checkpoint_data = engine.load_model_weights( single_model, config["CHECKPOINT_FILE"] )
        
        # resume criterion weights
        if config["USE_LEARNABLE_LOSS_WEIGHTS"] and "state_lossweights" in checkpoint_data:
            print("RESUME LOSS-WEIGHT VALUES")
            criterion.load_state_dict( checkpoint_data["state_lossweights"] )

    if config["NORM_LAYER"]=="batchnorm" and not args.no_parallel:
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(single_model)
        ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(single_model)
            
    single_model.to(device)


    # Wrap the model
    if args.no_parallel:
        model = single_model
    else:
        model = nn.parallel.DistributedDataParallel(single_model, device_ids=[gpu],find_unused_parameters=False)

    #print("RANK-%d Loaded Model"%(rank),model)
    print("RANK-%d Loaded Model"%(rank))    
    if not args.no_parallel:
        torch.distributed.barrier()

    print("model.parameters() type: ",type(model.parameters()))
    param_list = list(model.parameters())
    if config["USE_LEARNABLE_LOSS_WEIGHTS"]:
        param_list += list(criterion.parameters())

    # adjust the learning rate
    print("------------- ADJUST LEARNING RATES -----------")
    base_lr = float(config["LEARNING_RATE"])
    #lower_factor = 1.0e-3
    lower_factor = 1.0
    if args.no_parallel:
        xmodel = model
    else:
        xmodel = model.module
        
    set_param_list = [
        {"params":xmodel.stem.parameters(),"lr":base_lr*lower_factor},
        {"params":xmodel.encoder.parameters(),"lr":base_lr*lower_factor},
        {"params":xmodel.decoder.parameters(),"lr":base_lr*lower_factor},
        {"params":criterion.parameters(),"lr":base_lr*1.0e-1}
    ]
    if xmodel.run_lm:      set_param_list.append( {"params":xmodel.lm_classifier.parameters(),"lr":base_lr} )
    if xmodel.run_ssnet:   set_param_list.append( {"params":xmodel.ssnet_head.parameters(),"lr":base_lr} )
    if xmodel.run_kplabel: set_param_list.append( {"params":xmodel.kplabel_head.parameters(),"lr":base_lr} )    

    
        
    train_loader = larmatchMultiProcessDataloader(config["TRAIN_DATALOADER_CONFIG"],
                                                  config["TRAIN_DATALOADER_CONFIG"]["BATCHSIZE"],
                                                  num_workers=config["TRAIN_DATALOADER_CONFIG"]["NUM_WORKERS"],
                                                  prefetch_batches=1,
                                                  collate_fn=larmatchDataset.collate_fn)
    TRAIN_NENTRIES = train_loader.nentries
    print("RANK-%d TRAIN DATASET NENTRIES: "%(rank),TRAIN_NENTRIES," = 1 epoch")    
    sys.stdout.flush()

    if rank==0:
        valid_loader = larmatchMultiProcessDataloader(config["VALID_DATALOADER_CONFIG"],
                                                      config["VALID_DATALOADER_CONFIG"]["BATCHSIZE"],
                                                      num_workers=1,
                                                      prefetch_batches=2,
                                                      collate_fn=larmatchDataset.collate_fn)
        VALID_NENTRIES = valid_loader.nentries                
        print("RANK-%d: LOAD VALID DATASET NENTRIES: "%(rank),VALID_NENTRIES," = 1 epoch")


        fixed_loader = larmatchMultiProcessDataloader(config["FIXED_DATALOADER_CONFIG"],
                                                      config["FIXED_DATALOADER_CONFIG"]["BATCHSIZE"],
                                                      num_workers=1,
                                                      prefetch_batches=2,
                                                      collate_fn=larmatchDataset.collate_fn)
        FIXED_NENTRIES = fixed_loader.nentries
        print("RANK-%d: LOAD FIXED DATASET NENTRIES: "%(rank),FIXED_NENTRIES)

        # WATCH
        if config["WANDB_PROJECT"] != "NONE":
            wandb.watch(model,log="all",log_freq=100)

    # SEUTP OPTIMIZER ===============================================================
    if args.no_parallel:
        ITERS_PER_EPOCH = TRAIN_NENTRIES/(config["TRAIN_DATALOADER_CONFIG"]["BATCHSIZE"])
    else:
        ITERS_PER_EPOCH = TRAIN_NENTRIES/(config["TRAIN_DATALOADER_CONFIG"]["BATCHSIZE"]*args.gpus)
    optimizer = torch.optim.AdamW(set_param_list,
                                  lr=float(config["LEARNING_RATE"]), 
                                  weight_decay=config["WEIGHT_DECAY"])
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(ITERS_PER_EPOCH), eta_min=config["LEARNING_RATE_MIN"])
    
    #if config["USE_LEARNABLE_LOSS_WEIGHTS"]:
    #    print("optimizer params")
    #    for n,par in enumerate(optimizer.param_groups):
    #        print(n,": ",par)
    
    if config["RESUME_FROM_CHECKPOINT"] and config["RESUME_OPTIM_FROM_CHECKPOINT"]:
        print("RESUME OPTIM CHECKPOINT")
        optimizer.load_state_dict( checkpoint_data["optimizer"] )
        
        
    
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
            #iter_verbose = True
            iter_verbose = config["VERBOSE_ITER_LOOP"]
            engine.do_one_iteration(config,model,train_loader,criterion,optimizer,
                                    acc_meters,loss_meters,time_meters,True,device,
                                    verbose=iter_verbose)

            # increment the LR scheduler
            #scheduler.step()

            # periodic checkpoint
            if iiter>0 and train_iteration%config["ITER_PER_CHECKPOINT"]==0:
                if rank==0:
                    tag=None
                    if "CHECKPOINT_TAG" in config:
                        tag = config["CHECKPOINT_TAG"]
                    print("RANK-0: saving periodic checkpoint")
                    engine.save_checkpoint({
                        'iter':train_iteration,
                        'epoch': train_iteration/float(TRAIN_NENTRIES),
                        'state_larmatch': model.state_dict(),
                        'state_lossweights':criterion.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, False, train_iteration,tag=tag)
                else:
                    print("RANK-%d: waiting for RANK-0 to save checkpoint"%(rank))                
            
            if verbose: print("RANK-%d: current tree entry=%d"%(rank,train_dataset._current_entry))

            if train_iteration%int(config["TRAIN_ITER_PER_RECORD"])==0 and rank==0:
                # make averages and save to tensorboard, only if rank-0 process
                engine.prep_status_message( "Train-Iteration", train_iteration, acc_meters, loss_meters, time_meters )
                if config["USE_LEARNABLE_LOSS_WEIGHTS"]:
                    print("loss weights")
                    if config["RUN_LARMATCH"]:
                        print("  lm weight: ",torch.exp(-criterion.task_weights["lm"].detach()).item())
                    if config["RUN_SSNET"]:
                        print("  ssnet: ",torch.exp(-criterion.task_weights["ssnet"].detach()).item())
                    if config["RUN_KPLABEL"]:
                        print("  kp: ",torch.exp(-criterion.task_weights["kp"].detach()).item())

                # write to WANDB
                # --------------------

                logme = {}
                
                # losses go into same plot                
                loss_scalars = { "loss/"+x:y.avg for x,y in loss_meters.items() }
                logme.update(loss_scalars)
                
                # split acc into different types
                # larmatch
                acc_scalars = {}
                for accname in engine.LM_CLASS_NAMES:
                    if acc_meters[accname].count>0:
                        acc_scalars["lm/"+accname] = acc_meters[accname].avg
                if len(acc_scalars)>0:
                    logme.update(acc_scalars)

                # ssnet
                ssnet_scalars = {}
                for accname in engine.SSNET_CLASS_NAMES+["ssnet-all"]:
                    if acc_meters[accname].count>0:
                        ssnet_scalars["ssnet/"+accname] = acc_meters[accname].avg
                if len(ssnet_scalars)>0:
                    logme.update(ssnet_scalars)
                
                # keypoint
                kp_scalars = {}
                for accname in engine.KP_CLASS_NAMES:
                    for x in ["_pos","_neg"]:
                        if acc_meters[accname+x].count>0:
                            kp_scalars["kplabel/"+accname+x] = acc_meters[accname+x].avg
                if len(kp_scalars)>0:
                    logme.update(kp_scalars)                    
                
                # paf
                if acc_meters["paf"].count>0:
                    paf_acc_scalars = { "paf/paf":acc_meters["paf"].avg  }
                    logme.update(paf_acc_scalars)                    

                # loss params
                loss_weight_scalars = {}
                for k,par in criterion.named_parameters():
                    loss_weight_scalars["loss-weights/"+k] = torch.exp( -par.detach() ).item()
                logme.update(loss_weight_scalars)

                # add learning rate
                logme["lr"] = optimizer.param_groups[0]["lr"]

                # log into wandb
                logme_train = {}
                for x,y in logme.items():
                    logme_train["train/"+x] = y
                if config["WANDB_PROJECT"]!="NONE":                    
                    wandb.log(logme_train,step=train_iteration)


            if config["TRAIN_ITER_PER_VALIDPT"]>0 and train_iteration%int(config["TRAIN_ITER_PER_VALIDPT"])==0 and iiter>0:
                if rank==0:
                    valid_loss_meters,valid_acc_meters,valid_time_meters = engine.make_meters(config)                    
                    for viter in range(int(config["NUM_VALID_ITERS"])):
                        with torch.no_grad():
                            engine.do_one_iteration(config,model,
                                                    valid_loader,criterion,optimizer,
                                                    valid_acc_meters,valid_loss_meters,valid_time_meters,
                                                    False,device,verbose=False)
                    engine.prep_status_message( "Valid-Iteration", train_iteration,
                                                valid_acc_meters,
                                                valid_loss_meters,
                                                valid_time_meters )
                    # write to WANDB
                    # --------------------

                    logme = {}
                    
                    # losses go into same plot
                    loss_scalars = { "loss/"+x:y.avg for x,y in valid_loss_meters.items() }
                    logme.update(loss_scalars)
                
                    # split acc into different types
                    # larmatch
                    val_acc_scalars = {}
                    for accname in engine.LM_CLASS_NAMES:
                        if valid_acc_meters[accname].count>0:
                            val_acc_scalars["lm/"+accname] = valid_acc_meters[accname].avg
                    if len(val_acc_scalars)>0:
                        logme.update(val_acc_scalars)

                    # ssnet
                    val_ssnet_scalars = {}
                    for accname in engine.SSNET_CLASS_NAMES+["ssnet-all"]:
                        if valid_acc_meters[accname].count>0:
                            val_ssnet_scalars["ssnet/"+accname] = valid_acc_meters[accname].avg
                    if len(val_ssnet_scalars)>0:
                        logme.update( val_ssnet_scalars )
                
                    # keypoint
                    val_kp_scalars = {}
                    for accname in engine.KP_CLASS_NAMES:
                        for x in ["_pos","_neg"]:                        
                            if valid_acc_meters[accname+x].count>0:
                                val_kp_scalars["kplabel/"+accname+x] = valid_acc_meters[accname+x].avg
                    if len(val_kp_scalars)>0:
                        logme.update( val_kp_scalars )
                
                    # paf
                    if valid_acc_meters["paf"].count>0:
                        val_paf_acc_scalars = { "paf/paf":valid_acc_meters["paf"].avg  }
                        logme.update( val_paf_acc_scalars )

                    logme_valid = {}
                    for x,y in logme.items():
                        logme_valid["valid/"+x] = y
                    # log into wandb
                    if config["WANDB_PROJECT"]!="NONE":                    
                        wandb.log(logme_valid,step=train_iteration)
                    

                else:
                    if verbose: print("RANK-%d process waiting for RANK-0 validation run"%(rank))

            if rank==0 and config["TRAIN_ITER_PER_FIXEDPT"]>0 and train_iteration%int(config["TRAIN_ITER_PER_FIXEDPT"])==0 and iiter>0:
                fixed_loss_meters,fixed_acc_meters,fixed_time_meters = engine.make_meters(config)                    
                with torch.no_grad():
                    engine.do_one_iteration(config, model,
                                            fixed_loader,criterion,optimizer,
                                            fixed_acc_meters,fixed_loss_meters,fixed_time_meters,
                                            False,device,verbose=False)
                    #print("Fixed check")
                    if not args.no_parallel:
                        for k,par in model.module.named_parameters():
                            #print(k,type(par))
                            if torch.isnan(par.data).any():
                                raise ValueError("Found a NAN")
                    else:
                        for k,par in model.named_parameters():
                            #print(k,type(par))
                            if torch.isnan(par.data).any():
                                raise ValueError("Found a NAN")                        
                        
                    engine.prep_status_message( "Fixed-Iteration", train_iteration,
                                                fixed_acc_meters,
                                                fixed_loss_meters,
                                                fixed_time_meters )
                    # write to wandb
                    # --------------------

                    logme = {}
                    
                    # losses go into same plot
                    loss_scalars = { "loss/"+x:y.avg for x,y in fixed_loss_meters.items() }
                    logme.update(loss_scalars)
                    
                    # split acc into different types
                    # larmatch
                    fixed_acc_scalars = {}
                    for accname in engine.LM_CLASS_NAMES:
                        if fixed_acc_meters[accname].count>0:
                            fixed_acc_scalars["lm/"+accname] = fixed_acc_meters[accname].avg
                    if len(fixed_acc_scalars)>0:
                        logme.update(fixed_acc_scalars)

                    # ssnet
                    fixed_ssnet_scalars = {}
                    for accname in engine.SSNET_CLASS_NAMES+["ssnet-all"]:
                        if fixed_acc_meters[accname].count>0:
                            fixed_ssnet_scalars["ssnet/"+accname] = fixed_acc_meters[accname].avg
                    if len(fixed_ssnet_scalars)>0:
                        logme.update( fixed_ssnet_scalars )
                        
                    # keypoint
                    fixed_kp_scalars = {}
                    for accname in engine.KP_CLASS_NAMES:
                        for x in ["_pos","_neg"]:
                            if fixed_acc_meters[accname+x].count>0:
                                fixed_kp_scalars["kplabel/"+accname+x] = fixed_acc_meters[accname+x].avg
                    if len(fixed_kp_scalars)>0:
                        logme.update( fixed_kp_scalars )
                        
                    # paf
                    if fixed_acc_meters["paf"].count>0:
                        fixed_paf_acc_scalars = { "paf/paf":fixed_acc_meters["paf"].avg  }
                        logme.update( fixed_paf_acc_scalars )

                    logme_fixed = {}
                    for x,y in logme.items():
                        logme_fixed["fixed/"+x] = y
                        # log into wandb
                    if config["WANDB_PROJECT"]!="NONE":                           
                        wandb.log(logme_fixed,step=train_iteration)


        if verbose: print("RANK-%d process finished. Waiting to sync."%(rank))
        if rank==0:
            print("RANK-0: saving last checkpoint")
            tag=None
            if "CHECKPOINT_TAG" in config:
                tag = config["CHECKPOINT_TAG"]            
            engine.save_checkpoint({
                'iter':train_iteration,
                'epoch': train_iteration/float(TRAIN_NENTRIES),
                'state_larmatch': model.state_dict(),
                'state_lossweights':criterion.state_dict(),                
                'optimizer' : optimizer.state_dict(),
            }, False, train_iteration, tag=tag)
        
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
                        help='number of data loading workers (default: 1)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--no-parallel',default=False,action='store_true',help='if provided, will run without distributed training')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    
    if args.no_parallel:
        print("RUNNING WITHOUT USING TORCH DDP")
        run( 0, args )
    else:
        mp.spawn(run, nprocs=args.gpus, args=(args,), join=True)

    
    print("DISTRIBUTED MAIN DONE")
    
if __name__ == '__main__':

    main()
