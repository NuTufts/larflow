import os,sys,time
from collections import OrderedDict
import shutil
import torch
import yaml
import numpy as np
import torch.distributed as dist
import MinkowskiEngine as ME

from larvoxel.model.larvoxelclassifier import LArVoxelClassifier
from larvoxel.loss.larvoxel_class_loss import LArVoxelClassLoss
from larvoxel.larvoxelclass_dataset import larvoxelClassDataset

def load_config_file( config_file, dump_to_stdout=False ):
    stream = open(config_file, 'r')
    dictionary = yaml.load(stream, Loader=yaml.FullLoader)
    if dump_to_stdout:
        for key, value in dictionary.items():
            print (key + " : " + str(value))
    stream.close()
    return dictionary

def get_model( config, dump_model=False ):
    model = LArVoxelClassifier()
    return model

def load_model_weights( model, checkpoint_file ):

    # Map all weights back to cpu first
    loc_dict = {"cuda:%d"%(gpu):"cpu" for gpu in range(10) }

    # load weights
    checkpoint = torch.load( checkpoint_file, map_location=loc_dict )

    # change names if we saved the distributed data parallel model state
    model.load_state_dict( checkpoint["state_larmatch"] )

    return checkpoint

def get_loss( config ):
    return LArVoxelClassLoss()

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

def make_meters(config):
    
    # accruacy and loss meters
    loss_meters = {"total":AverageMeter()}

    acc_meters  = {}
    for n in larvoxelClassDataset.pdg_name:
        acc_meters[n] = AverageMeter()

    time_meters = {}
    for l in ["batch","data","forward","loss_calc","backward","accuracy"]:
        time_meters[l] = AverageMeter()

    return loss_meters,acc_meters,time_meters
        
def accuracy(pred_t, truth_t, acc_meters, verbose=False):
    """Computes the accuracy metrics."""

    # LARMATCH METRICS
    with torch.no_grad():        
        print("====== calc class accuracy ========")
        #print("truth_t: ",truth_t)
        #print("pred_t: ",pred_t.F.shape)
        #print(pred_t.F)
        match_pred = torch.nn.Softmax(dim=1)( pred_t.F.detach() )
        #print("match_pred: ",match_pred)        
        for iclass,classname in enumerate(larvoxelClassDataset.pdg_name):
            print("[%d] %s -------"%(iclass,classname))
            if (truth_t==iclass).sum()>0:
                subset = match_pred[ truth_t.eq(iclass) ]
                #print("subset: ",subset.shape)
                correct = subset[:,iclass].gt(0.5).sum().item()
                print("%s name: correct %d out of %d"%(classname,correct,(truth_t==iclass).sum().item()))
                acc_meters[ classname ].update( float(correct)/float(truth_t.eq(iclass).sum().item()) )
                
    return True

def do_one_iteration( config, model, data_loader, criterion, optimizer,
                      acc_meters, loss_meters, time_meters, is_train, 
                      verbose=False, rank=None ):
    """
    Perform one iteration, i.e. processes one batch. Handles both train and validation iteraction.
    """
    dt_all = time.time()
    
    if is_train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    dt_io = time.time()

    batch = next(iter(data_loader))
    batchsize = len(batch)
    device = torch.device( config["DEVICE"] )

    if is_train:
        if batchsize != config["BATCHSIZE_TRAIN"]:
            print("dataset returned too small a batch! %d vs %s"%(batchsize,config["BATCHSIZE_TRAIN"]))
    else:
        if batchsize != config["BATCHSIZE_VALID"]:
            print("dataset returned too small a batch! %d vs %s"%(batchsize,config["BATCHSIZE_VALID"]))            
            
    coord_v = []
    feat_v  = []
    for ib,data in enumerate(batch):
        #print("batchindex[",ib,"] num tries=",data["ntries"]," nvoxels=",data["coord"].shape[0]," tree-entry=",data["tree_entry"])
        coord_t = coord_v.append( torch.from_numpy( data["coord"] ).to(device) )
        feat_t  = feat_v.append( torch.from_numpy( data["feat"] ).to(device) )

    coords, feats = ME.utils.sparse_collate( coord_v, feat_v )
    tf = ME.TensorField(features=feats,coordinates=coords)
    #print(tf)

    # truth
    truth   = torch.zeros( batchsize, dtype=torch.long )
    for ib,data in enumerate(batch):
        truth[ib] = data["pid"][0]
    truth = truth.to(device)

    dt_io = time.time()-dt_io
    if verbose:
        print("loaded data. %.2f secs"%(dt_io)," root-tree-entry=",data["tree_entry"])
    time_meters["data"].update(dt_io)

    if config["RUNPROFILER"]:
        torch.cuda.synchronize()

    dt_forward = time.time()

    # run forward pass
    pred = model( tf )
    dt_forward = time.time()-dt_forward
            
    # Calculate the loss
    dt_loss = time.time()
    loss = criterion( pred.F, truth )
                        
    if config["RUNPROFILER"]:
        torch.cuda.synchronize()
    time_meters["loss_calc"].update(time.time()-dt_loss)
    
    if is_train:
        # calculate gradients for this batch
        dt_backward = time.time()
        
        loss.backward()

        # dump par and/or grad for debug
        #for n,p in model_dict["larmatch"].named_parameters():
        #    if "out" in n:
        #        #print(n,": grad: ",p.grad)
        #        print(n,": ",p)
        #torch.nn.utils.clip_grad_norm_(model_dict["larmatch"].parameters(), 1.0)
        optimizer.step()
    
        if config["RUNPROFILER"]:
            torch.cuda.synchronize()                
        time_meters["backward"].update(time.time()-dt_backward)

    # update loss meters
    loss_meters["total"].update( loss.detach().item() )
                
    # measure accuracy and update accuracy meters
    dt_acc = time.time()
    acc = accuracy(pred, truth, acc_meters, verbose=verbose)

    # update time meter
    time_meters["accuracy"].update(time.time()-dt_acc)

    # measure elapsed time for batch
    time_meters["batch"].update(time.time()-dt_all)

    # done with iteration
    return 0
    
    
def prep_status_message( descripter, iternum, acc_meters, loss_meters, timers ):
    print("------------------------------------------------------------------------")
    print(" Iter[",iternum,"] ",descripter)
    print("  Time (secs): iter[%.2f] batch[%.3f] Forward[%.3f/batch] Backward[%.3f/batch] Acc[%.3f/batch] Data[%.3f/batch]"%(timers["batch"].sum,
                                                                                                                             timers["batch"].avg,
                                                                                                                             timers["forward"].avg,
                                                                                                                             timers["backward"].avg,
                                                                                                                             timers["accuracy"].avg,
                                                                                                                             timers["data"].avg))
    print("  Losses: ")
    for name,meter in loss_meters.items():
        if meter.count>0:
            print("    ",name,": ",meter.avg)
        else:
            print("    ",name,": NULL")            
    print("  Accuracies: ")
    for name,meter in acc_meters.items():
        if meter.count>0:
            print("    ",name,": ",meter.avg," (count: ",meter.count,")")
        else:
            print("    ",name,": NULL")            
    print("------------------------------------------------------------------------")
    
    
def save_checkpoint(state, is_best, p, filename='checkpoint.%dth.tar'):

    if p>0:
        filename = filename%(p)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.tar')

if __name__ == "__main__":

    class argstest:
        def __init__(self):
            self.config_file = "config.yaml"

    args = argstest()
    config = load_config_file( args, dump_to_stdout=True ) 

    model = get_larmatch_model( config, config["DEVICE"], dump_model=True )
