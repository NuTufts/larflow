import os,sys,time
import shutil
import torch
import yaml
import numpy as np
from larmatchvoxel import LArMatchVoxel
from collections import OrderedDict
import MinkowskiEngine as ME
import torch.distributed as dist

SSNET_CLASS_NAMES=["bg","electron","gamma","muon","pion","proton","other"]
KP_CLASS_NAMES=["kp_nu","kp_trackstart","kp_trackend","kp_shower","kp_michel","kp_delta"]
LM_CLASS_NAMES=["lm_pos","lm_neg","lm_all"]

def get_larmatch_model( config, dump_model=False ):

    # create model, mark it to run on the device
    model = LArMatchVoxel(run_ssnet=config["RUN_SSNET"],
                          run_kplabel=config["RUN_KPLABEL"])

    if dump_model:
        # DUMP MODEL (for debugging)
        print(model)

    model_dict = {"larmatch":model}
    if model.run_ssnet:   model_dict["ssnet"] = model_dict["larmatch"].ssnet_classifier
    if model.run_kplabel: model_dict["kplabel"] = model_dict["larmatch"].kplabel_classifier
    
    return model, model_dict

def load_model_weights( model, checkpoint_file ):

    # Map all weights back to cpu first
    loc_dict = {"cuda:%d"%(gpu):"cpu" for gpu in range(10) }

    # load weights
    checkpoint = torch.load( checkpoint_file, map_location=loc_dict )

    # change names if we saved the distributed data parallel model state
    rename_distributed_checkpoint_par_names(checkpoint)

    model.load_state_dict( checkpoint["state_larmatch"] )

    return checkpoint

def load_config_file( args, dump_to_stdout=False ):
    stream = open(args.config_file, 'r')
    dictionary = yaml.load(stream, Loader=yaml.FullLoader)
    if dump_to_stdout:
        for key, value in dictionary.items():
            print (key + " : " + str(value))
    stream.close()
    return dictionary

def reshape_old_sparseconvnet_weights(checkpoint):

    # hack to be able to load sparseconvnet<1.3
    for name,arr in checkpoint["state_larmatch"].items():
        if ( ("resnet" in name and "weight" in name and len(arr.shape)==3) or
             ("stem" in name and "weight" in name and len(arr.shape)==3) or
             ("unet_layers" in name and "weight" in name and len(arr.shape)==3) or         
             ("feature_layer.weight" == name and len(arr.shape)==3 ) ):
            print("reshaping ",name)
            checkpoint["state_larmatch"][name] = arr.reshape( (arr.shape[0], 1, arr.shape[1], arr.shape[2]) )

    return

def rename_distributed_checkpoint_par_names(checkpoint):
    replacement = OrderedDict()
    notified = False
    for name,arr in checkpoint["state_larmatch"].items():
        if "module." in name and name[:len("module.")]=="module.":
            if not notified:
                print("renaming parameter by removing 'module.'")
                notified = True
            replacement[name[len("module."):]] = arr
        else:
            replacement[name] = arr
    checkpoint["state_larmatch"] = replacement
    return
            

def remake_separated_model_weightfile(checkpoint,model_dict,verbose=False):
    # copy items into larmatch dict
    larmatch_checkpoint_data = checkpoint["state_larmatch"]
    head_names = {"ssnet":"ssnet_head",
                  "kplabel":"kplabel_head",
                  "kpshift":"kpshift_head",
                  "paf":"affinity_head"}
    for name,model in model_dict.items():
        if name not in head_names:
            continue
        if verbose: print("STATE DATA: ",name)
        state_name = "stage_"+name
        if state_name not in checkpoint:
            continue
        checkpoint_data = checkpoint["state_"+name]
        for parname,arr in checkpoint_data.items():
            if verbose: print("append: ",head_names[name]+"."+parname,arr.shape)        
            larmatch_checkpoint_data[head_names[name]+"."+parname] = arr
    
    return larmatch_checkpoint_data



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
    loss_meters = {}
    for n in ("total","lm","ssnet","kp","paf"):
        loss_meters[n] = AverageMeter()
    
    accnames = LM_CLASS_NAMES+KP_CLASS_NAMES+["paf"]+SSNET_CLASS_NAMES+["ssnet-all"]
    acc_meters  = {}
    for n in accnames:
        acc_meters[n] = AverageMeter()

    time_meters = {}
    for l in ["batch","data","forward","loss_calc","backward","accuracy"]:
        time_meters[l] = AverageMeter()

    return loss_meters,acc_meters,time_meters
        
def accuracy(match_pred_t, match_label_t,
             ssnet_pred_t, ssnet_label_t,
             kp_pred_t, kp_label_t,
             paf_pred_t, paf_label_t,
             truematch_indices_t,
             acc_meters,
             verbose=False):
    """Computes the accuracy metrics."""

    # LARMATCH METRICS
    with torch.no_grad():
        match_pred = torch.nn.Softmax(dim=1)( match_pred_t.F.detach() )[:,1]
        npairs = match_pred.shape[0]
    
        pos_correct = (match_pred.gt(0.5)*match_label_t.eq(1)).sum().to(torch.device("cpu")).item()
        neg_correct = (match_pred.lt(0.5)*match_label_t.eq(0)).sum().to(torch.device("cpu")).item()
        npos = float(match_label_t.eq(1).sum().to(torch.device("cpu")).item())
        nneg = float(match_label_t.eq(0).sum().to(torch.device("cpu")).item())
        #print("larmatch npos=%d nneg=%d"%(npos,nneg))

    acc_meters["lm_pos"].update( float(pos_correct)/npos )
    acc_meters["lm_neg"].update( float(neg_correct)/nneg )
    acc_meters["lm_all"].update( float(pos_correct+neg_correct)/(npos+nneg) )

    # SSNET METRICS
    if ssnet_pred_t is not None:
        if ssnet_pred_t.shape[0]!=ssnet_label_t.shape[0]:
            ssnet_pred     = torch.index_select( ssnet_pred_t.detach(), 0, truematch_indices_t )
        else:
            ssnet_pred     = ssnet_pred_t.detach()
        ssnet_class    = torch.argmax( ssnet_pred, 1 )
        ssnet_correct  = ssnet_class.eq( ssnet_label_t )
        ssnet_tot_correct = ssnet_correct.sum().item()        
        for iclass,classname in enumerate(SSNET_CLASS_NAMES):
            if ssnet_label_t.eq(iclass).sum().item()>0:                
                ssnet_class_correct = ssnet_correct[ ssnet_label_t==iclass ].sum().item()    
                acc_meters[classname].update( float(ssnet_class_correct)/float(ssnet_label_t.eq(iclass).sum().item()) )
        acc_meters["ssnet-all"].update( ssnet_tot_correct/float(ssnet_label_t.shape[1]) )

    # KP METRIC
    if kp_pred_t is not None:
        if kp_pred_t.shape[0]!=kp_label_t.shape[0]:
            kp_pred  = torch.index_select( kp_pred_t.detach(),  0, truematch_indices_t )
            kp_label = torch.index_select( kp_label_t.detach(), 0, truematch_indices_t )
        else:
            kp_pred  = kp_pred_t.detach()
            kp_label = kp_label_t.detach()[:npairs]
        for c,kpname in enumerate(KP_CLASS_NAMES):
            kp_n_pos = float(kp_label[:,c].gt(0.5).sum().item())
            kp_pos   = float(kp_pred[:,c].gt(0.5)[ kp_label[:,c].gt(0.5) ].sum().item())
            #if verbose: print("kp[",c,"-",kpname,"] n_pos[>0.5]: ",kp_n_pos," pred[>0.5]: ",kp_pos)
            if kp_n_pos>0:
                acc_meters[kpname].update( kp_pos/kp_n_pos )

    # PARTICLE AFFINITY FLOW
    if paf_pred_t is not None:
        # we define accuracy with the direction is less than 20 degress
        if paf_pred_t.shape[0]!=paf_label_t.shape[0]:
            paf_pred  = torch.index_select( paf_pred_t.detach(),  0, truematch_indices_t )
            paf_label = torch.index_select( paf_label_t.detach(), 0, truematch_indices_t )
        else:
            paf_pred  = paf_pred_t.detach()
            paf_label = paf_label_t.detach()[:npairs]
        # calculate cosine
        paf_truth_lensum = torch.sum( paf_label*paf_label, 1 )
        paf_pred_lensum  = torch.sum( paf_pred*paf_pred, 1 )
        paf_pred_lensum  = torch.sqrt( paf_pred_lensum )
        paf_posexamples = paf_truth_lensum.gt(0.5)
        #print paf_pred[paf_posexamples,:].shape," ",paf_label[paf_posexamples,:].shape," ",paf_pred_lensum[paf_posexamples].shape
        paf_cos = torch.sum(paf_pred[paf_posexamples,:]*paf_label[paf_posexamples,:],1)/(paf_pred_lensum[paf_posexamples]+0.001)
        paf_npos  = paf_cos.shape[0]
        paf_ncorr = paf_cos.gt(0.94).sum().item()
        paf_acc = float(paf_ncorr)/float(paf_npos)
        #if verbose: print("paf: npos=",paf_npos," acc=",paf_acc)
        if paf_npos>0:
            acc_meters["paf"].update( paf_acc )
    
    
    return True

def do_one_iteration( config, model_dict, data_loader, criterion, optimizer,
                      acc_meters, loss_meters, time_meters, is_train, device,
                      verbose=False, rank=None ):
    """
    Perform one iteration, i.e. processes one batch. Handles both train and validation iteraction.
    """
    dt_all = time.time()
    
    if is_train:
        model_dict["larmatch"].train()
        #print("zero optimizer")
        optimizer.zero_grad()
    else:
        model_dict["larmatch"].eval()

    dt_io = time.time()

    data = next(iter(data_loader))[0]
    coordshape = data["voxcoord"].shape
    coord   = torch.from_numpy( data["voxcoord"] ).int().to(device)
    feat    = torch.from_numpy( np.clip( data["voxfeat"]/40.0, 0, 10.0 ) ).to(device)
    truth   = torch.from_numpy( data["truetriplet_t"] ).to(device)
    ssnet   = torch.from_numpy( data["ssnet_labels"] ).to(device)
    kplabel = torch.from_numpy( data["kplabel"] ).to(device)
    #print("feat: ",feat.shape," ",feat[:10])

    # check the input for NANs
    checklist = [ feat, coord, truth, ssnet, kplabel ]
    for iarr, arr in enumerate(checklist):
        if arr is not None and ( torch.isnan(arr).sum()>0 or torch.isinf(arr).sum()>0 ):
            print("RANK-%d Input array [%d] is nan or inf!"%(rank,iarr))
            print("RANK-%d is_train=%d root-tree-entry=%d"%(rank,is_train,data["tree_entry"]))
            print("KILL THE JOB")
            try:
                dist.destroy_process_group()  
            except KeyboardInterrupt: 
                os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
    
    coords, feats = ME.utils.sparse_collate(coords=[coord], feats=[feat])
    xinput = ME.SparseTensor( features=feats, coordinates=coords.to(device) )    

    dt_io = time.time()-dt_io
    if verbose:
        print("loaded data. %.2f secs"%(dt_io)," root-tree-entry=",data["tree_entry"])
    time_meters["data"].update(dt_io)

    if config["RUNPROFILER"]:
        torch.cuda.synchronize()

    dt_forward = time.time()

    # use UNET portion to first get feature vectors
    pred_dict = model_dict["larmatch"]( xinput )
    #for name,arr in pred_dict.items():
    #    if arr is not None: print(name," ",arr.shape)
    
    match_pred_t   = pred_dict["larmatch"]      
    ssnet_pred_t   = pred_dict["ssnet"]   if "ssnet" in pred_dict else None
    kplabel_pred_t = pred_dict["kplabel"] if "kplabel" in pred_dict else None
    kpshift_pred_t = pred_dict["kpshift"] if "kpshift" in pred_dict else None
    paf_pred_t     = pred_dict["paf"]     if "paf" in pred_dict else None

    match_label_t  = torch.from_numpy( data["voxlabel"] ).to(device)
    ssnet_label_t  = torch.from_numpy( data["ssnet_labels"] ).to(device).squeeze().unsqueeze(0)
    kp_label_t     = torch.from_numpy( np.transpose(data["kplabel"],(1,0)) ).to(device).unsqueeze(0)
    match_weight_t = torch.from_numpy( data["voxlmweight"] ).to(device)
    kp_weight_t    = torch.from_numpy( data["kpweight"] ).to(device).unsqueeze(0)
    ssnet_weight_t = torch.from_numpy( data["ssnet_weights"] ).to(device).unsqueeze(0)
    kpshift_t      = None
    paf_label_t    = None
    truematch_idx_t = None
    #print("lm pred: ",match_pred_t.F[:10])
    #print("match label: ",match_label_t.shape)
    #print("ssnet label: ",ssnet_label_t.shape)
    #print("kp label: ",kp_label_t.shape," kp predict: ",kplabel_pred_t.shape)
    #print("voxlmweight: ",match_weight_t.shape)
    #print("kp-weight: ",kp_weight_t.shape)
    #print("ssnet-weight: ",ssnet_weight_t.shape)

    # check the pred for NANs
    with torch.no_grad():
        checklist = [ match_pred_t.F, ssnet_pred_t, kplabel_pred_t ]        
        for iarr, arr in enumerate(checklist):
            if arr is not None and ( torch.isnan(arr).sum()>0 or torch.isinf(arr).sum()>0 ):
                print("RANK-%d PREDICTION ARRAY [%d] is nan or inf!"%(rank,iarr))
                print("RANK-%d is_train=%d root-tree-entry=%d"%(rank,is_train,data["tree_entry"]))
                print("KILL THE JOB")
                try:
                    dist.destroy_process_group()  
                except KeyboardInterrupt: 
                    os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
    

    # check the weights for NANs
    checklist = [ match_weight_t, ssnet_weight_t, kp_weight_t ]
    for iarr, arr in enumerate(checklist):
        if arr is not None and ( torch.isnan(arr).sum()>0 or torch.isinf(arr).sum()>0 ):
            print("RANK-%d WEIGHT array [%d] is nan or inf!"%(rank,iarr))
            print("RANK-%d is_train=%d root-tree-entry=%d"%(rank,is_train,data["tree_entry"]))
            print("KILL THE JOB")
            try:
                dist.destroy_process_group()  
            except KeyboardInterrupt: 
                os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
    
    
    if config["RUNPROFILER"]:
        torch.cuda.synchronize()
    time_meters["forward"].update(time.time()-dt_forward)

    dt_loss = time.time()
        
    # Calculate the loss
    totloss,larmatch_loss,ssnet_loss,kp_loss,paf_loss = criterion( match_pred_t,   ssnet_pred_t,  kplabel_pred_t, kpshift_pred_t, paf_pred_t,
                                                                   match_label_t,  ssnet_label_t, kp_label_t, kpshift_t, paf_label_t,
                                                                   #match_weight_t, ssnet_cls_weight_t*ssnet_top_weight_t, kp_weight_t, paf_weight_t,
                                                                   match_weight_t, ssnet_weight_t, kp_weight_t, None,
                                                                   #None, None, None, None,
                                                                   verbose=verbose )

    # check the total losses for NANs
    checklist = [ totloss ]
    with torch.no_grad():
        for iarr, arr in enumerate(checklist):
            if arr is not None and ( torch.isnan(arr).sum()>0 or torch.isinf(arr).sum()>0 ):
                print("RANK-%d LOSS [%d] is nan or inf!"%(rank,iarr))
                print("RANK-%d is_train=%d root-tree-entry=%d"%(rank,is_train,data["tree_entry"]))
                print("  lm=",larmatch_loss)
                print("  ssnet=",ssnet_loss)
                print("  kp=",kp_loss)
                print("  paf=",paf_loss)
                print("KILL THE JOB")
                try:
                    dist.destroy_process_group()  
                except KeyboardInterrupt: 
                    os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
                    
    if config["RUNPROFILER"]:
        torch.cuda.synchronize()
    time_meters["loss_calc"].update(time.time()-dt_loss)
    
    if is_train:
        # calculate gradients for this batch
        dt_backward = time.time()
        
        totloss.backward()

        # dump par and/or grad for debug
        #for n,p in model_dict["larmatch"].named_parameters():
        #    if "out" in n:
        #        #print(n,": grad: ",p.grad)
        #        print(n,": ",p)

        torch.nn.utils.clip_grad_norm_(model_dict["larmatch"].parameters(), 1.0)
        optimizer.step()

        #for name,p in model_dict["larmatch"].named_parameters():
        #    if "lmclassifier_out" in name:
        #        with torch.no_grad():
        #            print(name,": ",p.shape,torch.pow(p.grad,2).mean())
        
    
        if config["RUNPROFILER"]:
            torch.cuda.synchronize()                
        time_meters["backward"].update(time.time()-dt_backward)

    # update loss meters
    loss_meters["total"].update( totloss.detach().item() )
    loss_meters["lm"].update( larmatch_loss )
    loss_meters["ssnet"].update( ssnet_loss )
    loss_meters["kp"].update( kp_loss )
    loss_meters["paf"].update( paf_loss )
        
        
    # measure accuracy and update accuracy meters
    dt_acc = time.time()
    acc = accuracy(match_pred_t, match_label_t,
                   ssnet_pred_t, ssnet_label_t,
                   kplabel_pred_t, kp_label_t,
                   paf_pred_t, paf_label_t,
                   truematch_idx_t,
                   acc_meters,verbose=verbose)

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
            print("    ",name,": ",meter.avg)
        else:
            print("    ",name,": NULL")            
    print("------------------------------------------------------------------------")
    
    
def save_checkpoint(state, is_best, p, filename='checkpoint.pth.tar'):

    if p>0:
        filename = "checkpoint.%dth.tar"%(p)
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
