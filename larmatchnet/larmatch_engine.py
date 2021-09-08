import os,sys,time
import torch
import yaml
from larmatch import LArMatch

SSNET_CLASS_NAMES=["bg","electron","gamma","muon","pion","proton","other"]
KP_CLASS_NAMES=["kp_nu","kp_trackstart","kp_trackend","kp_shower","kp_michel","kp_delta"]
LM_CLASS_NAMES=["lm_pos","lm_neg","lm_all"]

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
        checkpoint_data = checkpoint["state_"+name]
        for parname,arr in checkpoint_data.items():
            if verbose: print("append: ",head_names[name]+"."+parname,arr.shape)        
            larmatch_checkpoint_data[head_names[name]+"."+parname] = arr
    
    return larmatch_checkpoint_data


def get_larmatch_model( config, dump_model=False ):

    # create model, mark it to run on the device
    model = LArMatch(use_unet=True,
                     run_ssnet=config["RUN_SSNET"],
                     run_kplabel=config["RUN_KPLABEL"],
                     run_kpshift=config["RUN_KPSHIFT"],
                     run_paf=config["RUN_PAF"])

    if dump_model:
        # DUMP MODEL (for debugging)
        print(model)

    model_dict = {"larmatch":model}
    if model.run_ssnet:   model_dict["ssnet"] = model_dict["larmatch"].ssnet_head
    if model.run_kplabel: model_dict["kplabel"] = model_dict["larmatch"].kplabel_head
    if model.run_kpshift: model_dict["kpshift"] = model_dict["larmatch"].kpshift_head
    if model.run_paf:     model_dict["paf"] = model_dict["larmatch"].affinity_head
    
    return model, model_dict

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
    match_pred = match_pred_t.detach()
    npairs = match_pred.shape[0]
    
    pos_correct = (match_pred.gt(0.0)*match_label_t[:npairs].eq(1)).sum().to(torch.device("cpu")).item()
    neg_correct = (match_pred.lt(0.0)*match_label_t[:npairs].eq(0)).sum().to(torch.device("cpu")).item()
    npos = float(match_label_t[:npairs].eq(1).sum().to(torch.device("cpu")).item())
    nneg = float(match_label_t[:npairs].eq(0).sum().to(torch.device("cpu")).item())

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
        acc_meters["ssnet-all"].update( ssnet_tot_correct/float(ssnet_label_t.shape[0]) )

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
            if verbose: print("kp[",c,"-",kpname,"] n_pos[>0.5]: ",kp_n_pos," pred[>0.5]: ",kp_pos)
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
        if verbose: print("paf: npos=",paf_npos," acc=",paf_acc)
        if paf_npos>0:
            acc_meters["paf"].update( paf_acc )
    
    
    return True

def do_one_iteration( config, model_dict, data_loader, criterion, optimizer,
                      acc_meters, loss_meters, time_meters, is_train, device,
                      verbose=False ):
    """
    Perform one iteration, i.e. processes one batch. Handles both train and validation iteraction.
    """
    dt_all = time.time()
    
    if is_train:
        optimizer.zero_grad()

    dt_io = time.time()
    
    flowdata = next(iter(data_loader))[0]

    # input: 2D image in sparse tensor form
    coord_t = [ torch.from_numpy( flowdata['coord_%s'%(p)] ).to(device) for p in [0,1,2] ]
    feat_t  = [ torch.from_numpy( flowdata['feat_%s'%(p)] ).to(device) for p in [0,1,2] ]

    # sample of 3D points coming from possilble wire intersections
    # represented as triplet of indices from above tensor
    npairs          = flowdata['npairs']
    match_t         = torch.from_numpy( flowdata['matchpairs'] ).to(device).requires_grad_(False)
    match_label_t   = torch.from_numpy( flowdata['larmatchlabels'] ).to(device).requires_grad_(False)

    # truth labels
    match_weight_t  = torch.from_numpy( flowdata['match_weight'] ).to(device).requires_grad_(False)
    truematch_idx_t = torch.from_numpy( flowdata['positive_indices'] ).to(device).requires_grad_(False)
        
    ssnet_label_t  = torch.from_numpy( flowdata['ssnet_label'] ).to(device).requires_grad_(False)
    ssnet_cls_weight_t = torch.from_numpy( flowdata['ssnet_class_weight'] ).to(device).requires_grad_(False)
    ssnet_top_weight_t = torch.from_numpy( flowdata['ssnet_top_weight'] ).to(device).requires_grad_(False)
        
    kp_label_t    = torch.from_numpy( flowdata['kplabel'] ).to(device).requires_grad_(False)
    kp_weight_t   = torch.from_numpy( flowdata['kplabel_weight'] ).to(device).requires_grad_(False)        
    kpshift_t     = torch.from_numpy( flowdata['kpshift'] ).to(device).requires_grad_(False)
    
    paf_label_t   = torch.from_numpy( flowdata['paf_label'] ).to(device).requires_grad_(False)
    paf_weight_t  = torch.from_numpy( flowdata['paf_weight'] ).to(device).requires_grad_(False)

    # handled in data loader now
    #for p in range(3):
    #    feat_t[p] = torch.clamp( feat_t[p], 0, ADC_MAX )
    dt_io = time.time()-dt_io
    if verbose:
        print("loaded data. %.2f secs"%(dt_io)," iteration=",entry," root-tree-entry=",flowdata["tree_entry"]," npairs=",npairs)
    time_meters["data"].update(dt_io)

    if config["RUNPROFILER"]:
        torch.cuda.synchronize()

    dt_forward = time.time()

    # use UNET portion to first get feature vectors
    pred_dict = model_dict["larmatch"]( coord_t, feat_t, match_t, flowdata["npairs"], device, verbose=config["TRAIN_VERBOSE"] )
    
    match_pred_t   = pred_dict["match"]      
    ssnet_pred_t   = pred_dict["ssnet"]   if "ssnet" in pred_dict else None
    kplabel_pred_t = pred_dict["kplabel"] if "kplabel" in pred_dict else None
    kpshift_pred_t = pred_dict["kpshift"] if "kpshift" in pred_dict else None
    paf_pred_t     = pred_dict["paf"]     if "paf" in pred_dict else None

                                           
    if config["RUNPROFILER"]:
        torch.cuda.synchronize()
    time_meters["forward"].update(time.time()-dt_forward)

    dt_loss = time.time()
        
    # Calculate the loss
    totloss,larmatch_loss,ssnet_loss,kp_loss,kpshift_loss, paf_loss = criterion( match_pred_t,   ssnet_pred_t,  kplabel_pred_t, kpshift_pred_t, paf_pred_t,
                                                                                 match_label_t,  ssnet_label_t, kp_label_t, kpshift_t, paf_label_t,
                                                                                 truematch_idx_t,
                                                                                 match_weight_t, ssnet_cls_weight_t*ssnet_top_weight_t, kp_weight_t, paf_weight_t,
                                                                                 verbose=verbose )
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
        print("    ",name,": ",meter.avg)
    print("  Accuracies: ")
    for name,meter in acc_meters.items():
        print("    ",name,": ",meter.avg)
    print("------------------------------------------------------------------------")
    
    

if __name__ == "__main__":

    class argstest:
        def __init__(self):
            self.config_file = "config.yaml"

    args = argstest()
    config = load_config_file( args, dump_to_stdout=True ) 

    model = get_larmatch_model( config, config["DEVICE"], dump_model=True )
