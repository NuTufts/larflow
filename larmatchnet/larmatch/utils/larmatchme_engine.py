import os,sys,time
import shutil
import numpy as np
import torch
import torch.distributed as dist
import MinkowskiEngine as ME
import yaml
import larmatch
from larmatch.model.larmatchminkowski import LArMatchMinkowski
from larmatch.loss.loss_larmatch_kps import SparseLArMatchKPSLoss
from collections import OrderedDict
from larmatch.utils.common import prepare_me_sparsetensor
from larmatch.utils.larmatch_hard_examples import larmatch_hard_example_mining

SSNET_CLASS_NAMES=["bg","electron","gamma","muon","pion","proton","other"]
KP_CLASS_NAMES=["kp_nu_pos",
                "kp_trackstart",
                "kp_trackend",
                "kp_shower",
                "kp_michel",
                "kp_delta"]
LM_CLASS_NAMES=["lm_pos","lm_neg","lm_all"]

def load_config_file( args, dump_to_stdout=False ):
    """
    opens file in YAML format and returns parameters
    """
    stream = open(args.config_file, 'r')
    dictionary = yaml.load(stream, Loader=yaml.FullLoader)
    if dump_to_stdout:
        for key, value in dictionary.items():
            print (key + " : " + str(value))
    stream.close()
    return dictionary



def get_model( config, dump_model=False ):

    # create model, mark it to run on the device
    model = LArMatchMinkowski(run_lm=config["RUN_LARMATCH"],
                              run_ssnet=config["RUN_SSNET"],
                              run_kp=config["RUN_KPLABEL"],
                              stem_nfeatures=config["NUM_STEM_FEATURES"],
                              norm_layer=config["NORM_LAYER"])

    if dump_model:
        # DUMP MODEL (for debugging)
        print(model)

    return model

def make_loss_fn( config ):
    device = torch.device(config["DEVICE"])
    kp_loss_weight = 0.0
    lm_loss_weight = 0.0
    if "INIT_LM_LOSS_WEIGHT" in config:
        lm_loss_weight = config["INIT_LM_LOSS_WEIGHT"]
    if "INIT_KP_LOSS_WEIGHT" in config:
        kp_loss_weight = config["INIT_KP_LOSS_WEIGHT"]
        
    criterion = SparseLArMatchKPSLoss( learnable_weights=config["USE_LEARNABLE_LOSS_WEIGHTS"],
                                       eval_lm=config["RUN_LARMATCH"],
                                       eval_ssnet=config["RUN_SSNET"],
                                       eval_keypoint_label=config["RUN_KPLABEL"],
                                       eval_keypoint_shift=config["RUN_KPSHIFT"],
                                       eval_affinity_field=config["RUN_PAF"],
                                       lm_loss_type='focal-soft-bse',
                                       init_lm_weight=lm_loss_weight,
                                       init_kp_weight=kp_loss_weight).to(device)
        
    print("made loss =============================")
    print("loss parameters: ")
    for k,x in criterion.named_parameters():
        print(k,": ",x)
    print("=======================================")
    return criterion
    

def load_model_weights( model, checkpoint_file ):

    # Map all weights back to cpu first
    loc_dict = {"cuda:%d"%(gpu):"cpu" for gpu in range(10) }

    # load weights
    checkpoint = torch.load( checkpoint_file, map_location=loc_dict )

    # change names if we saved the distributed data parallel model state
    rename_distributed_checkpoint_par_names(checkpoint)

    missing, extra = model.load_state_dict( checkpoint["state_larmatch"], strict=True )
    print("MISSING KEYS WHEN LOADING ============")
    for k in missing:
        print("  ",k)
    print("EXTRA KEYS WHEN LOADING ============")        
    for k in extra:        
        print("  ",k)

    return checkpoint

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
                print("renaming parameter by removing 'module': ",name)
                notified = False
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
        state_name = "state_"+name
        if state_name not in checkpoint:
            print(state_name," not in checkpoint")
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
        
def accuracy(predictions, truthdata, 
             acc_meters,
             verbose=False):
    """Computes the accuracy metrics."""

    batchsize = len(truthdata)

    for ibatch in range(batchsize):

        data = {}
        labels = truthdata
        for k in predictions.keys():
            data[k] = predictions[k][ibatch]
            if verbose: print("accuracy calc [",ibatch,"] key=",k,": ",data[k].shape)

        # get the data from the dictionaries
        if "lm" in data:
            match_pred  = torch.softmax( data["lm"].detach(), dim=0 )[1].squeeze()
            match_label_t = labels["lm"][ibatch].squeeze()
            
            # LARMATCH METRICS
            if verbose:
                print("match_pred: ",match_pred.shape)
                print("match_label: ",match_label_t.shape)
    
            pos_correct = (match_pred.gt(0.5)*match_label_t.gt(0.5)).sum().to(torch.device("cpu")).item()
            neg_correct = (match_pred.lt(0.5)*match_label_t.lt(0.5)).sum().to(torch.device("cpu")).item()
            npos = float(match_label_t.gt(0.5).sum().to(torch.device("cpu")).item())
            nneg = float(match_label_t.lt(0.5).sum().to(torch.device("cpu")).item())
            if verbose:
                print("npos: ",npos)
                print("nneg: ",nneg)
                print("pos correct: ",pos_correct)
                print("neg correct: ",neg_correct)

            if npos>0:
                acc_meters["lm_pos"].update( float(pos_correct)/npos )
            if nneg>0:
                acc_meters["lm_neg"].update( float(neg_correct)/nneg )
            if npos+nneg>0:
                acc_meters["lm_all"].update( float(pos_correct+neg_correct)/(npos+nneg) )

        # SSNET METRICS
        if "ssnet" in data:
            ssnet_pred_t  = data["ssnet"].detach().squeeze()
            ssnet_label_t = labels["ssnet"][ibatch]
            ssnet_class   = torch.argmax( ssnet_pred_t, 0 )
            ssnet_correct = ssnet_class.eq( ssnet_label_t )
            ssnet_tot_correct = 0
            ssnet_tot_labels  = 0
            for iclass,classname in enumerate(SSNET_CLASS_NAMES):
                if ssnet_label_t.eq(iclass).sum().item()>0:                
                    ssnet_class_correct = ssnet_correct[ ssnet_label_t==iclass ].sum().item()
                    ssnet_class_nlabels = ssnet_label_t.eq(iclass).sum().item()
                    acc_meters[classname].update( float(ssnet_class_correct)/float(ssnet_class_nlabels) )
                    if iclass>0:
                        # skip BG class
                        ssnet_tot_correct += ssnet_class_correct
                        ssnet_tot_labels  += ssnet_class_nlabels
            if ssnet_tot_labels>0:
                acc_meters["ssnet-all"].update( float(ssnet_tot_correct)/float(ssnet_tot_labels) )

        # KP METRIC
        if "kp" in data:
            kp_pred  = data["kp"].detach()
            kp_label = labels["kp"][ibatch].detach().squeeze()
            #print("kp_pred=",kp_pred.shape)
            #print("kp_label=",kp_label.shape)
            for c,kpname in enumerate(KP_CLASS_NAMES):
                kp_n_pos = float(kp_label[c,:].gt(0.05).sum().item())
                kp_pos   = float(kp_pred[c,:].gt(0.05)[ kp_label[c,:].gt(0.05) ].sum().item())
                if verbose: print("kp[",c,"-",kpname,"] n_pos[>0.5]: ",kp_n_pos," pred[>0.5]: ",kp_pos)
                if kp_n_pos>0:
                    acc_meters[kpname].update( kp_pos/kp_n_pos )

    # # PARTICLE AFFINITY FLOW
    # if paf_pred_t is not None:
    #     # we define accuracy with the direction is less than 20 degress
    #     if paf_pred_t.shape[0]!=paf_label_t.shape[0]:
    #         paf_pred  = torch.index_select( paf_pred_t.detach(),  0, truematch_indices_t )
    #         paf_label = torch.index_select( paf_label_t.detach(), 0, truematch_indices_t )
    #     else:
    #         paf_pred  = paf_pred_t.detach()
    #         paf_label = paf_label_t.detach()[:npairs]
    #     # calculate cosine
    #     paf_truth_lensum = torch.sum( paf_label*paf_label, 1 )
    #     paf_pred_lensum  = torch.sum( paf_pred*paf_pred, 1 )
    #     paf_pred_lensum  = torch.sqrt( paf_pred_lensum )
    #     paf_posexamples = paf_truth_lensum.gt(0.5)
    #     #print paf_pred[paf_posexamples,:].shape," ",paf_label[paf_posexamples,:].shape," ",paf_pred_lensum[paf_posexamples].shape
    #     paf_cos = torch.sum(paf_pred[paf_posexamples,:]*paf_label[paf_posexamples,:],1)/(paf_pred_lensum[paf_posexamples]+0.001)
    #     paf_npos  = paf_cos.shape[0]
    #     paf_ncorr = paf_cos.gt(0.94).sum().item()
    #     paf_acc = float(paf_ncorr)/float(paf_npos)
    #     if verbose: print("paf: npos=",paf_npos," acc=",paf_acc)
    #     if paf_npos>0:
    #         acc_meters["paf"].update( paf_acc )
    
    
    return True

def do_one_iteration( config, model, data_loader, criterion, optimizer,
                      acc_meters, loss_meters, time_meters, is_train, device,
                      verbose=False,no_step=False ):
    """
    Perform one iteration, i.e. processes one batch. Handles both train and validation iteraction.
    """
    DEVICE = torch.device(config["DEVICE"])
    
    dt_all = time.time()
    
    if is_train:
        model.train()
    else:
        model.eval()

    if verbose: print("larmatchme_engine.do_one_iteration: start loading data")
    dt_io = time.time()

    npts = None
    ntries = 0
    while (npts is None or npts>config["BATCH_TRIPLET_LIMIT"]) and ntries<20:
        batchdata = next(iter(data_loader))
        npts = 0
        for data in batchdata["matchtriplet_v"]:
            npts += data.shape[0]*data.shape[1]
        #print("Drawn total spacepoints [tries=%d]: "%(ntries),npts)
        sys.stdout.flush()
        ntries+=1
    batch_size = len(batchdata["matchtriplet_v"])

    if verbose: print("larmatchme_engine.do_one_iteration: prep batch and transfer to gpu")
    if config["USE_HARD_EXAMPLE_TRAINING"] and is_train:
        # HARD EXAMLPE MINING
        wireplane_sparsetensors, matchtriplet_v, batch_truth, batch_weight \
            = larmatch_hard_example_mining( model, batchdata, DEVICE, int(config["NUM_TRIPLET_SAMPLES"]), verbose=verbose )
        model.train()
    else:
        wireplane_sparsetensors, matchtriplet_v, batch_truth, batch_weight \
            = prepare_me_sparsetensor( batchdata, DEVICE, verbose=verbose )
    
    dt_io = time.time()-dt_io
    if verbose:
        print("larmatchme_engine.do_one_iteration: loaded data. %.2f secs"%(dt_io)," root-tree-entry=",batchdata["tree_entry"])
    time_meters["data"].update(dt_io)

    if verbose: print("larmatchme_engine.do_one_iteration: sync")    
    if config["RUN_PROFILER"]:
        torch.cuda.synchronize()

    if verbose: print("larmatchme_engine.do_one_iteration: zero gradients")            
    dt_forward = time.time()
    optimizer.zero_grad(set_to_none=True)    

    # use UNET portion to first get feature vectors
    if verbose: print("larmatchme_engine.do_one_iteration: run forward")    
    pred_dict = model( wireplane_sparsetensors, matchtriplet_v, batch_size )
    #if not args.no_parallel:
    #    torch.distributed.barrier()

    if config["RUN_PROFILER"]:
        torch.cuda.synchronize()
    time_meters["forward"].update(time.time()-dt_forward)

    dt_loss = time.time()
        
    # Calculate the loss
    if verbose: print("larmatchme_engine.do_one_iteration: calc loss")        
    loss_dict = criterion( pred_dict, batch_truth, batch_weight,
                           batch_size, DEVICE,
                           verbose=config["VERBOSE_LOSS"] )

    if config["RUN_PROFILER"]:
        torch.cuda.synchronize()
    time_meters["loss_calc"].update(time.time()-dt_loss)
    
    if is_train:
        # calculate gradients for this batch
        dt_backward = time.time()
        if verbose: print("larmatchme_engine.do_one_iteration: run backward")                
        loss_dict["tot"].backward()

        # dump par and/or grad for debug
        #for n,p in model_dict["larmatch"].named_parameters():
        #    if "out" in n:
        #        #print(n,": grad: ",p.grad)
        #        print(n,": ",p)

        if config["CLIP_GRAD_NORM"]:
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        if not no_step:
            optimizer.step()
            
        if config["RUN_PROFILER"]:
            torch.cuda.synchronize()                
        time_meters["backward"].update(time.time()-dt_backward)

    # update loss meters
    loss_meters["total"].update( loss_dict["tot"].detach().item() )
    loss_meters["lm"].update( loss_dict["lm"] )
    if "ssnet" in loss_dict: loss_meters["ssnet"].update( loss_dict["ssnet"] )
    if "kp" in loss_dict:    loss_meters["kp"].update( loss_dict["kp"] )
    if "paf" in loss_dict:   loss_meters["paf"].update( loss_dict["paf"] )
        
        
    # measure accuracy and update accuracy meters
    dt_acc = time.time()
    if verbose: print("larmatchme_engine.do_one_iteration: calc accuracies")                    
    acc = accuracy( pred_dict, batch_truth, acc_meters, verbose=config["VERBOSE_ACCURACY"] )

    # update time meter
    time_meters["accuracy"].update(time.time()-dt_acc)

    # measure elapsed time for batch
    time_meters["batch"].update(time.time()-dt_all)

    # done with iteration
    if verbose: print("larmatchme_engine.do_one_iteration: iteration complete")                        
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
        print("    ",name,": ",meter.avg," (from ",meter.count," counts)")
    print("  Accuracies: ")
    for name,meter in acc_meters.items():
        print("    ",name,": ",meter.avg," (from ",meter.count," counts)")
    print("------------------------------------------------------------------------")
    
    
def save_checkpoint(state, is_best, p, tag=None):

    stem = "checkpoint"
    if tag is not None:
        stem += ".%s"%(tag)

    filename = "%s.%dth.tar"%(stem,p)
    torch.save(state, filename)
    if is_best:
        bestname = "model_best"
        if tag is not None:
            bestname += ".%s"%(tag)
        bestname += ".tar"
        shutil.copyfile(filename, bestname )



if __name__ == "__main__":

    class argstest:
        def __init__(self):
            self.config_file = sys.argv[1]

    args = argstest()
    config = load_config_file( args, dump_to_stdout=True )
    config["DEVICE"] = "cpu"
    lmmodel = get_model( config, dump_model=True )


