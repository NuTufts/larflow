import os,sys
import torch
import yaml
from larmatch import LArMatch

def load_config_file( args, dump_to_stdout=False ):
    stream = open(args.config_file, 'r')
    dictionary = yaml.load(stream, Loader=yaml.FullLoader)
    if dump_to_stdout:
        for key, value in dictionary.items():
            print (key + " : " + str(value))
    stream.close()
    return dictionary

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

def get_larmatch_model( config, device, dump_model=False ):

    # create model, mark it to run on the device
    model = LArMatch(use_unet=True).to(device)

    if dump_model:
        # DUMP MODEL (for debugging)
        print(model)

    return model

if __name__ == "__main__":

    class argstest:
        def __init__(self):
            self.config_file = "config.yaml"

    args = argstest()
    config = load_config_file( args, dump_to_stdout=True ) 

    model = get_larmatch_model( config, config["DEVICE"], dump_model=True )
