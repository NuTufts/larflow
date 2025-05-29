from __future__ import print_function
import os,sys,argparse,time
sys.path.append(os.environ["LARFLOW_BASEDIR"]+"/larmatchnet")

parser = argparse.ArgumentParser(description='deploy larmatch model on microboone larcv/larlite input')
parser.add_argument('-c','--config-file',type=str,default="config.yaml",help="larmatch configuration file")
parser.add_argument('-w','--weights',required=True,type=str,help='weight file')
parser.add_argument('-p','--min-score',type=float,default=0.3,help="Minimum Score to save point [default: 0.3]")
parser.add_argument('-d','--device-name',default="cpu",type=str,help="Name of device. [default: cpu; e.g. cuda:0]")
parser.add_argument('-adc','--adc-name',default="wire",type=str,help="Name of ADC tree [default: wire]")
parser.add_argument('-v','--verbose',default=False,action='store_true',help='If flag given, just run 5 events for debugging')
parser.add_argument('-ilcv','--input-larcv', required=True,help="input larcv file")
parser.add_argument('-ill', '--input-larlite', required=True,help="input larlite file")
parser.add_argument('-ao', '--allow-output-overwrite', default=False, action='store_true', help="If flag given, allow output file to overwrite")
parser.add_argument('-tf','--tickforwards',action='store_true',default=False,help="Indicate that input larcv file is tick-forward [default: F]")
parser.add_argument('-o','--output',required=True,type=str,help="Filename stem for output files")
parser.add_argument('--save-larcv',default=False,action='store_true',help='If flag given, copy larcv -- useful for when running a subset of events and allowing synced larcv file for downstream input')
parser.add_argument('-e','--entry',type=int,default=0,help="Starting entry [default: 0]")
parser.add_argument('-n','--nentries',type=int,default=-1,help="(optional) sets number of entries to run. [default: -1, which runs all entries]")
parser.add_argument("--use-skip-limit",default=False,action='store_true',help="Specify a max triplet let. If surpassed, skip network eval.")

args = parser.parse_args()

# prepare network
import torch
import numpy as np
import larmatch.utils.larmatchme_engine as engine

DEVICE=torch.device(args.device_name)
config = engine.load_config_file(  args )
single_model    = engine.get_model( config )
# set to eval-mode for inference
single_model.eval()

# Loading weights
checkpointfile  = engine.get_weightfile( args.weights, config )
checkpoint_data = engine.load_model_weights( single_model, checkpointfile )

single_model.eval()
single_model.to(DEVICE)
print("loaded MODEL on ",DEVICE)
if args.verbose and False:
    print("MODEL")
    print("-----------------------------")
    print(single_model)
    print()
    print("=============================")
    print("Parameters")
    print("------------")
    print("------------")
    for name, par in single_model.named_parameters():
        print("---------------------------------")
        print(name," ",par.shape)
        print(par)



# Setup input and output files

input_larcv   = args.input_larcv
input_larlite = args.input_larlite
outdir = os.path.dirname( args.output )
if outdir=="":
    outdir="./"

# check file paths, input and outpiut
if not os.path.exists( input_larcv ):
    print("LARCV input file does not exist. path given: ",input_larcv)
    sys.exit(1)
if not os.path.exists( input_larlite ):
    print("larlite input file does not exist. path given: ", input_larlite)
    sys.exit(1)
if not args.allow_output_overwrite and os.path.exists( args.output ):
    print("output file exists. not allowing overwrites. path given: ",args.output)
    print("provide True to keyword argument 'allow_output_overwrite'")
    sys.exit(1)
if not os.path.exists( outdir  ):
    print("directory for output file does not exist. Given: ",outdir)
    sys.exit(1)

# setup larcv and larlite interfaces to data
# ROOT-based IO for data objects used in LArSoft (common framework used in LArTPC experiments )
import ROOT
from ROOT import std
from larcv import larcv
from larlite import larlite

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename( input_larlite )
ioll.open()

iolcv_mode = larcv.IOManager.kREAD
if args.save_larcv:
    iolcv_mode = larcv.IOManager.kBOTH
    iolcv_out = args.output.replace(".root","_larcvout.root")

if args.tickforwards:    
    iolcv = larcv.IOManager( iolcv_mode, "larcv", larcv.IOManager.kTickForward )
else:
    iolcv = larcv.IOManager( iolcv_mode, "larcv", larcv.IOManager.kTickBackward )

iolcv.add_in_file( input_larcv )
if args.save_larcv:
    iolcv.set_out_file( iolcv_out )
iolcv.reverse_all_products()
iolcv.initialize()

nentries_larcv = iolcv.get_n_entries()
start_entry = max(args.entry,0)
num_entries = nentries_larcv
print("Number of entries in file: ",nentries_larcv)
if start_entry>=nentries_larcv:
    print("Asking to start after last entry (%d) in file"%(nentries_larcv-1))
    sys.exit(1)

end_entry = start_entry + nentries_larcv
if num_entries>0:
    end_entry = start_entry + num_entries
if end_entry >= nentries_larcv:
    end_entry = nentries_larcv

if args.nentries>0:
    user_end_entry = start_entry + args.nentries
    if user_end_entry < end_entry:
        end_entry = user_end_entry

print("running entries [",start_entry,",",end_entry,"]")

outll = larlite.storage_manager( larlite.storage_manager.kWRITE )
outll.set_out_filename( args.output )
outll.open()

# event loop

# we use the LArMatchHDFWriter class to help us convert larcv/larlite data into numpy arrays
from larmatch.data.larmatch_hdf5_writer import LArMatchHDF5Writer
lmwriter = LArMatchHDF5Writer( use_triplet_skip_limit=args.use_skip_limit )
num_max_spacepoints = 10000000
process_truth_labels = False

# we use functions from the reader class to prepare the input for the network
from larmatch.data.larmatch_hdf5_reader import LArMatchHDF5Dataset

# setup the hit maker
from larflow import larflow
hitmaker = larflow.prep.FlowMatchHitMaker()
hitmaker.set_score_threshold( args.min_score )
triplet_key = 'matchtriplet_v'

for ientry in range(start_entry,end_entry):
    print("[[ RUN ENTRY %d ]]"%(ientry))

    tprep = time.time()

    ioll.go_to(ientry)
    iolcv.read_entry(ientry)

    ev_adc = iolcv.get_data( larcv.kProductImage2D, args.adc_name )
    ev_chstatus = iolcv.get_data( larcv.kProductChStatus, "wire" )
    adc_v = ev_adc.as_vector()

    evout_lfhits = outll.get_data(larlite.data.kLArFlow3DHit,"larmatch")
    evout_lfhits.clear()
    evout_lmsp = outll.get_data(larlite.data.kLArMatchSP,"splarmatch")
    evout_lmsp.clear()

    hitmaker.clear()

    # convert the data and store into self.entry_data
    lmwriter.larlite_larcv_to_hdf5_entry( ioll, iolcv, process_truth_labels, num_max_spacepoints )

    entrydata = lmwriter.entry_data.pop()
    inputdata = LArMatchHDF5Dataset.prepare_triplet_and_image_arrays_for_network( entrydata, triplet_key=triplet_key )
    # transfer keys from inputdata to entrydata
    for k,i in inputdata.items():
        entrydata[k] = i
        
    batch = [entrydata]
    
    batchsize = len(batch)
    batch_sparsetensors, batch_triplets, batch_coordqueries = LArMatchHDF5Dataset.make_batch_sparse_tensors( batch, DEVICE, triplet_key=triplet_key )
    dt_prep = time.time()-tprep

    ntriplets = 0
    for matchtriplet_b in batch_triplets:
        ntriplets += matchtriplet_b.shape[0]
    print("Number of spacepoints (i.e. triplets) to evaluate: ",ntriplets)

    with torch.no_grad():
        # run larmatch network
        # input: forward( self, input_wireplane_sparsetensors, matchtriplets, query_v, batch_size ):
        tstart_runnet = time.time()
        if ntriplets>0:
            larmatchout = single_model( batch_sparsetensors, batch_triplets, batch_coordqueries, batchsize )
        else:
            print("No spacepoitns to evaluate. make empty output dictionary")
            larmatchout = {}
        dt_runnet = time.time()-tstart_runnet


        # output is a dict with keys being the different output heads
        if True and ntriplets>0:
            print("-----------------------------------")
            #for ib,pred_dict in enumerate(larmatchout):
            pred_dict = larmatchout
            print("output: ")
            for k,v in pred_dict.items():
                print(k,": ",v.shape)
            print("-----------------------------------")

            if "cuda" in args.device_name:
                torch.cuda.synchronize()
            sys.stdout.flush()    
            
            # EVALUATE LARMATCH SCORES
            tstart = time.time()
            with torch.no_grad():
                lm_prob_t = torch.transpose(  pred_dict["lm"].squeeze(), 1, 0 )
                lm_prob_t = 1.0-torch.softmax( lm_prob_t, dim=1 )
                print("  lm_prob_t=",lm_prob_t.shape)
                #print(lm_prob_t[:10,:])

            # EVALUATE SSNET SCORES
            if config["RUN_SSNET"]:
                with torch.no_grad():
                    #print("  pred_dict[ssnet] shape: ",pred_dict["ssnet"].shape)        
                    ssnet_pred_t = torch.transpose( pred_dict["ssnet"].squeeze(), 1, 0 )
                    ssnet_pred_t = torch.softmax( ssnet_pred_t, dim=1 )
                    print("  ssnet_pred_t: ",ssnet_pred_t.shape)

            # EVALUATE KP-LABEL SCORES
            if config["RUN_KPLABEL"]:
                with torch.no_grad():
                    #print("  pred_dict[kplabel]: ",pred_dict["kp"].shape)
                    kplabel_pred_t = torch.transpose( pred_dict["kp"].squeeze(), 1, 0 )
                    print("  kplabel_pred_t: ",kplabel_pred_t.shape)

            print("prepare score arrays: ",time.time()-tstart," sec")
            
            # EVALUATE PAF SCORES
            if config["RUN_PAF"]:
                with torch.no_grad():
                    paf_pred_t = pred_dict['paf']
                    paf_pred_t = paf_pred_t.reshape( (paf_pred_t.shape[1],paf_pred_t.shape[2]) )
                    paf_pred_t = torch.transpose( paf_pred_t, 1, 0 )        
                    print("  paf-pred: ",paf_pred_t.shape)


            # PASS LARMATCH OUTPUTS to hitmaker
            matchtriplet_np = batch[0][triplet_key]
            #sparse_np_v = [ batch[ib]['wireimage_plane%d'%(p)] for p in range(3) ] 
            sparse_np_v = [ batch[0]['coord_%d'%(p)] for p in range(3) ] 
            prob_np = lm_prob_t.to(torch.device("cpu")).detach().numpy()
            #prob_np[:] = 1.0 # hack to check
            print("add larmatch output to hitmaker")
            sys.stdout.flush()   
            pos_v = std.vector("std::vector<float>")()
            hitmaker.add_triplet_match_data( prob_np,
                                            matchtriplet_np,
                                            sparse_np_v[0],
                                            sparse_np_v[1],
                                            sparse_np_v[2],
                                            pos_v,
                                            adc_v )

            if config["RUN_SSNET"]:
                print("  add ssnet data to hitmaker(...). probshape=",ssnet_pred_t.shape)
                sys.stdout.flush()   
                ssnet_np = ssnet_pred_t.to(torch.device("cpu")).detach().numpy()
                hitmaker.add_triplet_ssnet_scores(  matchtriplet_np, 
                                                    sparse_np_v[0],
                                                    sparse_np_v[1],
                                                    sparse_np_v[2],
                                                    adc_v.front().meta(),
                                                    ssnet_np )                                      

            if config["RUN_KPLABEL"]:
                print("  add kplabel to hitmaker(...). probshape=",kplabel_pred_t.shape)
                sys.stdout.flush()   
                kplabel_np = kplabel_pred_t.to(torch.device("cpu")).detach().numpy()
                hitmaker.add_triplet_keypoint_scores(  matchtriplet_np,
                                                    sparse_np_v[0],
                                                    sparse_np_v[1],
                                                    sparse_np_v[2],
                                                    adc_v.front().meta(),
                                                    kplabel_np )

            if config["RUN_PAF"]:
                print("  add affinity field prediction to hitmaker(...). probshape=",paf_pred_t.shape)
                sys.stdout.flush()   
                paf_np = paf_pred_t.to(torch.device("cpu")).detach().numpy()
                hitmaker.add_triplet_affinity_field(  matchtriplet_np, 
                                                    sparse_np_v[0],
                                                    sparse_np_v[1],
                                                    sparse_np_v[2],
                                                    adc_v.front().meta(),
                                                    paf_np )

            # make flow hits
            hitmaker.make_hits( ev_chstatus, adc_v, evout_lfhits )
            hitmaker.make_hits( ev_chstatus, adc_v, evout_lmsp )
            dt_make_hits = time.time()-tstart
            print("time to run net: ",dt_runnet," secs")
            print("time to make hits: ",dt_make_hits," secs")

        # End of flow direction loop
        print("number of hits made: ",evout_lfhits.size())

        # clear this out for now
        evout_lmsp.clear()
        
        outll.set_id( ioll.run_id(), ioll.subrun_id(), ioll.event_id() )
        outll.next_event(True)
        if args.save_larcv:
            iolcv.save_entry()
        sys.stdout.flush()
    print("End of entry[",ientry,"]")
    if False and ientry>=2:
        break

print("Finished")
print("Cleaning up")
outll.close()
ioll.close()
iolcv.finalize()
