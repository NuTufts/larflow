#!/usr/bin/env python
from __future__ import print_function
import os,sys,argparse,time


parser = argparse.ArgumentParser("run LArMatch (SparseConvNet) on data")
parser.add_argument("--config-file","-c",type=str,default="config.yaml",help="larmatch configuration file")
parser.add_argument("--supera","-su",required=True,type=str,help="LArCV file with ADC images")
parser.add_argument("--weights","-w",required=True,type=str,help="Weight files")
parser.add_argument("--output", "-o",required=True,type=str,help="Stem name of output files: [stem]_larlite.root, [stem]_larcv.root")
parser.add_argument("--tickbackwards","-tb",action='store_true',default=False,help="Indicate that input larcv file is tick-backward [default: F]")
parser.add_argument("--min-score","-p",type=float,default=0.5,help="Minimum Score to save point [default: 0.5]")
parser.add_argument("--num-events","-n",type=int,default=-1,help="Number of events [default: -1 -> All]")
parser.add_argument("--has-mc","-mc",action="store_true",default=False,help="If argument given, input file assumed to have mc truth [default: F]")
parser.add_argument("--larlite-mcinfo","-llmc",type=str,default=None,help="larlite file containing MC truth [default: None]")
parser.add_argument("--has-wirecell","-wc",action="store_true",default=False,help="If flag given, will use WC tagger image to mask cosmics [default: F]")
parser.add_argument("--adc-name","-adc",default="wire",type=str,help="Name of ADC tree [default: wire]")
parser.add_argument("--chstatus-name","-ch",default="wire",type=str,help="Name of the Channel Status tree [default: wire]")
parser.add_argument("--device-name","-d",default="cpu",type=str,help="Name of device. [default: cpu; e.g. cuda:0]")
parser.add_argument("--use-unet","-unet",default=False,action='store_true',help="Use UNet version")
parser.add_argument("--use-skip-limit",default=None,type=int,help="Specify a max triplet let. If surpassed, skip network eval.")
args = parser.parse_args( sys.argv[1:] )

from ctypes import c_int,c_double
import numpy as np
import torch

import utils.larmatchscn_engine as engine
import sparseconvnet as scn

import ROOT as rt
from ROOT import std
from larlite import larlite,larutil
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow

import larmatch.utils.larmatchscn_engine as engine

print(larutil.Geometry.GetME())
driftv = larutil.LArProperties.GetME().DriftVelocity()

DEVICE=torch.device(args.device_name)

config = engine.load_config_file( args.config_file )
single_model = engine.get_gen2_larmatch_model(dump_model=config["PRINT_MODEL"])
checkpointfile  = args.weights
engine.load_gen2_scn_model_weights( args.weights, single_model )

single_model.to(DEVICE)
single_model.eval()
print("loaded MODEL")
#print(single_model)

ADC_PRODUCER=args.adc_name
CHSTATUS_PRODUCER=args.chstatus_name
USE_GAPCH=True
RETURN_TRUTH=False
BATCHSIZE = 1

# DEFINE THE CLASSES THAT MAKE FLOW MATCH VECTORS
# we use a config file
preplarmatch = larflow.prep.PrepMatchTriplets()
preplarmatch.setDoDeadChannelBug( True )
if args.use_skip_limit is not None:
    print("Set Triplet Max where we will skip event: ",args.use_skip_limit)
    preplarmatch.setStopAtTripletMax( True, args.use_skip_limit )

#model_dict["larmatch"].eval()

# setup filename
outfilestem = args.output
if len(args.output)>=5 and args.output[-5:]==".root":
    outfilestem = args.output[:-5]
    
tickdir = larcv.IOManager.kTickForward
if args.tickbackwards:
    tickdir = larcv.IOManager.kTickBackward
io = larcv.IOManager( larcv.IOManager.kBOTH, "larcvio", tickdir )
io.add_in_file( args.supera )
io.set_out_file( "%s_larcv.root"%(outfilestem) )
io.set_verbosity(1)
io.specify_data_read( larcv.kProductImage2D,  "larflow" )
io.specify_data_read( larcv.kProductImage2D,  args.adc_name )
io.specify_data_read( larcv.kProductChStatus, args.chstatus_name )
io.specify_data_read( larcv.kProductSparseImage, "sparseuresnetout" )
if args.has_wirecell:
    io.specify_data_read( larcv.kProductChStatus, "thrumu" )
io.reverse_all_products()
io.initialize()

if args.has_mc and args.larlite_mcinfo is not None:
    ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
    ioll.add_in_filename( args.larlite_mcinfo )
    ioll.open()
else:
    ioll = None

out = larlite.storage_manager( larlite.storage_manager.kWRITE )
out.set_out_filename( "%s_larlite.root"%(outfilestem) )
out.open()

sigmoid = torch.nn.Sigmoid()
ssnet_softmax = torch.nn.Softmax(dim=1)

NENTRIES = io.get_n_entries()

if args.num_events>0 and args.num_events<NENTRIES:
    NENTRIES = args.num_events

dt_prep  = 0.
dt_chunk = 0.
dt_net   = 0.
dt_save  = 0.

# setup the hit maker
hitmaker = larflow.prep.FlowMatchHitMaker()
hitmaker.set_score_threshold( args.min_score )

# setup badch maker
badchmaker = ublarcvapp.EmptyChannelAlgo()

# flush standard out buffer before beginning
sys.stdout.flush()

print("Start event loop")

for ientry in range(NENTRIES):

    evout_lfhits = out.get_data(larlite.data.kLArFlow3DHit,"larmatch")
    evout_lfhits.clear()

    io.read_entry(ientry)
    if ioll is not None:
        ioll.go_to(ientry)
    
    print("==========================================")
    print("Entry {}".format(ientry))

    # get the adc larcv images
    ev_adc = io.get_data( larcv.kProductImage2D, args.adc_name )
    ev_chstatus = io.get_data( larcv.kProductChStatus, args.chstatus_name )
    adc_v = ev_adc.as_vector()
    
    # clear the hit maker
    hitmaker.clear();

    # make triplet proposals
    preplarmatch.process( io, args.adc_name, args.chstatus_name, 10.0, True )

    # make truth labels if possible
    if args.has_mc and ioll is not None:
        print("processing larflow truth...")
        preplarmatch.process_truth_labels( io, ioll )

    # turn shuffle off (to do, function should be kploader function)
    tripdata = preplarmatch.setShuffleWhenSampling( False )

    # get input data: 2d images
    sparse_np_v = []
    coord_v = []
    feat_v  = []    
    for p in range(3):
        wireimg = preplarmatch.make_sparse_image( p )
        wireimg_coord_np = wireimg[:,:2].astype(np.long)
        wireimg_coord = torch.from_numpy( wireimg_coord_np ).to(DEVICE)
        wireimg_feat  = torch.from_numpy( np.expand_dims( wireimg[:,2], 1 ) ).to(DEVICE)
        print("plane[%d]"%(p)," coord: ",wireimg_coord.shape,"  feat: ",wireimg_feat.shape)        
        coord_v.append( wireimg_coord )
        feat_v.append( wireimg_feat )
        sparse_np_v.append( wireimg_coord_np )
    
    # get 3d spacepoints (to do, function should be kploader function)
    matchtriplet_np = preplarmatch.get_all_triplet_data( False )
    print("matchtriplet_np: ",matchtriplet_np.shape)
    spacepoints = preplarmatch.make_spacepoint_charge_array()    
    nfilled = c_int(0)
    ntriplets = matchtriplet_np.shape[0]
    matchtriplet_t = torch.from_numpy(matchtriplet_np).to(DEVICE)

    print("Number of triplets: ",ntriplets)
    with torch.no_grad():
        pred_dict = single_model( coord_v, feat_v, matchtriplet_t, ntriplets, DEVICE, verbose=True )
    print("Ran model: ",pred_dict.keys())
    
    # EVALUATE LARMATCH SCORES
    tstart = time.time()
    with torch.no_grad():
        lm_prob_t = torch.sigmoid( pred_dict["lm"].squeeze() )
        print("  lm_prob_t=",lm_prob_t.shape)
        #print(lm_prob_t[:10,:])

    # EVALUATE SSNET SCORES
    if config["RUN_SSNET"]:
        with torch.no_grad():
            #print("  pred_dict[ssnet] shape: ",pred_dict["ssnet"].shape)        
            #ssnet_pred_t = torch.transpose( pred_dict["ssnet"].squeeze(), 1, 0 )
            ssnet_pred_t = torch.softmax( pred_dict["ssnet"], dim=1 )
            print("  ssnet_pred_t: ",ssnet_pred_t.shape)            

    # EVALUATE KP-LABEL SCORES
    if config["RUN_KEYPOINT"]:
        with torch.no_grad():
            #print("  pred_dict[kplabel]: ",pred_dict["kp"].shape)
            #kplabel_pred_t = torch.transpose( pred_dict["kp"].squeeze(), 1, 0 )
            kplabel_pred_t = pred_dict["kp"].squeeze()
            print("  kplabel_pred_t: ",kplabel_pred_t.shape)

    print("prepare score arrays: ",time.time()-tstart," sec")

    # EVALUATE PAF SCORES
    if config["RUN_PAF"]:
        with torch.no_grad():
            paf_pred_t = pred_dict["paf"]
        print("  paf-pred: ",paf_pred_t.shape)
        
    tstart = time.time()
    prob_np = lm_prob_t.to(torch.device("cpu")).detach().numpy()
    #prob_np[:] = 1.0 # hack to check

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
        ssnet_np = ssnet_pred_t.to(torch.device("cpu")).detach().numpy()
        hitmaker.add_triplet_ssnet_scores(  matchtriplet_np, 
                                            sparse_np_v[0],
                                            sparse_np_v[1],
                                            sparse_np_v[2],
                                            adc_v.front().meta(),
                                            ssnet_np )                                      

    print("  add kplabel to hitmaker(...). probshape=",kplabel_pred_t.shape)
    kplabel_np = kplabel_pred_t.to(torch.device("cpu")).detach().numpy()
    hitmaker.add_triplet_keypoint_scores(  matchtriplet_np,
                                           sparse_np_v[0],
                                           sparse_np_v[1],
                                           sparse_np_v[2],
                                           adc_v.front().meta(),
                                           kplabel_np )

    print("  add affinity field prediction to hitmaker(...). probshape=",paf_pred_t.shape)
    paf_np = paf_pred_t.to(torch.device("cpu")).detach().numpy()
    hitmaker.add_triplet_affinity_field(  matchtriplet_np, 
                                          sparse_np_v[0],
                                          sparse_np_v[1],
                                          sparse_np_v[2],
                                          adc_v.front().meta(),
                                          paf_np )
        
    dt_make_hits = time.time()-tstart
    dt_save += dt_make_hits

        
    print("end of loop over flow matches")

    # make flow hits
    tstart = time.time()    
    hitmaker.make_hits_v0_scn_network( ev_chstatus, adc_v, evout_lfhits )
    dt_make_hits = time.time()-tstart
    dt_save += dt_make_hits
    print("number of hits made: ",evout_lfhits.size())
    print("make hits: ",dt_make_hits," secs")
    print("time elapsed: prep=",dt_prep," chunk=",dt_chunk," net=",dt_net," save=",dt_save)
    print("try to store 2D ssnet data")
    hitmaker.store_2dssnet_score( io, evout_lfhits )

    # End of flow direction loop
    out.set_id( io.event_id().run(), io.event_id().subrun(), io.event_id().event() )
    out.next_event(True)
    io.save_entry()
    io.clear_entry()
    sys.stdout.flush()

print("Close output")
out.close()
io.finalize()

print("DONE")
