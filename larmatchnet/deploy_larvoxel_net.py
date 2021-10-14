#!/usr/bin/env python
from __future__ import print_function
import os,sys,argparse,time
import torch
import MinkowskiEngine as ME

parser = argparse.ArgumentParser("run LArFlow-LArMatch on data")
parser.add_argument("--config-file","-c",required=True,type=str,help="larmatch configuration file")
parser.add_argument("--supera","-su",required=True,type=str,help="LArCV file with ADC images")
parser.add_argument("--weights","-w",required=True,type=str,help="Weight files")
parser.add_argument("--output", "-o",required=True,type=str,help="Stem name of output files: [stem]_larlite.root, [stem]_larcv.root")
parser.add_argument("--tickbackwards","-tb",action='store_true',default=False,help="Indicate that input larcv file is tick-backward [default: F]")
parser.add_argument("--min-score","-p",type=float,default=0.5,help="Minimum Score to save point [default: 0.5]")
parser.add_argument("--num-events","-n",type=int,default=-1,help="Number of events [default: -1 -> All]")
parser.add_argument("--has-mc","-mc",action="store_true",default=False,help="If argument given, input file assumed to have mc truth [default: F]")
parser.add_argument("--has-wirecell","-wc",action="store_true",default=False,help="If flag given, will use WC tagger image to mask cosmics [default: F]")
parser.add_argument("--adc-name","-adc",default="wire",type=str,help="Name of ADC tree [default: wire]")
parser.add_argument("--chstatus-name","-ch",default="wire",type=str,help="Name of the Channel Status tree [default: wire]")
parser.add_argument("--device-name","-d",default="cpu",type=str,help="Name of device. [default: cpu; e.g. cuda:0]")
parser.add_argument("--use-skip-limit",default=None,type=int,help="Specify a max triplet let. If surpassed, skip network eval.")

args = parser.parse_args( sys.argv[1:] )

from ctypes import c_int,c_double
import numpy as np

import larvoxel_engine

import ROOT as rt
from ROOT import std
from larlite import larlite,larutil
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
import torch


print(larutil.Geometry.GetME())
driftv = larutil.LArProperties.GetME().DriftVelocity()

devname=args.device_name
device=torch.device(devname)
checkpointfile = args.weights
print("CHECKPOINT FILE: ",checkpointfile)
checkpoint = torch.load( checkpointfile, map_location={"cuda:0":"cpu",
                                                       "cuda:1":"cpu"} )
ADC_PRODUCER=args.adc_name
CHSTATUS_PRODUCER=args.chstatus_name
USE_GAPCH=True
RETURN_TRUTH=False
BATCHSIZE = 4

# DEFINE THE CLASSES THAT MAKE FLOW MATCH VECTORS
# we use a config file
preplarmatch = larflow.prep.PrepMatchTriplets()
if args.use_skip_limit is not None:
    print("Set Triplet Max where we will skip event: ",args.use_skip_limit)
    preplarmatch.setStopAtTripletMax( True, args.use_skip_limit )

# MULTI-HEAD LARMATCH MODEL
config = larvoxel_engine.load_config_file( args )
model, model_dict = larvoxel_engine.get_larmatch_model( config, dump_model=False )
model = model.to(device)
checkpoint_data = larvoxel_engine.load_model_weights( model, checkpointfile )
print("loaded MODEL")

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
if args.has_wirecell:
    io.specify_data_read( larcv.kProductChStatus, "thrumu" )
io.reverse_all_products()
io.initialize()

out = larlite.storage_manager( larlite.storage_manager.kWRITE )
out.set_out_filename( "%s_larlite.root"%(outfilestem) )
out.open()

sigmoid = torch.nn.Sigmoid()
ssnet_softmax = torch.nn.Softmax(dim=1)
larmatch_softmax = torch.nn.Softmax( dim=1 )

NENTRIES = io.get_n_entries()

if args.num_events>0 and args.num_events<NENTRIES:
    NENTRIES = args.num_events

dt_prep  = 0.
dt_chunk = 0.
dt_net   = 0.
dt_save  = 0.

# setup the hit maker
hitmaker = larflow.voxelizer.LArVoxelHitMaker()
hitmaker._voxelizer.set_voxel_size_cm( 1.0 )
hitmaker._hit_score_threshold = args.min_score

# setup badch maker
badchmaker = ublarcvapp.EmptyChannelAlgo()

# flush standard out buffer before beginning
sys.stdout.flush()

for ientry in range(NENTRIES):

    evout_lfhits = out.get_data(larlite.data.kLArFlow3DHit,"larmatch")
    evout_lfhits.clear()

    io.read_entry(ientry)
    
    print("==========================================")
    print("Entry {}".format(ientry))
    
    # clear the hit maker
    hitmaker.clear();

    adc_v = io.get_data(larcv.kProductImage2D,ADC_PRODUCER).Image2DArray()            
    ev_badch    = io.get_data(larcv.kProductChStatus,CHSTATUS_PRODUCER)
    if args.has_mc:
        print("Retrieving larflow truth...")
        ev_larflow = io.get_data(larcv.kProductImage2D,"larflow")
        flow_v     = ev_larflow.Image2DArray()

    if args.has_wirecell:
        # make wirecell masked image
        print("making wirecell masked image")
        start_wcmask = time.time()
        ev_wcthrumu = io.get_data(larcv.kProductImage2D,"thrumu")
        ev_wcwire   = io.get_data(larcv.kProductImage2D,"wirewc")
        for p in range(adc_v.size()):            
            adc = larcv.Image2D(adc_v[p]) # a copy
            np_adc = larcv.as_ndarray(adc)
            np_wc  = larcv.as_ndarray(ev_wcthrumu.Image2DArray()[p])
            np_adc[ np_wc>0.0 ] = 0.0
            masked = larcv.as_image2d_meta( np_adc, adc.meta() )
            ev_wcwire.Append(masked)
        adc_v = ev_wcwire.Image2DArray()
        end_wcmask = time.time()
        print("time to mask: ",end_wcmask-start_wcmask," secs")

    t_badch = time.time()
    badch_v = badchmaker.makeBadChImage( 4, 3, 2400, 6*1008, 3456, 6, 1, ev_badch )
    print("Number of badcv images: ",badch_v.size())
    gapch_v = badchmaker.findMissingBadChs( adc_v, badch_v, 10.0, 100 )
    for p in range(badch_v.size()):
        for c in range(badch_v[p].meta().cols()):
            if ( gapch_v[p].pixel(0,c)>0 ):
                badch_v[p].paint_col(c,255);
    dt_badch = time.time()-t_badch
    print( "Made EVENT Gap Channel Image: ",gapch_v.front().meta().dump(), " elasped=",dt_badch," secs")

    # run the larflow match prep classes
    t_prep = time.time()

    hitmaker._voxelizer.process_fullchain( io, args.adc_name, args.chstatus_name, False )
    
    if args.has_mc:
        print("processing larflow truth...")
        preplarmatch.make_truth_vector( flow_v )

    data = hitmaker._voxelizer.make_voxeldata_dict()
    print(data.keys())
    coord   = torch.from_numpy( data["voxcoord"] ).int().to(device)
    feat    = torch.from_numpy( np.clip( data["voxfeat"]/40.0, 0, 10.0 ) ).to(device)
    coords, feats = ME.utils.sparse_collate(coords=[coord], feats=[feat])
    xinput = ME.SparseTensor( features=feats, coordinates=coords.to(device) )
    
    t_prep = time.time()-t_prep
    print("  time to prep matches: ",t_prep,"secs")
    dt_prep += t_prep

    # we can run the whole sparse images through the network
    #  to get the individual feature vectors at each coodinate
    t_start = time.time()

    # use UNET portion to first get feature vectors
    model.eval()
    with torch.no_grad():
        pred_dict = model_dict["larmatch"]( xinput )
        for name,arr in pred_dict.items():
            if arr is not None and arr!="larmatch": print(name," ",arr.shape)

        match_pred_t   = pred_dict["larmatch"]      
        ssnet_pred_t   = pred_dict["ssnet"]   if "ssnet" in pred_dict else None
        kplabel_pred_t = pred_dict["kplabel"] if "kplabel" in pred_dict else None
        kpshift_pred_t = pred_dict["kpshift"] if "kpshift" in pred_dict else None
        paf_pred_t     = pred_dict["paf"]     if "paf" in pred_dict else None
        
        dt_net_feats = time.time()-t_start
        print("forward time: ",dt_net_feats,"secs")
        dt_net += dt_net_feats

        match_np   = larmatch_softmax( match_pred_t.F ).to(torch.device("cpu")).numpy()
        ssnet_np   = ssnet_softmax(ssnet_pred_t).to(torch.device("cpu")).numpy()[0]
        kplabel_np = kplabel_pred_t.to(torch.device("cpu")).numpy()[0]
        print("coord: ",data["voxcoord"].shape," ",data["voxcoord"].dtype)
        print("larmatch: ",match_np.shape)
        print("ssnet: ",ssnet_np.shape)
        print("kplabel_np: ",kplabel_np.shape)

    # make flow hits
    tstart = time.time()            
    hitmaker.add_voxel_labels( data["voxcoord"], match_np, ssnet_np, kplabel_np )
    hitmaker.make_labeled_larflow3dhits( hitmaker._voxelizer._triplet_maker, adc_v, evout_lfhits )    
    dt_make_hits = time.time()-tstart
    dt_save += dt_make_hits
    print("number of hits made: ",evout_lfhits.size())
    print("make hits: ",dt_make_hits," secs")
    print("time elapsed: prep=",dt_prep," net=",dt_net," save=",dt_save)    

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
