#!/usr/bin/env python
from __future__ import print_function
import os,sys,argparse,time

parser = argparse.ArgumentParser("run LArFlow-LArMatch on data")
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
args = parser.parse_args( sys.argv[1:] )

from ctypes import c_int,c_double
import numpy as np

import ROOT as rt
from ROOT import std
from larlite import larlite,larutil
from larcv import larcv
larcv.PSet
from ublarcvapp import ublarcvapp
from larflow import larflow
import torch

# dataset interface
from larcvdataset.larcvserver import LArCVServer

from larmatch import LArMatch

print(larutil.Geometry.GetME())
driftv = larutil.LArProperties.GetME().DriftVelocity()


devname=args.device_name
DEVICE=torch.device(devname)
checkpointfile = args.weights
checkpoint = torch.load( checkpointfile, map_location={"cuda:0":devname,
                                                       "cuda:1":devname} )
NUM_PAIRS=50000
ADC_PRODUCER=args.adc_name
CHSTATUS_PRODUCER=args.chstatus_name
USE_GAPCH=True
RETURN_TRUTH=False
BATCHSIZE = 1

# DEFINE THE CLASSES THAT MAKE FLOW MATCH VECTORS
# we use a config file
preplarmatch = larflow.PrepMatchTriplets()

model = LArMatch(neval=NUM_PAIRS).to(DEVICE)
model.load_state_dict(checkpoint["state_dict"])

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

NENTRIES = io.get_n_entries()

if args.num_events>0 and args.num_events<NENTRIES:
    NENTRIES = args.num_events

dt_prep  = 0.
dt_chunk = 0.
dt_net   = 0.
dt_save  = 0.

# setup the hit maker
hitmaker = larflow.FlowMatchHitMaker()
hitmaker.set_score_threshold( args.min_score )

# setup badch maker
badchmaker = ublarcvapp.EmptyChannelAlgo()

for ientry in range(NENTRIES):

    evout_lfhits = out.get_data(larlite.data.kLArFlow3DHit,"larmatch")
    evout_lfhits.clear()

    io.read_entry(ientry)
    
    print("==========================================")
    print("Entry {}".format(ientry))
    
    # clear the hit maker
    hitmaker.clear();

    adc_v       = io.get_data(larcv.kProductImage2D,ADC_PRODUCER).Image2DArray()
    ev_badch    = io.get_data(larcv.kProductChStatus,CHSTATUS_PRODUCER)

    if args.has_wirecell:
        # make wirecell masked image
        print("making wirecell masked image")
        start_wcmask = time.time()
        ev_wcthrumu = io.get_data(larcv.kProductImage2D,"thrumu")
        ev_wcwire   = io.get_data(larcv.kProductImage2D,"wirewc")
        for p in xrange(adc_v.size()):            
            adc = larcv.Image2D(adc_v[p]) # a copy
            np_adc = larcv.as_ndarray(adc)
            np_wc  = larcv.as_ndarray(ev_wcthrumu.Image2DArray()[p])
            np_adc[ np_wc>0.0 ] = 0.0
            masked = larcv.as_image2d_meta( np_adc, adc.meta() )
            ev_wcwire.Append(masked)
        end_wcmask = time.time()
        print("time to mask: ",end_wcmask-start_wcmask," secs")

    t_badch = time.time()
    badch_v = badchmaker.makeBadChImage( 4, 3, 2400, 6*1008, 3456, 6, 1, ev_badch )
    print("Number of badcv images: ",badch_v.size())
    gapch_v = badchmaker.findMissingBadChs( adc_v, badch_v, 10.0, 100 )
    for p in xrange(badch_v.size()):
        for c in xrange(badch_v[p].meta().cols()):
            if ( gapch_v[p].pixel(0,c)>0 ):
                badch_v[p].paint_col(c,255);
    dt_badch = time.time()-t_badch
    print( "Made EVENT Gap Channel Image: ",gapch_v.front().meta().dump(), " elasped=",dt_badch," secs")

    # run the larflow match prep classes
    t_prep = time.time()
    preplarmatch.process( adc_v, badch_v, 10.0, False )
    t_prep = time.time()-t_prep
    print("  time to prep matches: ",t_prep,"secs")
    dt_prep += t_prep
    
    
    # Prep sparse ADC numpy arrays
    sparse_np_v = [ preplarmatch.make_sparse_image(p) for p in xrange(3) ]
    coord_t = [ torch.from_numpy( sparse_np_v[p][:,0:2].astype(np.long) ).to(DEVICE) for p in xrange(3) ]
    feat_t  = [ torch.from_numpy( sparse_np_v[p][:,2].reshape(  (coord_t[p].shape[0], 1) ) ).to(DEVICE) for p in xrange(3) ]

    # we can run the whole sparse images through the network
    #  to get the individual feature vectors at each coodinate
    t_start = time.time()
    print("computing features")
    outfeat_u, outfeat_v, outfeat_y = model.forward_features( coord_t[0], feat_t[0],
                                                              coord_t[1], feat_t[1],
                                                              coord_t[2], feat_t[2], 1 )
    dt_net_feats = time.time()-t_start
    print("compute features: ",dt_net_feats,"secs")
    dt_net += dt_net_feats

    # get indices
    npairs      = c_int()
    npairs.value = 0
    last_index  = c_int()
    last_index.value = 0
    with_truth  = False
    tstart = time.time()

    ntriplets = preplarmatch._triplet_v.size()
    startidx = 0
    while startidx<ntriplets:
        print("create matchpairs: startidx=",startidx," of ",ntriplets)
        t_chunk = time.time()
        matchpair_np = preplarmatch.get_chunk_triplet_matches( startidx,
                                                               NUM_PAIRS,
                                                               last_index,
                                                               npairs,
                                                               with_truth )
        t_chunk = time.time()-t_chunk
        print("made matchpairs: npairs_filled=",npairs.value,"; time to make chunk=",t_chunk," secs") 
        dt_chunk += t_chunk
            
        startidx = int(last_index.value)

        # run matches through classifier portion of network
        matchpair_t = torch.from_numpy( matchpair_np.astype(np.long) ).to(DEVICE)
                
        if with_truth:
            truthvec = torch.from_numpy( sparse_np_v[:,3].astype(np.long) ).to(DEVICE)
        
        tstart = time.time()
        pred_t = model.classify_triplet( outfeat_u, outfeat_v, outfeat_y, matchpair_t, int(npairs.value), DEVICE )
        dt_net_classify = time.time()-tstart
        dt_net  += dt_net_classify
        prob_t = sigmoid(pred_t)
        print("prob_t=",prob_t.shape," time-elapsed=",dt_net_classify,"secs")


        tstart = time.time()
        prob_np = prob_t.to(torch.device("cpu")).detach().numpy().reshape( (prob_t.shape[-1]) )
        #prob_np[:] = 1.0 # hack to check
        print("  add data to hitmaker(...). probshape=",prob_np.shape)
        pos_v = std.vector("std::vector<float>")()
        hitmaker.add_triplet_match_data( prob_np[:int(npairs.value)],
                                         matchpair_np[:int(npairs.value),:],
                                         sparse_np_v[0],
                                         sparse_np_v[1],
                                         sparse_np_v[2],
                                         pos_v,
                                         adc_v )
        dt_make_hits = time.time()-tstart
        dt_save += dt_make_hits
            
        
    print("end of loop over flow matches")

    # make flow hits
    tstart = time.time()    
    hitmaker.make_hits( ev_badch, evout_lfhits )
    dt_make_hits = time.time()-tstart
    dt_save += dt_make_hits
    print("number of hits made: ",evout_lfhits.size())
    print("make hits: ",dt_make_hits," secs")
    print("time elapsed: prep=",dt_prep," chunk=",dt_chunk," net=",dt_net," save=",dt_save)    

    # End of flow direction loop
    out.set_id( io.event_id().run(), io.event_id().subrun(), io.event_id().event() )
    out.next_event(True)
    io.save_entry()
    io.clear_entry()

print("Close output")
out.close()
io.finalize()

print("DONE")
