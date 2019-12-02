from __future__ import print_function
import os,sys,argparse,time

parser = argparse.ArgumentParser("run LArFlow-LArMatch on data")
parser.add_argument("--supera","-su",required=True,type=str,help="LArCV file with ADC images")
parser.add_argument("--weights","-w",required=True,type=str,help="Weight files")
parser.add_argument("--output", "-o",required=True,type=str,help="Output file (larlite format)")
parser.add_argument("--tickbackwards","-tb",action='store_true',default=False,help="Indicate that input larcv file is tick-backward")
parser.add_argument("--min-score","-p",type=float,default=0.5,help="Minimum Score to save point")
parser.add_argument("--num-events","-n",type=int,default=-1,help="Number of events")
parser.add_argument("--has-mc","-mc",action="store_true",default=False,help="If argument given, input file assumed to have mc truth")
args = parser.parse_args( sys.argv[1:] )

from ctypes import c_int,c_double
import numpy as np

import ROOT as rt
from ROOT import std
from larlite import larlite,larutil
from larcv import larcv
larcv.PSet
from larflow import larflow
import torch

# dataset interface
from larcvdataset.larcvserver import LArCVServer

from larmatch import LArMatch
from load_larmatch_data import LArMatchDataset

print(larutil.Geometry.GetME())
driftv = larutil.LArProperties.GetME().DriftVelocity()

checkpointfile = args.weights
checkpoint = torch.load( checkpointfile, map_location={"cuda:0":"cpu",
                                                       "cuda:1":"cpu"} )
NUM_PAIRS=20000
ADC_PRODUCER="wire"
CHSTATUS_PRODUCER="wire"
DEVICE=torch.device("cpu")

# DEFINE THE CLASSES THAT MAKE FLOW MATCH VECTORS
# we use a config file
main_pset = larcv.CreatePSetFromFile("prepflowmatchdata.cfg","ProcessDriver")
driver_pset = main_pset.get_pset("ProcessDriver")
proclist_pset = driver_pset.get_pset("ProcessList")

preplarmatch = {}
for source_plane in ["Y","U","V"]:
    prepcfg = proclist_pset.get_pset("PrepFlowMatch%s"%(source_plane))
    print(prepcfg.dump())
    preplarmatch[source_plane] = larflow.PrepFlowMatchData("deploy%s"%(source_plane))
    preplarmatch[source_plane].configure( prepcfg )
    print("'HAS_MC' SET TO: ",args.has_mc)
    preplarmatch[source_plane].hasMCtruth( args.has_mc )
    
    preplarmatch[source_plane].setADCproducer(ADC_PRODUCER);
    preplarmatch[source_plane].setChStatusProducer(CHSTATUS_PRODUCER);

    preplarmatch[source_plane].initialize()

#preplarmatch = larflow.PrepFlowMatchData("deployY")
#preplarmatch.setSourcePlaneIndex(2)
#preplarmatch.initialize()


model = LArMatch(neval=NUM_PAIRS).to(DEVICE)
model.load_state_dict(checkpoint["state_dict"])

print("loaded MODEL")

tickdir = larcv.IOManager.kTickForward
if args.tickbackwards:
    tickdir = larcv.IOManager.kTickBackward
io = larcv.IOManager( larcv.IOManager.kBOTH, "larcvio", tickdir )
io.add_in_file( args.supera )
io.set_out_file( "test.root" )
io.set_verbosity(1)
io.reverse_all_products()
io.initialize()

out = larlite.storage_manager( larlite.storage_manager.kWRITE )
out.set_out_filename( args.output )
out.open()

sigmoid = torch.nn.Sigmoid()

NENTRIES = io.get_n_entries()

if args.num_events>0 and args.num_events<NENTRIES:
    NENTRIES = args.num_events

dt_net  = 0.
dt_save = 0.

for ientry in range(NENTRIES):

    evout_lfhits_y2u = out.get_data(larlite.data.kLArFlow3DHit,"larmatchy2u")
    evout_lfhits_y2v = out.get_data(larlite.data.kLArFlow3DHit,"larmatchy2v")
    evout_lfhits_y2u.clear()
    evout_lfhits_y2v.clear()

    io.read_entry(ientry)
    
    print("==========================================")
    print("Entry {}".format(ientry))

    for source_plane in ["Y"]:
        preplarmatch[source_plane].process( io )
    flowdata_v  =  preplarmatch[source_plane].getMatchData()
    sparseimg_v = io.get_data(larcv.kProductSparseImage,"larflow_plane2")
    adc_v       = io.get_data(larcv.kProductImage2D,ADC_PRODUCER).Image2DArray()
    srcmeta     = adc_v.at(2).meta()
    ev_badch    = io.get_data(larcv.kProductChStatus,"wire")

    nsparsepts = sparseimg_v.at(0).len()
    source_np  = larcv.as_sparseimg_ndarray( sparseimg_v.at(0) )
    target1_np = larcv.as_sparseimg_ndarray( sparseimg_v.at(1) )
    target2_np = larcv.as_sparseimg_ndarray( sparseimg_v.at(2) )
    
    coord_src_t  = torch.from_numpy( source_np[:,0:2].astype(np.long) )
    coord_tar1_t = torch.from_numpy( target1_np[:,0:2].astype(np.long) )
    coord_tar2_t = torch.from_numpy( target2_np[:,0:2].astype(np.long) )
    feat_src_t   = torch.from_numpy( source_np[:,2].reshape(  (coord_src_t.shape[0], 1) ) )
    feat_tar1_t  = torch.from_numpy( target1_np[:,2].reshape( (coord_tar1_t.shape[0],1) ) )
    feat_tar2_t  = torch.from_numpy( target2_np[:,2].reshape( (coord_tar2_t.shape[0],1) ) )
    
    print("number of flowmaps: {}".format(flowdata_v.size()))
    print("num sparse indices=",nsparsepts)

    sparse_index1 = 0
    sparse_index2 = 0

    y = c_double()
    z = c_double()

    while sparse_index1<nsparsepts and sparse_index2<nsparsepts:

        npairs1       = c_int()
        npairs2       = c_int()
        npairs1.value = 0
        npairs2.value = 0
        
        num_sparse_index1 = c_int()
        num_sparse_index2 = c_int()
        num_sparse_index1.value = 0
        num_sparse_index2.value = 0
    
        matchpair1 = larflow.get_chunk_pair_array( sparse_index1, NUM_PAIRS, flowdata_v.at(0), num_sparse_index1, npairs1 )
        matchpair2 = larflow.get_chunk_pair_array( sparse_index2, NUM_PAIRS, flowdata_v.at(1), num_sparse_index2, npairs2 )
    
        print("max(matchpair1)=",np.max(matchpair1))
        print("sparse_index1=",sparse_index1," npairs1_filled=",npairs1.value)
        print("sparse_index2=",sparse_index2," npairs2_filled=",npairs2.value)
        matchpair1_t = torch.from_numpy( matchpair1 )
        matchpair2_t = torch.from_numpy( matchpair2 )

        tstart = time.time()
        
        pred1_t, pred2_t = model( coord_src_t,  feat_src_t,
                                  coord_tar1_t, feat_tar1_t,
                                  coord_tar2_t, feat_tar2_t,
                                  [matchpair1_t], [matchpair2_t], 1,
                                  torch.device("cpu"), npts1=npairs1.value, npts2=npairs2.value )

        print("pred1_t=",pred1_t.shape," pred2_t=",pred2_t.shape)
        prob1 = sigmoid(pred1_t)
        prob2 = sigmoid(pred2_t)

        dt_net  += time.time()-tstart


        tstart = time.time()
        
        print("call make_larflow_hits(...)")
        larflow.make_larflow_hits_with_deadchs( prob1.detach().numpy().reshape( (1,pred1_t.shape[-1]) ),
                                                source_np, target1_np,
                                                matchpair1, 0,
                                                srcmeta, adc_v, ev_badch,
                                                evout_lfhits_y2u )
        larflow.make_larflow_hits_with_deadchs( prob2.detach().numpy().reshape( (1,pred2_t.shape[-1]) ),
                                                source_np, target2_np,
                                                matchpair2, 1,
                                                srcmeta, adc_v, ev_badch, 
                                                evout_lfhits_y2v )

        print("  lfhits(y->u)=",evout_lfhits_y2u.size()," lfhits(y->v)=",evout_lfhits_y2v.size())

        dt_save += time.time()-tstart
            
        sparse_index1 += num_sparse_index1.value+1
        sparse_index2 += num_sparse_index2.value+1

    print("save hits: Y2U=",evout_lfhits_y2u.size()," Y2V=",evout_lfhits_y2v.size())
    print("time elapsed: net=",dt_net," save=",dt_save)
    out.set_id( io.event_id().run(), io.event_id().subrun(), io.event_id().event() )
    out.next_event(True)


print("Close output")
out.close()

print("DONE")
