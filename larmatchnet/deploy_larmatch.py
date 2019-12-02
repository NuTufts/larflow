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
RETURN_TRUTH=False
BATCHSIZE = 1
SOURCE_PLANE_LIST = ["Y"]

# DEFINE THE CLASSES THAT MAKE FLOW MATCH VECTORS
# we use a config file
main_pset = larcv.CreatePSetFromFile("prepflowmatchdata.cfg","ProcessDriver")
driver_pset = main_pset.get_pset("ProcessDriver")
proclist_pset = driver_pset.get_pset("ProcessList")

preplarmatch = {}
for source_plane in SOURCE_PLANE_LIST:
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

dt_prep  = 0.
dt_chunk = 0.
dt_net   = 0.
dt_save  = 0.

for ientry in range(NENTRIES):

    evout_lfhits_y2u = out.get_data(larlite.data.kLArFlow3DHit,"larmatchy2u")
    evout_lfhits_y2v = out.get_data(larlite.data.kLArFlow3DHit,"larmatchy2v")
    evout_lfhits_y2u.clear()
    evout_lfhits_y2v.clear()

    io.read_entry(ientry)
    
    print("==========================================")
    print("Entry {}".format(ientry))

    adc_v       = io.get_data(larcv.kProductImage2D,ADC_PRODUCER).Image2DArray()
    ev_badch    = io.get_data(larcv.kProductChStatus,CHSTATUS_PRODUCER)

    # run the larflow match prep classes
    t_prep = time.time()
    for source_plane in SOURCE_PLANE_LIST:
        # parse the image and produce candidate pixel matches
        print(" made match pairs for source plane=",source_plane)
        preplarmatch[source_plane].process( io )
    t_prep = time.time()-t_prep
    print("  time to prep matches: ",t_prep,"secs")
    dt_prep += t_prep
    
    # we get the sparse images for each plane
    # we get sparse info from output of prepflowmatch class
    source_plane = "Y"
    plane_index  = 2

    # get the sparse ADC images produced by the preplarmatch class above
    sparseimg_v = io.get_data(larcv.kProductSparseImage,"larflow_plane%d"%(plane_index))
    
    # Prep sparse ADC numpy arrays
    nsparsepts = sparseimg_v.at(0).len()
    sparse_adc_np = { "Y":larcv.as_sparseimg_ndarray( sparseimg_v.at(0) ),
                      "U":larcv.as_sparseimg_ndarray( sparseimg_v.at(1) ),
                      "V":larcv.as_sparseimg_ndarray( sparseimg_v.at(2) ) }
    coord_t = {"Y":torch.from_numpy( sparse_adc_np["Y"][:,0:2].astype(np.long) ),
               "U":torch.from_numpy( sparse_adc_np["U"][:,0:2].astype(np.long) ),
               "V":torch.from_numpy( sparse_adc_np["V"][:,0:2].astype(np.long) )}
    feat_t  = {"Y":torch.from_numpy( sparse_adc_np["Y"][:,2].reshape(  (coord_t["Y"].shape[0], 1) ) ),
               "U":torch.from_numpy( sparse_adc_np["U"][:,2].reshape(  (coord_t["U"].shape[0], 1) ) ),
               "V":torch.from_numpy( sparse_adc_np["V"][:,2].reshape(  (coord_t["V"].shape[0], 1) ) )}


    # we can run the whole sparse images through the network
    #  to get the individual feature vectors at each coodinate
    t_start = time.time()
    print("computing features")
    outfeat_y, outfeat_u, outfeat_v = model.forward_features( coord_t["Y"], feat_t["Y"],
                                                              coord_t["U"], feat_t["U"],
                                                              coord_t["V"], feat_t["V"], 1 )
    outfeat_t = {"Y":outfeat_y,
                 "U":outfeat_u,
                 "V":outfeat_v}
    
    dt_net_feats = time.time()-t_start
    print("compute features: ",dt_net_feats,"secs")
    dt_net += dt_net_feats

    for (source_plane,tar1_plane,tar2_plane,src_idx,tar1_idx,tar2_idx) in [ ("Y","U","V",2,0,1) ]:
        # match data for this source plane
        flowdata_v  =  preplarmatch[source_plane].getMatchData()
        # get the image meta of the source image
        srcmeta     = adc_v.at(src_idx).meta()

        # get the sparse ADC images produced by this source plane's preplarmatch class
        sparseimg_v = io.get_data(larcv.kProductSparseImage,"larflow_plane%d"%(src_idx))
    
        # Get source and target planes
        nsparsepts = sparseimg_v.at(0).len()
        source_np  = sparse_adc_np[source_plane]
        target1_np = sparse_adc_np[tar1_plane]
        target2_np = sparse_adc_np[tar2_plane]
    
        coord_src_t  = coord_t[source_plane]
        coord_tar1_t = coord_t[tar1_plane]
        coord_tar2_t = coord_t[tar2_plane]
        feat_src_t   = feat_t[source_plane]
        feat_tar1_t  = feat_t[tar1_plane]
        feat_tar2_t  = feat_t[tar2_plane]
        
        print("number of flowmaps: {}".format(flowdata_v.size()))
        print("num sparse indices=",nsparsepts)

        # we tackle each flow direction separately
        for (iflow,tar_idx,tar_plane,target_np,evout_lfhits) in [ (0,tar1_idx,tar1_plane,target1_np,evout_lfhits_y2u),
                                                                  (1,tar2_idx,tar2_plane,target2_np,evout_lfhits_y2v)]:
            # track the source pixel we've evaluated
            sparse_index = 0
            y = c_double()
            z = c_double()
            coord_tar_t = coord_t[tar_plane]
            
            # loop until we've evaluated the matches for each source point
            print("== flow direction [",iflow,"] ",source_plane,"->",tar_plane,"==")
            while sparse_index<nsparsepts:

                npairs = c_int()
                npairs.value = 0
                num_sparse_index = c_int()
                num_sparse_index.value = 0
                tstart = time.time()
                matchpair_np = larflow.get_chunk_pair_array( sparse_index, NUM_PAIRS,
                                                             flowdata_v.at(iflow), num_sparse_index, npairs )
                #print("  max(matchpair)=",np.max(matchpair_np))
                t_chunk = time.time()-tstart
                print("  sparse_index=",sparse_index," npairs_filled=",npairs.value," time to make chunk=",t_chunk," secs")
                dt_chunk += t_chunk

                
                # run matches through classifier portion of network
                matchpair_t = torch.from_numpy( matchpair_np )
                
                # track indices through the batch
                bstart_src = 0
                bstart_tar = 0
                if RETURN_TRUTH:
                    truthvec = torch.zeros( (1,1,npairs.value), requires_grad=False, dtype=torch.int32 ).to( DEVICE )
        
                for b in range(BATCHSIZE):
                    if BATCHSIZE>1:
                        nbatch_src = coord_src_t[:,2].eq(b).sum()
                        nbatch_tar = coord_tar_t[:,2].eq(b).sum()
                    else:
                        nbatch_src = coord_src_t.shape[0]
                        nbatch_tar = coord_tar_t.shape[0]
            
                    bend_src = bstart_src + nbatch_src
                    bend_tar = bstart_tar + nbatch_tar
                    tstart = time.time()
                    pred_t, truth_t = model.classify_sample( coord_src_t[bstart_src:bend_src,:],
                                                             outfeat_t[source_plane][bstart_src:bend_src,:],
                                                             outfeat_t[tar_plane][bstart_tar:bend_tar,:],
                                                             matchpair_t, torch.device("cpu"),
                                                             RETURN_TRUTH, npairs.value )
                    dt_net_classify = time.time()-tstart
                    dt_net  += dt_net_classify
                    prob = sigmoid(pred_t)                
                    print("  batch[",b,"] pred_t=",pred_t.shape," time-elapsed=",dt_net_classify,"secs")


                    tstart = time.time()
                    print("  call make_larflow_hits(...)")
                    larflow.make_larflow_hits_with_deadchs( prob.detach().numpy().reshape( (1,pred_t.shape[-1]) ),
                                                            source_np, target_np,
                                                            matchpair_np, tar_idx,
                                                            srcmeta, adc_v, ev_badch,
                                                            evout_lfhits )

                    dt_make_hits = time.time()-tstart
                    print("  lfhits(",source_plane,"->",tar_plane,")=",evout_lfhits.size()," secs elapsed=",dt_make_hits)
                    dt_save += dt_make_hits
            
                sparse_index += num_sparse_index.value+1
                
            # end of flow loop
            print("end of flow loop")


        # End of while loop
        print("end of flow direction loop")
        
        print("saved hits. Source Plane=",source_plane," Y2U=",evout_lfhits_y2u.size()," Y2V=",evout_lfhits_y2v.size())
        print("time elapsed: prep=",dt_prep," chunk=",dt_chunk," net=",dt_net," save=",dt_save)
        
    print("end of source plane loop")

    # End of flow direction loop
    out.set_id( io.event_id().run(), io.event_id().subrun(), io.event_id().event() )
    out.next_event(True)


print("Close output")
out.close()

print("DONE")
