#!/usr/bin/env python
from __future__ import print_function
import os,sys,argparse,time


parser = argparse.ArgumentParser("run LArMatchMinkowskiEngine on data")
parser.add_argument("--config-file","-c",type=str,default="config.yaml",help="larmatch configuration file")
parser.add_argument("--detector","-geo",required=True,type=str,help="Detector geometry to apply. Options [uboone,sbnd,icarus]")
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
parser.add_argument("--use-skip-limit",default=None,type=int,help="Specify a max triplet let. If surpassed, skip network eval.")
parser.add_argument("--run-kpreco",default=False,action='store_true',help="If flag is given, will run KeypointReco from larflow/Reco")
args = parser.parse_args( sys.argv[1:] )

if args.detector not in ["uboone","sbnd","icarus"]:
    raise ValueError("Invalid detector [%s]. options are {\"uboone\",\"sbnd\",\"icarus\"}")

from ctypes import c_int,c_double
import numpy as np
import torch

import utils.larmatchme_engine as engine
import MinkowskiEngine as ME

import ROOT as rt
from ROOT import std
from ROOT import larutil
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow

import larmatch.utils.larmatchme_engine as engine

if args.detector == "icarus":
    detid = larlite.geo.kICARUS
    overlap_matrix_file = os.environ["LARFLOW_BASEDIR"]+"/larflow/PrepFlowMatchData/test/output_icarus_wireoverlap_matrices.root"
elif args.detector == "uboone":
    detid = larlite.geo.kMicroBooNE
    overlap_matrix_file = os.environ["LARFLOW_BASEDIR"]+"/larflow/PrepFlowMatchData/test/output_microboone_wireoverlap_matrices.root"    
elif args.detector == "sbnd":
    detid = larlite.geo.kSBND
    
larutil.LArUtilConfig.SetDetector(detid)

DEVICE=torch.device(args.device_name)
config = engine.load_config_file(  args )
single_model    = engine.get_model( config )
checkpointfile  = args.weights
checkpoint_data = engine.load_model_weights( single_model, checkpointfile )

single_model.eval()
single_model.to(DEVICE)
print("loaded MODEL on ",DEVICE)
#print(single_model)

#for name, par in single_model.named_parameters():
#    print("---------------------------------")
#    print(name," ",par.shape)
#    print(par)
#sys.exit(0)

ADC_PRODUCER=args.adc_name
CHSTATUS_PRODUCER=args.chstatus_name
USE_GAPCH=True
RETURN_TRUTH=False
BATCHSIZE = 1

# DEFINE THE CLASSES THAT MAKE FLOW MATCH VECTORS
# we use a config file
preplarmatch = larflow.prep.PrepMatchTriplets()
preplarmatch.set_wireoverlap_filepath( overlap_matrix_file  )    
if args.use_skip_limit is not None:
    print("Set Triplet Max where we will skip event: ",args.use_skip_limit)
    preplarmatch.setStopAtTripletMax( True, args.use_skip_limit )


if args.run_kpreco:
    kprecoalgo = larflow.reco.EventKeypointReco()
    kprecoalgo.kpalgo.set_min_cluster_size( 5,  0 )
    kprecoalgo.kpalgo.set_min_cluster_size( 10, 1 )    

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

# setup geometry
geom = larlite.larutil.Geometry.GetME()

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
    print("Make larmatch spacepoint proposals")
    preplarmatch.process( io, args.adc_name, args.chstatus_name, 10.0, True )

    # make truth labels if possible
    if args.has_mc and ioll is not None:
        print("processing larflow truth...")
        preplarmatch.process_truth_labels( io, ioll )

    # turn shuffle off (to do, function should be kploader function)
    #preplarmatch.setShuffleWhenSampling( False )

    # we run over different TPC plane sets
    ntpcsets = preplarmatch._match_triplet_v.size()
    for itpc in range(ntpcsets):
        matchdata = preplarmatch._match_triplet_v.at(itpc)
        ntriplets = matchdata._triplet_v.size()
        if ntriplets==0:
            continue

        cryoid = matchdata._trip_cryo_tpc_v.at(0)[0]
        tpcid  = matchdata._trip_cryo_tpc_v.at(0)[1]
        #if cryoid!=0 or tpcid!=0:
        #    continue
        
        img_indices_v = std.vector("int")(3,0)
        print("Run larmatch for (TPCID,CRYOID)=(%d,%d)"%(tpcid,cryoid))

        nplanes = geom.Nplanes(tpcid,cryoid)
        start_plane_index = geom.GetSimplePlaneIndexFromCTP(cryoid,tpcid,0)
        for iplane in range( adc_v.size() ):
            planeidx = adc_v.at(iplane).meta().id()
            if planeidx==start_plane_index:
                img_indices_v.resize( nplanes, 0 )
                for p in range(nplanes):
                    img_indices_v[p] = iplane+p

        strplane = ""
        for ip in range( img_indices_v.size() ):
            strplane += " %d "%(img_indices_v[ip])
        print("  plane indices (i.e. positions in adc_v): ",strplane)
            
        # get input data: 2d images
        sparse_np_v = []
        wireplane_sparsetensors = []
        for p in range(3):
            wireimg = matchdata.make_sparse_image( p )
            wireimg_coord_np = wireimg[:,:2].astype(np.int64)
            #print(wireimg_coord_np[:10,:])
        
            wireimg_coord = torch.from_numpy( wireimg_coord_np ).to(DEVICE)
            wireimg_feat  = torch.from_numpy( np.clip( np.expand_dims( wireimg[:,2], 1 )/50.0, 0, 10.0 ) ).to(DEVICE)
            #print("== plane[%d] =="%(p))
            #print("wireimg_coord: ",wireimg_coord.shape)        
            #print("wireimg_feat:  ",wireimg_feat.shape)
            #print("  ",wireimg_coord[:10,:])
            coord_v = [ wireimg_coord ]
            feat_v  = [ wireimg_feat ]
            # prep data to pass into model, single batch
            coords, feats = ME.utils.sparse_collate(coord_v, feat_v)
            #print("sparse_collate out")
            #print("coords")
            #print(coords)
            #print("feats")
            #print(feats)
            wireplane_sparsetensors.append( ME.SparseTensor(features=feats, coordinates=coords) )
            sparse_np_v.append( wireimg_coord_np )

        # get 3d spacepoints (to do, function should be kploader function)
        matchtriplet_np = matchdata.get_all_triplet_data( False )
        print("matchtriplet_np: ",matchtriplet_np.shape," ",matchtriplet_np.dtype)
        spacepoints = matchdata.make_spacepoint_charge_array()    
        nfilled = c_int(0)
        ntriplets = matchtriplet_np.shape[0]
    
        # check
        #print("matchtriplet_np: ",matchtriplet_np[:20,:])
        idxlist = np.arange(ntriplets)
        np.random.shuffle(idxlist)
        #matchtriplet_np = matchtriplet_np[idxlist[:50000],:]
        print("matchtriplet_np (post-sample): ",matchtriplet_np.shape," ",matchtriplet_np.dtype)
                
        matchtriplet_v = [ torch.from_numpy(matchtriplet_np).to(DEVICE) ]

        print("Number of triplets: ",ntriplets)
        with torch.no_grad():
            pred_dict = single_model( wireplane_sparsetensors, matchtriplet_v, 1 )
        print("Ran model: ",pred_dict.keys())
        
        if args.device_name != "cpu":
            torch.cuda.synchronize()
        sys.stdout.flush()    
    
        # EVALUATE LARMATCH SCORES
        tstart = time.time()
        with torch.no_grad():
            print("pred_dict: ",pred_dict["lm"].shape)
            lm_prob_t = pred_dict["lm"]
            lm_prob_t = torch.softmax( lm_prob_t, dim=1 )
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
        #with torch.no_grad():
        #    paf_pred_t = model_dict['paf'].forward( feat_triplet_t )
        #    paf_pred_t = paf_pred_t.reshape( (paf_pred_t.shape[1],paf_pred_t.shape[2]) )
        #    paf_pred_t = torch.transpose( paf_pred_t, 1, 0 )        
        #print("  paf-pred: ",paf_pred_t.shape)
        
        tstart = time.time()
        prob_np = lm_prob_t[0,1,:].to(torch.device("cpu")).detach().numpy().astype(np.float32)
        print("prob_np: ",prob_np.shape)
        print("np.min: ",np.min(prob_np))
        print("np.max: ",np.max(prob_np))
        #prob_np[:] = 1.0 # hack to check

        hitmaker.add_triplet_match_data( prob_np,
                                         matchtriplet_np,
                                         sparse_np_v[0],
                                         sparse_np_v[1],
                                         sparse_np_v[2],
                                         matchdata._pos_v,
                                         adc_v,
                                         img_indices_v,
                                         tpcid, cryoid )

        if config["RUN_SSNET"]:
            print("  add ssnet data to hitmaker(...). probshape=",ssnet_pred_t.shape)
            ssnet_np = ssnet_pred_t.to(torch.device("cpu")).detach().numpy()
            hitmaker.add_triplet_ssnet_scores(  matchtriplet_np, 
                                                sparse_np_v[0],
                                                sparse_np_v[1],
                                                sparse_np_v[2],
                                                adc_v.front().meta(),
                                                ssnet_np )                                      
        if config["RUN_KPLABEL"]:
            print("  add kplabel to hitmaker(...). probshape=",kplabel_pred_t.shape)
            kplabel_np = kplabel_pred_t.to(torch.device("cpu")).detach().numpy()
            hitmaker.add_triplet_keypoint_scores(  matchtriplet_np,
                                                   sparse_np_v[0],
                                                   sparse_np_v[1],
                                                   sparse_np_v[2],
                                                   adc_v.front().meta(),
                                                   kplabel_np,
                                                   tpcid, cryoid )
        
        # deprecated
        #print("  add affinity field prediction to hitmaker(...). probshape=",paf_pred_t.shape)
        #paf_np = paf_pred_t.to(torch.device("cpu")).detach().numpy()
        #hitmaker.add_triplet_affinity_field(  matchtriplet_np, 
        #                                      sparse_np_v[0],
        #                                      sparse_np_v[1],
        #                                      sparse_np_v[2],
        #                                      adc_v.front().meta(),
        #                                      paf_np[:int(npairs.value)] )
        if False:
            # early stopping on first filled tpc data for debugging
            break
        
    # end of tpc loop
        
    dt_make_hits = time.time()-tstart
    dt_save += dt_make_hits

        
    print("end of loop over flow matches")

    # make flow hits
    tstart = time.time()    
    hitmaker.make_hits( ev_chstatus, adc_v, evout_lfhits )
    dt_make_hits = time.time()-tstart
    dt_save += dt_make_hits
    print("number of hits made: ",evout_lfhits.size())
    print("make hits: ",dt_make_hits," secs")
    print("time elapsed: prep=",dt_prep," chunk=",dt_chunk," net=",dt_net," save=",dt_save)
    print("try to store 2D ssnet data")
    #hitmaker.store_2dssnet_score( io, evout_lfhits )

    if args.run_kpreco:
        kprecoalgo.process_larmatch_v2( out, "larmatch" )

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
