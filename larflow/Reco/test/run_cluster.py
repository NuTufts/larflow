import os,sys,argparse

"""
Run the PCA-based clustering routine for track space-points.
Uses 3D points saved in larflow3dhit objects.
"""

parser = argparse.ArgumentParser("Run larflow3dhit clustering algorith")
parser.add_argument('-i','--input-dlmerged',type=str,required=True,help="Input file containing ADC, ssnet, badch images/info")
parser.add_argument('-l','--input-larflow',type=str,required=True,help="Input file containing larlite::larflow3dhit objects")
parser.add_argument('-o','--output',type=str,required=True,help="Name of output file. Will not overwrite")
parser.add_argument('-rw','--rewrite',action='store_true',default=False,help="Rewrite output file")
parser.add_argument('-n','--num-entries',type=int,default=None,help="Number of entries to run")
parser.add_argument('-e','--start-entry',type=int,default=0,help="Starting entry")
parser.add_argument('-tb','--tickbackwards',action='store_true',default=False,help="Input larcv images are tick-backward")
parser.add_argument('-min','--min-score',type=float,default=0.2,help='Minimum larmatch score')
parser.add_argument('-ds','--ds-fraction',type=float,default=0.10,help='Down-sampling fraction')

args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

print larflow.reco
larflow.reco.ClusterFunctions()
print larflow.reco.ClusterFunctions

io = larlite.storage_manager( larlite.storage_manager.kBOTH )
iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )

print "[DL MERGED] ",args.input_dlmerged
print "[LARMATCH]  ",args.input_larflow
print "[OUTPUT]    ",args.output

io.add_in_filename(  args.input_dlmerged )
io.add_in_filename(  args.input_larflow )
io.set_data_to_read( larlite.data.kLArFlow3DHit, "larmatch" )
io.set_out_filename( args.output )
iolcv.add_in_file(   args.input_dlmerged )

#io.add_in_filename( "larmatch-testfile.root" )
#io.add_in_filename( "merged_dlreco_larmatch_run3_bnboverlay.root" )
#io.add_in_filename( "larmatch_wctagger_example.root" )

# eLEE sample
#io.add_in_filename(  "larmatch_triplet_eLEE_sample2_larlite.root" )
#io.set_out_filename( "larflow_triplet_cluster_eLEE_sample2.root" )
#iolcv.add_in_file( "merged_dlreco_eLEE_sample2.root" )

# BNB RUN 3 TEST
#io.add_in_filename(  "../../../../testdata/mcc9_v28_wc_bnb5e19/larmatch_run3g_test_larlite.root" )
#io.set_out_filename( "larflow_triplet_reco_bnb_run3test.root" )
#iolcv.add_in_file(   "../../../../testdata/mcc9_v28_wc_bnb5e19/merged_dlreco_run3g_test.root" )

# BNB RUN 3 OVERLAY TEST
#io.add_in_filename(  "../../../../testdata/mcc9_v28_wc_bnb_overlay_run3g/larmatch_dlreco_run3g_bnboverlay_test_calib_larlite.root" )
#io.set_out_filename( "larflow_triplet_reco_bnb_overlay_run3test.root" )
#iolcv.add_in_file(   "../../../../testdata/mcc9_v28_wc_bnb_overlay_run3g/merged_dlreco_run3g_bnboverlay_test_calib.root" )

# BNB RUN 3 OVERLAY PRODUCTION TEST
#io.add_in_filename(  "testdata/mcc9_v29e_wc_bnb_overlay_run3g/larmatch_run3g_wc_bnboverlay_test_larlite.root" )
#io.set_out_filename( "testdata/mcc9_v29e_wc_bnb_overlay_run3g/triplet_reco_bnb_overlay_run3g_test.root" )
#iolcv.add_in_file(   "testdata/mcc9_v29e_wc_bnb_overlay_run3g/ssnet_run3g_wc_bnboverlay_test.root" )

io.open()
if args.tickbackwards:
    iolcv.reverse_all_products()
iolcv.initialize()

pcacluster = larflow.reco.PCACluster()
pcacluster.set_min_larmatch_score( args.min_score )
pcacluster.set_downsample_fraction( args.ds_fraction )
pcacluster.set_dbscan_pars( 10.0, 5, 10 )

lcv_nentries = iolcv.get_n_entries()
ll_nentries  = io.get_entries()
if lcv_nentries<ll_nentries:
    nentries = lcv_nentries
else:
    nentries = ll_nentries
    
if args.num_entries is not None:
    end_entry = args.start_entry + args.num_entries
    if end_entry>nentries:
        end_entry = nentries
else:
    end_entry = nentries

#io.go_to( args.start_entry )
io.next_event()
#io.go_to( args.start_entry )
for ientry in xrange( args.start_entry, end_entry ):
    print "[ENTRY ",ientry,"]"
    iolcv.read_entry(ientry)

    ev_lfhits = io.get_data( larlite.data.kLArFlow3DHit, "larmatch" )
    print "num of hits: ",ev_lfhits.size()

    pcacluster.process( iolcv, io )

    io.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    io.next_event()    

io.close()
iolcv.finalize()
