import os,sys,time

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

print larflow.reco.cluster_t
print larflow.reco.cluster_larflow3dhits


iolcv = larcv.IOManager( larcv.IOManager.kBOTH, "larcv", larcv.IOManager.kTickBackward )
iolcv.add_in_file( "../../../../testdata/mcc9_v29e_intrinsic_nue_LowE/merged_dlreco_4c558c3b-344b-4f5a-b319-6ac339aa82b3.root" )
iolcv.set_out_file("outtest_showerbuilder_larcv.root")
iolcv.reverse_all_products()
iolcv.addto_storeonly_list( larcv.kProductImage2D, "trueshoweradc" )
iolcv.addto_storeonly_list( larcv.kProductImage2D, "segment" )
iolcv.initialize()

io = larlite.storage_manager( larlite.storage_manager.kBOTH )
io.add_in_filename( "../../../../testdata/mcc9_v29e_intrinsic_nue_LowE/merged_dlreco_4c558c3b-344b-4f5a-b319-6ac339aa82b3.root" )
io.add_in_filename( "../../../../testdata/mcc9_v29e_intrinsic_nue_LowE/larmatch_wckps_intrinsic_nue_LowE_4c558c3b-344b-4f5a-b319-6ac339aa82b3_larlite.root" )
io.set_out_filename("outtest_showerbuilder.root")
io.set_data_to_read( larlite.data.kMCShower, "mcreco" )
io.set_data_to_read( larlite.data.kMCTrack, "mcreco" )
io.set_data_to_read( larlite.data.kMCTruth, "generator" )
io.set_data_to_read( larlite.data.kLArFlow3DHit, "larmatch" )
io.set_data_to_write( larlite.data.kLArFlow3DHit, "trueshowerhits" )
io.open()

outio = larlite.storage_manager( larlite.storage_manager.kWRITE )

outana_name = "outtest_showerbuilder_ana.root"
outana = rt.TFile(outana_name,"recreate")
builder = larflow.reco.ShowerLikelihoodBuilder()

nentries = iolcv.get_n_entries()
print "Number of entries: ",nentries
#nentries = 3

print "Start loop."
raw_input()

io.next_event()
for ientry in xrange( nentries ):

    iolcv.read_entry(ientry)

    builder.process( iolcv, io )
    iolcv.set_id( iolcv.event_id().run(), iolcv.event_id().subrun(), iolcv.event_id().event() )
    io.set_id( iolcv.event_id().run(), iolcv.event_id().subrun(), iolcv.event_id().event() )
    io.next_event()
    iolcv.save_entry()

outana.Write()
outana.Close()
io.close()
iolcv.finalize()
