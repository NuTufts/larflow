from __future__ import print_function
import os,sys
import ROOT as rt
from larlite import larlite
from larcv import larcv
from larflow import larflow

iolcv = larcv.IOManager(larcv.IOManager.kREAD,"larcvio",larcv.IOManager.kTickBackward)
iolcv.add_in_file( "maskrcnn_testfile/merged_dlana_1d64fc18-3d97-4254-bd52-e528983db358.root" )
iolcv.add_in_file( "maskrcnn_testfile/hadd_mrcnnproposals_merged_dlana_1d64fc18-3d97-4254-bd52-e528983db358.root" )
iolcv.specify_data_read( larcv.kProductImage2D, "wire" );
iolcv.specify_data_read( larcv.kProductImage2D, "thrumu" );
iolcv.specify_data_read( larcv.kProductImage2D, "ancestor" );
iolcv.specify_data_read( larcv.kProductImage2D, "segment" );
iolcv.specify_data_read( larcv.kProductImage2D, "instance" );
iolcv.specify_data_read( larcv.kProductImage2D, "larflow" );
iolcv.specify_data_read( larcv.kProductChStatus, "wire" );
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane0" )
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane1" )
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane2" )
iolcv.specify_data_read( larcv.kProductClusterMask, "mask_proposals_y" )
iolcv.specify_data_read( larcv.kProductSparseImage, "sparseuresnetout" )
iolcv.reverse_all_products()
iolcv.initialize()

ioll = larlite.storage_manager(larlite.storage_manager.kBOTH)
ioll.add_in_filename( "maskrcnn_testfile/larmatch_kps_fileid0001_1d64fc18-3d97-4254-bd52-e528983db358_larlite.root" )
ioll.set_out_filename( "outtest_maskrcnn_reco.root" )
ioll.set_data_to_read( larlite.data.kLArFlow3DHit, "larmatch" )
ioll.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
ioll.set_data_to_read( larlite.data.kMCShower, "mcreco" )
ioll.set_data_to_read( larlite.data.kMCTruth,  "generator" )
ioll.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashBeam" )
ioll.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashCosmic" )
ioll.set_verbosity(0)
ioll.open()

mrcnnreco = larflow.mrcnnreco.MaskRCNNreco()
mrcnnreco.set_verbosity(larcv.msg.kDEBUG)

nentries = iolcv.get_n_entries()
ioll.go_to(0)
for ientry in range(nentries):
    print("[ENTRY ",ientry,"]")
    iolcv.read_entry(ientry)

    mrcnnreco.process( iolcv, ioll )
    ioll.next_event()
    break

ioll.close()
iolcv.finalize()

