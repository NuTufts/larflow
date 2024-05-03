from __future__ import print_function
import os,sys,argparse

# parser = argparse.ArgumentParser(description='Make MC flashmatch training data from ROOT file. Store into petastorm.')
# parser.add_argument('-db',"--db-folder",required=True,type=str,help="path to directory storing PySpark database")
# parser.add_argument('-lcv',"--in-larcvtruth",required=True,type=str,help="path to larcv truth root file")
# parser.add_argument('-mc',"--in-mcinfo",required=True,type=str,help="path to mcinfo root file")
# parser.add_argument('-op',"--in-opreco",required=True,type=str,help="path to opreco root file")
# parser.add_argument('-v',"--verbosity",type=int,default=0,help='Set Verbosity Level [0=quiet, 2=debug]')
# parser.add_argument('-e',"--entry",type=int,default=None,help='Run specific entry')
# args = parser.parse_args(sys.argv[1:])

import ROOT as rt
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
from ROOT import std
from ROOT import larutil

from opmodeldatautil import get_reco_flash_vectors

rt.gStyle.SetOptStat(0)
rt.gROOT.ProcessLine( "gErrorIgnoreLevel = 3002;" )

start_entry = 0
end_entry = 1
#if args.entry is not None:
#    start_entry = args.entry
#    end_entry = start_entry+1
adc_name = "wiremc"
    
opdataprep = larflow.opticalmodel.OpModelMCDataPrep()
fm_verbosity = 0
opdataprep.setVerboseLevel(fm_verbosity)

data_folder = "~/working/data/mcc9_v13_bnbnue_corsika/"
sourcefile = data_folder + "/mcinfo-Run000001-SubRun000001.root"
input_larcv_rootfile_v = [ data_folder + "/larcvtruth-Run000001-SubRun000001.root" ]
input_larlite_rootfile_v = [ data_folder + "/opreco-Run000001-SubRun000001.root",
                            data_folder + "/mcinfo-Run000001-SubRun000001.root" ]

# c++ classes that provides spacepoint labels
larproperties = larutil.LArProperties.GetME()
voxelsize = 5.0
voxel_origin = std.vector("float")(3)
voxel_origin[0] = (2400.0-3200.0)*0.5*larproperties.DriftVelocity()
voxel_origin[1] = -120.0
voxel_origin[2] = -10.0

voxel_len = std.vector("float")(3)
voxel_len[0] = (2400+6.0*1010 - 3200)*0.5*larproperties.DriftVelocity() - voxel_origin[0]
voxel_len[1] = 240.0
voxel_len[2] = 1050.0

print("voxel_origin: ",voxel_origin)
print("voxel_len: ",voxel_len)

voxelizer = larflow.voxelizer.VoxelizeTriplets( voxel_origin, voxel_len, 5.0 )
ndims_v   = voxelizer.get_dim_len()
origin_v  = voxelizer.get_origin()
tpc_origin = std.vector("float")(3)
tpc_origin[0] = 0.3
tpc_origin[1] = -117.0
tpc_origin[2] = 0.3

tpc_end = std.vector("float")(3)
tpc_end[0] = 256.0
tpc_end[1] = 117.0
tpc_end[2] = 1035.7

index_tpc_origin = [ voxelizer.get_axis_voxel(i,tpc_origin[i]) for i in range(3) ]
index_tpc_end    = [ voxelizer.get_axis_voxel(i,tpc_end[i]) for i in range(3) ]

print("VOXELIZER SETUP =====================")
print("origin: (",origin_v[0],",",origin_v[1],",",origin_v[2],")")
print("ndims: (",ndims_v[0],",",ndims_v[1],",",ndims_v[2],")")
print("index-tpc-origin: ",index_tpc_origin)
print("index-tpc-end: ",index_tpc_end)

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "io", larcv.IOManager.kTickBackward )
for f in input_larcv_rootfile_v:
    iolcv.add_in_file( f )
iolcv.specify_data_read( larcv.kProductImage2D,  "wire" )
iolcv.specify_data_read( larcv.kProductImage2D,  "wiremc" )
iolcv.specify_data_read( larcv.kProductChStatus, "wire" )
iolcv.specify_data_read( larcv.kProductChStatus, "wiremc" )
iolcv.specify_data_read( larcv.kProductImage2D,  "ancestor" )
iolcv.specify_data_read( larcv.kProductImage2D,  "instance" )
iolcv.specify_data_read( larcv.kProductImage2D,  "segment" )
iolcv.specify_data_read( larcv.kProductImage2D,  "larflow" )
iolcv.reverse_all_products()
iolcv.initialize()

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
for f in input_larlite_rootfile_v:
    ioll.add_in_filename( f )    
ioll.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
ioll.set_data_to_read( larlite.data.kMCShower, "mcreco" )
ioll.set_data_to_read( larlite.data.kMCTruth,  "generator" )
ioll.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashBeam" )
ioll.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashCosmic" )
ioll.open()

nentries = ioll.get_entries()
print("Number of entries: ",nentries)

for ientry in range( start_entry, end_entry ):

    print()
    print("==========================")
    print("===[ EVENT ",ientry," ]===")

    iolcv.read_entry(ientry)
    ioll.go_to(ientry)
    
    run     = ioll.run_id()
    subrun  = ioll.subrun_id()
    eventid = ioll.event_id()

    # build candidate spacepoints and true labels
    print("Run Voxelizer")
    truth_correct_tdrift = True
    voxelizer.process_fullchain_withtruth( iolcv, ioll, adc_name, adc_name, truth_correct_tdrift )
    
    # match reco flashes to true track and shower information
    print("Run opdataprep")    
    opdataprep.process( ioll, voxelizer )
    opdataprep.printMatches()
    #opdataprep.printFiltered()
    #opdataprep.tagBadFlashMatches( voxelizer, ioll )    

    # make vectors of reco opflashes
    flash_np_v = get_reco_flash_vectors( ioll )
    
    for iflash in range( opdataprep.recoflash_v.size() ):

        flash = opdataprep.recoflash_v.at(iflash)
    
        # get flash match vectors
        #coord_v = std.vector("std::vector<int>")()
        #feat_v  = std.vector("std::vector<float>")()
        #opdataprep.getChargeVoxelsForFlash( flash, voxelizer, coord_v, feat_v )

        # get the right flash pe vector
        #if flash.producerid>=0:
        #    flash_np = flash_np_v[ (flash.producerid,flash.index) ]
        #else:
        #    flash_np = np.zeros( 32, dtype=np.float32 )            
        
        data_dict = opdataprep.make_opmodel_data_dict( flash, voxelizer, ioll )
        
        print("flash[",iflash,"]")
        print("  ",opdataprep.strRecoMatchInfo( flash, iflash ))
        print("  pe: ",data_dict["flashpe"].shape)        
        print("  voxel coord: ",data_dict["voxcoord"].shape)
        print("  voxel charge: ",data_dict["voxcharge"].shape)
        print("  frac of track traj. in tpc with voxel charge: ",opdataprep.flash_track_frac_intpc_w_charge.at(iflash))            

            


    


    
