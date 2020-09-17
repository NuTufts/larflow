import os,sys,argparse,time

"""
Run the PCA-based clustering routine for track space-points.
Uses 3D points saved in larflow3dhit objects.
"""

parser = argparse.ArgumentParser("Run larflow3dhit clustering algorith")
# required
parser.add_argument('-i','--input-dlmerged',type=str,required=True,help="Input file containing ADC, ssnet, badch images/info")
parser.add_argument('-l','--input-larflow',type=str,required=True,help="Input file containing larlite::larflow3dhit objects")
parser.add_argument('-o','--output',type=str,required=True,help="Name of output file. Will not overwrite")
# optional
parser.add_argument('-n','--num-entries',type=int,default=None,help="Number of entries to run")
parser.add_argument('-e','--start-entry',type=int,default=0,help="Starting entry")
parser.add_argument('-tb','--tickbackwards',action='store_true',default=False,help="Input larcv images are tick-backward")
parser.add_argument("-mc",'--ismc',action='store_true',default=False,help="If true, store MC information")

args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow


io = larlite.storage_manager( larlite.storage_manager.kBOTH )
iolcv = larcv.IOManager( larcv.IOManager.kBOTH, "larcv", larcv.IOManager.kTickBackward )

print "[INPUT: DL MERGED] ",args.input_dlmerged
print "[INPUT: LARMATCH-KPS]  ",args.input_larflow
print "[OUTPUT]    ",args.output

io.add_in_filename(  args.input_dlmerged )
io.add_in_filename(  args.input_larflow )
io.set_data_to_read( larlite.data.kLArFlow3DHit, "larmatch" )
io.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
io.set_data_to_read( larlite.data.kMCShower, "mcreco" )
io.set_data_to_read( larlite.data.kMCTruth,  "generator" )
io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashBeam" )
io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashCosmic" )


iolcv.add_in_file(   args.input_dlmerged )
iolcv.specify_data_read( larcv.kProductImage2D, "wire" );
iolcv.specify_data_read( larcv.kProductImage2D, "thrumu" );
iolcv.specify_data_read( larcv.kProductImage2D, "ancestor" );
iolcv.specify_data_read( larcv.kProductImage2D, "instance" );
iolcv.specify_data_read( larcv.kProductImage2D, "larflow" );
iolcv.specify_data_read( larcv.kProductChStatus, "wire" );
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane0" )
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane1" )
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane2" )
#iolcv.addto_storeonly_list( ... )
iolcv.reverse_all_products()

io.set_out_filename( args.output.replace(".root","_larlite.root") )
iolcv.set_out_file( args.output.replace(".root","_larcv.root") )

io.open()
iolcv.initialize()

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

# ALGORITHMS
recoman = larflow.reco.KPSRecoManager( args.output.replace(".root","_kpsrecomanagerana.root") )
recoman.set_verbosity(1)
if args.ismc:
    recoman.saveEventMCinfo( args.ismc )

io.go_to( args.start_entry )
#io.next_event()
#io.go_to( args.start_entry )
for ientry in xrange( args.start_entry, end_entry ):
    print "[ENTRY ",ientry,"]"
    iolcv.read_entry(ientry)

    recoman.process( iolcv, io )
    recoman.truthAna( iolcv, io )    
    
  #   # make bad channel image
  #   t_badch = time.time()
  #   """
  # std::vector<larcv::Image2D> EmptyChannelAlgo::makeGapChannelImage( const std::vector<larcv::Image2D>& img_v,
  #                                                                    const larcv::EventChStatus& ev_status, int minstatus,
  #                                                                    int nplanes, int start_tick, int nticks, int nchannels,
  #                                                                    int time_downsample_factor, int wire_downsample_factor,
  #                                                                    const float empty_ch_threshold,
  #                                                                    const int max_empty_gap,
  #                                                                    const float empty_ch_max ) {
  #   """
  #   ev_adc = iolcv.get_data(larcv.kProductImage2D, "wire")
  #   ev_chstatus = iolcv.get_data(larcv.kProductChStatus, "wire")
  #   adc_v = ev_adc.Image2DArray()
  #   gapch_v = badchmaker.makeGapChannelImage( adc_v, ev_chstatus,
  #                                             4, 3, 2400, 6*1008, 3456, 6, 1,
  #                                             5.0,
  #                                             50,
  #                                             -1.0 )
    
  #   print("Number of badcv images: ",gapch_v.size())
  #   dt_badch = time.time()-t_badch
  #   ev_badch = iolcv.get_data(larcv.kProductImage2D,"badch")
  #   for iimg in range(gapch_v.size()):
  #       ev_badch.Append( gapch_v[iimg] )
  #   print( "Made EVENT Gap Channel Image: ",gapch_v.front().meta().dump(), " elasped=",dt_badch," secs")        

  #   ev_lfhits = io.get_data( larlite.data.kLArFlow3DHit, "larmatch" )
  #   print("num of hits: ",ev_lfhits.size())

  #   splithits.process( iolcv, io )
  #   print("number of track hits: ",splithits.get_track_hits().size())
    
  #   tracker.process( iolcv, io, kpreco.kpcluster_v )

    io.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    io.next_event()
    iolcv.save_entry()

print "Event Loop finished"
#del kpsrecoman

io.close()
iolcv.finalize()
recoman.write_ana_file()
