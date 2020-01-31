import os,sys,time

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

#print larflow.reco.cluster_t
#print larflow.reco.cluster_larflow3dhits


io = larlite.storage_manager( larlite.storage_manager.kREAD )
#io.add_in_filename(  "larflow_reco_extbnb_run3.root" )
io.add_in_filename(  "merged_dlreco_extbnb_run3_821c2dfc-96b0-4725-a187-6b628cbbbea3.root" )
#io.set_out_filename( "crttrack_match_reco_extbnb_run3.root" )
io.open()

outio = larlite.storage_manager( larlite.storage_manager.kWRITE )
outio.set_out_filename( "crttrack_match_reco_extbnb_run3.root" )
outio.open()

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
iolcv.add_in_file( "merged_dlreco_extbnb_run3_821c2dfc-96b0-4725-a187-6b628cbbbea3.root" )
iolcv.reverse_all_products()
iolcv.initialize()

crtmatch = larflow.crtmatch.CRTTrackMatch()
crtmatch.set_max_iters( 25 )
crtmatch.make_debug_images( False )

nentries = iolcv.get_n_entries()
print "Number of entries: ",nentries
#nentries = 5

print "Start loop."

#outio.next_event()
for ientry in xrange( 0, nentries ):
#for ientry in xrange( 0, 2 ):

    io.go_to(ientry)    
    iolcv.read_entry(ientry)

    #crtmatch.clear()    

    dtload = time.time()
    
    # get opflashes
    beamflash   = io.get_data( larlite.data.kOpFlash, "simpleFlashBeam" )
    cosmicflash = io.get_data( larlite.data.kOpFlash, "simpleFlashCosmic" )
    #crtmatch.addIntimeOpFlashes( beamflash )
    #crtmatch.addCosmicOpFlashes( cosmicflash )

    # get crt hits
    #crthit_v   = io.get_data( larlite.data.kCRTHit, "crthitcorr" )
    #crttrack_v = io.get_data( larlite.data.kCRTTrack, "crttrack" )
    #crtmatch.addCRThits(   crthit_v )
    #crtmatch.addCRTtracks( crttrack_v )

    # get clusters
    #lfclusters_v = io.get_data( larlite.data.kLArFlowCluster, "pcacluster" )
    #pcaxis_v     = io.get_data( larlite.data.kPCAxis, "pcacluster" )
    #crtmatch.addLArFlowClusters( lfclusters_v, pcaxis_v )
    
    dtload = time.time()-dtload
    print "time to load inputs: ",dtload," secs"

    crtmatch.process( iolcv, io )

    crtmatch.save_to_file( outio )
    outio.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    outio.next_event()
    #break


io.close()
iolcv.finalize()
outio.close()
