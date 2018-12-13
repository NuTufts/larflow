#include <iostream>
#include <string>

// larlite
#include "LArUtil/Geometry.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"

#include "LArFlowFlashMatch.h"

// ==================================
// FLASHMATCH DEV
// ==================================
// Take in larflow3dhit clusters
// output
// --------
//  -- larlite::larflowclusters with flash match information
//     1) primary solution for each cluster
//     2) scores for other solutions
//     3) flashpe hypo+data for primary solution
//     4) tick of flash
//     5) (real) time of event relative to trigger (usec)
//     6) index of cluster mask object (see next)
//     7) larflow3dhits
//  -- cluster masks
//     1) pixels associated with each larflowcluster
//     2) bounding box
//     3) class index
//  -- image2d tagged image
//     1) 

int main( int nargs, char** argv ) {

  std::string input_cluster = argv[1];
  std::string input_larlite = argv[2];
  std::string input_larcv   = argv[3];
  std::string input_mcinfo  = argv[4];

  std::string output_flashmatch = argv[5];
  std::string output_larcvfile  = argv[6];
  std::string output_anafile    = argv[7];

  std::cout << "=======================================================================================" << std::endl;
  std::cout << " dev_flashmatch" << std::endl;
  std::cout << " --------------" << std::endl;
  std::cout << " input cluster file: " << input_cluster << std::endl;
  std::cout << " input opreco larlite file: " << input_larlite << std::endl;
  std::cout << " input supera larcv2 file: " << input_larcv << std::endl;
  std::cout << " input mcinfo larlite file: " << input_mcinfo << std::endl;
  std::cout << " " << std::endl;
  std::cout << " output flashmatch-larlite file (larflow clusters): " << output_flashmatch << std::endl;
  std::cout << " output flashmatch-larcv file (cluster masks): " << output_larcvfile << std::endl;
  std::cout << " output ana-file (ntuples for tuning/performance analysis): " << output_anafile << std::endl;
  std::cout << "=======================================================================================" << std::endl;
    
  
  // input
  larlite::storage_manager io( larlite::storage_manager::kREAD );
  io.add_in_filename( input_cluster );
  io.add_in_filename( input_larlite );
  io.add_in_filename( input_mcinfo  );
  io.open();

  larcv::IOManager iolarcv( larcv::IOManager::kREAD );
  iolarcv.add_in_file( input_larcv );
  iolarcv.initialize();

  // output
  larlite::storage_manager outlarlite( larlite::storage_manager::kWRITE );
  outlarlite.set_out_filename( output_flashmatch );
  outlarlite.open();

  larcv::IOManager outlarcv( larcv::IOManager::kWRITE );
  outlarcv.set_out_file( output_larcvfile );
  outlarcv.initialize();
  
  int nentries       = io.get_entries();
  int nentries_larcv = iolarcv.get_n_entries();

  std::cout << "larlite entries: " << nentries << std::endl;
  std::cout << "larcv entries: "   << nentries_larcv << std::endl;

  // dump pmt pos
  const larutil::Geometry* geo = larutil::Geometry::GetME();

  std::cout << "--------------" << std::endl;
  std::cout << "PMT POSITIONS" << std::endl;
  std::cout << "--------------" << std::endl;
  for (int ich=0; ich<32; ich++) {
    int opdet = geo->OpDetFromOpChannel(ich);
    double xyz[3];
    geo->GetOpChannelPosition( ich, xyz );
    std::cout << "[ch " << ich << "] (" << xyz[0] << "," << xyz[1] << "," << xyz[2] << ")" << std::endl;
  }
  std::cout << "-------------------" << std::endl;
  

  // algo
  larflow::LArFlowFlashMatch algo;
  algo.saveAnaVariables( output_anafile  );

  int istart = 11;
  int nprocessed = 0;
  for (int ientry=istart; ientry<nentries; ientry++) {

    std::cout << "//////////////////////////////////////" << std::endl;
    std::cout << "[ ENTRY " << ientry << "]" << std::endl;
    std::cout << "//////////////////////////////////////" << std::endl;
    
    io.go_to( ientry );
    iolarcv.read_entry(ientry);
  
    larlite::event_larflowcluster* ev_cluster = (larlite::event_larflowcluster*)io.get_data( larlite::data::kLArFlowCluster, "flowtruthclusters" );

    larlite::event_opflash* ev_opflash_beam   = (larlite::event_opflash*)io.get_data( larlite::data::kOpFlash, "simpleFlashBeam" );
    larlite::event_opflash* ev_opflash_cosmic = (larlite::event_opflash*)io.get_data( larlite::data::kOpFlash, "simpleFlashCosmic" );

    larcv::EventImage2D*  ev_larcv  = (larcv::EventImage2D*) iolarcv.get_data( "image2d",  "wire" );
    larcv::EventChStatus* ev_status = (larcv::EventChStatus*)iolarcv.get_data( "chstatus", "wire" );

    std::cout << "number of charge clusters: " << ev_cluster->size() << std::endl;
    std::cout << "number of beam flashes:    " << ev_opflash_beam->size() << std::endl;
    std::cout << "number of cosmic flashes:  " << ev_opflash_cosmic->size() << std::endl;
    std::cout << "number of adc images:      " << ev_larcv->as_vector().size() << std::endl;
    
    std::cout << "== [cosmic op flashes] ======================" << std::endl;
    for ( int icosmic = 0; icosmic < (int)ev_opflash_cosmic->size(); icosmic++ )
      std::cout << " cosmic[" << icosmic << "] " << ev_opflash_cosmic->at(icosmic).TotalPE() << " nopdets=" << ev_opflash_cosmic->at(icosmic).nOpDets() << std::endl;
    
    larlite::event_mctrack*  ev_mctrack  = (larlite::event_mctrack*) io.get_data( larlite::data::kMCTrack,  "mcreco" );
    larlite::event_mcshower* ev_mcshower = (larlite::event_mcshower*)io.get_data( larlite::data::kMCShower, "mcreco" );
    std::cout << "== [MC Truth] ======================" << std::endl;
    std::cout << "  number of mctracks: "  << ev_mctrack->size()  << std::endl;
    std::cout << "  number of mcshowers: " << ev_mcshower->size() << std::endl;    

    // prep algo
    algo.loadChStatus( ev_status );    
    algo.loadMCTrackInfo( *ev_mctrack, *ev_mcshower, true );
    algo.setRSE( io.run_id(), io.subrun_id(), io.event_id() );

    algo.dumpPrefitImages(false);
    algo.dumpPostfitImages(false);
    algo.match( *ev_opflash_beam, *ev_opflash_cosmic, *ev_cluster, ev_larcv->as_vector() );
    std::cout << "== [dev_flashmatch][INFO] result run ==========" << std::endl;

    // save output products:
    //  larflowclusters and clustermasks
    // ------------------------------------------
    larlite::event_larflowcluster* match_lfcluster  = (larlite::event_larflowcluster*) outlarlite.get_data( larlite::data::kLArFlowCluster, "allflashmatched" );
    larlite::event_larflowcluster* intime_lfcluster = (larlite::event_larflowcluster*) outlarlite.get_data( larlite::data::kLArFlowCluster, "intimeflashmatched" );

    for ( auto& lfcluster : algo._final_lfcluster_v )
      match_lfcluster->emplace_back( std::move(lfcluster) );
    for ( auto& lfcluster : algo._intime_lfcluster_v )
      intime_lfcluster->emplace_back( std::move(lfcluster) );

    larcv::EventClusterMask* match_clustermask  = (larcv::EventClusterMask*) outlarcv.get_data( "clustermask", "allflashmatched" );
    larcv::EventClusterMask* intime_clustermask = (larcv::EventClusterMask*) outlarcv.get_data( "clustermask", "intimeflashmatched" );
    for ( auto& mask_v : algo._final_clustermask_v )
      match_clustermask->emplace( std::move( mask_v ) );
    for ( auto& mask_v : algo._intime_clustermask_v )
      intime_clustermask->emplace( std::move(mask_v) );
    
    // save some output for analysis, cut tuning
    // ------------------------------------------
    // ana variables for analysis and setting parameters

    // mask imnon-in-time flashmatched clusters to mask ADC image

    // get flash-matched clusters with correct x-position

    // 
    
    outlarlite.set_id( io.run_id(), io.subrun_id(), io.event_id() );
    outlarcv.set_id( io.run_id(), io.subrun_id(), io.event_id() );

    outlarlite.next_event(); // saves and clears
    outlarcv.save_entry();
    std::cout << "== [dev_flashmatch][INFO] results stored ==========" << std::endl;
    
    algo.clearEvent();

    std::cout << "== [dev_flashmatch][INFO] algo cleared ==========" << std::endl;    
    nprocessed++;
    
    //if ( nprocessed>=1 )
    //break;
  }

  std::cout << "== [dev_flashmatch] write ana file ===============" << std::endl;
  algo.writeAnaFile();

  std::cout << "== [dev_flashmatch] close files and clean up =========" << std::endl;  
  outlarlite.close();
  io.close();
  iolarcv.finalize();
  
  return 0;
}
