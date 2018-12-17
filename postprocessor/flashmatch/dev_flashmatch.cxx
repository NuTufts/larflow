#include <iostream>
#include <string>

// larlite
#include "LArUtil/Geometry.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/pixelmask.h"

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

  int istart = 0;
  int nprocessed = 0;
  for (int ientry=istart; ientry<nentries; ientry++) {

    std::cout << "//////////////////////////////////////" << std::endl;
    std::cout << "[ ENTRY " << ientry << "]" << std::endl;
    std::cout << "//////////////////////////////////////" << std::endl;
    
    io.go_to( ientry );
    iolarcv.read_entry(ientry);

    // get input
    // ---------
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

    // get output containers
    // ----------------------
    larlite::event_larflowcluster* match_lfcluster  = (larlite::event_larflowcluster*) outlarlite.get_data( larlite::data::kLArFlowCluster, "allflashmatched" );
    larlite::event_larflowcluster* intime_lfcluster = (larlite::event_larflowcluster*) outlarlite.get_data( larlite::data::kLArFlowCluster, "intimeflashmatched" );

    larlite::event_pixelmask* match_clustermask[3]  = { nullptr, nullptr, nullptr };
    larlite::event_pixelmask* intime_clustermask[3] = { nullptr, nullptr, nullptr };
    for ( size_t p=0; p<3; p++ ) {
      char branchname1[100];
      sprintf( branchname1, "allflashmatchedp%d", (int)p );
      match_clustermask[p] = (larlite::event_pixelmask*) outlarlite.get_data( larlite::data::kPixelMask, branchname1 );

      char branchname2[100];
      sprintf( branchname2, "intimeflashmatchedp%d", (int)p );
      match_clustermask[p] = (larlite::event_pixelmask*) outlarlite.get_data( larlite::data::kPixelMask, branchname2 );
    }

    // clean up clusters
    // -----------------
    int nclusters = ev_cluster->size();
    larlite::event_larflowcluster filtered_cluster_v;
    const larcv::ImageMeta& meta = ev_larcv->as_vector().at(2).meta();
    for ( auto& cluster : *ev_cluster ) {
      int ngood = 0;
      for ( auto const& hit : cluster ) {
	bool isok = true;
	for (int i=0; i<3; i++) {
	  if ( std::isnan(hit[i]) ) isok = false;
	}
	if ( hit[1]<-118.0 || hit[1]>118 )  isok = false;
	else if ( hit[2]<0 || hit[2]>1050 ) isok = false;

	if ( !meta.contains(hit.tick,hit.srcwire) ) isok = false;
	
	if (isok) ngood++;
      }//end of cluster loop
      if (ngood>5)
	filtered_cluster_v.emplace_back( std::move(cluster) );
    }
    std::cout << "== [dev_flashmatch][INFO] filtered clusters " << filtered_cluster_v.size() << " out of " << nclusters << " ======" << std::endl;
    
    // prep algo
    algo.loadChStatus( ev_status );    
    algo.loadMCTrackInfo( *ev_mctrack, *ev_mcshower, true );
    algo.setRSE( io.run_id(), io.subrun_id(), io.event_id() );

    algo.dumpPrefitImages(false);
    algo.dumpPostfitImages(false);
    algo.match( *ev_opflash_beam, *ev_opflash_cosmic, filtered_cluster_v, ev_larcv->as_vector() );
    std::cout << "== [dev_flashmatch][INFO] result run ==========" << std::endl;

    // save output products:
    //  larflowclusters and clustermasks
    // ------------------------------------------

    for ( auto& lfcluster : algo._final_lfcluster_v ) {
      match_lfcluster->emplace_back( std::move(lfcluster) );
    }
    for ( auto& lfcluster : algo._intime_lfcluster_v ) {
      intime_lfcluster->emplace_back( std::move(lfcluster) );
    }
    
    for ( auto& mask_v : algo._final_clustermask_v ) {
      for ( size_t p=0; p<mask_v.size(); p++ )
	match_clustermask[p]->emplace_back( std::move( mask_v[p]) );
    }
    for ( auto& mask_v : algo._intime_clustermask_v ) {
      for ( size_t p=0; p<mask_v.size(); p++ )
	intime_clustermask[p]->emplace_back( std::move(mask_v[p]) );
    }
    
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
