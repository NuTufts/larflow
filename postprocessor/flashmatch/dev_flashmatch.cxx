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

int main( int nargs, char** argv ) {

  std::string input_cluster = argv[1];
  std::string input_larlite = argv[2];
  std::string input_larcv   = argv[3];
  std::string input_mcinfo  = argv[4];

  std::string output_flashmatch = argv[5];
  
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
  io.set_out_filename( output_flashmatch );
  io.open();
  
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
  algo.saveAnaVariables( "out_larflow_flashmatch_ana.root" );
  
  int nprocessed = 0;
  for (int ientry=0; ientry<nentries; ientry++) {
    
    io.go_to( ientry );
  
    larlite::event_larflowcluster* ev_cluster = (larlite::event_larflowcluster*)io.get_data( larlite::data::kLArFlowCluster, "flowtruthclusters" );


    larlite::event_opflash* ev_opflash_beam   = (larlite::event_opflash*)io.get_data( larlite::data::kOpFlash, "simpleFlashBeam" );
    larlite::event_opflash* ev_opflash_cosmic = (larlite::event_opflash*)io.get_data( larlite::data::kOpFlash, "simpleFlashCosmic" );

    larcv::EventImage2D* ev_larcv   = (larcv::EventImage2D*) iolarcv.get_data( "image2d",  "wire" );
    larcv::EventChStatus* ev_status = (larcv::EventChStatus*)iolarcv.get_data( "chstatus", "wire" );

    std::cout << "number of clusters: " << ev_cluster->size() << std::endl;
    std::cout << "number of beam flashes: " << ev_opflash_beam->size() << std::endl;
    std::cout << "number of cosmic flashes: " << ev_opflash_cosmic->size() << std::endl;
    std::cout << "number of images: " << ev_larcv->as_vector().size() << std::endl;
    
    std::cout << "[cosmic hits] ======================" << std::endl;
    for ( int icosmic = 0; icosmic < (int)ev_opflash_cosmic->size(); icosmic++ )
      std::cout << " cosmic[" << icosmic << "] " << ev_opflash_cosmic->at(icosmic).TotalPE() << " nopdets=" << ev_opflash_cosmic->at(icosmic).nOpDets() << std::endl;
    
    larlite::event_mctrack*  ev_mctrack  = (larlite::event_mctrack*) io.get_data( larlite::data::kMCTrack,  "mcreco" );
    larlite::event_mcshower* ev_mcshower = (larlite::event_mcshower*)io.get_data( larlite::data::kMCShower, "mcreco" );    
    std::cout << "number of mctracks: "  << ev_mctrack->size()  << std::endl;
    std::cout << "number of mcshowers: " << ev_mcshower->size() << std::endl;    

    // prep algo
    algo.loadChStatus( ev_status );    
    algo.loadMCTrackInfo( *ev_mctrack, *ev_mcshower, true );
    algo.setRSE( io.run_id(), io.subrun_id(), io.event_id() );
    
    algo.match( *ev_opflash_beam, *ev_opflash_cosmic, *ev_cluster, ev_larcv->as_vector() );
    std::cout << "[dev_flashmatch][INFO] result run" << std::endl;
    
    algo.saveAnaMatchData();
    
    outlarlite.set_id( io.run_id(), io.subrun_id(), io.event_id() );
    outlarlite.next_event(); // saves and clears

    algo.clearEvent();    
    nprocessed++;
    
    if ( nprocessed==1 )
      break;
  }

  algo.writeAnaFile();
  
  outlarlite.close();
  io.close();
  iolarcv.finalize();
  
  return 0;
}
