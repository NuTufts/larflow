#include <iostream>
#include <string>

// larlite
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "LArFlowFlashMatch.h"

int main( int nargs, char** argv ) {

  std::string input_cluster = argv[1];
  std::string input_opflash = argv[2];
  std::string input_larcv   = argv[3];

  // input
  larlite::storage_manager io( larlite::storage_manager::kREAD );
  io.add_in_filename( input_cluster );
  io.add_in_filename( input_opflash );  
  io.open();

  larcv::IOManager iolarcv( larcv::IOManager::kREAD );
  iolarcv.add_in_file( input_larcv );
  iolarcv.initialize();

  int nentries = io.get_entries();
  int nentries_larcv = iolarcv.get_n_entries();

  std::cout << "larlite entries: " << nentries << std::endl;
  std::cout << "larcv entries: " << nentries_larcv << std::endl;

  // algo
  larflow::LArFlowFlashMatch algo;

  for (int ientry=0; ientry<nentries; ientry++) {
    
    io.go_to( ientry );
  
    larlite::event_larflowcluster* ev_cluster = (larlite::event_larflowcluster*)io.get_data( larlite::data::kLArFlowCluster, "flowtruthclusters" );


    larlite::event_opflash* ev_opflash_beam   = (larlite::event_opflash*)io.get_data( larlite::data::kOpFlash, "simpleFlashBeam" );
    larlite::event_opflash* ev_opflash_cosmic = (larlite::event_opflash*)io.get_data( larlite::data::kOpFlash, "simpleFlashCosmic" );

    larcv::EventImage2D* ev_larcv = (larcv::EventImage2D*)iolarcv.get_data( "image2d", "wire" );

    std::cout << "number of clusters: " << ev_cluster->size() << std::endl;
    std::cout << "number of beam flashes: " << ev_opflash_beam->size() << std::endl;
    std::cout << "number of cosmic flashes: " << ev_opflash_cosmic->size() << std::endl;
    std::cout << "number of images: " << ev_larcv->as_vector().size() << std::endl;
    
    larflow::LArFlowFlashMatch::Results_t result = algo.match( *ev_opflash_beam, *ev_opflash_cosmic, *ev_cluster, ev_larcv->as_vector() );

    break;
  }
  
  return 0;
}
