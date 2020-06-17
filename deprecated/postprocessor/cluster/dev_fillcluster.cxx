#include <iostream>
#include <vector>
#include <string>

// larlite
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pixelmask.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"

// postprocessor/cluster
#include "FillPixelMask.h"


/** 
 * Simple routine to run one mask for testing and visualize output
 *
 */
int main(int nargs, char** argv ) {

  std::string input_larlite = argv[1];
  std::string input_larcv   = argv[2];

  std::string output_larlite = argv[3];

  larlite::storage_manager iolarlite( larlite::storage_manager::kREAD );
  iolarlite.add_in_filename( input_larlite );
  iolarlite.open();

  larcv::IOManager iolarcv( larcv::IOManager::kREAD );
  iolarcv.add_in_file( input_larcv );
  iolarcv.initialize();

  larlite::storage_manager outlarlite( larlite::storage_manager::kWRITE );
  outlarlite.set_out_filename( output_larlite );
  outlarlite.open();

  // algo instance
  larflow::FillPixelMask algo;
  algo.setDebugMode(false);
  
  // Event loop
  int nentries = iolarlite.get_entries();

  for ( int ientry=0; ientry<nentries; ientry++ ) {
    
    iolarlite.go_to( ientry );
    iolarcv.read_entry( ientry );

    // adc images
    larcv::EventImage2D* evimg = (larcv::EventImage2D*)iolarcv.get_data( larcv::kProductImage2D, "wire" );
    const std::vector< larcv::Image2D >& img_v = evimg->as_vector();

    // clusters
    larlite::event_larflowcluster* evcluster_v =
      (larlite::event_larflowcluster*)iolarlite.get_data( larlite::data::kLArFlowCluster, "intimeflashmatched" );

    // pixel masks (for each plane)
    std::vector< const larlite::event_pixelmask* > evmask_vv;    
    //std::vector< larlite::pixelmask > mask_v;
    for ( size_t p=0; p<img_v.size(); p++ ) {
      char treename[50];
      sprintf( treename, "intimeflashmatchedp%d", (int)p );
      const larlite::event_pixelmask* evmask_v = (larlite::event_pixelmask*)iolarlite.get_data( larlite::data::kPixelMask, treename );
      evmask_vv.push_back( evmask_v );
      //mask_v.push_back( evmask_v->at(0) );
    }

    // run code
    //  (one instance example)
    //std::vector< std::vector< larlite::pixelmask > > outmasks_vv = algo.fillMask( img_v, evcluster_v->at(0), mask_v );

    // all instances
    std::vector< std::vector< larlite::pixelmask > > outmasks_vv = algo.fillMasks( img_v, *evcluster_v, evmask_vv );

    // output containers
    auto evout_clusters = (larlite::event_larflowcluster*)outlarlite.get_data( larlite::data::kLArFlowCluster, "intimeflashmatched" );
    larlite::event_pixelmask* evout_origmask_v[3];
    larlite::event_pixelmask* evout_filledmask_v[3];
    for ( size_t p=0; p<3; p++ ) {
      char treename1[200];
      sprintf( treename1, "intimeflashmatchedp%d", (int)p );
      evout_origmask_v[p] = (larlite::event_pixelmask*)outlarlite.get_data( larlite::data::kPixelMask, treename1 );

      char treename2[200];
      sprintf( treename2, "intimefilledp%d", (int)p );
      evout_filledmask_v[p] = (larlite::event_pixelmask*)outlarlite.get_data( larlite::data::kPixelMask, treename2 );
    }

    // original masks
    for ( size_t p=0; p<3; p++ ) {
      for ( auto const& mask : *evmask_vv[p] )
	evout_origmask_v[p]->push_back( mask );
    }

    // filled-in masks
    for ( auto& mask_v : outmasks_vv ) {
      for ( size_t p=0; p<3; p++ ) {
	evout_filledmask_v[p]->emplace_back( std::move( mask_v.at(p) ) );
      }
    }

    // larlite clusters
    for ( auto& cluster : *evcluster_v  ) {
      evout_clusters->emplace_back( std::move(cluster) );
    }

    // set id
    outlarlite.set_id( iolarlite.run_id(), iolarlite.subrun_id(), iolarlite.event_id() );

    outlarlite.next_event();

  }//end of entry loop

  outlarlite.close();
  
  iolarlite.close();
  iolarcv.finalize();
  
  std::cout << "End of dev" << std::endl;
}
