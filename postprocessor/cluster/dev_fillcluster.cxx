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

  larlite::storage_manager iolarlite( larlite::storage_manager::kREAD );
  iolarlite.add_in_filename( input_larlite );
  iolarlite.open();

  larcv::IOManager iolarcv( larcv::IOManager::kREAD );
  iolarcv.add_in_file( input_larcv );
  iolarcv.initialize();

  // algo instance
  larflow::FillPixelMask algo;

  // get one instance of object
  iolarlite.go_to( 1 );
  iolarcv.read_entry( 1 );

  // adc images
  larcv::EventImage2D* evimg = (larcv::EventImage2D*)iolarcv.get_data( "image2d", "wire" );
  const std::vector< larcv::Image2D >& img_v = evimg->as_vector();

  // clusters
  larlite::event_larflowcluster* evcluster_v = (larlite::event_larflowcluster*)iolarlite.get_data( larlite::data::kPixelMask, "intimeflashmatched" );

  // pixel masks (for each plane)
  std::vector< const larlite::event_pixelmask* > evmask_vv;
  std::vector< larlite::pixelmask > mask_v;
  for ( size_t p=0; p<img_v.size(); p++ ) {
    char treename[50];
    sprintf( treename, "intimeflashmatchedp%d", (int)p );
    const larlite::event_pixelmask* evmask_v = (larlite::event_pixelmask*)iolarlite.get_data( larlite::data::kPixelMask, treename );
    evmask_vv.push_back( evmask_v );
    mask_v.push_back( evmask_v->at(0) );
  }

  // ready
  std::vector< larlite::pixelmask > outmask_v = algo.fillMask( img_v, evcluster_v->at(0), mask_v );

  // visualize man
  
}
