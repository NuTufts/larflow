#include <iostream>
#include <vector>
#include <string>

// larlite
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflowcluster.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventClusterMask.h"

// postprocessor/cluster
#include "MRCNNClusterMaker.h"


/** 
 * Simple routine to run one mask for testing and visualize output
 *
 */
int main(int nargs, char** argv ) {

  std::string input_larlite_larflowhits = argv[1];
  std::string input_larcv_supera        = argv[2];
  std::string input_larcv_mrcnnout      = argv[3];

  std::string output_larlite_clusters = argv[4];

  larlite::storage_manager iolarlite( larlite::storage_manager::kREAD );
  iolarlite.add_in_filename( input_larlite_larflowhits );
  iolarlite.open();
  
  larcv::IOManager iolarcv_supera( larcv::IOManager::kREAD, "supera", larcv::IOManager::kTickBackward );
  iolarcv_supera.add_in_file( input_larcv_supera );
  iolarcv_supera.initialize();
  
  larcv::IOManager iolarcv_mrcnn( larcv::IOManager::kREAD );
  iolarcv_mrcnn.add_in_file( input_larcv_mrcnnout );  
  iolarcv_mrcnn.initialize();

  larlite::storage_manager outlarlite( larlite::storage_manager::kWRITE );
  outlarlite.set_out_filename( output_larlite_clusters );
  outlarlite.open();

  // algo instance
  larflow::MRCNNClusterMaker algo;
  
  // Event loop
  int nentries = iolarlite.get_entries();

  for ( int ientry=0; ientry<nentries; ientry++ ) {

    std::cout << "==============" << std::endl;
    std::cout << " Entry [" << ientry << "]" << std::endl;
    std::cout << "==============" << std::endl;
    iolarlite.go_to( ientry );
    iolarcv_supera.read_entry( ientry );
    iolarcv_mrcnn.read_entry( ientry );    

    // adc images
    larcv::EventImage2D* evimg = (larcv::EventImage2D*)iolarcv_supera.get_data( larcv::kProductImage2D, "wire" );
    const std::vector< larcv::Image2D >& img_v = evimg->as_vector();
    std::cout << "  adc images: " << img_v.size() << std::endl;

    // clusters
    larlite::event_larflow3dhit* evcluster_v =
      (larlite::event_larflow3dhit*)iolarlite.get_data( larlite::data::kLArFlow3DHit, "flowhits" );
    std::cout << "  flowhits: " << evcluster_v->size() << std::endl;

    // pixel masks (for each plane)
    larcv::EventClusterMask* evmask_v = (larcv::EventClusterMask*)iolarcv_mrcnn.get_data( larcv::kProductClusterMask, "mrcnn_masks" );
    std::cout << "  masks: " << evmask_v->as_vector().size()  << std::endl;
    for ( size_t p=0; p<3; p++ ) {
      std::cout << "    masks p[" << p << "]: " << evmask_v->as_vector()[p].size()  << std::endl;    
    }

    std::vector<larlite::larflowcluster> clusters_v = algo.makeSimpleClusters( evmask_v->as_vector().at(2),
                                                                               *evcluster_v,
                                                                               img_v );
    
    // output containers
    auto evout_clusters = (larlite::event_larflowcluster*)outlarlite.get_data( larlite::data::kLArFlowCluster, "rawmcrcnn" );
    // larlite::event_pixelmask* evout_origmask_v[3];
    // larlite::event_pixelmask* evout_filledmask_v[3];
    // for ( size_t p=0; p<3; p++ ) {
    //   char treename1[200];
    //   sprintf( treename1, "intimeflashmatchedp%d", (int)p );
    //   evout_origmask_v[p] = (larlite::event_pixelmask*)outlarlite.get_data( larlite::data::kPixelMask, treename1 );

    //   char treename2[200];
    //   sprintf( treename2, "intimefilledp%d", (int)p );
    //   evout_filledmask_v[p] = (larlite::event_pixelmask*)outlarlite.get_data( larlite::data::kPixelMask, treename2 );
    // }

    // // original masks
    // for ( size_t p=0; p<3; p++ ) {
    //   for ( auto const& mask : *evmask_vv[p] )
    //     evout_origmask_v[p]->push_back( mask );
    // }

    // // filled-in masks
    // for ( auto& mask_v : outmasks_vv ) {
    //   for ( size_t p=0; p<3; p++ ) {
    //     evout_filledmask_v[p]->emplace_back( std::move( mask_v.at(p) ) );
    //   }
    // }

    // larlite clusters
    for ( auto& cluster : clusters_v  ) {
      evout_clusters->emplace_back( std::move(cluster) );
    }
    
    // set id
    outlarlite.set_id( iolarlite.run_id(), iolarlite.subrun_id(), iolarlite.event_id() );
    outlarlite.next_event();

  }//end of entry loop

  outlarlite.close();
  
  iolarlite.close();
  iolarcv_supera.finalize();
  iolarcv_mrcnn.finalize();  
  
  std::cout << "End of dev" << std::endl;
}
