#include <iostream>
#include <string>
#include "pangolin/utils/file_utils.h"

// larlite
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflowcluster.h"

// Cilantro
#include "cilantro/point_cloud.hpp"

/**
 * Simple utility to convert a larflow cluster object 
 * into the Pangolin PLY format. For visualizing Cilantro 
 * fun.
 *
 */
int main( int nargs, char** argv ) {

  std::string input_larlite = argv[1];
  std::string producer      = argv[2];
  int         entry         = std::atoi(argv[3]);
  int         instance      = std::atoi(argv[4]);

  larlite::storage_manager io( larlite::storage_manager::kREAD );
  io.add_in_filename( input_larlite );
  io.open();
  
  bool ok = io.go_to( entry );

  if ( !ok ) {
    std::cout << "Entry " << entry << " could not load" << std::endl;
    return 1;
  }

  auto ev_cluster = (larlite::event_larflowcluster*)io.get_data( larlite::data::kLArFlowCluster, producer );
  if ( ev_cluster->size()<instance ) {
    std::cout << "Producer " << producer << " does not contain an instance " << instance << std::endl;
    return 1;
  }

  auto const& cluster = ev_cluster->at(instance);
  int npts = cluster.size();
  
  std::cout << "LArFlowCluster instance " << instance << " has " << npts << " points" << std::endl;
  
  // define a cilantro point cloud
  cilantro::PointCloud3f cloud;
  
  
  return 0;
}
