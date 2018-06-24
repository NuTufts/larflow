#include <iostream>
#include <string>

// larlite
#include "DataFormat/hit.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"

// larlitecv
#include "Base/DataCoordinator.h"

// #ifdef USE_OPENCV
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #endif

// ContourTools
//#include "ContourTools/ContourShapeMeta.h"
#include "ContourTools/ContourCluster.h"

int main( int nargs, char** argv ) {

  std::cout << "larflow post-processor dev" << std::endl;

  // use hard-coded test-paths for now
  std::string input_larflow_file = "../testdata_tmp/larflow_test_8541376_98.root";
  std::string input_reco2d_file  = "../testdata_tmp/larlite_reco2d_8541376_98.root";

  larlitecv::DataCoordinator dataco;
  dataco.add_inputfile( input_larflow_file, "larcv" );
  dataco.add_inputfile( input_reco2d_file,  "larlite" );
  dataco.initialize();

  // cluster algo
  larlitecv::ContourCluster cluster_algo;
  
  int nentries = dataco.get_nentries( "larcv" );

  for (int ientry=0; ientry<nentries; ientry++) {

    dataco.goto_entry(ientry,"larcv");
  
    // input data
    larcv::EventImage2D* ev_wire      = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "wire");
    larcv::EventImage2D* ev_flow      = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "larflow_y2u");
    const larlite::event_hit&  ev_hit = *((larlite::event_hit*)dataco.get_larlite_data(larlite::data::kHit, "gaushit"));
  
    const std::vector<larcv::Image2D>& wire_v = ev_wire->image2d_array();
    const std::vector<larcv::Image2D>& flow_v = ev_flow->image2d_array();
    
    // get cluster atomics for u and y ADC image

  }
  
  
  
  return 0;

}
