#include <iostream>

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventSparseImage.h"

// larlite
#include "DataFormat/larflow3dhit.h"

// larflow
#include "larflow/FlowContourMatch/FlowContourMatch.h"

int main( int nargs, char** argv ) {

  std::cout << "================================" << std::endl;
  std::cout << "sparseflow2hits" << std::endl;
  std::cout << "--------------------------------" << std::endl;

  larcv::IOManager io(larcv::IOManager::kREAD,"SparseInput");
  io.add_in_file( "test_sparseout_stitched.root" );
  io.initialize();

  int nentries = io.get_n_entries();

  std::cout << "Number of entries: " << nentries << std::endl;

  for ( size_t i=0; i<nentries; i++ ) {
    io.read_entry(i);
    larcv::EventSparseImage* ev_crop_dualflow = (larcv::EventSparseImage*)io.get_data(larcv::kProductSparseImage,"cropdualflow");
    auto const& dualflow_v = ev_crop_dualflow->SparseImageArray();
    std::cout << "ENTRY[" << i << "] num dualflow crops: " << dualflow_v.size() << std::endl;

    // eventual goal
    //std::vector<larflow3dhit> flowhit = makeFlowHitsFromSparseCrops( dualflow_v, ... );
    
  }
  
  return 0;
}
