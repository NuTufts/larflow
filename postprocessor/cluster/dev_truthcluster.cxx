#include <iostream>

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"

#include "TruthCluster.h"

int main( int nargs, char** argv ) {

  std::cout << "Dev Truth Cluster" << std::endl;

  std::string inputfile = argv[1];

  larlite::storage_manager io( larlite::storage_manager::kREAD );
  io.add_in_filename( inputfile );
  io.open();

  // Truth Cluster
  larflow::TruthCluster clusteralgo;

  
  int nentries = io.get_entries();

  for (int ientry=0; ientry<nentries; ientry++) {
    
    io.go_to( ientry );
  
    larlite::event_larflow3dhit* ev_hits = (larlite::event_larflow3dhit*)io.get_data( larlite::data::kLArFlow3DHit, "flowhits" );

    std::cout << "number of hits: " << ev_hits->size() << std::endl;
    std::vector< std::vector<larlite::larflow3dhit*> > clusters = clusteralgo.clusterHits( *ev_hits );
    
  }
  
}
