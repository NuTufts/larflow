#include <iostream>

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"

#include "TruthCluster.h"

int main( int nargs, char** argv ) {

  std::cout << "Dev Truth Cluster" << std::endl;

  std::string inputfile = argv[1];

  // input
  larlite::storage_manager io( larlite::storage_manager::kREAD );
  io.add_in_filename( inputfile );
  io.open();

  // output
  larlite::storage_manager io_out( larlite::storage_manager::kWRITE );
  io_out.set_out_filename( "output_dev_truthcluster.root" );
  io_out.open();

  // Truth Cluster
  larflow::TruthCluster clusteralgo;

  
  int nentries = io.get_entries();

  for (int ientry=0; ientry<nentries; ientry++) {
    
    io.go_to( ientry );
  
    larlite::event_larflow3dhit* ev_hits = (larlite::event_larflow3dhit*)io.get_data( larlite::data::kLArFlow3DHit, "flowhits" );

    std::cout << "number of hits: " << ev_hits->size() << std::endl;
    std::vector< std::vector<const larlite::larflow3dhit*> > clusters = clusteralgo.clusterHits( *ev_hits );

    std::cout << "truthcluster returned with " << clusters.size() << " clusters" << std::endl;
    larlite::event_larflowcluster* ev_outcluster = (larlite::event_larflowcluster*)io_out.get_data( larlite::data::kLArFlowCluster,"flowtruthclusters");
    
    for ( auto& hit_v : clusters ) {
      larlite::larflowcluster flowcluster;
      flowcluster.reserve( hit_v.size() );
      for ( auto const& phit : hit_v ) {
	flowcluster.push_back( *phit );
      }
      ev_outcluster->emplace_back( std::move(flowcluster) );
    }

    io_out.set_id( io.run_id(), io.subrun_id(), io.event_id() );

    io_out.next_event();
    break;
  }

  std::cout << "finished" << std::endl;
  io_out.close();
  io.close();
  
}
