#include <iostream>

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"

#include "TruthCluster.h"

#include "CilantroPCA.h"

int main( int nargs, char** argv ) {

  std::cout << "Dev Truth Cluster" << std::endl;

  std::string larflow_input   = argv[1];
  // std::string larcv_input     = argv[2];
  // std::string opreco_input    = argv[3];
  // std::string mcinfo_input    = argv[4];

  // input
  larlite::storage_manager io( larlite::storage_manager::kREAD );
  io.add_in_filename( larflow_input );
  io.open();

  // eventually want to merge inputs here
  // larlite::storage_manager io_larlite( larlite::storage_manager::kREAD );
  // io_larlite.add_in_filename( opreco_input );
  // io_larlite.add_in_filename( mcinfo_input );
  // io_larlite.open();

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
    larlite::event_pcaxis* ev_outpca             = (larlite::event_pcaxis*)io_out.get_data( larlite::data::kPCAxis,"flowtruthclusters");    
    
    for ( auto& hit_v : clusters ) {
      larlite::larflowcluster flowcluster;
      flowcluster.reserve( hit_v.size() );
      for ( auto const& phit : hit_v ) {
	flowcluster.push_back( *phit );
      }
      larflow::CilantroPCA pca( flowcluster );
      larlite::pcaxis pcainfo = pca.getpcaxis();
      
      ev_outpca->emplace_back( std::move(pcainfo) );
      ev_outcluster->emplace_back( std::move(flowcluster) );
    }

    io_out.set_id( io.run_id(), io.subrun_id(), io.event_id() );

    io_out.next_event();
  }

  std::cout << "finished" << std::endl;
  io_out.close();
  io.close();
  
}
