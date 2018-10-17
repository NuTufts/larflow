#include <iostream>

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"

#include "TruthCluster.h"

#include "CilantroPCA.h"
#include "CoreFilter.h"

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
  io_out.set_out_filename( "output_dev_truthcluster_truepixflow.root" );
  //io_out.set_out_filename( "output_dev_truthcluster_recopixflow.root" );
  io_out.open();

  // Truth Cluster
  larflow::TruthCluster clusteralgo;
  
  int nentries = io.get_entries();

  for (int ientry=0; ientry<nentries; ientry++) {
    
    io.go_to( ientry );
    
    larlite::event_larflow3dhit* ev_hits = (larlite::event_larflow3dhit*)io.get_data( larlite::data::kLArFlow3DHit, "flowhits" );

    std::cout << "number of hits: " << ev_hits->size() << std::endl;
    std::vector< std::vector<const larlite::larflow3dhit*> > clusters = clusteralgo.clusterHits( *ev_hits, true );

    std::cout << "truthcluster returned with " << clusters.size() << " clusters" << std::endl;
    larlite::event_larflowcluster* ev_outcluster = (larlite::event_larflowcluster*)io_out.get_data( larlite::data::kLArFlowCluster,"flowtruthclusters");
    larlite::event_larflowcluster* ev_outcore    = (larlite::event_larflowcluster*)io_out.get_data( larlite::data::kLArFlowCluster,"flowtruthcore"); // after dbscan to identify core
    larlite::event_pcaxis* ev_outpca             = (larlite::event_pcaxis*)io_out.get_data( larlite::data::kPCAxis,"flowtruthclusters");
    larlite::event_pcaxis* ev_outcorepca         = (larlite::event_pcaxis*)io_out.get_data( larlite::data::kPCAxis,"flowtruthcore");
    
    for ( auto& hit_v : clusters ) {
      larlite::larflowcluster flowcluster;
      flowcluster.reserve( hit_v.size() );
      std::cout << "truth cluster has " << hit_v.size() << " hits" << std::endl;
      for ( auto const& phit : hit_v ) {
	flowcluster.push_back( *phit );
	//std::cout << " truthhit (" << (*phit)[0] << "," << (*phit)[1] << "," << (*phit)[2] << ")" << std::endl;
      }
      larflow::CilantroPCA pca( flowcluster );
      larlite::pcaxis pcainfo = pca.getpcaxis();

      // calculate core
      larflow::CoreFilter corealgo( flowcluster, 3, 15.0 );
      larlite::larflowcluster corecluster = corealgo.getCore();
      larflow::CilantroPCA pcacore( corecluster );
      larlite::pcaxis pcacoreinfo = pcacore.getpcaxis();
            
      ev_outpca->emplace_back( std::move(pcainfo) );
      ev_outcluster->emplace_back( std::move(flowcluster) );
      if ( corecluster.size()>0 ) {
	std::cout << "core of cluster has " << corecluster.size() << " of " << hit_v.size() << " hits" << std::endl;
	// for ( auto const& hit : corecluster ) {
	//   std::cout << " corehit (" << hit[0] << "," << hit[1] << "," << hit[2] << ")" << std::endl;
	// }
	ev_outcore->emplace_back( std::move(corecluster) );
	ev_outcorepca->emplace_back( std::move( pcacoreinfo) );
      }
    }

    io_out.set_id( io.run_id(), io.subrun_id(), io.event_id() );

    io_out.next_event();
  }

  std::cout << "finished" << std::endl;
  io_out.close();
  io.close();
  
}
