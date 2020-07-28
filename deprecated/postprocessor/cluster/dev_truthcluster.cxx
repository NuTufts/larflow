#include <iostream>

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"

#include "TruthCluster.h"

#include "CilantroPCA.h"
#include "CoreFilter.h"

int main( int nargs, char** argv ) {

  std::cout << "Dev Truth Cluster" << std::endl;

  std::string larflow_input   = argv[1];
  std::string cluster_outputfile = argv[2];

  bool use_ancestor_id   = true;   
  bool return_unassigned = true;
  
  // input
  larlite::storage_manager io( larlite::storage_manager::kREAD );
  io.add_in_filename( larflow_input );
  io.open();

  // output
  larlite::storage_manager io_out( larlite::storage_manager::kWRITE );
  io_out.set_out_filename( cluster_outputfile );
  io_out.open();

  // Truth Cluster
  larflow::TruthCluster clusteralgo;
  
  int nentries = io.get_entries();

  for (int ientry=0; ientry<nentries; ientry++) {

    std::cout << "==========================================================" << std::endl;
    std::cout << "[dev_truthcluster] Entry " << ientry << " of " << nentries << std::endl;
    
    io.go_to( ientry );
    
    larlite::event_larflow3dhit* ev_hits = (larlite::event_larflow3dhit*)io.get_data( larlite::data::kLArFlow3DHit, "flowhits" );
    larlite::event_mctrack*  ev_mctrack   = (larlite::event_mctrack*) io.get_data( larlite::data::kMCTrack,  "mcreco" );
    larlite::event_mcshower* ev_mcshower  = (larlite::event_mcshower*)io.get_data( larlite::data::kMCShower, "mcreco" );    

    int numhits = ev_hits->size();
    std::cout << "number of hits: " << numhits << std::endl;
    std::vector< std::vector<const larlite::larflow3dhit*> > clusters = clusteralgo.clusterHits( *ev_hits, *ev_mctrack, *ev_mcshower, use_ancestor_id, return_unassigned );

    std::cout << "truthcluster returned with " << clusters.size() << " clusters" << std::endl;
    larlite::event_larflowcluster* ev_outcluster = (larlite::event_larflowcluster*)io_out.get_data( larlite::data::kLArFlowCluster,"flowtruthclusters");
    larlite::event_larflowcluster* ev_outcore    = (larlite::event_larflowcluster*)io_out.get_data( larlite::data::kLArFlowCluster,"flowtruthcore"); // after dbscan to identify core
    larlite::event_pcaxis* ev_outpca             = (larlite::event_pcaxis*)io_out.get_data( larlite::data::kPCAxis,"flowtruthclusters");
    larlite::event_pcaxis* ev_outcorepca         = (larlite::event_pcaxis*)io_out.get_data( larlite::data::kPCAxis,"flowtruthcore");

    io_out.set_id( io.run_id(), io.subrun_id(), io.event_id() );
    if ( numhits== 0 ) {
      std::cout << "Event is empty. save empty entry and move on." << std::endl;
      io_out.next_event();
      continue;
    }

    
    int iclust=-1;
    for ( auto& hit_v : clusters ) {
      iclust++;
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
      larflow::CoreFilter     corealgo( flowcluster, 3, 10.0 );
      larlite::larflowcluster corecluster = corealgo.getCore(5,flowcluster);

      ev_outpca->emplace_back( std::move(pcainfo) );
      ev_outcluster->emplace_back( std::move(flowcluster) );	
      
      if ( corecluster.size()>3 && iclust+1!=(int)clusters.size() ) {
	// no core for last cluster which is unassigned hits
	std::cout << "core of cluster has " << corecluster.size() << " of " << hit_v.size() << " hits" << std::endl;
	// for ( auto const& hit : corecluster ) {
	//   std::cout << " corehit (" << hit[0] << "," << hit[1] << "," << hit[2] << ")" << std::endl;
	// }
	larflow::CilantroPCA pcacore( corecluster );
	larlite::pcaxis pcacoreinfo = pcacore.getpcaxis();
	
	ev_outcore->emplace_back( std::move(corecluster) );
	ev_outcorepca->emplace_back( std::move( pcacoreinfo) );
      }
    }



    io_out.next_event();
  }

  std::cout << "finished" << std::endl;
  io_out.close();
  io.close();
  
}
