#include <iostream>

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"

#include "DBSCAN.h"
#include <cilantro/timer.hpp>

#include "TTree.h"
#include "TCanvas.h"
#include "TH1D.h"


int main( int nargs, char** argv ) {

  std::cout << "Dev Reco Cluster" << std::endl;

  std::string inputfile = argv[1];

  // input
  larlite::storage_manager io( larlite::storage_manager::kREAD );
  io.add_in_filename( inputfile );
  io.open();

  // output
  int idx=0; // reco clust index, arbitrary
  int truth_idx=0; //track id
  int truthflag = 0;
  float x[3] = {0.,0.,0.};
    
  //TFile* fout = new TFile("simple_clusterout.root","recreate");
  TTree* tree = new TTree("tree","simple tree");
  tree->Branch("reco_idx",&idx,"reco_idx/I");
  tree->Branch("truthflag",&truthflag,"truthflag/I");
  tree->Branch("truth_idx",&truth_idx,"truth_idx/I");
  tree->Branch("x",x,"x[3]/F");

  
  larlite::storage_manager io_out( larlite::storage_manager::kWRITE );
  io_out.set_out_filename( "outputdev_dbscancluster.root" );
  io_out.open();

  // Algo
  larflow::DBSCAN algo;
  
  int nentries = io.get_entries();
  nentries = 1;
  for (int ientry=0; ientry<nentries; ientry++) {
    
    io.go_to( ientry );
  
    larlite::event_larflow3dhit& ev_hits       = *((larlite::event_larflow3dhit*)  io.get_data( larlite::data::kLArFlow3DHit,  "flowhits" ));
    std::cout << "number of hits: " << ev_hits.size() << std::endl;

    // translate hits into vector< vector<float> >
    std::vector< std::vector<float> > ptlist(ev_hits.size());
    for ( int ihit=0; ihit<(int)ev_hits.size(); ihit++ ) {
      auto const& hit = ev_hits[ihit];
      ptlist[ihit].resize(3,0);
      for (int i=0; i<3; i++) ptlist[ihit][i] = hit[i];
    }
      
    //
    cilantro::Timer timer;
    timer.start();
    std::vector< std::vector<int> > dbscan_clusters = algo.makeCluster( 5, 10, 0, ptlist );
    timer.stop();
    std::cout << "Clustering time: " << timer.getElapsedTime() << "ms" << std::endl;
    
    std::cout << "Cluster results ------------------" << std::endl;
    std::cout << "  point indices size=" << dbscan_clusters.size() << std::endl;

    larlite::event_larflowcluster* ev_outcluster = (larlite::event_larflowcluster*)io_out.get_data( larlite::data::kLArFlowCluster, "flowtruthclusters" );
    
    for(int i=0; i<dbscan_clusters.size(); i++){
      std::cout <<"  cluster[" << i << "] size=" << dbscan_clusters[i].size() << std::endl;
      larlite::larflowcluster lfcluster;
      
      for ( auto const& idx : dbscan_clusters[i] ) {
	lfcluster.push_back( ev_hits.at(idx) );
      }

      ev_outcluster->emplace_back( std::move(lfcluster) );
    }
    
    io_out.set_id( io.run_id(), io.subrun_id(), io.event_id() );
    io_out.next_event();
    break;
  }//end of entry loop
  
  std::cout << "finished" << std::endl;
  //fout->cd();
  //tree->Write();
  //fout->Close();
  
  io_out.close();
  io.close();
  return 0;
  
}
