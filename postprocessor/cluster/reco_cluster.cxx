#include <iostream>

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"

#include "RecoCluster.h"
#include "CilantroSpectral.h"

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
  //larlite::storage_manager io_out( larlite::storage_manager::kWRITE );
  //io_out.set_out_filename( "output_dev_truthcluster.root" );
  //io_out.open();

  // RecoCluster
  larflow::RecoCluster clusteralgo;
  
  int nentries = io.get_entries();
  nentries = 1;
  for (int ientry=0; ientry<nentries; ientry++) {
    
    io.go_to( ientry );
  
    larlite::event_larflow3dhit& ev_hits = *((larlite::event_larflow3dhit*)io.get_data( larlite::data::kLArFlow3DHit, "flowhits" ));
    larlite::event_larflowcluster& ev_clusters = *((larlite::event_larflowcluster*)io.get_data( larlite::data::kLArFlowCluster,"dbscan"));
    std::cout << "number of hits: " << ev_hits.size() << std::endl;
    std::cout << "number of dbscan clusters: " << ev_clusters.size() << std::endl;
    
    for (int i=0; i<ev_clusters.size(); i++){
      //std::cout << "hits in clust "<< ev_clusters.at(i).size() << std::endl;
    }
    /*
    idx=0;
    for ( auto& cluster : ev_clusters ) {
      for ( auto const& phit : cluster ) {
	truthflag = phit.truthflag;
	truth_idx = phit.trackid;
	x[0] = phit.at(0);
	x[1] = phit.at(1);
	x[2] = phit.at(2);
	tree->Fill();
      }
      idx++;
    }
    */
    std::vector<larlite::larflow3dhit> fhits;
    clusteralgo.filter_hits(fhits,ev_hits);
    larflow::CilantroSpectral sc( ev_hits,40,5 );
    std::vector<std::vector<long unsigned int> > cpi;
    std::vector<long unsigned int> idx_mat;
    sc.get_cluster_indeces(cpi,idx_mat);
    std::cout << cpi.size() <<" "<<idx_mat.size() << std::endl;
    for(int i=0; i<10; i++){
      std::cout << cpi[0][i] <<" "<< idx_mat[i] << std::endl;
    }

    //io_out.set_id( io.run_id(), io.subrun_id(), io.event_id() );
    //io_out.next_event();
    break;
  }

  std::cout << "finished" << std::endl;
  //fout->cd();
  //tree->Write();
  //fout->Close();
  //io_out.close();
  io.close();
  return 0;
  
}
