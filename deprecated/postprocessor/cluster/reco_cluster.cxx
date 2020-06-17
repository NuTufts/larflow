#include <iostream>

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"

#include "RecoCluster.h"
#include <cilantro/timer.hpp>

#include "TTree.h"
#include "TCanvas.h"
#include "TH1D.h"


int main( int nargs, char** argv ) {

  std::cout << "Dev Reco Cluster" << std::endl;

  std::string inputfile = argv[1];

  int saveHitTree = atoi(argv[2]);
  int saveClustTree = atoi(argv[3]);

  //input DBSCAN
  float MAXDIST = std::atof(argv[4]);
  float MINHITS = std::atof(argv[5]);
  int MAXKDN    = std::atoi(argv[6]);

  //input stitching
  float ANGLE   = std::atof(argv[7]);
  float ENDDIST = std::atof(argv[8]);
  
  // input
  larlite::storage_manager io( larlite::storage_manager::kREAD );
  io.add_in_filename( inputfile );
  io.open();

  // event
  int run=0;
  int subrun=0;
  int evt=0;
  // hit variables
  int idx=0; // reco clust index
  int truth_idx=0; //track id
  int truthflag = 0;
  float x[3] = {0.,0.,0.};
  //cluster variables
  int Nhits = 0; //num hits in cluster
  int purity=0;
  float efficiency=0;
  
  //dbscan params
  float maxdist;
  float minhits;
  int maxkdn;

  //stitch params
  float angledist;
  float enddist;
  
  TFile* fout = new TFile("clusterana.root","recreate");
  TTree* tree = NULL;
  TTree* ctree = NULL;
  TTree* cstree = NULL;
  
 if(saveHitTree){
    tree = new TTree("htree","hit tree");
    tree->Branch("reco_idx",&idx,"reco_idx/I");
    tree->Branch("truthflag",&truthflag,"truthflag/I");
    tree->Branch("truth_idx",&truth_idx,"truth_idx/I");
    tree->Branch("x",x,"x[3]/F");
  }
  if(saveClustTree){
    ctree = new TTree("ctree","cluster tree");
    ctree->Branch("run",&run,"run/I");
    ctree->Branch("subrun",&subrun,"subrun/I");
    ctree->Branch("event",&evt,"event/I");
    ctree->Branch("dbscan_maxdist",&maxdist,"dbscan_maxdist/F");
    ctree->Branch("dbscan_minhits",&minhits,"dbscan_minhits/F");
    ctree->Branch("dbscan_maxkdn",&maxkdn,"dbscan_maxkdn/I");
    ctree->Branch("angledist",&angledist,"angledist/F");
    ctree->Branch("enddist",&enddist,"enddist/F");
    ctree->Branch("Nhits",&Nhits,"Nhits/I");
    ctree->Branch("purity",&purity,"purity/I");
    ctree->Branch("efficiency",&efficiency,"efficiency/F");
    //
    cstree = new TTree("cstree","stitch cluster tree");
    cstree->Branch("run",&run,"run/I");
    cstree->Branch("subrun",&subrun,"subrun/I");
    cstree->Branch("event",&evt,"event/I");
    cstree->Branch("dbscan_maxdist",&maxdist,"dbscan_maxdist/F");
    cstree->Branch("dbscan_minhits",&minhits,"dbscan_minhits/F");
    cstree->Branch("dbscan_maxkdn",&maxkdn,"dbscan_maxkdn/I");
    cstree->Branch("angledist",&angledist,"angledist/F");
    cstree->Branch("enddist",&enddist,"enddist/F");
    cstree->Branch("Nhits",&Nhits,"Nhits/I");
    cstree->Branch("purity",&purity,"purity/I");
    cstree->Branch("efficiency",&efficiency,"efficiency/F");

  }
  
  larlite::storage_manager io_out( larlite::storage_manager::kWRITE );
  io_out.set_out_filename( "output_dev_recocluster.root" );
  io_out.open();

  // RecoCluster
  larflow::RecoCluster clusteralgo;
  
  int nentries = io.get_entries();
  std::cout << nentries << std::endl;
  nentries = 1;
  for (int ientry=0; ientry<nentries; ientry++) {
    
    io.go_to( ientry );
  
    larlite::event_larflow3dhit& ev_hits = *((larlite::event_larflow3dhit*)io.get_data( larlite::data::kLArFlow3DHit, "flowhits" ));
    larlite::event_larflowcluster& ev_clusters = *((larlite::event_larflowcluster*)io_out.get_data( larlite::data::kLArFlowCluster,"dbscan"));
    larlite::event_larflowcluster& ev_clusters2 = *((larlite::event_larflowcluster*)io_out.get_data( larlite::data::kLArFlowCluster,"dbscanStitch"));
    larlite::event_pcaxis& ev_pcaout = *((larlite::event_pcaxis*)io_out.get_data( larlite::data::kPCAxis,"dbscan"));
    std::cout << "number of hits: " << ev_hits.size() << std::endl;

    std::vector<larlite::larflow3dhit> fhits;
    fhits.reserve(ev_hits.size());
    clusteralgo.filter_hits(ev_hits,fhits,3.,1.);
    std::cout << "number of filtered hits: " << fhits.size() << std::endl;
    //
    // (1) CLUSTER
    clusteralgo.set_dbscan_param(MAXDIST,MINHITS,MAXKDN);
    std::vector< std::vector<larlite::larflow3dhit> > clusters = clusteralgo.clusterHits( fhits, "DBSCAN", false );    
    // (2) Filter line clusters
    std::vector<int> isline;
    std::vector<larlite::pcaxis> pcainfos;
    std::vector<std::vector<std::pair<float, larlite::larflow3dhit> > > sorted_flowhits;
    clusteralgo.filterLineClusters(clusters, pcainfos, sorted_flowhits, isline);
    // (3) Select clusters to stitch
    std::vector<std::vector<int> > parallel_idx(clusters.size(),std::vector<int>(0));
    std::vector<std::vector<int> > stitch_idx(clusters.size(),std::vector<int>(0));
    clusteralgo.selectClustToConnect(sorted_flowhits, pcainfos, isline, parallel_idx, stitch_idx,ANGLE,ENDDIST);
    // (4) Stitch clusters
    std::vector<std::vector<larlite::larflow3dhit> > newclusters;
    clusteralgo.ConnectClusters(clusters, stitch_idx, newclusters);
    // (5) Evaluate clusters
    std::vector<int> clust_purity;
    std::vector<float> clust_efficiency;
    clusteralgo.EvaluateClusters(clusters, clust_purity, clust_efficiency);
    std::vector<int> clust_purity_s;
    std::vector<float> clust_efficiency_s;
    clusteralgo.EvaluateClusters(newclusters, clust_purity_s, clust_efficiency_s);
    
    std::cout << "number of dbscan clusters: " << clusters.size() << std::endl;
    std::cout << "number of stitched clusters: " << newclusters.size() << std::endl;

    // fill larlite output
    for (int i=0; i<clusters.size(); i++){
      larlite::larflowcluster lf;
      for(auto& hit : clusters.at(i)){
	lf.push_back(hit);
      }
      ev_clusters.emplace_back(std::move(lf));
      ev_pcaout.emplace_back(std::move(pcainfos.at(i)));
      //std::cout << "hits in clust "<< ev_clusters.at(i).size() << std::endl; 
    }
    for (int i=0; i<newclusters.size(); i++){
      larlite::larflowcluster lf;
      for(auto& hit : newclusters.at(i)){
	lf.push_back(hit);
      }
      ev_clusters2.emplace_back(std::move(lf));
      //std::cout << "hits in stitched clust "<< ev_clusters2.at(i).size() << std::endl;
     
    }

    // fill ana trees
    run = io.run_id();
    subrun = io.subrun_id();
    evt = io.event_id();
    enddist = ENDDIST;
    angledist = ANGLE;
    maxdist = MAXDIST;
    minhits = MINHITS;
    maxkdn = MAXKDN;
    
    if(saveHitTree){
      for ( int i=0; i< ev_clusters.size(); i++ ) {
	for ( auto const& phit : ev_clusters.at(i) ) {
	  idx = i;
	  truthflag = phit.truthflag;
	  truth_idx = phit.trackid;
	  x[0] = phit.at(0);
	  x[1] = phit.at(1);
	  x[2] = phit.at(2);
	  tree->Fill();
	}
      }
    }
    if(saveClustTree){
      for(int i=0; i<ev_clusters.size(); i++){
	Nhits = ev_clusters.at(i).size();
	purity = clust_purity.at(i);
	efficiency = clust_efficiency.at(i);
	ctree->Fill();
      }
      for(int i=0; i<ev_clusters2.size(); i++){
	Nhits = ev_clusters2.at(i).size();
	purity = clust_purity_s.at(i);
	efficiency = clust_efficiency_s.at(i);
	cstree->Fill();
      }
    }

    
    

    io_out.set_id( io.run_id(), io.subrun_id(), io.event_id() );
    io_out.next_event();
    //break;
  }

  std::cout << "finished" << std::endl;
  fout->cd();
  if(saveHitTree) tree->Write();
  if(saveClustTree){
    ctree->Write();
    cstree->Write();
  }
  fout->Close();
  io_out.close();
  io.close();
  return 0;
  
}
