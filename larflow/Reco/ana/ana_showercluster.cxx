#include <iostream>
#include <vector>
#include <set>

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/mctrajectory.h"
#include "DataFormat/mctruth.h"

#include "larflow/Reco/MCPixelPGraph.h"
#include "larflow/Reco/cluster_functions.h"


int main( int nargs, char** argv ) {

  std::cout << "Ana ShowerCluster" << std::endl;

  // we want to know
  // 1) which shower cluster, if any, matches best to the truth shower-trunk
  // 2) purity of that best shower cluster
  // 3) efficiency of that shower cluster
  // 4) pca-line versus shower direction
  // 5) statistics of matched clusters and false clusters

  larlite::storage_manager ioll( larlite::storage_manager::kREAD );
  ioll.add_in_filename( "../test/merged_dlreco_eLEE_sample2.root" );
  ioll.add_in_filename( "../test/larmatch_eLEE_sample2.root" );
  ioll.add_in_filename( "../test/larflow_cluster_eLEE_sample2_full.root" );
  ioll.open();

  larcv::IOManager iolcv( larcv::IOManager::kREAD, "iolcv", larcv::IOManager::kTickBackward );
  iolcv.add_in_file( "../test/merged_dlreco_eLEE_sample2.root" );
  iolcv.reverse_all_products();
  iolcv.initialize();

  int nentries_iolcv = iolcv.get_n_entries();
  int nentries_ioll  = ioll.get_entries();

  int nentries = std::min(nentries_iolcv,nentries_ioll);

  std::cout << "Num entries: " << nentries << std::endl;

  TFile* out = new TFile("out_ana_showercluster.root","recreate");

  TTree* truthmatchana = new TTree("truthmatchana","Info for best Truth-Matched Cluster");
  int entry;
  float EnuMeV;
  float EeMeV;
  int   cluster_max_truthpix;
  int   cluster_max_recopix;  
  float cluster_max_truthfraction;
  float cluster_max_pixelsum[3];
  float truetrunk_pix_planeave;
  truthmatchana->Branch( "entry",  &entry,  "entry/I" );  
  truthmatchana->Branch( "EnuMeV", &EnuMeV, "ENuMeV/F" );
  truthmatchana->Branch( "EeMeV",  &EeMeV,  "EeMeV/F" );
  truthmatchana->Branch( "cluster_max_recopix",       &cluster_max_recopix,       "cluster_max_recopix/I" );  
  truthmatchana->Branch( "cluster_max_truthpix",      &cluster_max_truthpix,      "cluster_max_truthpix/I" );
  truthmatchana->Branch( "cluster_max_truthfraction", &cluster_max_truthfraction, "cluster_max_truthfraction/F" );
  truthmatchana->Branch( "cluster_max_pixelsum",      cluster_max_pixelsum,       "cluster_max_pixelsum[3]/F" );  
  truthmatchana->Branch( "truetrunk_pix_planeave",    &truetrunk_pix_planeave,    "truetrunk_pix_planeave/F" );

  larflow::reco::MCPixelPGraph mcpg;

  for ( int ientry=0; ientry<nentries; ientry++ ) {
    
    std::cout << "===============" << std::endl;
    std::cout << " Entry " << ientry << std::endl;
    std::cout << "===============" << std::endl;

    entry = ientry;

    // load entry
    ioll.go_to(ientry);
    iolcv.read_entry( ientry );

    // build MC pgraph and assign image pixels to them
    mcpg.buildgraph( iolcv, ioll );
    mcpg.printGraph();

    // get reco shower clusters
    larlite::event_larflowcluster* shower_lfcluster_v
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "lfshower" );

    // get back to cluster_t objects
    std::vector< larflow::reco::cluster_t > shower_cluster_v;
    for ( auto& lfcluster : *shower_lfcluster_v ) {
      larflow::reco::cluster_t c = larflow::reco::cluster_from_larflowcluster( lfcluster );
      shower_cluster_v.emplace_back( std::move(c) );
    }

    // get ADC images
    larcv::EventImage2D* ev_adc = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_adc->Image2DArray();

    // we are after the primary electron, get it's data
    std::vector<larflow::reco::MCPixelPGraph::Node_t*> nodelist
      = mcpg.getPrimaryParticles();

    larflow::reco::MCPixelPGraph::Node_t* prim_electron = nullptr;
    for ( auto& pnode : nodelist ) {
      if ( std::abs(pnode->pid)==11 ) {
        prim_electron = pnode;
        break; // don't expect another primary
      }
    }

    if ( prim_electron==nullptr )
      continue;

    // we build a set of (tick,wire) pairs for look up reasons
    std::set< std::pair<int,int> > electron_pixels_vv[3];

    truetrunk_pix_planeave = 0.;
    int nplane_w_truth = 0;
    for ( size_t p=0; p<3; p++ ) {

      int ntruthpixels = prim_electron->pix_vv[p].size()/2;
      truetrunk_pix_planeave += (float)ntruthpixels;
      if ( ntruthpixels>0 )
        nplane_w_truth++;
      
      for ( int ipix=0; ipix<ntruthpixels; ipix++ ) {
        int tick = prim_electron->pix_vv[p][ 2*ipix ];
        int wire = prim_electron->pix_vv[p][ 2*ipix+1 ];
        electron_pixels_vv[p].insert( std::pair<int,int>(tick,wire) );
      }

      if ( nplane_w_truth>0 )
        truetrunk_pix_planeave /= (float)nplane_w_truth;
    }

    // get truth info for primary electron
    larlite::event_mctrack* mctrack_v
      = (larlite::event_mctrack*)ioll.get_data(larlite::data::kMCTrack,"mcreco");
    larlite::event_mcshower* mcshower_v
      = (larlite::event_mcshower*)ioll.get_data(larlite::data::kMCShower,"mcreco");
    double profE = mcshower_v->at( prim_electron->vidx ).DetProfile().E();
    double stepE = mcshower_v->at( prim_electron->vidx ).Start().E();
    std::cout << "truth shower profileE=" << profE << " MeV stepE=" << stepE << " MeV" << std::endl;
    EeMeV = profE;

    // neutrino energy
    const larlite::mctruth& mctruth
      = ((larlite::event_mctruth*)ioll.get_data(larlite::data::kMCTruth,"generator"))->front();
    float trueNuE = mctruth.GetNeutrino().Nu().Trajectory().front().E()*1000.0;
    std::cout << "truth neutrino energy= " << trueNuE << " MeV" << std::endl;
    EnuMeV = trueNuE;

    // loop over clusters, get the fraction that lie on truth pixels
    // for larflow, might make mistakes, so 2 of 3 planes must be on pixels
    std::vector<int>   truth_pixels_v(   shower_cluster_v.size(), 0 );
    std::vector<float> truth_fraction_v( shower_cluster_v.size(), 0 );

    for ( size_t i=0; i<shower_cluster_v.size(); i++ ) {
      // scan over hits
      auto const& cluster = shower_cluster_v[i];
      for ( size_t ihit=0; ihit<cluster.imgcoord_v.size(); ihit++ ) {
        const std::vector<int>& coord = cluster.imgcoord_v[ihit];
        int noverlap = 0;
        for ( size_t p=0; p<3; p++ ) {
          std::pair<int,int> tw(coord[3],coord[p]);
          if ( electron_pixels_vv[p].find( tw )!=electron_pixels_vv[p].end() ) {
            noverlap++;
          }
        }
        if (noverlap>=2 ) {
          truth_pixels_v[i]++;
          truth_fraction_v[i] += 1.0;
        }          
      }
      if ( cluster.imgcoord_v.size()>0 ) {
        truth_fraction_v[i] /= (float)cluster.imgcoord_v.size();
      }
    }

    int max_cluster = -1;
    int max_pixels  = 0;
    for ( int i=0; i<truth_pixels_v.size(); i++ ) {
      if ( truth_pixels_v[i]>max_pixels ) {
        max_pixels = truth_pixels_v[i];
        max_cluster = i;
      }
    }
    if ( max_cluster>=0 ) {
      cluster_max_truthpix = max_pixels;
      cluster_max_truthfraction = truth_fraction_v[max_cluster];
      cluster_max_recopix  = (int)shower_cluster_v[max_cluster].imgcoord_v.size();
      std::vector<float> pixsum = larflow::reco::cluster_pixelsum( shower_cluster_v[max_cluster], adc_v );
      for (int i=0; i<3; i++ ) cluster_max_pixelsum[i] = pixsum[i];
    }
    else {
      cluster_max_truthpix = 0;
      cluster_max_recopix  = 0;
      cluster_max_truthfraction = 0.0;
      for (int i=0; i<3; i++ ) cluster_max_pixelsum[i] = 0.;
    }
        

    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << "Max shower reco cluster=" << max_cluster
              << " max pixels in cluster=" << max_pixels
              << " fraction=" << truth_fraction_v[max_cluster]
              << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;

    truthmatchana->Fill();
    
  }

  truthmatchana->Write();

  out->Close();
  
  return 0;
  
}
