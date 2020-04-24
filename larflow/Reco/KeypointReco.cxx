#include "KeypointReco.h"

#include <iostream>
#include <fstream>

#include "nlohmann/json.hpp"

#include "cluster_functions.h"

namespace larflow {
namespace reco {

  /**
   * take in storage manager, get larflow3dhits, which stores keypoint scores, 
   * make candidate KPCluster_t, and store them as larflow3dhit for now. (tech-debt!)
   *
   */  
  void KeypointReco::process( larlite::storage_manager& io_ll )
  {
    
  }
  
  /**
   * take in larflow3dhits, which stores keypoint scores, and make candidate KPCluster_t
   *
   */
  void KeypointReco::process( const std::vector<larlite::larflow3dhit>& input_lfhits )
  {

    output_pt_v.clear();
    
    _make_initial_pt_data( input_lfhits, 0.25 );
    _make_kpclusters( 0.7 );

    std::cout << "[ KeypointReco::process ] num kpclusters = " << output_pt_v.size() << std::endl;
  }
  
  /**
   * scan larflow3dhit input and assemble 3D keypoint data we will work on
   *
   * clears and fills:
   *  _initial_pt_pos_v;
   *  _initial_pt_used_v;
   *
   */
  void KeypointReco::_make_initial_pt_data( const std::vector<larlite::larflow3dhit>& lfhits,
                                            float score_threshold )
  {

    _initial_pt_pos_v.clear();
    _initial_pt_used_v.clear();

    for (auto const& lfhit : lfhits ) {
      float score = lfhit[13];
      if ( score>score_threshold ) {
        std::vector<float> pos3d(4,0);
        for (int i=0; i<3; i++) pos3d[i] = lfhit[i];
        pos3d[3] = score;
        _initial_pt_pos_v.push_back( pos3d );
      }
    }

    _initial_pt_used_v.resize( _initial_pt_pos_v.size(), 0 );

    std::cout << "[larflow::reco::KeypointReco::_make_initial_pt_data] number of points stored for keypoint reco: "
              << _initial_pt_pos_v.size()
              << "/"
              << lfhits.size()
              << std::endl;


    
    
  }

  void KeypointReco::_make_kpclusters( float round_score_threshold )
  {

    std::vector< std::vector<float> > skimmed_pt_v;
    std::vector< int > skimmed_index_v;

    _skim_remaining_points( round_score_threshold,
                            skimmed_pt_v,
                            skimmed_index_v );


    // cluster the points
    std::vector< cluster_t > cluster_v;
    float maxdist = 2.0;
    int minsize = 5;
    int maxkd   = 5;
    cluster_spacepoint_v( skimmed_pt_v, cluster_v, maxdist, minsize, maxkd );

    for ( auto& cluster : cluster_v ) {
      // make kpcluster
      KPCluster_t kpc = _characterize_cluster( cluster, skimmed_pt_v, skimmed_index_v );
      auto& kpc_cluster = _cluster_v[ kpc._cluster_idx ];
      // tag points in total-set (_intial_pt_pos_v)
      for ( auto const& idx : kpc_cluster.hitidx_v ) {
        _initial_pt_used_v[idx] = 1;
      }

      _expand_kpcluster( kpc );
      
      output_pt_v.emplace_back( std::move(kpc) );
    }
    std::cout << "[larflow::KeypointReco::_make_kpclusters] number of clusters=" << output_pt_v.size() << std::endl;
    
  }

  void KeypointReco::_skim_remaining_points( float score_threshold,
                                             std::vector<std::vector<float> >& skimmed_pt_v,
                                             std::vector<int>& skimmed_index_v )
  {

    for ( size_t i=0; i<_initial_pt_pos_v.size(); i++ ) {
      if ( _initial_pt_used_v[i]==1 ) continue;

      if ( _initial_pt_pos_v[i][3]>score_threshold ) {
        skimmed_pt_v.push_back( _initial_pt_pos_v[i] );
        skimmed_index_v.push_back(i);
      }
    }
  }

  /**
   * we take the cluster we've made using dbscan and make a KPCluster_t object
   * we define the centroid using a weighted score
   * we define the pca as well, to help us absorb points
   *
   */
  KPCluster_t KeypointReco::_characterize_cluster( cluster_t& cluster,
                                                   std::vector< std::vector<float> >& skimmed_pt_v,
                                                   std::vector< int >& skimmed_index_v )
  {

    // run pca
    cluster_pca( cluster );

    KPCluster_t kpc;
    kpc.center_pt_v.resize(3,0.0);

    float totw = 0.;
    for ( int i=0; i<(int)cluster.points_v.size(); i++ ) {

      int skimidx = cluster.hitidx_v[i];
      
      float w = skimmed_pt_v[ skimidx ][3];
      if ( w>1.0 ) w = 1.0;
      if ( w<0.0 ) w = 0.0;
      
      for (int v=0; v<3; v++ ) {
        kpc.center_pt_v[v] += w*w*skimmed_pt_v[ skimidx ][v];
      }
      totw += w*w;
        
      // load up the cluster
      kpc.pt_pos_v.push_back(   skimmed_pt_v[ skimidx ] );
      kpc.pt_score_v.push_back( skimmed_pt_v[ skimidx ][3] );

      // update the hitindex to use the total-set indexing
      cluster.hitidx_v[i] =  skimmed_index_v[skimidx];
    }
    if ( totw>0.0 ) {
      for (int v=0; v<3; v++ ) {
        kpc.center_pt_v[v] /= totw;
      }
    }
    //kpc.cluster = cluster; // this seems dumb. just for now/development.

    // copy pca info
    kpc.pca_axis_v      = cluster.pca_axis_v;
    kpc.pca_center      = cluster.pca_center;
    kpc.pca_eigenvalues = cluster.pca_eigenvalues;
    kpc.pca_ends_v      = cluster.pca_ends_v;
    kpc.bbox_v          = cluster.bbox_v;
    kpc.pca_max_r       = cluster.pca_max_r;
    kpc.pca_ave_r2      = cluster.pca_ave_r2;
    kpc.pca_len         = cluster.pca_len;
    
    std::cout << "[KeypointReco::_characterize_cluster]" << std::endl;
    std::cout << "  center: (" << kpc.center_pt_v[0] << "," << kpc.center_pt_v[1] << "," << kpc.center_pt_v[2] << ")" << std::endl;
    std::cout << "  pca: (" << cluster.pca_axis_v[0][0] << "," << cluster.pca_axis_v[0][1] << "," << cluster.pca_axis_v[0][2] << ")" << std::endl;

    // insert cluster into class continer
    _cluster_v.emplace_back( std::move(cluster) );
    kpc._cluster_idx = (int)_cluster_v.size()-1;

    
    return kpc;
  }

  /**
   * we absorb points to clusters, using pca-line, proximity, and confidence score
   *
   */
  void KeypointReco::_expand_kpcluster( KPCluster_t& kp )
  {
    // to do
  }
  
  /**
   * dump output to json for development
   *
   */
  void KeypointReco::dump2json( std::string outfilename )
  {
    
    nlohmann::json j;
    std::vector< nlohmann::json > jcluster_v;
    j["keypoints"] = jcluster_v;

    for ( auto const& kpc : output_pt_v ) {
      //std::cout << "cluster: nhits=" << cluster.points_v.size() << std::endl;
      nlohmann::json jkpc;
      jkpc["center"]   = kpc.center_pt_v;
      auto& clust = _cluster_v[ kpc._cluster_idx ];
      jkpc["clusters"] = cluster_json(clust);
      j["keypoints"].emplace_back( std::move(jkpc) );
      std::cout << "dump cluster" << std::endl;
    }
    
    std::ofstream o(outfilename.c_str());
    j >> o;
    o.close();
    
  }
  

}
}
