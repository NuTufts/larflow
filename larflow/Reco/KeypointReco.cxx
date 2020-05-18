#include "KeypointReco.h"

#include <iostream>
#include <fstream>

#include "nlohmann/json.hpp"

#include "cluster_functions.h"

namespace larflow {
namespace reco {

  void KeypointReco::set_param_defaults()
  {
    _sigma = 10.0; // cm
    _score_threshold = 0.5;
    _num_passes = 2;
    _min_cluster_size = 10;
    _max_dbscan_dist = 2.0;
  }
  
  /**
   * take in storage manager, get larflow3dhits, which stores keypoint scores, 
   * make candidate KPCluster, and store them as larflow3dhit for now. (tech-debt!)
   *
   */  
  void KeypointReco::process( larlite::storage_manager& io_ll )
  {
    
  }
  
  /**
   * take in larflow3dhits, which stores keypoint scores, and make candidate KPCluster
   *
   */
  void KeypointReco::process( const std::vector<larlite::larflow3dhit>& input_lfhits )
  {

    output_pt_v.clear();
    
    _make_initial_pt_data( input_lfhits, _score_threshold );

    for (int i=0; i<_num_passes; i++ ) {
      std::cout << "[KeypointReco::process] Pass " << i+1 << std::endl;
      _make_kpclusters( _score_threshold );
      std::cout << "[KeypointReco::process] Pass " << i+1 << ", clusters formed: " << output_pt_v.size() << std::endl;
      int nabove=0;
      for (auto& posv : _initial_pt_pos_v ) {
        if (posv[3]>_score_threshold) nabove++;
      }
      std::cout << "[KeypointReco::process] Pass " << i+1 << ", points remaining above threshold" << nabove << "/" << _initial_pt_pos_v.size() << std::endl;
    }

    printAllKPClusterInfo();
    std::cout << "[ KeypointReco::process ] num kpclusters = " << output_pt_v.size() << std::endl;
  }
  
  /**
   * scan larflow3dhit input and assemble 3D keypoint data we will work on
   *
   * clears and fills:
   *  _initial_pt_pos_v;
   *  _initial_pt_used_v;
   *
   * @param[in] lfhits          LArFlow hits with keypoint network info
   * @param[in] score_threshold Keep hits above these thresholds
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

  /**
   * Make clusters with remaining points
   *
   * fills member cluster container, output_pt_v.
   * 
   * @param[in] round_score_threshold 
   *
   */
  void KeypointReco::_make_kpclusters( float round_score_threshold )
  {

    std::vector< std::vector<float> > skimmed_pt_v;
    std::vector< int > skimmed_index_v;

    _skim_remaining_points( round_score_threshold,
                            skimmed_pt_v,
                            skimmed_index_v );

    // cluster the points
    std::vector< cluster_t > cluster_v;
    float maxdist = _max_dbscan_dist;
    int minsize   = _min_cluster_size;
    int maxkd     = 5;
    cluster_spacepoint_v( skimmed_pt_v, cluster_v, maxdist, minsize, maxkd );

    float sigma = _sigma; // bandwidth

    for ( auto& cluster : cluster_v ) {
      // make kpcluster
      KPCluster kpc = _characterize_cluster( cluster, skimmed_pt_v, skimmed_index_v );
      auto& kpc_cluster = _cluster_v[ kpc._cluster_idx ];

      // We now subtract the point score from nearby points
      for ( auto const& idx : kpc_cluster.hitidx_v ) {

        float dist = 0.;
        for ( int i=0; i<3; i++ )
          dist += ( kpc.center_pt_v[i]-_initial_pt_pos_v[idx][i] )*(kpc.center_pt_v[i]-_initial_pt_pos_v[idx][i] );

        float current_score = _initial_pt_pos_v[idx][3];

        // suppress score based on distance from cluster centroid
        float newscore = current_score - kpc.max_score*exp(-0.5*dist/(sigma*sigma));
        if (newscore<0) {
          newscore = 0.0;
          _initial_pt_used_v[idx] = 1;
        }
        _initial_pt_pos_v[idx][3] = newscore;
      }

      // does nothing right now
      _expand_kpcluster( kpc );
      
      output_pt_v.emplace_back( std::move(kpc) );
    }
    std::cout << "[larflow::KeypointReco::_make_kpclusters] number of clusters=" << output_pt_v.size() << std::endl;
    
  }

  /**
   * get list of points to cluster above threshold
   *
   * @param[in]  score_threshold   Keep points above threshold.
   * @param[out] skimmed_pt_v      Return 3D points.
   * @param[out] skimmed_index_v   Index of point in the Original Point list.
   *
   */
  void KeypointReco::_skim_remaining_points( float score_threshold,
                                             std::vector<std::vector<float> >& skimmed_pt_v,
                                             std::vector<int>& skimmed_index_v )
  {

    for ( size_t i=0; i<_initial_pt_pos_v.size(); i++ ) {
      if ( _initial_pt_pos_v[i][3]>score_threshold ) {
        skimmed_pt_v.push_back( _initial_pt_pos_v[i] );
        skimmed_index_v.push_back(i);
      }
    }
  }

  /**
   * we take the cluster we've made using dbscan and make a KPCluster object
   * we define the centroid using a weighted score
   * we define the pca as well, to help us absorb points
   *
   */
  KPCluster KeypointReco::_characterize_cluster( cluster_t& cluster,
                                                   std::vector< std::vector<float> >& skimmed_pt_v,
                                                   std::vector< int >& skimmed_index_v )
  {

    // run pca
    cluster_pca( cluster );

    KPCluster kpc;
    kpc.center_pt_v.resize(3,0.0);
    kpc.max_pt_v.resize(4,0.0);

    float totw = 0.;
    float max_score = 0.;
    int   max_idx = -1;
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

      if ( skimmed_pt_v[skimidx][3]>max_score ) {
        max_score = skimmed_pt_v[skimidx][3];
        max_idx   = i;
        kpc.max_pt_v = skimmed_pt_v[skimidx];        
      }

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

    // store max info
    kpc.max_score       = max_score;
    kpc.max_idx         = max_idx;
    
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
  void KeypointReco::_expand_kpcluster( KPCluster& kp )
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
      jkpc["maxpt"]    = kpc.max_pt_v;
      auto& clust = _cluster_v[ kpc._cluster_idx ];
      jkpc["clusters"] = cluster_json(clust);
      j["keypoints"].emplace_back( std::move(jkpc) );
    }
    
    std::ofstream o(outfilename.c_str());
    j >> o;
    o.close();
    
  }

  /**
   * Dump all cluster info to standard out
   *
   * prints info for clusters in output_pt_v.
   *
   */
  void KeypointReco::printAllKPClusterInfo()
  {
    for ( auto const& kp : output_pt_v )
      kp.printInfo();
  }

}
}
