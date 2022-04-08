#include "KeypointReco.h"

#include <iostream>
#include <fstream>

#include "TMath.h"
#include <Eigen/Dense>

#include "nlohmann/json.hpp"

#include "larlite/DataFormat/larflow3dhit.h"
#include "larlite/DataFormat/pcaxis.h"
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/Geometry.h"

#include "cluster_functions.h"



namespace larflow {
namespace reco {

  /** 
   * @brief set default parameter values
   *
   */
  void KeypointReco::set_param_defaults()
  {
    _sigma = 5.0; // cm
    _larmatch_score_threshold = 0.5;    
    _num_passes = 1;
    _keypoint_score_threshold_v = std::vector<float>( 2, 0.5 );
    _min_cluster_size_v = std::vector<int>(2,50);
    _max_dbscan_dist = 2.0;
    _input_larflowhit_tree_name = "larmatch";
    _output_tree_name = "keypoint";
    _keypoint_type = -1;
    _threshold_cluster_max_score = 0.75;
  }
  
  /**
   * @brief take in storage manager, get larflow3dhits, which stores keypoint scores, 
   * make candidate KPCluster, and store them as larflow3dhit 
   *
   */  
  void KeypointReco::process( larlite::storage_manager& io_ll )
  {
    larlite::event_larflow3dhit* ev_larflow_hit
      = (larlite::event_larflow3dhit*)io_ll.get_data( larlite::data::kLArFlow3DHit, _input_larflowhit_tree_name );

    process( *ev_larflow_hit );

    // save into larlite::storage_manager
    // we need our own data product, but for now we use abuse the larflow3dhit
    larlite::event_larflow3dhit* evout_keypoint =
      (larlite::event_larflow3dhit*)io_ll.get_data( larlite::data::kLArFlow3DHit, _output_tree_name );
    larlite::event_pcaxis* evout_pcaxis =
      (larlite::event_pcaxis*)io_ll.get_data( larlite::data::kPCAxis, _output_tree_name );

    int cidx=0;
    for ( auto const& kpc : output_pt_v ) {
      larlite::larflow3dhit hit;
      std::vector<double> vtxpos(3);
      hit.resize( 5, 0 ); // [0-2]: hit pos, [3]: type, [4]: max net score
      for (int i=0; i<3; i++) {
        hit[i] = kpc.max_pt_v[i];
        vtxpos[i] = kpc.max_pt_v[i];
      }
      hit[3] = kpc._cluster_type;
      hit[4] = kpc.max_score;

      hit.targetwire.resize( 3, 0 );
      for  (int p=0; p<3; p++) 
        hit.targetwire[p] = larutil::Geometry::GetME()->WireCoordinate( vtxpos, p );
      hit.tick = vtxpos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
      

      // pca-axis
      larlite::pcaxis::EigenVectors e_v;
      // just std::vector< std::vector<double> >
      // we store axes (3) and then the 1st axis end points. So five vectors.
      for ( auto const& a_v : kpc.pca_axis_v ) {
        std::vector<double> da_v = { (double)a_v[0], (double)a_v[1], (double) a_v[2] };
        e_v.push_back( da_v );
      }
      // start and end points
      for ( auto const& p_v : kpc.pca_ends_v ) {
        std::vector<double> dp_v = { (double)p_v[0], (double)p_v[1], (double)p_v[2] };
        e_v.push_back( dp_v );
      }
      double eigenval[3] = { kpc.pca_eigenvalues[0], kpc.pca_eigenvalues[1], kpc.pca_eigenvalues[2] };
      double centroid[3] = { kpc.pca_center[0], kpc.pca_center[1], kpc.pca_center[2] };
      larlite::pcaxis llpca( true, kpc.pt_pos_v.size(), eigenval, e_v, centroid, 0, cidx);

      evout_keypoint->emplace_back( std::move(hit) );
      evout_pcaxis->emplace_back( std::move(llpca) );
      cidx++;
    }
  }
  
  /**
   * @brief take in larflow3dhits, which stores keypoint scores, and make candidate KPCluster
   *
   * @param[in] input_lfhits Vector of larflow3dhit with larmatch and keypoint scores
   *
   */
  void KeypointReco::process( const std::vector<larlite::larflow3dhit>& input_lfhits )
  {

    output_pt_v.clear();
    
    _make_initial_pt_data( input_lfhits, _keypoint_score_threshold_v.front(), _larmatch_score_threshold );

    for (int i=0; i<_num_passes; i++ ) {
      LARCV_INFO() << "[KeypointReco::process] Pass " << i+1 << std::endl;
      _make_kpclusters( _keypoint_score_threshold_v[i], _min_cluster_size_v[i] );
      LARCV_INFO() << "[KeypointReco::process] Pass " << i+1 << ", clusters formed: " << output_pt_v.size() << std::endl;
      int nabove=0;
      for (auto& posv : _initial_pt_pos_v ) {
        if (posv[3]>_keypoint_score_threshold_v[i]) nabove++;
      }
      LARCV_INFO() << "[KeypointReco::process] Pass " << i+1 << ", points remaining above threshold" << nabove << "/" << _initial_pt_pos_v.size() << std::endl;
    }

    if ( logger().level()<=larcv::msg::kINFO )
      printAllKPClusterInfo();
    LARCV_NORMAL() << "[ KeypointReco::process (type=" << _keypoint_type << ", hitindex=" << _lfhit_score_index << ") ] "
		   << "num kpclusters = " << output_pt_v.size()
		   << std::endl;
  }
  
  /**
   * @brief scan larflow3dhit input and assemble 3D keypoint data we will work on.
   *
   * we select points by filttering based on the score_threshold.
   * clears and fills:
   * @verbatim embed:rst:leading-asterisk
   *  * _initial_pt_pos_v;
   *  * _initial_pt_used_v;
   * @endverbatim
   *
   * @param[in] lfhits          LArFlow hits with keypoint network info
   * @param[in] keypoint_score_threshold Only cluster hits with keypoint score above this threshold
   * @param[in] larmatch_score_threshold Only cluster hits with larmatch score above this threshold
   *
   */
  void KeypointReco::_make_initial_pt_data( const std::vector<larlite::larflow3dhit>& lfhits,
                                            const float keypoint_score_threshold,
                                            const float larmatch_score_threshold )
  {

    _initial_pt_pos_v.clear();
    _initial_pt_used_v.clear();

    for (auto const& lfhit : lfhits ) {
      const float& kp_score = lfhit[_lfhit_score_index];
      const float& lm_score = lfhit[9];
      if ( kp_score>keypoint_score_threshold && lm_score>larmatch_score_threshold ) {
        std::vector<float> pos3d(5,0);
        for (int i=0; i<3; i++) pos3d[i] = lfhit[i];
        //pos3d[3] = (kp_score<1.0) ? kp_score : 1.0;
        pos3d[3] = kp_score;
        pos3d[4] = lm_score;
        _initial_pt_pos_v.push_back( pos3d );
      }
    }

    _initial_pt_used_v.resize( _initial_pt_pos_v.size(), 0 );

    LARCV_INFO() << "[larflow::reco::KeypointReco::_make_initial_pt_data] number of points stored for keypoint reco: "
                 << _initial_pt_pos_v.size()
                 << "/"
                 << lfhits.size()
                 << std::endl;
    
  }

  /**
   * @brief Make clusters with remaining points
   *
   * fills member cluster container, output_pt_v.
   * internal data members used:
   * @verbatim embed:rst:leading-asterisk
   *  * _initial_pt_pos_v: the list of points (x,y,z,current score)
   *  * _initial_pt_used_v: ==1 if the point has been claimed
   * @endverbatim
   * 
   * @param[in] keypoint_score_threshold Keypoint score threshold
   * @param[in] min_cluster_size Minimum size of keypoint hit cluster
   *
   */
  void KeypointReco::_make_kpclusters( float keypoint_score_threshold, int min_cluster_size )
  {

    std::vector< std::vector<float> > skimmed_pt_v;
    std::vector< int > skimmed_index_v;

    // collect spacepoints with scores above a threshold
    _skim_remaining_points( keypoint_score_threshold,
                            skimmed_pt_v,
                            skimmed_index_v );


    // cluster the points
    std::vector< cluster_t > cluster_v;
    float maxdist = _max_dbscan_dist;
    int maxkd     = 100;

    LARCV_INFO() << "finding keypoint clusters using " << skimmed_pt_v.size() << " points" << std::endl;
    LARCV_DEBUG() << "  clustering pars: maxdist=" << _max_dbscan_dist
                  << " minsize=" << min_cluster_size
                  << " maxkd=" <<  maxkd
                  << std::endl;
    
    cluster_sdbscan_spacepoints( skimmed_pt_v, cluster_v, maxdist, min_cluster_size, maxkd );    

    float sigma = _sigma; // bandwidth

    for ( auto& cluster : cluster_v ) {

      if ( cluster.points_v.size()<4 )
	continue;
      
      // make kpcluster
      KPCluster kpc     = _characterize_cluster( cluster, skimmed_pt_v, skimmed_index_v );
      KPCluster kpc_fit = _fit_cluster_CARUANA(  cluster, skimmed_pt_v, skimmed_index_v );
      kpc.center_pt_v = kpc_fit.center_pt_v;
      if ( kpc.max_score < _threshold_cluster_max_score )
	continue;
      
      // insert cluster into class continer
      _cluster_v.emplace_back( std::move(cluster) );
      kpc._cluster_idx = (int)_cluster_v.size()-1;      

      kpc._cluster_type = _keypoint_type;
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
    LARCV_INFO() << "[larflow::KeypointReco::_make_kpclusters] number of clusters=" << output_pt_v.size() << std::endl;
    
  }

  /**
   * @brief get list of points to cluster above threshold
   *
   * @param[in]  keypoint_score_threshold   Keep points above threshold.
   * @param[out] skimmed_pt_v      Returned 3D points.
   * @param[out] skimmed_index_v   Index of point in the Original Point list, _initial_pt_pos_v.
   *
   */
  void KeypointReco::_skim_remaining_points( float keypoint_score_threshold,
                                             std::vector<std::vector<float> >& skimmed_pt_v,
                                             std::vector<int>& skimmed_index_v )
  {

    for ( size_t i=0; i<_initial_pt_pos_v.size(); i++ ) {
      if ( _initial_pt_pos_v[i][3]>keypoint_score_threshold ) {
        skimmed_pt_v.push_back( _initial_pt_pos_v[i] );
        skimmed_index_v.push_back(i);
      }
    }
  }

  /**
   * @brief Characterize the keypoint cluster
   *
   * we take the cluster we've made using dbscan and make a KPCluster object
   * we define the centroid using a weighted score
   * we define the pca as well, to help us absorb points
   *
   * @param[in] cluster Cluster to characterize
   * @param[in] skimmed_pt_v     Points used to cluster
   * @param[in] skimmed_index_v  Index of point in the Original Point list, _initial_pt_pos_v.
   * @return Keypoint cluster represented as KPCluster object
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
      
      float w = skimmed_pt_v[ skimidx ][3]*skimmed_pt_v[ skimidx ][4]; // (score * charge)
      if ( w>10.0 ) w = 10.0;
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
    
    LARCV_DEBUG() << "[KeypointReco::_characterize_cluster]" << std::endl;
    LARCV_DEBUG() << "  center: (" << kpc.center_pt_v[0] << "," << kpc.center_pt_v[1] << "," << kpc.center_pt_v[2] << ")" << std::endl;
    LARCV_DEBUG() << "  pca: (" << cluster.pca_axis_v[0][0] << "," << cluster.pca_axis_v[0][1] << "," << cluster.pca_axis_v[0][2] << ")" << std::endl;

    return kpc;
  }

  /**
   * @brief Find the center of the cluster by fitting score pattern
   *
   * The network is trained to predict a score for each spacepoint.
   * The target score is a gaussian with mean at the location of true best point
   * We use Caruana's algorithm taken from a description in https://arxiv.org/pdf/1907.07241.pdf
   * We can solve a linear system of equations by finding a bunch of moments.
   * We solve it 3-times, one for each spatial dimension, as the target Gaussian has no correlations by construction.
   *
   * @param[in] cluster Cluster to characterize
   * @param[in] skimmed_pt_v     Points used to cluster
   * @param[in] skimmed_index_v  Index of point in the Original Point list, _initial_pt_pos_v.
   * @return Keypoint cluster represented as KPCluster object
   */
  KPCluster KeypointReco::_fit_cluster_CARUANA( cluster_t& cluster,
						std::vector< std::vector<float> >& skimmed_pt_v,
						std::vector< int >& skimmed_index_v )
  {

    std::vector<double> mean(3,0);

    for (int dim=0; dim<3; dim++) {

      // we calculate 4 moments
      std::vector<double> x_sum(4,0);
      std::vector<double> lny_terms(3,0);

      for (auto const& idx : cluster.hitidx_v ) {
	auto const& pt = skimmed_pt_v.at( idx );
	for (int n=1; n<=4; n++)
	  x_sum[n-1] += TMath::Power(pt[dim],n);
	double lny = TMath::Log(pt[3]);
	lny_terms[0] += lny;
	lny_terms[1] += pt[dim]*lny;
	lny_terms[2] += pt[dim]*pt[dim]*lny;	
      }
      double N = cluster.hitidx_v.size();
      
      // construct the matrix
      Eigen::Matrix3d A;
      A << N, x_sum[0], x_sum[1],
	   x_sum[0], x_sum[1], x_sum[2],
	   x_sum[1], x_sum[2], x_sum[3];
      Eigen::Vector3d b;
      b << lny_terms[0], lny_terms[1], lny_terms[2];
      
      Eigen::Matrix3d invA = A.inverse();
      Eigen::Vector3d sol = invA*b;
      mean[dim] = -sol(1)/(2*sol(2));
    }

    LARCV_DEBUG() << "Solved for mean position, N=" << cluster.hitidx_v.size() << ": "
		  << "(" << mean[0] << "," << mean[1] << "," << mean[2] << ")" << std::endl;
    
    // return a KPCluster object
    //cluster_pca( cluster );
    
    KPCluster kpc;
    kpc.center_pt_v = { mean[0], mean[1], mean[2] };
    
    return kpc;
  }
  
  /**
   * @brief we absorb points to clusters, using pca-line, proximity, and confidence score
   *
   * unwritten
   * @param[in] kp KPCluster object to expand
   *
   */
  void KeypointReco::_expand_kpcluster( KPCluster& kp )
  {
    // to do
  }
  
  /**
   * @brief dump output to json for development
   *
   * @param[in] outfilename Path of json file to write output to
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
   * @brief Dump all cluster info to standard out
   *
   * prints info for clusters in output_pt_v. calls KPCluster::printInfo()
   *
   */
  void KeypointReco::printAllKPClusterInfo()
  {
    for ( auto const& kp : output_pt_v )
      kp.printInfo();
  }

  /**
   * @brief make branch with pointer to output cluster container
   *
   * Adds a single branch, `kpcluster_v`, to the given ROOT TTree.
   * The branch will hold a vector of KPCluster objects.
   * The branch is given a pointer to the member container output_pt_v.
   * 
   * @param[in] out ROOT TTree to add branch to
   */
  void KeypointReco::bindKPClusterContainerToTree( TTree* out )
  {
    out->Branch( "kpcluster_v", &output_pt_v );
  }


  /** 
   * @brief create TTree instance in class and use it to store output container contents
   *
   * Creates tree named `larflow_keypointreco` and adds branch with
   * KPCluster objects via bindKPClusterContainerToTree().
   * 
   */
  void KeypointReco::setupOwnTree()
  {
    _output_tree = new TTree("larflow_keypointreco", "Reconstructed keypoint clusters");
    bindKPClusterContainerToTree( _output_tree );
  }

  /**
   * @brief parse the output of v2 larmatch keypoint output and produce keypoints
   */

  
}
}
