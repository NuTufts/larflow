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

#include "larflow/RecoUtils/cluster_functions.h"



namespace larflow {
namespace reco {

  /** 
   * @brief Initialize algorithm parameters to sensible defaults for neutrino vertex reconstruction
   *
   * Sets parameters optimized for MicroBooNE neutrino vertex finding:
   * - Gaussian suppression: 5.0 cm (typical vertex size scale)
   * - Score thresholds: 0.5 for both LArMatch and keypoint networks
   * - DBSCAN clustering: 2.0 cm radius (vertex-scale feature grouping)
   * - Minimum cluster size: 50 hits (substantial feature requirement)
   * - Single reconstruction pass (simple case)
   * - Standard I/O tree names for LArFlow workflow integration
   * - Physics keypoint type names for human-readable logging
   *
   * These defaults work well for most neutrino vertex reconstruction tasks.
   * Specific keypoint types or detector configurations may require tuning.
   */
  void KeypointReco::set_param_defaults()
  {
    _sigma = 5.0; // cm - Gaussian suppression bandwidth
    _larmatch_score_threshold = 0.5;    // Medium quality spacepoint requirement
    _num_passes = 1;                    // Single pass reconstruction
    _keypoint_score_threshold_v = std::vector<float>( 2, 0.5 );  // Medium confidence threshold
    _min_cluster_size_v = std::vector<int>(2,50);               // Substantial cluster requirement
    _max_dbscan_dist = 2.0;            // cm - Vertex-scale clustering
    _input_larflowhit_tree_name = "larmatch";  // Standard LArFlow input
    _output_tree_name = "keypoint";            // Standard keypoint output
    _keypoint_type = -1;                       // Unspecified type (must be set)
    _threshold_cluster_max_score = 0.5;        // Quality control threshold
    
    // Human-readable keypoint type names for logging and debugging
    __keypoint_type_names.resize(6);
    __keypoint_type_names[0] = "nu";          // Neutrino interaction vertex
    __keypoint_type_names[1] = "trackstart";  // Track starting point
    __keypoint_type_names[2] = "trackend";    // Track ending point
    __keypoint_type_names[3] = "shower";      // Electromagnetic shower start
    __keypoint_type_names[4] = "michel";      // Michel electron vertex
    __keypoint_type_names[5] = "delta";       // Delta ray interaction point
  }

  /**
   * @brief Reset all output containers and algorithm state for new event processing
   *
   * Clears all data structures to prepare for processing a new event:
   * - output_pt_v: Reconstructed keypoint clusters from previous event
   * - _cluster_v: Intermediate DBSCAN cluster objects
   * - _initial_pt_pos_v: Working point data with updated scores
   * - _initial_pt_used_v: Point usage flags
   *
   * Call this before processing each new event to ensure clean algorithm state.
   * Essential for proper multi-event processing in analysis frameworks.
   */
  void KeypointReco::clear_output()
  {
    output_pt_v.clear();              // Clear reconstructed keypoint clusters
    _cluster_v.clear();               // Clear intermediate cluster data
    _initial_pt_pos_v.clear();        // Clear working point coordinates/scores
    _initial_pt_used_v.clear();       // Clear point usage tracking
  }
  
  /**
   * @brief Process keypoint reconstruction using I/O framework integration
   *
   * High-level interface that manages data I/O using the larlite storage framework.
   * Loads spacepoints from the configured input tree, executes reconstruction,
   * and saves results to output trees with proper data format conversion.
   *
   * Algorithm workflow:
   * 1. **Data Loading**: Read spacepoints from _input_larflowhit_tree_name
   * 2. **Reconstruction**: Execute core algorithm via process(vector) method
   * 3. **Format Conversion**: Convert KPCluster objects to larlite format
   * 4. **Data Saving**: Store keypoints and PCA data to output trees
   *
   * Output data products:
   * - larflow3dhit objects containing keypoint positions and metadata
   * - pcaxis objects containing PCA analysis (shape characterization)
   *
   * @param io_ll larlite storage manager for reading input and writing output
   */
  void KeypointReco::process( larlite::storage_manager& io_ll )
  {
    // Load input spacepoints with keypoint network scores
    larlite::event_larflow3dhit* ev_larflow_hit
      = (larlite::event_larflow3dhit*)io_ll.get_data( larlite::data::kLArFlow3DHit, _input_larflowhit_tree_name );

    LARCV_NORMAL() << "Processing " << ev_larflow_hit->size() << " input spacepoints from tree=\"" 
                   << _input_larflowhit_tree_name << "\"" << std::endl;
    
    // Execute core reconstruction algorithm
    process( *ev_larflow_hit );

    // Prepare output containers for results
    larlite::event_larflow3dhit* evout_keypoint =
      (larlite::event_larflow3dhit*)io_ll.get_data( larlite::data::kLArFlow3DHit, _output_tree_name );
    larlite::event_pcaxis* evout_pcaxis =
      (larlite::event_pcaxis*)io_ll.get_data( larlite::data::kPCAxis, _output_tree_name );

    // Convert KPCluster objects to larlite data format and save
    int cidx = 0;
    for ( auto const& kpc : output_pt_v ) {
      
      // Convert cluster to standard larflow3dhit format
      // This includes 3D position, keypoint type, and confidence score
      larlite::larflow3dhit hit = kpc.as_larflow_hit();

      // Convert cluster PCA analysis to standard pcaxis format
      // This includes eigenvalues, eigenvectors, and shape characterization
      larlite::pcaxis llpca = kpc.get_pcaxis(cidx);
      
      // Store both data products with matching indices
      evout_keypoint->emplace_back( std::move(hit) );
      evout_pcaxis->emplace_back( std::move(llpca) );
      cidx++;
    }
    
    // Log reconstruction results
    std::string kptypename = (_keypoint_type >= 0 && _keypoint_type < 6) ? 
                             __keypoint_type_names[_keypoint_type] : "(unspecified)";
    LARCV_NORMAL() << "Reconstructed " << output_pt_v.size() << " keypoints of type \"" 
                   << kptypename << "\" (ID=" << _keypoint_type << ")" << std::endl;
  }
  
  /**
   * @brief Core keypoint reconstruction algorithm operating on spacepoint vectors
   *
   * This is the main algorithmic entry point that implements the multi-pass keypoint
   * reconstruction with score suppression. The algorithm identifies prominent features
   * in the keypoint score landscape through iterative clustering and suppression.
   *
   * Algorithm phases:
   * 1. **Data Preparation**: Filter and organize input spacepoints by score thresholds
   * 2. **Multi-pass Clustering**: For each configured pass:
   *    - Select points above current threshold
   *    - Apply DBSCAN spatial clustering
   *    - Characterize and fit cluster centroids
   *    - Suppress nearby point scores to prevent double-counting
   * 3. **Quality Control**: Apply minimum cluster size and score requirements
   * 4. **Logging**: Output reconstruction statistics and debug information
   *
   * The multi-pass approach captures features at different prominence scales:
   * - Early passes find the most confident, prominent features
   * - Later passes with lower thresholds find weaker features
   * - Score suppression prevents the same physical feature from being found multiple times
   *
   * Results stored in output_pt_v as fully characterized KPCluster objects.
   *
   * @param input_lfhits Vector of spacepoints with LArMatch and keypoint network scores
   */
  void KeypointReco::process( const std::vector<larlite::larflow3dhit>& input_lfhits )
  {
    // Phase 1: Prepare filtered point data for clustering
    _make_initial_pt_data( input_lfhits, _keypoint_score_threshold_v.front(), _larmatch_score_threshold );

    // Phase 2: Multi-pass clustering with progressively relaxed thresholds
    for (int i=0; i<_num_passes; i++ ) {
      LARCV_INFO() << "Keypoint reconstruction pass " << i+1 << "/" << _num_passes << std::endl;
      
      // Execute clustering for this pass
      _make_kpclusters( _keypoint_score_threshold_v[i], _min_cluster_size_v[i] );
      
      // Log progress for this pass
      LARCV_INFO() << "Pass " << i+1 << ": " << output_pt_v.size() << " total clusters found" << std::endl;
      
      // Count remaining candidate points above threshold
      int nabove = 0;
      for (auto& posv : _initial_pt_pos_v ) {
        if (posv[3] > _keypoint_score_threshold_v[i]) nabove++;
      }
      LARCV_INFO() << "Pass " << i+1 << ": " << nabove << "/" << _initial_pt_pos_v.size() 
                   << " points remain above threshold" << std::endl;
    }

    // Phase 3: Debug output and final logging
    if ( logger().level() <= larcv::msg::kDEBUG ) {
      printAllKPClusterInfo();  // Detailed cluster information for debugging
    }
    
    // Generate human-readable summary
    std::string kptypename = (_keypoint_type >= 0 && _keypoint_type < 6) ? 
                             __keypoint_type_names[_keypoint_type] : "(unspecified)";
    LARCV_NORMAL() << "Keypoint reconstruction complete: type=\"" << kptypename 
                   << "\" (ID=" << _keypoint_type << ", score_index=" << _lfhit_score_index 
                   << ") → " << output_pt_v.size() << " keypoint clusters" << std::endl;
  }
  
  /**
   * @brief Filter and prepare spacepoint data for clustering algorithm
   *
   * This method performs the first phase of keypoint reconstruction by filtering
   * the input spacepoints based on quality thresholds and organizing them into
   * the internal working format used by the clustering algorithm.
   *
   * Filtering criteria:
   * - Keypoint network score ≥ keypoint_score_threshold
   * - LArMatch reconstruction score ≥ larmatch_score_threshold
   *
   * Data organization:
   * - Creates working point array: (x, y, z, current_keypoint_score, larmatch_score)
   * - Initializes usage flags for score suppression tracking
   * - Computes score statistics for algorithm monitoring
   *
   * The current_keypoint_score (index 3) will be modified during reconstruction
   * via Gaussian suppression to prevent double-counting of the same features.
   *
   * @param lfhits Input spacepoints with network scores
   * @param keypoint_score_threshold Minimum keypoint network confidence
   * @param larmatch_score_threshold Minimum LArMatch reconstruction quality
   *
   * Populates: _initial_pt_pos_v (working coordinates/scores), _initial_pt_used_v (usage tracking)
   */
  void KeypointReco::_make_initial_pt_data( const std::vector<larlite::larflow3dhit>& lfhits,
                                            const float keypoint_score_threshold,
                                            const float larmatch_score_threshold )
  {
    // Clear previous event data
    _initial_pt_pos_v.clear();
    _initial_pt_used_v.clear();

    // Track score statistics for algorithm monitoring
    float min_score = 100.0;
    float max_score = 0.0;

    // Filter spacepoints based on quality thresholds
    for (auto const& lfhit : lfhits ) {
      const float& kp_score = lfhit[_lfhit_score_index];  // Keypoint network score
      const float& lm_score = lfhit[9];                   // LArMatch score (standard index)
      
      // Apply dual quality filter
      if ( kp_score > keypoint_score_threshold && lm_score > larmatch_score_threshold ) {
        
        // Create working point data: (x, y, z, current_kp_score, lm_score)
        std::vector<float> pos3d(5, 0);
        for (int i=0; i<3; i++) pos3d[i] = lfhit[i];  // 3D coordinates
        pos3d[3] = kp_score;  // Current keypoint score (will be modified by suppression)
        pos3d[4] = lm_score;  // LArMatch score (unchanged)
        
        // Update score statistics
        if ( kp_score < min_score ) min_score = kp_score;
        if ( kp_score > max_score ) max_score = kp_score;
        
        _initial_pt_pos_v.push_back( pos3d );
      }
    }

    // Initialize usage tracking (0 = available, 1 = consumed by cluster)
    _initial_pt_used_v.resize( _initial_pt_pos_v.size(), 0 );

    // Log data preparation results
    LARCV_NORMAL() << "Data preparation complete: " << _initial_pt_pos_v.size() << " / "
		   << lfhits.size() << " spacepoints passed thresholds, score range: " 
		   << min_score << " → " << max_score << std::endl;
  }

  /**
   * @brief Execute one pass of the clustering and suppression algorithm
   *
   * This method implements the core reconstruction logic for a single pass:
   * spatial clustering of high-scoring points, cluster characterization and fitting,
   * quality control, and Gaussian score suppression to prevent double-counting.
   *
   * Algorithm steps:
   * 1. **Point Selection**: Collect unused points above current threshold
   * 2. **Spatial Clustering**: Apply DBSCAN to group nearby high-scoring points
   * 3. **Cluster Analysis**: Characterize each cluster (basic stats + advanced fitting)
   * 4. **Quality Control**: Filter by cluster size and maximum score requirements
   * 5. **Score Suppression**: Reduce scores of nearby points using Gaussian kernel
   * 6. **Storage**: Add accepted clusters to output container
   *
   * The Gaussian suppression is critical for multi-pass reconstruction:
   * - Suppression radius controlled by _sigma parameter (typically 3-8 cm)
   * - Score reduction: new_score = old_score - cluster_max_score * exp(-dist²/2σ²)
   * - Points with suppressed scores ≤ 0 are marked as used
   * - Prevents the same physical feature from being reconstructed multiple times
   *
   * Quality control ensures robust reconstruction:
   * - Minimum 4 points per cluster (basic geometric requirement)
   * - Configurable minimum cluster size (statistical significance)
   * - Maximum score threshold (confidence requirement)
   *
   * @param keypoint_score_threshold Minimum score for point inclusion in this pass
   * @param min_cluster_size Minimum number of points required for cluster acceptance
   *
   * Results added to output_pt_v container and _cluster_v for analysis.
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
    std::vector< recoutils::cluster_t > cluster_v;
    float maxdist = _max_dbscan_dist;
    int maxkd     = 100;

    LARCV_INFO() << "finding keypoint clusters using " << skimmed_pt_v.size() << " points" << std::endl;
    LARCV_INFO() << "  clustering pars: maxdist=" << _max_dbscan_dist
                  << " minsize=" << min_cluster_size
                  << " maxkd=" <<  maxkd
                  << std::endl;
    
    cluster_sdbscan_spacepoints( skimmed_pt_v, cluster_v, maxdist, min_cluster_size, maxkd );    

    float sigma = _sigma; // bandwidth

    LARCV_INFO() << "  dbscan returns with " << cluster_v.size() << " clusters" << std::endl;

    for ( auto& cluster : cluster_v ) {

      if ( cluster.points_v.size()<4 )
      	continue;
      
      // make kpcluster
      KPCluster kpc     = _characterize_cluster( cluster, skimmed_pt_v, skimmed_index_v );
      KPCluster kpc_fit = _fit_cluster_CARUANA(  cluster, skimmed_pt_v, skimmed_index_v );
      kpc.center_pt_v = kpc_fit.center_pt_v;
      kpc.center_pt_rmse_v = kpc_fit.center_pt_rmse_v;
      kpc.center_pt_rsqr_v = kpc_fit.center_pt_rsqr_v;
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
    LARCV_INFO() << "number of clusters=" << output_pt_v.size() << std::endl;
    
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
  KPCluster KeypointReco::_characterize_cluster( recoutils::cluster_t& cluster,
                                                 std::vector< std::vector<float> >& skimmed_pt_v,
                                                 std::vector< int >& skimmed_index_v )
  {

    // run pca
    cluster_pca( cluster );

    KPCluster kpc;
    kpc.center_pt_v.resize(3,0.0); //will get overwritten by Gaussian fit
    kpc.center_avg_pt_v.resize(3,0.0);
    kpc.max_pt_v.resize(4,0.0);

    float totw = 0.;
    float max_score = 0.;
    int   max_idx = -1;
    for ( int i=0; i<(int)cluster.points_v.size(); i++ ) {

      int skimidx = cluster.hitidx_v[i];
      
      //float w = skimmed_pt_v[ skimidx ][3]*skimmed_pt_v[ skimidx ][4]; // (score * charge)
      float w = skimmed_pt_v[ skimidx ][3]; // (keypoint score)
      //if ( w>10.0 ) w = 10.0; // only needed if charge is combined with score
      if ( w<0.0 ) w = 0.0;
      
      for (int v=0; v<3; v++ ) {
        kpc.center_pt_v[v] += w*w*skimmed_pt_v[ skimidx ][v];
        kpc.center_avg_pt_v[v] += w*w*skimmed_pt_v[ skimidx ][v];
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
        kpc.center_avg_pt_v[v] /= totw;
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
  KPCluster KeypointReco::_fit_cluster_CARUANA( recoutils::cluster_t& cluster,
						std::vector< std::vector<float> >& skimmed_pt_v,
						std::vector< int >& skimmed_index_v )
  {

    std::vector<double> mean(3,0);
    std::vector<float> rmse(3,-1);
    std::vector<float> rsqr(3,-1);
    float avg_score = 0.;

    for (int dim=0; dim<3; dim++) {

      // we calculate 4 moments
      std::vector<double> x_sum(4,0);
      std::vector<double> lny_terms(3,0);

      for (auto const& idx : cluster.hitidx_v ) {
        auto const& pt = skimmed_pt_v.at( idx );
	if(dim == 0) avg_score += pt[3];
        for (int n=1; n<=4; n++)
          x_sum[n-1] += TMath::Power(pt[dim],n);
        double lny = TMath::Log(pt[3]);
        lny_terms[0] += lny;
        lny_terms[1] += pt[dim]*lny;
        lny_terms[2] += pt[dim]*pt[dim]*lny;	
      }
      double N = cluster.hitidx_v.size();
      if(N > 0. && dim == 0) avg_score /= N;
      
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

      if(sol(2) <= 0.){ //standard deviation is a real number
        // calculate goodness of fit metrics
        float fit_stnd = TMath::Sqrt(-1.0/(2*sol(2)));
        float fit_norm = TMath::Exp(sol(0) - (sol(1)*sol(1))/(4*sol(2)));
        float res_sq = 0.;
        float dev_sq = 0.;
        for (auto const& idx : cluster.hitidx_v ) {
          auto const& pt = skimmed_pt_v.at( idx );
          float val_fit = fit_norm*TMath::Exp(-0.5*TMath::Power( (pt[dim] - mean[dim])/fit_stnd, 2));
          res_sq += TMath::Power(pt[3] - val_fit, 2);
          dev_sq += TMath::Power(pt[3] - avg_score, 2);
        }
        if(N > 0) rmse[dim] = TMath::Sqrt(res_sq/N);
        if(dev_sq > 0) rsqr[dim] = 1.0 - res_sq/dev_sq;
      }
    }

    LARCV_DEBUG() << "Solved for mean position, N=" << cluster.hitidx_v.size() << ": "
		  << "(" << mean[0] << "," << mean[1] << "," << mean[2] << ")" << std::endl;
    
    // return a KPCluster object
    //cluster_pca( cluster );
    
    KPCluster kpc;
    kpc.center_pt_v = { (float)mean[0], (float)mean[1], (float)mean[2] };
    kpc.center_pt_rmse_v = { (float)rmse[0], (float)rmse[1], (float)rmse[2] };
    kpc.center_pt_rsqr_v = { (float)rsqr[0], (float)rsqr[1], (float)rsqr[2] };
    
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
    o << j;
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
