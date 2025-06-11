#ifndef __LARFLOW_RECO_KEYPOINTRECO_H__
#define __LARFLOW_RECO_KEYPOINTRECO_H__

#include <vector>

#include "TTree.h"

// larcv
#include "larcv/core/Base/larcv_base.h"

// larlite
#include "larlite/DataFormat/larflow3dhit.h"
#include "larlite/DataFormat/storage_manager.h"

#include "larflow/RecoUtils/cluster_functions.h"
#include "larflow/Reco/KPCluster.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class KeypointReco
   * @brief Reconstructs physics keypoints from neural network scores using clustering and fitting
   *
   * This class transforms dense 3D spacepoint clouds with keypoint network scores into
   * discrete, well-characterized physics features such as neutrino vertices, track endpoints,
   * shower starts, and Michel electron vertices. It uses an iterative clustering approach
   * combined with score suppression to identify and characterize the most prominent features
   * while avoiding double-counting.
   *
   * ## Physics Motivation
   *
   * Physics keypoints represent critical features in neutrino interactions:
   * - **Neutrino vertex**: Primary interaction point where neutrino converts to charged leptons
   * - **Track endpoints**: Start/end points of muon or proton tracks
   * - **Shower starts**: Electromagnetic cascade initiation points from electrons/photons
   * - **Michel electrons**: Decay electrons from stopped muons (delayed signature)
   * - **Delta rays**: High-energy electrons knocked out by ionizing particles
   *
   * Accurate keypoint reconstruction is essential for:
   * - Neutrino vertex finding and energy reconstruction
   * - Track and shower direction determination
   * - Background rejection (cosmic vs neutrino discrimination)
   * - Physics analysis (interaction topology, kinematics)
   *
   * ## Algorithm Overview
   *
   * The reconstruction proceeds through multiple passes with progressively relaxed thresholds:
   *
   * ### Phase 1: Data Preparation
   * 1. **Filtering**: Select spacepoints above configurable LArMatch and keypoint score thresholds
   * 2. **Data Structure**: Organize as (x,y,z,keypoint_score,larmatch_score) tuples
   *
   * ### Phase 2: Iterative Clustering (Multi-pass)
   * For each pass (typically 1-3 passes with decreasing thresholds):
   * 1. **Point Selection**: Collect unused points above current threshold
   * 2. **Spatial Clustering**: Apply DBSCAN clustering to group nearby high-scoring points
   * 3. **Cluster Characterization**: Fit Gaussian centroids and compute statistics
   * 4. **Score Suppression**: Reduce scores of nearby points using Gaussian kernel
   * 5. **Quality Control**: Filter clusters by size and maximum score requirements
   *
   * ### Phase 3: Cluster Analysis
   * 1. **PCA Analysis**: Compute principal component axes for cluster shape characterization
   * 2. **Centroid Fitting**: Determine best-fit cluster center using multiple methods
   * 3. **Feature Extraction**: Calculate cluster statistics (size, density, elongation)
   * 4. **Wire Projection**: Convert 3D positions to 2D wire plane coordinates
   *
   * ## Key Features
   *
   * - **Multi-pass processing**: Captures features at different prominence scales
   * - **Score suppression**: Prevents double-counting of the same physical feature
   * - **Adaptive thresholds**: Configurable sensitivity for different keypoint types
   * - **Cluster characterization**: Rich statistical description of each keypoint
   * - **Quality filtering**: Minimum cluster size and score requirements
   * - **PCA analysis**: Shape and orientation information for downstream algorithms
   *
   * ## Algorithm Parameters
   *
   * Critical parameters affecting performance:
   * - **keypoint_score_threshold**: Neural network confidence threshold (0.3-0.8 typical)
   * - **larmatch_score_threshold**: 3D reconstruction quality threshold (0.1-0.5)
   * - **sigma**: Gaussian suppression bandwidth (3-8 cm typical)
   * - **max_dbscan_dist**: Spatial clustering radius (1.5-3.0 cm)
   * - **min_cluster_size**: Minimum points per cluster (20-100 typical)
   * - **num_passes**: Number of reconstruction iterations (1-3 typical)
   *
   * ## Output Products
   *
   * Each reconstructed keypoint is represented by a KPCluster containing:
   * - **3D position**: Best-fit centroid coordinates
   * - **Quality metrics**: Maximum score, cluster size, fitting residuals
   * - **Shape analysis**: PCA axes, eigenvalues, elongation measures
   * - **Wire projections**: 2D coordinates for all three wire planes
   * - **Point lists**: All constituent spacepoints for detailed analysis
   *
   * ## Integration Notes
   *
   * - Designed for processing LArMatch network outputs with keypoint scores
   * - Can process multiple keypoint types (vertex, track ends, showers, etc.)
   * - Outputs compatible with downstream tracking and shower reconstruction
   * - Supports both standalone and I/O framework integration
   * - Optional ROOT tree output for analysis and debugging
   *
   * ## Performance Considerations
   *
   * - DBSCAN clustering is O(n log n) with KD-tree optimization
   * - Memory usage scales with input hit density and clustering parameters
   * - Processing time dominated by clustering step (1-10 seconds typical)
   * - Score suppression requires O(n²) distance calculations for dense regions
   *
   * ## Typical Usage Patterns
   *
   * ```cpp
   * // Configure for neutrino vertex reconstruction
   * KeypointReco vertex_reco;
   * vertex_reco.set_keypoint_type(0);  // neutrino vertex
   * vertex_reco.set_keypoint_threshold(0.5);
   * vertex_reco.set_min_cluster_size(50);
   * vertex_reco.set_sigma(5.0);
   * 
   * // Process spacepoints
   * vertex_reco.process(larmatch_hits);
   * auto& vertices = vertex_reco.output_pt_v;
   * ```
   */
  class KeypointReco : public larcv::larcv_base {

  public:

    /**
     * @brief Default constructor with standard parameter initialization
     *
     * Initializes keypoint reconstruction with commonly used default parameters:
     * - Gaussian suppression sigma: 5.0 cm (typical vertex size)
     * - LArMatch score threshold: 0.5 (medium quality spacepoints)
     * - Keypoint score threshold: 0.5 (medium confidence features)
     * - DBSCAN clustering distance: 2.0 cm (vertex-scale clustering)
     * - Minimum cluster size: 50 hits (substantial feature requirement)
     * - Single pass processing (simple cases)
     * - Input tree: "larmatch", Output tree: "keypoint"
     *
     * These defaults work well for neutrino vertex reconstruction in MicroBooNE.
     * Specific keypoint types may require parameter tuning for optimal performance.
     */
    KeypointReco()
      : larcv::larcv_base("KeypointReco"),
      _output_tree(nullptr)
    { set_param_defaults(); };

    /**
     * @brief Virtual destructor with resource cleanup
     *
     * Properly manages memory for optional ROOT tree output. The tree is owned
     * by this class and will be deleted if it was created via setupOwnTree().
     * Trees bound via bindKPClusterContainerToTree() are not deleted.
     */
    virtual ~KeypointReco()
      {
        if (_output_tree) {
          delete _output_tree;
          _output_tree = nullptr;
        }
      };

    // Algorithm parameters
  protected:

    float _sigma;                                   ///< Gaussian suppression bandwidth in cm (typical: 3-8 cm). Controls how aggressively nearby points are suppressed after cluster formation
    float _larmatch_score_threshold;                ///< Minimum LArMatch reconstruction confidence (typical: 0.1-0.5). Filters poor quality 3D spacepoints
    int   _num_passes;                              ///< Number of reconstruction passes (typical: 1-3). Multiple passes capture features at different prominence scales
    std::vector<float> _keypoint_score_threshold_v; ///< Keypoint network confidence thresholds per pass (typical: 0.3-0.8). Usually decreasing values for subsequent passes
    std::vector<int>   _min_cluster_size_v;         ///< Minimum cluster size per pass (typical: 20-100 hits). Prevents spurious small clusters from being keypoints
    float _max_dbscan_dist;                         ///< DBSCAN clustering radius in cm (typical: 1.5-3.0 cm). Sets spatial scale for grouping nearby high-scoring points
    int   _max_clustering_points;                   ///< Maximum points for clustering (default: no limit). Large values can slow processing; random sampling if exceeded
    std::string _input_larflowhit_tree_name;        ///< Name of input tree containing LArMatch spacepoints with keypoint scores (default: "larmatch")
    std::string _output_tree_name;                  ///< Name of output tree for reconstructed keypoints (default: "keypoint")
    int   _keypoint_type;                           ///< Physics keypoint type ID: 0=neutrino, 1=track_start, 2=track_end, 3=shower, 4=michel, 5=delta
    int   _lfhit_score_index;                       ///< Index in larflow3dhit feature vector containing keypoint network score (keypoint-type dependent)
    float _threshold_cluster_max_score;             ///< Minimum maximum score within cluster to accept as keypoint (typical: 0.3-0.7). Quality control filter
    std::vector< std::string > __keypoint_type_names; ///< Human-readable names for keypoint types (for logging and debugging)
    
  public:
    
    /**
     * @brief Initialize all algorithm parameters to sensible defaults
     *
     * Sets default parameters suitable for neutrino vertex reconstruction:
     * - Suppression sigma: 5.0 cm
     * - LArMatch threshold: 0.5
     * - Keypoint threshold: 0.5 (single pass)
     * - Min cluster size: 50 hits
     * - DBSCAN distance: 2.0 cm
     * - Keypoint type names and I/O tree names
     */
    void set_param_defaults();

    // Configuration setters

    /**
     * @brief Set keypoint score threshold for reconstruction pass
     * @param threshold Minimum keypoint network confidence (0.0-1.0, typical: 0.3-0.8)
     * @param pass Pass number (0-based, default: 0)
     *
     * Higher thresholds are more selective but may miss weak features.
     * Lower thresholds capture more features but increase noise.
     * Multi-pass strategies often use decreasing thresholds.
     */
    void set_keypoint_threshold( float threshold, int pass=0 )    { _keypoint_score_threshold_v[pass] = threshold; };

    /**
     * @brief Set minimum LArMatch reconstruction quality threshold
     * @param threshold Minimum LArMatch confidence (0.0-1.0, typical: 0.1-0.5)
     *
     * Filters out poor quality 3D spacepoint reconstructions that could
     * create spurious keypoints. Higher values improve purity but may
     * reject valid points in complex regions.
     */
    void set_larmatch_threshold( float threshold )    { _larmatch_score_threshold=threshold; };

    /**
     * @brief Set Gaussian suppression bandwidth
     * @param sigma Suppression radius in cm (typical: 3-8 cm)
     *
     * Controls how aggressively nearby points are suppressed after cluster
     * formation. Larger values prevent closer keypoints, smaller values
     * allow finer feature separation. Should match typical physics scales.
     */
    void set_sigma( float sigma )            { _sigma=sigma; };

    /**
     * @brief Set minimum cluster size requirement
     * @param minsize Minimum hits per cluster (typical: 20-100)
     * @param pass Pass number (0-based, default: 0)
     *
     * Prevents small spurious clusters from being accepted as keypoints.
     * Larger values improve purity but may reject real small features.
     * Can be adjusted per pass for multi-scale reconstruction.
     */
    void set_min_cluster_size( int minsize, int pass=0 ) { _min_cluster_size_v[pass] = minsize;  };

    /**
     * @brief Set minimum maximum score within cluster for acceptance
     * @param threshold Minimum peak score in cluster (typical: 0.3-0.7)
     *
     * Quality control filter ensuring that accepted clusters contain
     * at least one high-confidence point. Prevents acceptance of
     * clusters with only marginal keypoint scores.
     */
    void set_threshold_cluster_max_kpscore( float threshold ) { _threshold_cluster_max_score=threshold; };

    /**
     * @brief Set number of reconstruction passes
     * @param npasses Number of passes (typical: 1-3)
     *
     * Multi-pass reconstruction captures features at different prominence
     * scales. Automatically resizes threshold and cluster size vectors
     * with default values. Configure individual pass parameters separately.
     */
    void set_num_passes( int npasses )       {
      _num_passes = npasses;
      _keypoint_score_threshold_v.resize(npasses,0.5);
      _min_cluster_size_v.resize(npasses,50);
    };

    /**
     * @brief Set DBSCAN clustering radius
     * @param dist Maximum distance for point connectivity in cm (typical: 1.5-3.0)
     *
     * Determines spatial scale for clustering nearby high-scoring points.
     * Larger values merge distant features, smaller values allow finer
     * separation. Should match expected keypoint spatial extent.
     */
    void set_max_dbscan_dist( float dist )   { _max_dbscan_dist = dist; };

    /**
     * @brief Set maximum number of points for clustering
     * @param maxpts Maximum points to cluster (default: unlimited)
     *
     * Performance optimization for very dense events. If exceeded,
     * points are randomly sampled. Set based on available memory
     * and acceptable processing time.
     */
    void set_max_clustering_points( int maxpts ) { _max_clustering_points = maxpts; };

    /**
     * @brief Set physics keypoint type for this reconstructor instance
     * @param kptype Keypoint type ID: 0=neutrino, 1=track_start, 2=track_end, 3=shower, 4=michel, 5=delta
     *
     * Each keypoint type may require different network score indices and
     * algorithm parameters. This setting affects which score column is
     * used from the input larflow3dhit data structure.
     */
    void set_keypoint_type (int kptype ) { _keypoint_type=kptype; };

    /**
     * @brief Set input tree name containing LArMatch spacepoints
     * @param name Tree name with larflow3dhit objects (default: "larmatch")
     *
     * Input tree should contain spacepoints with both LArMatch reconstruction
     * scores and keypoint network scores in the feature vector.
     */
    void set_input_larmatch_tree_name( std::string name ) { _input_larflowhit_tree_name=name; };

    /**
     * @brief Set output tree name for reconstructed keypoints
     * @param name Tree name for keypoint output (default: "keypoint")
     *
     * Output contains reconstructed keypoints as larflow3dhit objects
     * with associated PCA analysis data for shape characterization.
     */
    void set_output_tree_name( std::string name ) { _output_tree_name=name; };

    /**
     * @brief Set feature vector index containing keypoint scores
     * @param idx Index in larflow3dhit info vector (keypoint-type dependent)
     *
     * Different keypoint types store their network scores in different
     * columns of the larflow3dhit feature vector. This index determines
     * which score column is used for reconstruction.
     */
    void set_lfhit_score_index( int idx ) { _lfhit_score_index=idx; };

    /**
     * @brief Clear all output containers and intermediate data
     *
     * Resets the algorithm state for processing a new event. Clears:
     * - Reconstructed keypoint clusters (output_pt_v)
     * - Intermediate cluster data (_cluster_v)
     * - Initial point data and usage flags
     *
     * Call before processing each new event to ensure clean state.
     */
    void clear_output();

    // Main processing methods

    /**
     * @brief Process keypoint reconstruction using I/O framework
     *
     * High-level interface that reads spacepoints from the configured input tree,
     * performs keypoint reconstruction, and saves results to the output tree.
     * Also saves associated PCA analysis data for cluster shape characterization.
     *
     * Algorithm flow:
     * 1. Load spacepoints from input tree (_input_larflowhit_tree_name)
     * 2. Execute reconstruction via process(vector) method
     * 3. Convert KPCluster objects to larflow3dhit format
     * 4. Save to output tree (_output_tree_name) with PCA data
     *
     * @param io_ll larlite storage manager for I/O operations
     */
    void process( larlite::storage_manager& io_ll );

    /**
     * @brief Process keypoint reconstruction from spacepoint vector
     *
     * Core reconstruction method that operates on a vector of 3D spacepoints
     * with keypoint network scores. This is the main algorithmic entry point.
     *
     * Algorithm execution:
     * 1. **Data Preparation**: Filter and organize input spacepoints
     * 2. **Multi-pass Clustering**: Execute configured number of passes
     * 3. **Cluster Characterization**: Fit centroids and compute statistics
     * 4. **Score Suppression**: Prevent double-counting via Gaussian suppression
     * 5. **Quality Control**: Apply size and score thresholds
     *
     * Results stored in output_pt_v container as KPCluster objects.
     *
     * @param input_lfhits Vector of spacepoints with LArMatch and keypoint scores
     */
    void process( const std::vector<larlite::larflow3dhit>& input_lfhits );

    /**
     * @brief Export reconstruction results to JSON format
     * @param outfilename Output JSON filename (default: "dump_keypointreco.json")
     *
     * Saves detailed reconstruction information in human-readable JSON format
     * for debugging, analysis, and visualization. Includes cluster positions,
     * scores, PCA analysis, and constituent point lists.
     */
    void dump2json( std::string outfilename="dump_keypointreco.json" );    

    // Output containers
    std::vector< KPCluster > output_pt_v;  ///< Reconstructed keypoint clusters with full characterization data
    std::vector< recoutils::cluster_t >   _cluster_v; ///< Intermediate DBSCAN cluster objects (for debugging and analysis)

    // Internal algorithm data structures
    std::vector< std::vector<float> > _initial_pt_pos_v;  ///< Working point data: (x,y,z,current_keypoint_score,larmatch_score). Scores updated during suppression
    std::vector< int >                _initial_pt_used_v; ///< Usage flags for each point: 0=available, 1=consumed by a cluster

  protected:
    
    // Core algorithm methods

    /**
     * @brief Prepare filtered point data for clustering
     *
     * Filters input spacepoints based on score thresholds and organizes them
     * into the internal working format. This is the first step of reconstruction.
     *
     * @param lfhits Input spacepoints with network scores
     * @param keypoint_score_threshold Minimum keypoint network confidence
     * @param larmatch_score_threshold Minimum LArMatch reconstruction quality
     *
     * Populates _initial_pt_pos_v and _initial_pt_used_v containers.
     */
    void _make_initial_pt_data( const std::vector<larlite::larflow3dhit>& lfhits,
                                const float keypoint_score_threshold,
                                const float larmatch_score_threshold );

    /**
     * @brief Execute one pass of keypoint clustering
     *
     * Core clustering algorithm that finds keypoint candidates above threshold,
     * applies DBSCAN clustering, characterizes clusters, and suppresses scores
     * of nearby points to prevent double-counting.
     *
     * @param round_score_threshold Keypoint score threshold for this pass
     * @param min_cluster_size Minimum cluster size for acceptance
     *
     * Results added to output_pt_v container.
     */
    void _make_kpclusters( float round_score_threshold, int min_cluster_size );
    
    // Algorithm helper methods

    /**
     * @brief Collect unused points above threshold for clustering
     *
     * Extracts points from _initial_pt_pos_v that haven't been used and
     * have scores above the threshold. Prepares data for DBSCAN clustering.
     *
     * @param score_threshold Minimum score for point inclusion
     * @param skimmed_pt_v Output: filtered point coordinates
     * @param skimmed_index_v Output: indices into original _initial_pt_pos_v
     */
    void _skim_remaining_points( float score_threshold,
                                 std::vector<std::vector<float> >& skimmed_pt_v,
                                 std::vector<int>& skimmed_index_v );

    /**
     * @brief Compute basic cluster statistics and properties
     *
     * Analyzes a DBSCAN cluster to extract basic properties like centroid,
     * maximum score, size, and constituent point information.
     *
     * @param cluster DBSCAN cluster object
     * @param skimmed_pt_v Point coordinates used in clustering
     * @param skimmed_index_v Indices mapping to original point data
     * @return KPCluster with basic characterization data
     */
    KPCluster _characterize_cluster( recoutils::cluster_t& cluster,
                                     std::vector< std::vector<float> >& skimmed_pt_v,
                                     std::vector< int >& skimmed_index_v );

    /**
     * @brief Fit cluster centroid using Caruana algorithm
     *
     * Advanced centroid fitting method that provides more accurate position
     * estimates than simple averaging. Includes uncertainty quantification.
     *
     * @param cluster DBSCAN cluster object
     * @param skimmed_pt_v Point coordinates used in clustering
     * @param skimmed_index_v Indices mapping to original point data
     * @return KPCluster with fitted centroid and uncertainties
     */
    KPCluster _fit_cluster_CARUANA( recoutils::cluster_t& cluster,
				    std::vector< std::vector<float> >& skimmed_pt_v,
				    std::vector< int >& skimmed_index_v );
    
    /**
     * @brief Expand cluster with additional nearby points (placeholder)
     *
     * Reserved for future algorithm enhancement. Currently does nothing
     * but provides framework for cluster expansion post-processing.
     *
     * @param kp Keypoint cluster to potentially expand
     */
    void _expand_kpcluster( KPCluster& kp );

    /**
     * @brief Print detailed information about all reconstructed clusters
     *
     * Debug utility that outputs comprehensive cluster information including
     * positions, scores, sizes, and fitting statistics. Enabled only when
     * logging level is set to DEBUG or higher.
     */
    void printAllKPClusterInfo();

  protected:

    TTree* _output_tree; ///< Optional ROOT tree for detailed analysis output. Managed by this class if created via setupOwnTree()

  public:

    // ROOT tree interface (optional analysis output)

    /**
     * @brief Bind reconstruction output to external ROOT tree
     * @param out ROOT tree to bind output containers to
     *
     * Links the output_pt_v container to branches in an external ROOT tree
     * for detailed analysis. The tree is NOT owned by this class and will
     * not be deleted in the destructor.
     */
    void bindKPClusterContainerToTree( TTree* out );    

    /**
     * @brief Create internal ROOT tree for analysis output
     *
     * Creates a ROOT tree owned by this class with branches for detailed
     * reconstruction analysis. Tree will be deleted in destructor.
     * Use writeTree() to save to file.
     */
    void setupOwnTree();

    /**
     * @brief Write ROOT tree to current ROOT file
     *
     * Saves the ROOT tree (if configured) to the currently open ROOT file.
     * Only works if tree was created via setupOwnTree() or bound via
     * bindKPClusterContainerToTree().
     */
    void writeTree() { if ( _output_tree ) _output_tree->Write(); };

    /**
     * @brief Fill one entry in ROOT tree
     *
     * Adds current reconstruction results as one entry in the ROOT tree.
     * Call after each event's reconstruction is complete. Only works if
     * tree interface has been configured.
     */
    void fillTree() {  if ( _output_tree ) _output_tree->Fill(); };

  };

}
}

#endif
