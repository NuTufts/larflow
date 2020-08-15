#ifndef __LARFLOW_PROJECTION_DEFECT_SPLITTER_H__
#define __LARFLOW_PROJECTION_DEFECT_SPLITTER_H__

#include <vector>
#include <string>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/track.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class ProjectionDefectSplitter
   * @brief Splits 3D clusters by projecting into 2D and breaking cluster based on cluster defect points
   *
   * This is a way to try and produce very straight clusters, which we can then piece 
   * back together to make tracks.
   *
   */
  class ProjectionDefectSplitter : public larcv::larcv_base {
    
  public:

    /** @brief default construct */
    ProjectionDefectSplitter()
      : larcv::larcv_base("ProjectionDefectSplitter"),
      _input_lfhit_tree_name("larmatch"),
      _output_cluster_tree_name("projsplit"),
      _min_larmatch_score(0.0),
      _maxdist(2.0),
      _minsize(20),
      _maxkd(30),
      _veto_hits_around_keypoints(false),
      _fit_line_segments_to_clusters(false)      
      {};
    virtual ~ProjectionDefectSplitter() {};

    void process( larcv::IOManager& iolc, larlite::storage_manager& ioll );

    int split_clusters( std::vector<cluster_t>& cluster_v,
                        const std::vector<larcv::Image2D>& adc_v,
                        const float min_second_pca_len );

  protected:

    larlite::larflowcluster
      _makeLArFlowCluster( cluster_t& cluster,
                           const larlite::event_larflow3dhit& source_lfhit_v );
    
    cluster_t _absorb_nearby_hits( const cluster_t& cluster,
                                   const std::vector<larlite::larflow3dhit>& hit_v,
                                   std::vector<int>& used_hits_v,
                                   std::vector<larlite::larflow3dhit>& downsample_hit_v,
                                   std::vector<int>& orig_idx_v,                       
                                   float max_dist2line );
    
    void _runSplitter( const larlite::event_larflow3dhit& inputhits,
                       const std::vector<larcv::Image2D>& adc_v,
                       std::vector<int>& used_hits_v,
                       std::vector<cluster_t>& output_cluster_v );

    void _defragment_clusters( std::vector<cluster_t>& cluster_v,
                               const float max_2nd_pca_eigenvalue );
    
    // PARAMETER NAMES
  protected:
    
    std::string _input_lfhit_tree_name;     ///< name of tree to get larflow hits to cluster
    std::string _output_cluster_tree_name;  ///< name of tree to store output clusters
    float _min_larmatch_score;              ///< minimum larmatch score spacepoint must have to be included
    float _maxdist;                         ///< maximum distance two spacepoints can be connected for dbscan
    int   _minsize;                         ///< minimum cluster size for dbscan
    int   _maxkd;                           ///< max number of neighbors per spaecepoint for dbscan

  public:

    /** @brief set name of tree with larflow3dhit instnces to cluster */
    void set_input_larmatchhit_tree_name( std::string name ) { _input_lfhit_tree_name=name; };

    /** @brief set name of the tree to write output clusters */
    void set_output_tree_name( std::string name ) { _output_cluster_tree_name=name; };

    /** @brief set minimum larmatch score must have to be included in clusters */
    void set_min_larmatch_score( float min_score ) { _min_larmatch_score = min_score; };


    /** 
     * @brief set the dbscan parameters for clustering 
     * @param[in] maxdist maximum distance two spacepoints can be connected for dbscan
     * @param[in] minsize minimum cluster size for dbscan
     * @param[in] maxkd   max number of neighbors per spaecepoint for dbscan
     */
    void set_dbscan_pars( float maxdist, int minsize, int maxkd ) {
      _maxdist = maxdist;
      _minsize = minsize;
      _maxkd = maxkd;
    };

    // OPTIONAL INPUTS

  public:
    
    void add_input_keypoint_treename_for_hitveto( std::string name );
    
  protected:

    bool _veto_hits_around_keypoints; ///< if true, veto hits around keypoints to help separate particle clusters
    std::vector< std::string > _keypoint_veto_trees_v; ///< contains name of keypoint tree names for vetoing hits
    std::vector< const larlite::event_larflow3dhit* > _event_keypoint_for_veto_v;  
    int _veto_hits_using_keypoints( const larlite::event_larflow3dhit& inputhits,
                                    std::vector<int>& used_hits_v );

    bool _fit_line_segments_to_clusters; ///< if true, run routine to fit line segments to the different clusters

  public:
    
    static void fitLineSegmentsToClusters( const std::vector<larflow::reco::cluster_t>& cluster_v,
                                           const larlite::event_larflow3dhit& lfhit_v,
                                           const std::vector<larcv::Image2D>& adc_v,
                                           larlite::event_track& evout_track );
    
    static larlite::track fitLineSegmentToCluster( const larflow::reco::cluster_t& cluster,
                                                   const larlite::event_larflow3dhit& lfhit_v,
                                                   const std::vector<larcv::Image2D>& adc_v );
  public:

    /** @brief set the flag that determines if linesegments are fit to the clusters */
    void set_fit_line_segments_to_clusters( bool fit ) { _fit_line_segments_to_clusters=fit; };
    
    
    
  };

}
}

#endif
