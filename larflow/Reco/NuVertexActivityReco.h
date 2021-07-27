#ifndef __LARFLOW_RECO_NUVERTEX_ACTIVITYRECO_H__
#define __LARFLOW_RECO_NUVERTEX_ACTIVITYRECO_H__

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/track.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/LArbysMC.h"
#include "cluster_functions.h"
#include "TGraph.h"
#include "ShowerdQdx.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class NuVertexActivityReco
   *
   * Uses larmatch points to find 3D-consistent point-like deposition.
   * Filter out background and real candiates using larmatch keypoint candidates.
   * Emphasis is end of showers.
   *
   */
  class NuVertexActivityReco : public larcv::larcv_base {
    
  public:
    
    NuVertexActivityReco()
      : larcv::larcv_base("NuVertexActivityReco"),
      _output_treename("vacand"),      
      _va_ana_tree(nullptr),
      _kown_tree(false)
        {};
    virtual ~NuVertexActivityReco() { if ( _kown_tree && _va_ana_tree ) delete _va_ana_tree; };


    /**
     * @struct VACandidate_t
     * @brief Internal struct for storing VACandidate variables for selection
     */
    struct VACandidate_t {
      larlite::larflow3dhit lfhit; ///< the hit we propose
      std::vector<float> va_dir;   ///< direction of va candidate, from 1st pca-axis of attached cluster
      int hit_index; ///< index of hit in the source hit container
      const larflow::reco::cluster_t* pattached; //< pointer to attached cluster
      int attached_cluster_index; ///< index of cluster in cluster container

      larlite::track trunk;
      std::vector<TGraph> seg_dqdx_v;

      float attclust_length; ///< attached cluster: 1st pca-axis length
      int attclust_nallhits; ///< attached cluster: number of hits
      int attclust_ntrackhits; ///< attached cluster: number of track hits
      int attclust_nshowerhits; ///< attached cluster: number of shower hits
      float backward_length;  ///< length of line of hits in opposite direction
      int nhits_inside_cone;  ///< num shower hits inside cone around va_dir
      int nhits_outside_cone; ///< num shower hits outside cone around va_dir

      std::vector< const larflow::reco::cluster_t* > subcluster_v; ///< added shower subclusters
      int nhits_all; ///< number of hits inside attached + subclusters
      int nhits_all_shower; ///< number of shower hits inside attached + subclusters
      int nhits_all_track;  ///< number of track hits inside attached + subclusters

      int num_pix_on_thrumu[4]; ///< number of pixels tagged by WC as part of off-time clusters

      int truth_num_nupix[4]; ///< number of pixels on true simulated neutrino pixels, each plane + total
    };

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void make_tree();
    void bind_to_tree(TTree* tree );

    /** @brief write ana tree to root file */
    void write_tree() { _va_ana_tree->Write(); };
    
    void clear_ana_variables();

    /** @brief save event variables */
    void fill_tree() { _va_ana_tree->Fill(); };
    
    void calcTruthVariables( larlite::storage_manager& ioll,
                             larcv::IOManager& iolcv,
                             const ublarcvapp::mctools::LArbysMC& truedata );

    /** @brief set name of larflowcluster trees to search for vertex activity locations on them */
    void set_input_cluster_list( const std::vector<std::string>& clist ) { _input_clustertree_list = clist; };

    /** @brief set name of larflow3dhit tree to search for vertex activity locations */
    void set_input_hit_list( const std::vector<std::string>& hlist ) { _input_hittree_list = hlist; };

    /** @brief set name of larflow3dhit tree to save candidate vertices */
    void set_output_treename( std::string name ) { _output_treename=name; };

    int numCandidates() const { return (int)vtxact_v.size(); };
    std::vector<TGraph> debug_vacandidates_as_tgraph();

    struct DebugVis_t {
      std::vector< std::vector<TGraph> > plane_end_vv;
      std::vector< TGraph > seg_dqdx_v;
    }; ///< info for visualizing VA cluster candidates
    DebugVis_t get_debug_vis(int icandidate);
    

  protected:

    void makeClusters( larlite::storage_manager& ioll,
                       std::vector<larflow::reco::cluster_t>& cluster_v,
                       const float larmatch_threshold );

    std::vector<larlite::track> getClusterTrunks( const larflow::reco::cluster_t& cluster,
                                                  std::vector<float>& pca_ratio );
    
    std::vector<larflow::reco::NuVertexActivityReco::VACandidate_t>      
      findVertexActivityCandidates( larlite::storage_manager& ioll,
                                    larcv::IOManager& iolcv,
                                    std::vector<larflow::reco::cluster_t>& cluster_v,
                                    const float va_threshold );      
    
    std::vector<float> calcPlanePixSum( const larlite::larflow3dhit& hit,
                                        const std::vector<larcv::Image2D>& adc_v );


    void analyzeVertexActivityCandidates( larflow::reco::NuVertexActivityReco::VACandidate_t& va_cand,
                                          std::vector<larflow::reco::cluster_t>& cluster_v,
                                          larlite::storage_manager& ioll,
                                          larcv::IOManager& iolcv,
                                          const float min_dist2cluster );

    void checkWireCellCosmicMask( NuVertexActivityReco::VACandidate_t& va, larcv::IOManager& iolcv );
                                  
    void analyzeAttachedCluster( larflow::reco::NuVertexActivityReco::VACandidate_t& vacand,
                                 std::vector<larflow::reco::cluster_t>& cluster_v,
                                 larlite::storage_manager& ioll,
                                 larcv::IOManager& iolcv );

    void calcTruthNeutrinoPixels( std::vector<VACandidate_t>& valist_v,
                                  larcv::IOManager& iolcv );

    // parameter data members
    std::vector<std::string> _input_hittree_list;  ///< list of tress to get input hits
    std::vector<std::string> _input_clustertree_list; ///< list of trees to get input clusters
    std::vector<larlite::larflow3dhit>    _input_hit_v; ///< collection of hits from input hit trees
    std::map< int, int > _input_hit_origin_v; ///< save map back to source of input hits
    std::vector<larflow::reco::cluster_t> _event_cluster_v; ///< list of clusters for making VA selection variables
    std::string _output_treename; ///< name of tree to store larlite product
    
    
    // output data members
    std::vector<VACandidate_t> vtxact_v; ///< The container of found vertex activity candidates
    
    TTree* _va_ana_tree;  ///< event level tree with data for each reco VA candidate
    bool _kown_tree; ///< if true, then class instance assumes it owns _va_ana_tree
    std::vector< std::vector<float> > pca_dir_vv;  ///< 1st principle component of cluster VA is attached to
    std::vector<int> nbackwards_shower_pts; ///< num of shower pixels in the opposite direction of attached cluster
    std::vector<int> nbackwards_track_pts; ///< number of track pixels in the opposite direction of attached cluster
    std::vector<int> nforwards_shower_pts; ///< number of shower pixels in the forward cone direction of attached custer
    std::vector<int> nforwards_track_pts; ///< number of track pixels in the forward cone direction of attached custer
    std::vector<int> npix_on_cosmic_v; ///< number of pixels on WC-tagged off-time pixels
    std::vector<int> attcluster_nall_v; ///< number of pixels on attached cluster
    std::vector<int> attcluster_nshower_v; ///< number of shower pixels on attached cluster
    std::vector<int> attcluster_ntrack_v; ///< number of track pixels on attached cluster
    std::vector<float> dist_closest_forwardshower; ///< distance from vertex to start point to closest shower cluster
    std::vector<float> shower_likelihood; ///< likelihood based on distance of shower hits from trunk axis
    std::vector<float> dist2truescevtx; ///< distance of reco vertex to space-charge translated true vertex
    std::vector<int> ntrue_nupix_v; ///< number of pixels around vertex on true neutrino pixels    
    float min_dist2truescevtx; ///< for an event, min distance to the true neutrino vertex

    larflow::reco::ShowerdQdx dqdx_algo;
    
  };

  
}
}

#endif
