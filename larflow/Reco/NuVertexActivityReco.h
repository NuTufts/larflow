#ifndef __LARFLOW_RECO_NUVERTEX_ACTIVITYRECO_H__
#define __LARFLOW_RECO_NUVERTEX_ACTIVITYRECO_H__

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/LArbysMC.h"
#include "cluster_functions.h"

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
      int attached_cluster_index; //< index of cluster in cluster container

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

      int num_pix_on_thrumu[4];

      int truth_num_nupix[4]; ///< number of pixels on true simulated neutrino pixels, each plane + total
    };

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void make_tree();
    void bind_to_tree(TTree* tree );
    void write_tree() { _va_ana_tree->Write(); };
    void clear_ana_variables();
    void fill_tree() { _va_ana_tree->Fill(); };
    void calcTruthVariables( larlite::storage_manager& ioll,
                             larcv::IOManager& iolcv,
                             const ublarcvapp::mctools::LArbysMC& truedata );

  protected:
    
    void makeClusters( larlite::storage_manager& ioll,
                       std::vector<larflow::reco::cluster_t>& cluster_v,
                       const float larmatch_threshold );

    
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
    
    std::vector<VACandidate_t> vtxact_v; // The container of found vertex activity candidates
    
    
    TTree* _va_ana_tree;  //< event level tree with data for each reco VA candidate
    bool _kown_tree;
    std::vector< std::vector<float> > pca_dir_vv;
    std::vector<int> nbackwards_shower_pts;
    std::vector<int> nbackwards_track_pts;
    std::vector<int> nforwards_shower_pts;
    std::vector<int> nforwards_track_pts;
    std::vector<int> npix_on_cosmic_v;
    std::vector<int> attcluster_nall_v;
    std::vector<int> attcluster_nshower_v;
    std::vector<int> attcluster_ntrack_v;
    std::vector<float> dist_closest_forwardshower;
    std::vector<float> shower_likelihood;
    std::vector<float> dist2truescevtx;
    std::vector<int> ntrue_nupix_v;
    

    float min_dist2truescevtx;
    
    
  };

  
}
}

#endif
