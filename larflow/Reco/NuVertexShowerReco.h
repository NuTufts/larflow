#ifndef __LARFLOW_RECO_NUVERTEX_SHOWER_RECO_H__
#define __LARFLOW_RECO_NUVERTEX_SHOWER_RECO_H__

#include <vector>
#include <map>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "NuVertexCandidate.h"
#include "cluster_functions.h"
#include "ClusterBookKeeper.h"

#include <xgboost/c_api.h>

#define nuvertexshowerreco_safe_xgboost(call) {  \
  int err = (call); \
  if (err != 0) { \
    std::cout << "xgboost error: " << XGBGetLastError() << std::endl; \
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                        ": error in " + #call + ":" + XGBGetLastError());  \
  } \
}

namespace larflow {
namespace reco {

  /** 
   * @ingroup Reco
   * @class NuVertexShowerReco
   * @brief Build showers by making simple cone to shower fragments starting from vertex.
   *
   * Take shower clusters assigned to vertex and build final shower objects for vertex.
   * Do this by simple cone algorithm. Provide dE/dx measure.
   *
   */
  class NuVertexShowerReco : public larcv::larcv_base {

  public:

    NuVertexShowerReco();
    virtual ~NuVertexShowerReco();


    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  std::vector<NuVertexCandidate>& nu_candidate_v,
		  std::vector<ClusterBookKeeper>& nu_cluster_book_v );
    void loadClusters( larlite::storage_manager& ioll );
    void use_shower_keypoints( bool use, std::string container_name ) {
      _use_showerkp=use;
      _keypoint_container_name=container_name;
    };
    void save_detectable_photon_info( larlite::storage_manager& ioll );

  protected:

    std::map<std::string, larlite::event_larflowcluster* >   _cluster_producers;     ///< map from tree name to event container for larflowcluster
    std::map<std::string, larlite::event_pcaxis* >           _cluster_pca_producers; ///< map from tree name to pca info for cluster
    std::map<std::string, NuVertexCandidate::ClusterType_t > _cluster_type;          ///< cluster type
    std::vector< NuVertexCandidate::VtxCluster_t >           _showercluster_candidates_v;

    typedef struct {
      int nabove_showerkp_threshold;
      float maxscore;
      std::vector<float> maxscore_pos;
    } ShowerClusterKeypointVars_t;
    std::vector< ShowerClusterKeypointVars_t > _showercluster_keypoint_vars_v;

  public:
    
    /** @brief add name of tree to get shower clusters from. call before running process. */
    void add_cluster_producer( std::string name, NuVertexCandidate::ClusterType_t ctype ) {
      _cluster_producers[name] = nullptr;
      _cluster_pca_producers[name] = nullptr;
      _cluster_type[name] = ctype;      
    };

    void build_vertex_showers( NuVertexCandidate& nuvtx,
			       ClusterBookKeeper& nuclusterbook,
			       larcv::IOManager& iolcv, 
			       larlite::storage_manager& ioll );

    void calcShowerKeypointVariables( const larlite::larflowcluster& cluster, 
                                      const float& score_threshold,
                                      std::vector<float>& maxscore_pos, 
                                      float& maxscore,
                                      int& nabove_threshold );

    // =============================================================================
    // mc analysis variables/functions
  public:
    void activateMCanalysisMode( bool doit=true) { _mc_analysis_mode=doit; }; ///< if MC analysis mode activated, will record information to study decision parameters for tuning
    bool isMCanaModeActive() { return _mc_analysis_mode; };

    typedef enum { kAccept=0, kSubCluster, kFailPreCuts, kFailAttachment } RecoOutCome_t;
    struct RecoShowerInfo_t {
      int   _trueprong_trackid;  //< geant4 trackid of photon prong best matched to this shower fragment
      float _cluster_pixsum_MeV; //< pixelsum of true trunk fragment
      float _frac_truetrunk;     //< fraction that reco fragment pixels overlap with true trunk pixels 
      float _frac_recopurity;    //< fraction that true trunk pixels overlap with reco fragment pixels
      std::vector<float> _true_trunkdir; //< true trunk direction using 3D point 1st principle component
      float _trueprong_dist2vtx;  //< distance to reco vertex to true prong trunk
      float _recoshower_dist2vtx; //< distance to reco vertex to reco fragment
      float _recoshower_impactpar; //< distance of vtx along shower trunk line
      float _recoshower_cosine;   //< cosine between trunk dir and line from vertex to shower start
      float _recoshower_pixsum_MeV; //< total energy
      float _recoshower_cosmic_pixsum; //< overlap of trunk points onto cosmic-tagged pixels
      std::vector<float> _recoshower_trunkdir;
      int   _reco_outcome;  //< outcome of reco for shower fragment
      int   _correct_outcome;  ///< correct outcome label for this shower fragment

      public:
      RecoShowerInfo_t()
      : _trueprong_trackid(0),
      _cluster_pixsum_MeV(0.0),
      _frac_truetrunk(0.0),
      _frac_recopurity(0.0),
      _true_trunkdir({0.,0.,0.}),
      _trueprong_dist2vtx(-1.0),
      _recoshower_dist2vtx(-1.0),
      _recoshower_impactpar(-1.0),
      _recoshower_cosine(0.0),
      _recoshower_pixsum_MeV(0.0),
      _recoshower_cosmic_pixsum(0.0),
      _recoshower_trunkdir({0.,0.,0.}),
      _reco_outcome(3),
      _correct_outcome(3)
      {};
    };
    
    void createMCAnalysisTree( TFile* outfile );
    void writeAnaTree();

  protected:

    struct ProngRank_t {
      std::string producer;
      int prong_idx;
      int container_idx;
      float score;
      float dist2vtx;
      float impactpar;
      float cosine;
      float cos_paf;
      float pixsum;
      float cosmic;
      int ikpbest;
      float kpdist;
      float kpmax;
      std::vector<float> axis;
      std::vector<float> axis_start;
      std::vector<float> axis_end;
      ProngRank_t( std::string p, int pi, int ci, float s )
        : producer(p), prong_idx(pi), container_idx(ci), score(s)
      {};
      // the following operator is used to sort the prongs by score for seeding priority
      bool operator<( const ProngRank_t& rhs ) {
        // threshold on hits, else rank on hits        
        if ( score>rhs.score ) return true;
        return false;
      };
    };

    void getBDTseedscore( std::vector< NuVertexShowerReco::ProngRank_t >& seed_v );

  protected:

    bool _mc_analysis_mode;
    ublarcvapp::mctools::MCPixelPGraph* _mcpg;
    void _gatherTruthShowerFeatures( larflow::reco::cluster_t& prong, 
      larflow::reco::NuVertexCandidate& vtx,
      RecoShowerInfo_t& showerinfo );

    bool _mc_analysis_saveinfo_for_this_vertex;
    std::map< int, RecoShowerInfo_t > _map_prongindex_to_mcanainfo;

    // root tree and branch variables we use to save mc analysis output
    TTree* _mcana_per_recoshower_tree; ///< Analysis tree that saves info per reco shower, relative to good reco vertices (within 3 cm)
    int   _mcana_index_closest_recovtx; ///< index of reco vertex of closest to true vertex
    float _mcana_closest_recovtx_dist;  ///< distance to reco vertex
    float _mcana_trueprong_pixsum_MeV;  ///< observable energy deposited by shower trunk
    float _mcana_trueprong_efficiency;
    float _mcana_trueprong_dist2vtx;
    float _mcana_recofragment_purity;
    float _mcana_recofragment_dist2vtx;
    float _mcana_recofragment_impactpar;
    float _mcana_recofragment_cosine;
    float _mcana_recofragment_pixsum;
    int   _mcana_reco_outcome;
    int   _mcana_groundtruth_outcome;
    float _mcana_trueprong_trunkdir[3];
    float _mcana_recofragment_trunkdir[3];
    void _set_default_mcana_variable_values();
    void _fill_mcanalysis_tree(); ///< save variables for each shower fragment to the tree
    // end of mc analysis functions and variables ==================================================
    
  protected:

    float _trunk_maxdist_from_closest_cm;
    bool  _calc_cosmic_overlap;
    int _make_trunk_cand( const std::vector<float>& pos,
                           const larlite::larflowcluster& lfcluster,
                           std::vector<float>& shower_start,
                           std::vector<float>& shower_dir,
                           std::vector<float>& paf_dir,
                           float& shower_ll );
    std::vector<float> _get_cluster_pixsum( const std::vector<larcv::Image2D>& adc_v,
                                            const larlite::larflowcluster& lfcluster );

    // XGBoost 
  protected:
    BoosterHandle* _boosterhandle;

    // Use Shower Keypoints to rank and seed prongs
  protected:
    bool _use_showerkp;
    std::string _keypoint_container_name;

    
  };

}
}


#endif
