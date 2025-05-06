#ifndef __LARFLOW_RECO_NUSHOWER_BUILDER_H__
#define __LARFLOW_RECO_NUSHOWER_BUILDER_H__

#include <vector>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/DataFormat/storage_manager.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "larflow/RecoUtils/cluster_functions.h"
#include "larflow/Reco/NuVertexCandidate.h"
#include "larflow/Reco/ClusterBookKeeper.h"



namespace larflow {
namespace reco {

  /** 
   * @ingroup NuShowerBuilder
   * @class NuShowerBuilder
   * @brief Build tracks by assembling clusters, starting from neutrino vertices
   *
   * Inherits from TrackClusterBuilder. The base class provides the track buiding algorithms.
   * This class provides interface to the NuVertexCandidate inputs.
   *
   */
  class NuShowerBuilder : public larcv::larcv_base {

  public:

    NuShowerBuilder()
      : larcv::larcv_base("NuShowerBuilder"),
      _mc_analysis_mode(false),
      _mcpg(nullptr)
    {};
    virtual ~NuShowerBuilder() {};


    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  std::vector<NuVertexCandidate>& nu_candidate_v,
		  std::vector<ClusterBookKeeper>& nu_cluster_book_v );

    // mc analysis variables/functions
  public:
    void activateMCanalysisMode( bool doit=true) { _mc_analysis_mode=doit; }; ///< if MC analysis mode activated, will record information to study decision parameters for tuning
    typedef enum { kAccept=0, kFailPreCuts, kFailAttachment } RecoOutCome_t;
    typedef struct {
      int   _trueprong_trackid;  //< geant4 trackid of photon prong best matched to this shower fragment
      float _cluster_pixsum_MeV; //< pixelsum of true trunk fragment
      float _frac_truetrunk;     //< fraction that reco fragment pixels overlap with true trunk pixels 
      float _frac_recopurity;    //< fraction that true trunk pixels overlap with reco fragment pixels
      std::vector<float> _true_trunkdir; //< true trunk direction using 3D point 1st principle component
      float _trueprong_dist2vtx;  //< distance to reco vertex to true prong trunk
      float _recoshower_dist2vtx; //< distance to reco vertex to reco fragment
      std::vector<float> _reco_trunkdir;      //< trunk direction
      float _vtx_impactpar; //< impact parameter to reco vtx: determines if shower attaches
      int   _reco_outcome;  //< outcome of reco for shower fragment
      int   _correct_outcome;  ///< correct outcome label for this shower fragment
    } RecoShowerInfo_t;
    void createMCAnalysisTree( TFile* outfile );

  protected:

    bool _mc_analysis_mode;
    ublarcvapp::mctools::MCPixelPGraph* _mcpg;
    void _truthMatchShowerFragments(); ///< use truth to match reco clusters to true shower trunks
    void _gatherTruthShowerFeatures( larflow::recoutils::cluster_t& prong, 
      larflow::reco::NuVertexCandidate& vtx,
      RecoShowerInfo_t& showerinfo );

    std::map< int, RecoShowerInfo_t > _map_prongindex_to_mcanainfo;

    // root tree and branch variables we use to save mc analysis output
    TTree* _mcana_per_recoshower_tree; ///< Analysis tree that saves info per reco shower, relative to good reco vertices (within 3 cm)
    int   _mcana_index_closest_recovtx; ///< index of reco vertex of closest to true vertex
    float _mcana_closest_recovtx_dist;  ///< distance to reco vertex
    float _mcana_trueprong_pixsum_MeV;  ///< observable energy deposited by shower trunk
    float _mcana_trueprong_efficiency;
    float _mcana_recofragment_purity;
    float _mcana_trueprong_dist2vtx;
    float _mcana_recofragment_dist2vtx;
    float _mcana_recofragment_impactpar;
    int   _mcana_reco_outcome;
    int   _mcana_groundtruth_outcome;
    float _mcana_trueprong_trunkdir[3];
    float _mcana_recofragment_trunkdir[3];
    void _fill_mcanalysis_tree(); ///< save variables for each shower fragment to the tree

    
  };

}
}


#endif
