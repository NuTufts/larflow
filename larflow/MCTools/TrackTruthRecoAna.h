#ifndef __LARFLOW_RECO_TRACK_TRUTH_RECO_ANA_H__
#define __LARFLOW_RECO_TRACK_TRUTH_RECO_ANA_H__

#include <vector>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/track.h"
#include "DataFormat/larflowcluster.h"

#include "larflow/Reco/NuVertexCandidate.h"



namespace larflow {
namespace mctools {

  class TrackTruthRecoInfo {
    
  public:
    
    TrackTruthRecoInfo() {};
    virtual ~TrackTruthRecoInfo() {};

    int matched_true_trackid;
    int matched_true_pid;
    float matched_mse;    
    float truetrack_completeness;
    float dist_to_trueend;
    
  };

  class VertexTrackTruthRecoInfo {
    
  public:
    
    VertexTrackTruthRecoInfo() {};
    virtual ~VertexTrackTruthRecoInfo() {};

    int vtxid;
    std::vector< TrackTruthRecoInfo > trackinfo_v;
    
  };
  
  class TrackTruthRecoAna : public larcv::larcv_base {

  public:

    TrackTruthRecoAna();
    virtual ~TrackTruthRecoAna();

    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  std::vector< larflow::reco::NuVertexCandidate >& nuvtx_v );

    void bindAnaVariables( TTree* ana_tree );
    

    std::vector<VertexTrackTruthRecoInfo> vtxinfo_v;

    /** @brief clear event info */
    void clearVertexInfo() { vtxinfo_v.clear(); };

    static std::vector< std::vector<float> >
      getSCEtrueTrackPath( const larlite::mctrack& mct,      
                           const larutil::SpaceChargeMicroBooNE* psce );

    
    float _get_mse_cluster_truetrack_v( const std::vector< std::vector<float> >& truetrack_path_v,
                                        const larlite::larflowcluster& track_hit_cluster );

    TrackTruthRecoInfo _make_truthmatch_info( const larlite::track& track,
                                              const larlite::larflowcluster& track_hit_cluster,
                                              ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                              const larlite::event_mctrack& ev_mctrack );
    

  protected:

    larutil::SpaceChargeMicroBooNE *_psce;


  };
  
}
}

#endif
