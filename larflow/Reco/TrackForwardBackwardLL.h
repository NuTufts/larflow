#ifndef __LARFLOW_RECO_TRACK_FORWARD_BACKWARD_LL_H__
#define __LARFLOW_RECO_TRACK_FORWARD_BACKWARD_LL_H__

#include "TGraphErrors.h"
#include "TSpline.h"
#include "TFile.h"
#include "larcv/core/Base/larcv_base.h"

#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"

namespace larflow {
namespace reco {

  class TrackForwardBackwardLL : public larcv::larcv_base
  {
    
  public:

    TrackForwardBackwardLL();
    virtual ~TrackForwardBackwardLL();

    struct Pt_t {
      int ipt;      ///< index of point in track
      float x;      ///< position along track
      float dqdx;   ///< dq/dx value
      float ddqdx2; ///< derivative
      float var;    ///< variance of smoothed value
    };
    typedef std::vector<Pt_t> Track_t;
    
    void analyze( larflow::reco::NuVertexCandidate& nuvtx,
                  larflow::reco::NuSelectionVariables& nusel );

    // graphs for plotting/debugging/validating
    std::vector< std::vector<TGraphErrors> > graph_vv; ///< for debugging
    std::vector< TGraph > proton_v; // forward proton dedx curves
    std::vector< TGraph > muon_v;   // backward muon dedx curves
    

    Track_t _smooth( Track_t& track, const float maxdist, const int nn );    


    // data for de/dx versus range
    TFile*    _splinefile_rootfile;
    TSpline3* _sMuonRange2dEdx;
    TSpline3* _sProtonRange2dEdx;
    void _load_data();

    
  };
  
}
}


#endif
