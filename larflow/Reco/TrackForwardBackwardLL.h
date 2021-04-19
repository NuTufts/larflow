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
    std::vector< int >    muon_bestfit_plane_v;
    std::vector< int >    proton_bestfit_plane_v;
    std::vector< float >  muon_chi2_v;
    std::vector< float >  proton_chi2_v;
    std::vector< float >  best_llr_v;    

    Track_t _smooth( Track_t& track, const float maxdist, const int nn );    


    Track_t _generate_muon_expectation( const Track_t& data_track,
                                        float x_shift, float y_scale );
    
    Track_t _generate_proton_expectation( const Track_t& data_track,
                                          float x_shift, float y_scale );

    float _calc_chi2( const TrackForwardBackwardLL::Track_t& data_track,
                      const TrackForwardBackwardLL::Track_t& expect_track );

    Track_t _scan_muon_comparison( const Track_t& data_track,
                                   const int nx, const float xstep,
                                   const int ny, const float ystep,
                                   float& best_xshift, float& best_yscale, float& min_chi2 );

    Track_t _scan_proton_comparison( const Track_t& data_track,
                                     const int nx, const float xstep,
                                     const int ny, const float ystep,
                                     float& best_xshift, float& best_yscale, float& min_chi2 );
    
    float _get_backwardmu_vs_forwardproton_ll( const TrackForwardBackwardLL::Track_t& data_track,
                                               const TrackForwardBackwardLL::Track_t& backmu,
                                               const TrackForwardBackwardLL::Track_t& forwp );
    
    // data for de/dx versus range
    TFile*    _splinefile_rootfile;
    TSpline3* _sMuonRange2dEdx;
    TSpline3* _sProtonRange2dEdx;
    void _load_data();

    
  };
  
}
}


#endif
