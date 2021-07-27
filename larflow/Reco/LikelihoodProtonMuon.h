#ifndef __LARFLOW_RECO_LIKELIHOOD_PROTON_MUON_H__
#define __LARFLOW_RECO_LIKELIHOOD_PROTON_MUON_H__

#include "DataFormat/track.h"
#include "larcv/core/Base/larcv_base.h"

class TFile;
class TSpline3;

namespace larflow {
namespace reco {

  class LikelihoodProtonMuon : public larcv::larcv_base {

  public:

    LikelihoodProtonMuon();
    virtual ~LikelihoodProtonMuon();

    double calculateLL( const larlite::track& track, bool reverse=false ) const;    
    double calculateLL( const larlite::track& track, const std::vector<float>& vertex_pos ) const;
    std::vector<double> calculateLLseparate( const larlite::track& track, const std::vector<float>& vertex_pos ) const;

  protected:

    double _q2adc;
    TFile*    _splinefile_rootfile;
    TSpline3* _sMuonRange2dEdx;
    TSpline3* _sProtonRange2dEdx;
    
  };

}
}

#endif
