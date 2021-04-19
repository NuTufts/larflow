#ifndef __LARFLOW_RECO_NU_SHOWER_KINEMATICS_H__
#define __LARFLOW_RECO_NU_SHOWER_KINEMATICS_H__

#include "TLorentzVector.h"
#include "TVector.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/track.h"
#include "DataFormat/larflowcluster.h"
#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"

namespace larflow {
namespace reco {

  /**
   * @class NuShowerKinematics
   * @brief ADC pixelsum to MeV conversion
   *
   * Uses K. Mason's algorithm
   *
   */
  class NuShowerKinematics : public larcv::larcv_base {

  public:

  NuShowerKinematics()
    : larcv::larcv_base("NuShowerKinematics")
      {};

    virtual ~NuShowerKinematics() {};


    void analyze( larflow::reco::NuVertexCandidate& nuvtx,
                  larflow::reco::NuSelectionVariables& nusel,
                  larcv::IOManager& iolcv );

    std::vector< std::vector<TLorentzVector> > _shower_mom_v; // outer loop is the plane
    
    std::vector<float> GetADCSum(const larlite::larflowcluster& shower,
                                 const std::vector<larcv::Image2D>& wire_img,
                                 const int threshold,
                                 const int dpix );
                                 
    TVector3 get_showerdir( const larlite::track& shower_trunk,
                            const std::vector<float>& vtxpos );
      
    
  };
  
}
}

#endif
