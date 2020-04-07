#ifndef __PREP_KEYPOINT_DATA_H__
#define __PREP_KEYPOINT_DATA_H__

/**
 * 
 * This class is responsible for making the training data for the key-point+larmatch task
 *
 * The job is to provide labels for the proposed matches for the network.
 * It can either remake proposals or it can make labels for old proposals.
 *
 * The cases we are trying to label:
 *
 * cosmic muon start and stop, primary ancestor only.
 * cosmic proton start and stop, primary ancestor only.
 * neutrino primary track start and stop
 * neutrino primary shower start and stop
 * cosmic shower start, primary ancestor only (hard)
 *
 * maybe, depending on quality of truth data
 * neutrino secondaries -- scattering secondaries
 * 
 *
 */

#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include <string>

#include "KPdata.h"

namespace larcv {
  class Image2D;
  class IOManager;
}

namespace larlite {
  class event_mctrack;
  class event_mcshower;
  class event_mctruth;
  class storage_manager;
}

namespace larutil {
  class SpaceChargeMicroBooNE;
}

namespace ublarcvapp {
namespace mctools {
  class MCPixelPGraph;
}
}

namespace larflow {
namespace keypoints {

  class PrepKeypointData {
  public:

    PrepKeypointData() {};
    virtual ~PrepKeypointData() {};

    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll );
    
    void process( const std::vector<larcv::Image2D>&    adc_v,
                  const std::vector<larcv::Image2D>&    badch_v,
                  const std::vector<larcv::Image2D>&    segment_v,
                  const std::vector<larcv::Image2D>&    instance_v,
                  const std::vector<larcv::Image2D>&    ancestor_v,
                  const larlite::event_mctrack&  mctrack_v,
                  const larlite::event_mcshower& mcshower_v,
                  const larlite::event_mctruth&  mctruth_v );

  protected:

    /* struct KPdata { */
    /*   int crossingtype; */
    /*   std::vector<int> imgcoord_start; */
    /*   std::vector<int> imgcoord_end; */
    /*   std::vector<float> startpt; */
    /*   std::vector<float> endpt; */
    /*   int trackid; */
    /*   int pid; */
    /*   int vid; */
    /*   int is_shower; */
    /*   int origin; */
    /*   KPdata() { */
    /*     crossingtype = -1; */
    /*     imgcoord_start.clear(); */
    /*     imgcoord_end.clear(); */
    /*     startpt.clear(); */
    /*     endpt.clear(); */
    /*     trackid = 0; */
    /*     pid = 0; */
    /*     vid = 0; */
    /*     is_shower = 0; */
    /*     origin = -1; */
    /*   }; */
    /*   ~KPdata() {}; */
    /* }; */
    std::vector<KPdata> _kpd_v;
    
    std::vector<KPdata>    
      getMuonEndpoints( ublarcvapp::mctools::MCPixelPGraph& mcpg,
                        const std::vector<larcv::Image2D>& adc_v,
                        const larlite::event_mctrack& mctrack_v,
                        larutil::SpaceChargeMicroBooNE* psce );
    
    std::vector<KPdata>
      getShowerStarts( ublarcvapp::mctools::MCPixelPGraph& mcpg,
                       const std::vector<larcv::Image2D>& adc_v,
                       const larlite::event_mcshower& mcshower_v,
                       larutil::SpaceChargeMicroBooNE* psce );

    std::string str( const KPdata& kpd );

    PyObject* get_keypoint_array() const;


  private:

    static bool _setup_numpy;
    
  };

}
}

#endif
