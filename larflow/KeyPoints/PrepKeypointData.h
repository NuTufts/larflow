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

#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"

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

    void make_proposal_labels( const larflow::PrepMatchTriplets& match_proposals );
    
  protected:

    // KPdata in KPdata.h    
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

  public:

    struct bvhnode_t {
      float bounds[3][2]; //bounding box
      int splitdim;       // dimension we split with (-1) is a leaf
      int kpdidx;         // index to point in the boundary volume
      bvhnode_t* left;
      bvhnode_t* right;
      bvhnode_t()
      : splitdim(-1),
        kpdidx(-1),
        left(nullptr),
        right(nullptr)
      {};
    };

  protected:
    
    std::vector< bvhnode_t* > _bvhnodes_v;
    bvhnode_t* _bvhroot;
    
  public:

    PyObject* get_keypoint_array() const;


  protected:

    // this provides the labels for each triplet proposal made by
    // larflow::PrepMatchTriplets
    std::vector< std::vector<float> > _match_proposal_labels_v;
    
  private:

    static bool _setup_numpy;
    
  };

}
}

#endif
