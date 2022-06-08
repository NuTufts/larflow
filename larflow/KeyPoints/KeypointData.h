#ifndef __LARFLOW_KEYPOINT_KEYPOINTDATA_H__
#define __LARFLOW_KEYPOINT_KEYPOINTDATA_H__

#include <Python.h>
#include "bytesobject.h"

#include "KPdata.h"

namespace larflow {
namespace keypoints {

  class KeypointData {

  public:
    KeypointData();
    virtual ~KeypointData() {};

    int tpcid;
    int cryoid;
    // std::vector< std::vector<float> > _kppos_v[6]; ///< container of true keypoint 3D positions in cm, for each of the 6 classes
    // std::vector< std::vector<int> >   _kp_pdg_trackid_v[6]; ///< each entry maps (pdg, trackid) for truth meta-data matching
    std::vector<KPdata> _kpd_v; ///< info on true keypoints found using MC truth
    std::vector< std::vector<float> > _match_proposal_labels_v[6]; ///< provides the labels for triplet proposals made by larflow::prep::PrepMatchTriplets

    PyObject* get_keypoint_array( int iclass ) const;        
    PyObject* get_triplet_score_array( float sig ) const;

  private:
    static bool _setup_numpy;
  };

}
}

#endif
