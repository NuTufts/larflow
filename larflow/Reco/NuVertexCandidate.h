#ifndef __LARFLOW_RECO_NUVERTEX_CANDIDATE_H__
#define __LARFLOW_RECO_NUVERTEX_CANDIDATE_H__

/**
 *
 * class to store neutrino interaction vertices and the constituents
 *
 */

#include <string>
#include <vector>

namespace larflow {
namespace reco {


  class NuVertexCandidate {

  public:

    NuVertexCandidate() {};
    virtual ~NuVertexCandidate() {};

    /**
       structure representing particle cluster associated to vertex
    */
    typedef enum { kTrack=0, kShowerKP, kShower } ClusterType_t;
    struct VtxCluster_t {
      std::string producer;
      int index;
      std::vector<float> dir;
      std::vector<float> pos;
      float gap;
      float impact;
      int npts;
      ClusterType_t type;
    };
    
    std::string keypoint_producer;      
    int keypoint_index;
    std::vector<float> pos;
    std::vector< VtxCluster_t > cluster_v;
    float score;
    bool operator<(const NuVertexCandidate& rhs) const {
      if ( score>rhs.score ) return true;
      return false;
    };
    
  };
  

  

}
}

#endif
