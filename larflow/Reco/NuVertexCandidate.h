#ifndef __LARFLOW_RECO_NUVERTEX_CANDIDATE_H__
#define __LARFLOW_RECO_NUVERTEX_CANDIDATE_H__


#include <string>
#include <vector>
#include "DataFormat/track.h"
#include "DataFormat/larflowcluster.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class NuVertexCandidate
   * @brief class to store neutrino interaction vertices and their constituents
   *
   * Made by NuVertexMaker.
   *
   */
  class NuVertexCandidate {

  public:

    NuVertexCandidate() {};
    virtual ~NuVertexCandidate() {};

    /**
     * @brief Type of Vertex Candidate
     */
    typedef enum { kTrack=0, kShowerKP, kShower } ClusterType_t;

    /**
     * @struct VtxCluster_t
     * @brief structure representing particle cluster associated to vertex
    */    
    struct VtxCluster_t {
      std::string producer;    ///< larflowcluster tree name this cluster came from
      int index;               ///< the cluster's index in the cluster container
      std::vector<float> dir;  ///< direction along first principle component
      std::vector<float> pos;  ///< start position
      float gap;               ///< distance from vertex
      float impact;            ///< distance of first pc axis to the vertex position
      int npts;                ///< number of points in cluster
      ClusterType_t type;      ///< type of cluster
    };
    
    std::string keypoint_producer;  ///< name of tree containing keypoints used to seed candidates
    int keypoint_index;             ///< index of vertex candidate in container above
    std::vector<float> pos;         ///< keypoint position
    std::vector< VtxCluster_t > cluster_v; ///< clusters assigned to vertex
    float score;                    ///< vertex candidate score based on number of clusters assigned and the impact parameter of each cluster

    std::vector<larlite::track>  track_v;  ///< track candidates
    std::vector<larlite::larflowcluster> shower_v; ///< shower candidates

    /** @brief comparator to sort candidates by highest score */
    bool operator<(const NuVertexCandidate& rhs) const {
      if ( score>rhs.score ) return true;
      return false;
    };
    
  };
  

  

}
}

#endif
