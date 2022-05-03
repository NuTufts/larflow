#ifndef __LARFLOW_RECO_CLUSTER_BOOK_KEEPER_H__
#define __LARFLOW_RECO_CLUSTER_BOOK_KEEPER_H__

#include <vector>
#include "larcv/core/Base/larcv_base.h"

namespace larflow {
namespace reco {

  /**
   * @brief Keep track of which clusters have been assigned to a track or shower object
   * 
   * This class is so embarassing. And we'll need one for each NuVertexCandidate
   * Helps us track which clusters have been incorporated into the vertex candidate.
   *
   */

  class ClusterBookKeeper : public larcv::larcv_base {
    
  public:
    ClusterBookKeeper()
      : larcv::larcv_base("ClusterBookKeeper")
      {};
    virtual ~ClusterBookKeeper() {};

    void set_cluster_status( int clusterid, int status );
    int get_cluster_status( int clusterid ) const;
    int numUsed() const;
    
    std::vector<int> cluster_status_v;
      
  };
  
}
}

#endif
