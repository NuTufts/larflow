#ifndef __LARFLOW_DBSCAN_H__
#define __LARFLOW_DBSCAN_H__

#include <vector>

// larlite
#include "DataFormat/larflowcluster.h"

namespace larflow {

  class DBSCAN {

  public:
    DBSCAN() {};
    virtual ~DBSCAN() {};

    typedef std::vector<int> Cluster_t;
    
    std::vector< Cluster_t > makeCluster( const float maxdist, const float minhits, const int maxkdneighbors, const larlite::larflowcluster& clust );
    
  };


}


#endif
