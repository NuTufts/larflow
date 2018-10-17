#ifndef __LARFLOW_CORE_FILTER_H__
#define __LARFLOW_CORE_FILTER_H__

#include "DataFormat/larflowcluster.h"

/** ===============================================
 *  CoreFilter
 *  ------------------
 *
 *  Take a cluster, identify the "core", using dbscan.
 * 
 *  Evaluate quality of core using PCA. Want one
 *  strong axis. Other axes should be weak.
 *  
 *  ============================================== */

#include "./DBSCAN.h"

namespace larflow {

  class CoreFilter {
  public:
    
    CoreFilter( const larlite::larflowcluster& cluster, const int min_neighbors, const float maxdist );
    virtual ~CoreFilter();

    larlite::larflowcluster getCore( int min_hits_in_subcluster );
    larlite::larflowcluster getNonCore();

  protected:

    int _min_neighbors;
    float _maxdist;
    const larlite::larflowcluster* _cluster;
    std::vector< DBSCAN::Cluster_t > _clusters_v; // from the dbscan results
    larlite::larflowcluster getPoints( bool core, int min_hits_in_subcluster );
    
    

    
    
  };

}

#endif
