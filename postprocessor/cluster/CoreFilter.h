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
    
    CoreFilter( const std::vector< std::vector<float> >& cluster, const int min_neighbors, const float maxdist );
    CoreFilter( const larlite::larflowcluster& cluster, const int min_neighbors, const float maxdist );
    virtual ~CoreFilter();

    larlite::larflowcluster           getCore( int min_hits_in_subcluster, const larlite::larflowcluster& cluster );
    std::vector< std::vector<float> > getCore( int min_hits_in_subcluster, const std::vector< std::vector<float> >& cluster );
    larlite::larflowcluster getNonCore( int min_hits_in_subcluster, const larlite::larflowcluster& cluster );
    std::vector< int > getPointIndices( bool core, int min_hits_in_subcluster );
    const std::vector< DBSCAN::Cluster_t > getClusterIndices() const { return _clusters_v; };
    int getIndexOfLargestCluster() const;
    
  protected:

    int _min_neighbors;
    float _maxdist;
    std::vector< DBSCAN::Cluster_t > _clusters_v; // from the dbscan results

    
    

    
    
  };

}

#endif
