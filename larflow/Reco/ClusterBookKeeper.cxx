#include "ClusterBookKeeper.h"

namespace larflow {
namespace reco {

  void ClusterBookKeeper::clear()
  {
    cluster_producer_v.clear();
    cluster_container_index_v.clear();
    cluster_status_v.clear();
    cluster_type_v.clear();
  }
  
  void ClusterBookKeeper::set_cluster_status( int clusterid, int status )
  {
    cluster_status_v.at(clusterid) = status;
  }
  
  int ClusterBookKeeper::get_cluster_status( int clusterid ) const
  {
    return cluster_status_v.at(clusterid);
  }

  int ClusterBookKeeper::numUsed() const
  {
    int num = 0;
    for ( auto const&  status : cluster_status_v ) {
      if ( status!=0 )
	      num++;
    }
    return num;
  }
  
  
}
}
