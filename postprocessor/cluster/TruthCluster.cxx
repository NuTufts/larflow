#include "TruthCluster.h"

namespace larflow {

  std::vector< std::vector<larlite::larflow3dhit*> > TruthCluster::clusterHits( const std::vector<larlite::larflow3dhit>& hits ) {
    std::vector< std::vector<larlite::larflow3dhit*> > output;

    return output;
  }


  float TruthCluster::distToCluster( const larlite::larflow3dhit* phit, const Cluster_t* pcluster ) {
    return 0.;
  }

  
  void  TruthCluster::assignToClosestCluster( const larlite::larflow3dhit* phit, std::vector<Cluster_t>& clusters ) {
    
  }

  
  std::vector<TruthCluster::Cluster_t> TruthCluster::createClustersByTrackID( const std::vector<larlite::larflow3dhit>& hits ) {
    std::vector<TruthCluster::Cluster_t> output;

    return output;
  }
  


}
