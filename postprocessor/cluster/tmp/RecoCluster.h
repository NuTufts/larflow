#ifndef __LARFLOW_RECO_CLUSTER__
#define __LARFLOW_RECO_CLUSTER__

/** /////////////////////////////////////////
 *  Reco Cluster
 *  makes cluster using either 
 *  Cilantro (via CilantroXX) or 
 *  Scikit learn (wrapper of apply_cluster_algo)
 * 
 */ ////////////////////////////////////////

#include <vector>

#include "DataFormat/larflow3dhit.h"
#include "CilantroSpectral.h"

namespace larflow {


  class RecoCluster {

  public:

    RecoCluster() {};
    virtual ~RecoCluster() {};

    void filter_hits(const std::vector<larlite::larflow3dhit>& hits, std::vector<larlite::larflow3dhit>& fhits, int min_nn, float nn_dist, float fraction_kept=1.0);
    std::vector< std::vector<const larlite::larflow3dhit*> > clusterHits( const std::vector<larlite::larflow3dhit>& hits, std::string algo, bool return_unassigned=true );

  protected:
    struct Cluster_t {
      std::vector< const larlite::larflow3dhit* > phits; // pointer of hits
      float aabbox[3][2]; // axis-aligned bounding box (for faster nn tests)
      int trackid;
    };

    std::vector<Cluster_t> createClusters( const std::vector<larlite::larflow3dhit>& hits );
    std::vector<Cluster_t> createClustersPy( const std::vector<larlite::larflow3dhit>& hits, std::string algo);
    Cluster_t assignUnmatchedToClusters( const std::vector<const larlite::larflow3dhit*>& unmatchedhit_v, std::vector<Cluster_t>& cluster_v );    
  };


}

#endif
