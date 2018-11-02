#ifndef __LARFLOW_TRUTH_CLUSTER__
#define __LARFLOW_TRUTH_CLUSTER__

/** /////////////////////////////////////////
 *  Truth Cluster
 *  makes cluster from truth information
 *
 *  first clusters by trackid the core and edge truth-matched hits
 *  then sweeps up remaining hits to closest cluster
 *  
 *  these clusters are meant to provide a tool for developing
 *   flash matching code and for comparing tuning.
 *   
 *  maybe these will be useful for diagnostics ...
 * 
 */ ////////////////////////////////////////

#include <vector>

#include "DataFormat/larflow3dhit.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/mctrack.h"

namespace larflow {


  class TruthCluster {

  public:

    TruthCluster() {};
    virtual ~TruthCluster() {};

    std::vector< std::vector<const larlite::larflow3dhit*> > clusterHits( const std::vector<larlite::larflow3dhit>& hits,
									  const std::vector<larlite::mctrack>&  mctrack_v,
									  const std::vector<larlite::mcshower>& mcshower_v,
									  bool use_ancestorid,
									  bool return_unassigned=true );


  protected:
    struct Cluster_t {
      std::vector< const larlite::larflow3dhit* > phits; // pointer of hits
      float aabbox[3][2]; // axis-aligned bounding box (for faster nn tests)
      int trackid;
    };

    float distToCluster( const larlite::larflow3dhit* phit, const Cluster_t* pcluster );
    void  assignToClosestCluster( const larlite::larflow3dhit* phit, std::vector<Cluster_t>& clusters );
    std::vector<Cluster_t> createClustersByTrackID( const std::vector<larlite::larflow3dhit>& hits,
						    const std::vector<larlite::mctrack>&  mctrack_v,
						    const std::vector<larlite::mcshower>& mcshower_v,
						    bool use_ancestor_id );
    Cluster_t assignUnmatchedToClusters( const std::vector<const larlite::larflow3dhit*>& unmatchedhit_v, std::vector<Cluster_t>& cluster_v );    
  };


}

#endif
