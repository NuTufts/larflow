#include "CoreFilter.h"

namespace larflow {

  CoreFilter::CoreFilter( const larlite::larflowcluster& cluster, const int min_neighbors, const float maxdist  )
    : _cluster(&cluster),
      _min_neighbors(min_neighbors),
      _maxdist(maxdist)
  {

    // create kdtree
    // for each point do a neighbor search
    // make a queue for connected neighbors: append if not-labeled
    // noise points with non-minimum neighbors is labeled noise point

    // core is the largest cluster
    // provide pca of this core cluster
    //
    larflow::DBSCAN algo;
    _clusters_v.clear();
    _clusters_v = algo.makeCluster( _maxdist, _min_neighbors, 100, cluster );
         
  }

  CoreFilter::~CoreFilter() {
    _clusters_v.clear();
  }


  larlite::larflowcluster CoreFilter::getPoints( bool core, int minhits_in_subcluster ) {

    larlite::larflowcluster out;
    
    if ( core ) {
      for (int iclust=0; iclust<(int)_clusters_v.size()-1; iclust++) {
	if ( (int)_clusters_v[iclust].size()>=minhits_in_subcluster ) {
	  for ( auto const& hitidx : _clusters_v[iclust] ) {
	    out.push_back( (*_cluster)[hitidx] );
	  }
	}
      }
    }
    else {
      // non-core
      for ( auto const& hitidx : _clusters_v.back() ) {
	out.push_back( (*_cluster)[hitidx] );
      }
    }
    
    std::cout << "[CoreFilter::getPoints] GetCore=" << core
	      << ": " << out.size() << " of " << _cluster->size() << std::endl;
    
    return out;
  }
  
  larlite::larflowcluster CoreFilter::getCore(int min_hits_in_subcluster) {
    return getPoints( true, min_hits_in_subcluster );
  }
  
  larlite::larflowcluster CoreFilter::getNonCore() {
    return getPoints( false, -1 );
  }
  
  



}
