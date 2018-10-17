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


  larlite::larflowcluster CoreFilter::getPoints( bool core ) {
    int maxcluster = -1;
    int maxnhits = 0;
    if ( core ) {
      for (int iclust=0; iclust<(int)_clusters_v.size()-1; iclust++) {
	if ( (int)_clusters_v[iclust].size()>maxnhits ) {
	  maxnhits   = _clusters_v[iclust].size();
	  maxcluster = iclust;
	}
      }
    }
    else {
      // non-core
      maxcluster = _clusters_v.size()-1;
    }
    
    std::cout << "[CoreFilter::getPoints] GetCore=" << core
	      << ": maxcluster=" << maxcluster << " of " << _clusters_v.size()
	      << " npts=" << maxnhits << " of " << _cluster->size() << std::endl;
    
    larlite::larflowcluster out;
    if ( core && maxnhits>0 ) {
      for ( auto const& hitidx : _clusters_v[maxcluster] ) {
	out.push_back( (*_cluster)[hitidx] );
      }
    }
    
    return out;
  }
  
  larlite::larflowcluster CoreFilter::getCore() {
    return getPoints( true );
  }
  const larlite::larflowcluster& getNonCore();
  
  



}
