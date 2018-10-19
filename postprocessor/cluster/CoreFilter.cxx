#include "CoreFilter.h"

namespace larflow {

  CoreFilter::CoreFilter( const std::vector< std::vector<float> >& cluster, const int min_neighbors, const float maxdist  )
    : _min_neighbors(min_neighbors),
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
  
  CoreFilter::CoreFilter( const larlite::larflowcluster& cluster , const int min_neighbors, const float maxdist  )
    : _min_neighbors(min_neighbors),
      _maxdist(maxdist)
  {

    // create kdtree
    // for each point do a neighbor search
    // make a queue for connected neighbors: append if not-labeled
    // noise points with non-minimum neighbors is labeled noise point

    // core is the largest cluster
    // provide pca of this core cluster
    std::vector< std::vector<float> > fcluster( cluster.size() );
    for ( size_t ilfhit=0; ilfhit<cluster.size(); ilfhit++ ) {
      fcluster[ilfhit] = cluster[ilfhit];
    }

    larflow::DBSCAN algo;
    _clusters_v.clear();
    _clusters_v = algo.makeCluster( _maxdist, _min_neighbors, 100, fcluster );
    
  }
  
  CoreFilter::~CoreFilter() {
    _clusters_v.clear();
  }


  std::vector<int> CoreFilter::getPointIndices( bool core, int minhits_in_subcluster ) {

    std::vector< int > out;
    
    if ( core ) {
      for (int iclust=0; iclust<(int)_clusters_v.size()-1; iclust++) {
	if ( (int)_clusters_v[iclust].size()>=minhits_in_subcluster ) {
	  for ( auto const& hitidx : _clusters_v[iclust] ) {
	    out.push_back( hitidx );
	  }
	}
      }
    }
    else {
      // non-core
      for ( auto const& hitidx : _clusters_v.back() ) {
	out.push_back( hitidx );
      }
    }
    
    // std::cout << "[CoreFilter::getPoints] GetCore=" << core
    // 	      << ": " << out.size() << " of " << _cluster->size() << std::endl;
    
    return out;
  }
  
  std::vector< std::vector<float> > CoreFilter::getCore(int min_hits_in_subcluster, const std::vector< std::vector<float> >& cluster ) {
    std::vector<int> coreidx = getPointIndices( true, min_hits_in_subcluster );
    std::vector< std::vector<float> > out;
    out.reserve( cluster.size() );
    
    for (auto& idx : coreidx )
      out.push_back( cluster[idx] );
    return out;
  }
  
  larlite::larflowcluster CoreFilter::getCore(int min_hits_in_subcluster, const larlite::larflowcluster& cluster ) {
    std::vector<int> coreidx = getPointIndices( true, min_hits_in_subcluster );
    larlite::larflowcluster out;
    for (auto& idx : coreidx )
      out.push_back( cluster[idx] );
    return out;
  }
  
  larlite::larflowcluster CoreFilter::getNonCore( int min_hits_in_subcluster, const larlite::larflowcluster& cluster ) {
    std::vector<int> coreidx = getPointIndices( false, min_hits_in_subcluster );
    larlite::larflowcluster out;
    for (auto& idx : coreidx )
      out.push_back( cluster[idx] );
    return out;
  }
  
  



}
