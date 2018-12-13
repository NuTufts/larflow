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
    std::cout << "[CoreFilter::CoreFilter] (from vector<float>) number of hits into dbscan: " << cluster.size() << std::endl;
    larflow::DBSCAN algo;
    _clusters_v.clear();
    _clusters_v = algo.makeCluster( _maxdist, _min_neighbors, 100, cluster );
    std::cout << "number of hits out of dbscan: " << _clusters_v.size() << std::endl;
    
    std::cout << "[CoreFilter::CoreFilter][DEBUG] dbscan made " << _clusters_v.size() << " clusters" << std::endl;
    int totcheck = 0;
    for ( auto& c : _clusters_v ) {
      totcheck += c.size();
    }
    std::cout << "[CoreFilter::CoreFilter][DEBUG] total hits " << totcheck << " in DBSCAN clusters" << std::endl;    
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

    std::cout << "[CoreFilter::CoreFilter] (from larflowcluster) number of hits into dbscan: " << fcluster.size() << std::endl;    
    larflow::DBSCAN algo;
    _clusters_v.clear();
    _clusters_v = algo.makeCluster( _maxdist, _min_neighbors, 100, fcluster );
    std::cout << "[CoreFilter::CoreFilter][DEBUG] dbscan made " << _clusters_v.size() << " clusters" << std::endl;
    int totcheck = 0;
    for ( auto& c : _clusters_v ) {
      totcheck += c.size();
    }
    std::cout << "[CoreFilter::CoreFilter][DEBUG] total hits " << totcheck << " in DBSCAN clusters" << std::endl;
    
  }
  
  CoreFilter::~CoreFilter() {
    _clusters_v.clear();
  }


  std::vector<int> CoreFilter::getPointIndices( bool core, int minhits_in_subcluster ) {

    std::vector< int > out;
    out.clear();
    
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
    
    // std::cout << "[CoreFilter::getPointIndices] GetCore=" << core
    //  	      << ": size=" << out.size() << std::endl;
    
    return out;
  }

  std::vector<int> CoreFilter::getClusterIndices( int iclust) {

    std::vector< int > out;
    out.clear();

    if ( iclust<0 || iclust >= (int)_clusters_v.size() ) {
      //std::cout << "return index list with size=" << out.size() << " for clust=" << iclust << std::endl;
      return out;
    }
    
    for ( auto const& hitidx : _clusters_v[iclust] ) {
      //std::cout << " [" << iclust << "]: " << hitidx << std::endl;
      out.push_back( hitidx );
    }

    return out;
  }
  
  std::vector< std::vector<float> > CoreFilter::getCore(int min_hits_in_subcluster, const std::vector< std::vector<float> >& cluster ) {
    std::vector<int> coreidx = getPointIndices( true, min_hits_in_subcluster );
    std::vector< std::vector<float> > out;
    if ( coreidx.size()==0 )
      return out;
    
    out.reserve( cluster.size() );
    
    for (auto& idx : coreidx )
      out.push_back( cluster[idx] );
    return out;
  }
  
  larlite::larflowcluster CoreFilter::getCore(int min_hits_in_subcluster, const larlite::larflowcluster& cluster ) {
    std::vector<int> coreidx = getPointIndices( true, min_hits_in_subcluster );
    larlite::larflowcluster out;    
    if ( coreidx.size()==0 )
      return out;
    
    for (auto& idx : coreidx )
      out.push_back( cluster[idx] );
    return out;
  }
  
  larlite::larflowcluster CoreFilter::getNonCore( int min_hits_in_subcluster, const larlite::larflowcluster& cluster ) {
    std::vector<int> coreidx = getPointIndices( false, min_hits_in_subcluster );
    larlite::larflowcluster out;
    if ( coreidx.size()==0 )
      return out;
    
    for (auto& idx : coreidx )
      out.push_back( cluster[idx] );
    return out;
  }
  
  int CoreFilter::getIndexOfLargestCluster() const {
    int maxn = 0;
    int maxn_idx = 0;
    for (int idx=0; idx<(int)_clusters_v.size(); idx++) {
      if ( (int)_clusters_v[idx].size()>maxn ) {
	maxn = _clusters_v[idx].size();
	maxn_idx = idx;
      }
    }
    return maxn_idx;
  }



}
