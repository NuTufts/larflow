#include "DBSCAN.h"

#include <map>

#include <cilantro/kd_tree.hpp>

namespace larflow {


  std::vector< DBSCAN::Cluster_t > DBSCAN::makeCluster( const float maxdist, const float minhits, const int maxkdneighbors,
							const std::vector<std::vector<float> >& clust ) {

    // get points (maybe one day I figure out how to do this without copying
    std::vector<Eigen::Vector3f> _points;
    _points.reserve( clust.size() );
    std::map<int,int> indexmap; // 
    for ( int iorig=0; iorig<(int)clust.size(); iorig++) {
      const std::vector<float>& hit = clust[iorig];
      bool hitok = true;
      for (int i=0; i<3; i++) {
	if ( std::isnan( hit[i] ) ) hitok = false;         // bad value
	if ( hit[0]==hit[1]==hit[2]==-1.0 ) hitok = false; // uninitialized
      }
      if ( hitok ) {
	indexmap[_points.size()] = iorig;
	_points.push_back( Eigen::Vector3f(hit[0],hit[1],hit[2]) );
      }
    }
    int npoints = _points.size();
    
    cilantro::KDTree3f tree(_points);    
    
    int iidx = -1;
    std::vector<int> clusterlabel_v(npoints,-1); // -2=noise, -1=unvisited
    std::vector<Cluster_t> dbscan_clust_v;

    enum { kNoise=-2, kUnlabeled=-1 };
    
    // visit every point at least once
    for ( int ipt=0; ipt<npoints; ipt++ ) {

      if ( clusterlabel_v[ipt]!=kUnlabeled ) {
	// previously labeled
	continue;
      }

      // get point
      const Eigen::Vector3f& pt = _points[ipt];
      
      // get neighbors to point
      cilantro::NeighborSet<float> nn;
      tree.radiusSearch( pt, maxdist, nn);

      // is it dense enough?
      if ( nn.size()<minhits ) {
	// label point as noise (-2)
	clusterlabel_v[ipt] = kNoise;
	continue;
      }      

      // dense enough, get a new cluster label and define new cluster
      int clustlabel = dbscan_clust_v.size();
      Cluster_t cluster;
      cluster.clear();
      cluster.reserve(100);
      // add current point
      cluster.push_back( ipt ); 
      
      // establish a queue for the set of neighbors to visit
      std::vector<int> queue_v;
      queue_v.reserve(npoints);

      // load with initial neighbors
      for ( auto& neighbor : nn )
	queue_v.push_back(neighbor.index);

      // traverse queue
      int iqueue = 0;      
      while ( iqueue<queue_v.size() ) {
	int index = queue_v[iqueue];      
	iqueue++;

	int pastlabel = clusterlabel_v[index];
	//std::cout << "pt[" << ipt << "] label=" << pastlabel << " queuesize=" << queue_v.size() << " queuepos=" << ipos << std::endl;      
	//std::cout << "[larflow::DBSCAN::makeClusters][DEBUG] ipt=" << " nneighbors=" << nn.size() << " pastlabel=" << pastlabel << std::endl;

	if ( pastlabel==kNoise ) {
	  // override noise label with this one
	  clusterlabel_v[index] = clustlabel;
	  cluster.push_back(index);
	}
	else if (pastlabel>=0 ) {
	  // already labeled
	}
	else {
	  // unlabeled (-1)
	  
	  // assign label
	  clusterlabel_v[index] = clustlabel;
	  cluster.push_back(index);	  

	  // find this one's neighbors
	  const Eigen::Vector3f& pt2 = _points[index];	  
	  cilantro::NeighborSet<float> nn2;
	  tree.radiusSearch( pt2, maxdist, nn2);
	  
	  if ( nn2.size()>=minhits ) {
	    // point is dense enough, add neighbors to queue if unlabeled
	    for ( auto& neighbor : nn2 ) {
	      if ( clusterlabel_v[ neighbor.index ]==kUnlabeled )
		queue_v.push_back( neighbor.index );
	    }
	  }
	}
      }//while queue is not finished

      // store cluster
      dbscan_clust_v.emplace_back( std::move(cluster) );
      
    }//end of outer point loop

    // collect noise points
    Cluster_t noise;
    noise.clear();
    for ( int ipt=0; ipt<npoints; ipt++ ) {
      if ( clusterlabel_v[ipt]==kNoise )
	noise.push_back( ipt );
    }
    
    // append noise to cluster vector
    dbscan_clust_v.emplace_back( std::move(noise) );

    // now translate cluster back to original indexes (if needed)
    if ( _points.size()!=clust.size() ){
      for ( auto& dbscanclust : dbscan_clust_v ) {
	for (int ii=0; ii<(int)dbscanclust.size(); ii++) {
	  // translation
	  auto it = indexmap.find( dbscanclust[ii] );
	  if ( it!=indexmap.end() ) 
	    dbscanclust[ii] = it->second;
	  else
	    throw std::runtime_error("[larflow::DBSCAN::makeCluster][ERROR] missing index");
	}
      }
    }
    
    return dbscan_clust_v;
  }
  
}
