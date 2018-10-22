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
    Cluster_t noise;
    noise.clear();

    int ipos = 0;
    std::vector<int> queue_v;
    queue_v.reserve(npoints);
    queue_v.push_back(0);

    while ( ipos<queue_v.size() || queue_v.size()<clusterlabel_v.size() ) {
      int ipt = queue_v[ipos];      
      ipos++;
      const Eigen::Vector3f& pt = _points[ipt];

      int pastlabel = clusterlabel_v[ipt];
      //std::cout << "pt[" << ipt << "] label=" << pastlabel << " queuesize=" << queue_v.size() << " queuepos=" << ipos << std::endl;      
      
      // get neighbors to point
      cilantro::NeighborSet<float> nn;
      tree.kNNInRadiusSearch( pt, maxkdneighbors, maxdist, nn);
      
      //std::cout << "[larflow::DBSCAN::makeClusters][DEBUG] ipt=" << " nneighbors=" << nn.size() << " pastlabel=" << pastlabel << std::endl;
      
      if ( nn.size()<minhits ) {
	// label point as noise (-2)
	clusterlabel_v[ipt] = -2;
	noise.push_back( ipt );
      }
      else {
	if ( pastlabel==-1 ) {
	  // start a cluster index
	  iidx++;
	  clusterlabel_v[ipt] = iidx;
	  pastlabel = iidx;
	  // create a new cluster (empty)
	  Cluster_t newclust;
	  newclust.clear();
	  dbscan_clust_v.emplace_back( std::move(newclust) );
	}
	// node, which has min neighbors to the cluster it is now labeled as
	// note, we only get added to cluster if we past the min neighbors requirement
	dbscan_clust_v[pastlabel].push_back(ipt);
	
	// set labels of neighbors
	for ( int inn=0; inn<(int)nn.size(); inn++) {
	  int nn_label = clusterlabel_v[ nn[inn].index ];
	  if ( nn_label==-1 ) {
	    clusterlabel_v[ nn[inn].index ] = pastlabel; // set to this cluster's index
	    // push into queue
	    queue_v.push_back( nn[inn].index );
	  }
	  else if ( nn_label>=0 && nn_label!=pastlabel ) {
	    //std::cout << "[larflow::DBSCAN][ERROR] neighbor points got different labels: current=" << pastlabel << " neighbor=" << nn_label << "!" << std::endl;
	    throw std::runtime_error("[larflow::DBSCAN][ERROR] neighbor points got different labels!");
	  }

	}
      }

      if ( ipos==queue_v.size() && queue_v.size()<clusterlabel_v.size() ) {
	// find next seed
	for ( int i=0; i<clusterlabel_v.size(); i++) {
	  if ( clusterlabel_v[i]==-1 ) {
	    queue_v.push_back( i );
	    break;
	  }
	}
      }//end of reseed queue
    }//end of pt loop

    // std::cout << "[larflow::DBSCAN::makeCluster][DEBUG2] enter to continue." << std::endl;
    // std::cin.get();

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