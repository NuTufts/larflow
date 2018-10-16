#include "RecoCluster.h"

#include <map>

namespace larflow {

  std::vector< std::vector<const larlite::larflow3dhit*> > RecoCluster::clusterHits( const std::vector<larlite::larflow3dhit>& hits, std::string algo, bool return_unassigned ) {

    std::vector<Cluster_t> cluster_info;
    if(algo=="dbscan" || algo=="DBSCAN"){
      cluster_info = createClustersPy( hits, algo );
    }
    else{
      cluster_info = createClusters( hits );
    }
    Cluster_t unmatched = cluster_info.front(); // first entry is the unmatched hits cluster (at least for DBSCAN)
    cluster_info.erase(cluster_info.begin()); // remove first element from list

    std::cout << "Created " << cluster_info.size() << " clusters using " << algo << std::endl;
    std::cout << "Number of unassigned hits: " << unmatched.phits.size() << " of " << hits.size() << std::endl;

    
    Cluster_t unassigned = assignUnmatchedToClusters( unmatched.phits, cluster_info );
    std::cout << "after NN assignement of non-truthmatched hits, remaining unassigned: " << unassigned.phits.size() << std::endl;

    // for the output
    int nclusters = cluster_info.size();
    if ( return_unassigned )
      nclusters++;
    
    std::vector< std::vector<const larlite::larflow3dhit*> > output(nclusters);
    int icluster = -1;
    for ( auto& cluster : cluster_info ) {
      icluster++;
      for ( auto& phit : cluster.phits ) {
	output[icluster].push_back( phit );
      }
    }

    if ( return_unassigned ) {
      output.at(nclusters-1) = unassigned.phits;
    }
    
    return output;
  }
  
  std::vector<RecoCluster::Cluster_t> RecoCluster::createClusters( const std::vector<larlite::larflow3dhit>& hit_v ) {
    std::vector<RecoCluster::Cluster_t> output;
    output.reserve(50);

    Cluster_t noclusterhits;
    noclusterhits.trackid = -1;
    
    larflow::CilantroSpectral sc( hit_v,40,5 );
    std::vector<std::vector<long unsigned int> > cpi;    
    std::vector<long unsigned int> idx_mat;
    larflow::CilantroSpectral::get_cluster_indeces(sc._sc,cpi,idx_mat);
    // I need to check how to use cluster indeces properly
    
    int idx = 0;
    for (auto const& cl : cpi ) {
      // create new cluster
      Cluster_t info;
      info.trackid = hit_v.at(idx).trackid;
      info.phits.push_back( &hit_v.at(idx) ); // store address: for now 1st hit only
      // put into vector
      output.emplace_back( std::move(info) );
      /*
	if(no cluster id){
	noclusterhits.phits.push_back( &hit );
	}
      */
      idx += cl.size();
    }
    
    // calculate axis-aligned bounding boxes for clusters
    for ( auto& cluster : output ) {
      // set first location
      for (int i=0; i<3; i++)
	cluster.aabbox[i][0] = cluster.aabbox[i][1] = cluster.phits.front()->X_truth[i];

      for ( auto& phit : cluster.phits ) {
	for (int i=0; i<3; i++) {
	  if ( phit->X_truth[i] < cluster.aabbox[i][0] )
	    cluster.aabbox[i][0] = phit->X_truth[i];
	  if ( phit->X_truth[i] > cluster.aabbox[i][1] )
	    cluster.aabbox[i][1] = phit->X_truth[i];
	}
      }
    }
    
    // append notruthhits cluster to end of cluster vector
    output.emplace_back( std::move(noclusterhits) );
    
    return output;
  }
  
  std::vector<RecoCluster::Cluster_t> RecoCluster::createClustersPy( const std::vector<larlite::larflow3dhit>& hit_v, std::string algo ) {
    std::vector<RecoCluster::Cluster_t> output;
    output.reserve(50);

    Cluster_t noclusterhits;
    noclusterhits.trackid = -1;

    // ======================================
    // here we need to call python wrapper
    // it calls functions from apply_cluster_algo.py
    //
    //             TO BE DONE
    // ======================================
    
    // calculate axis-aligned bounding boxes for clusters
    for ( auto& cluster : output ) {
      // set first location
      for (int i=0; i<3; i++)
	cluster.aabbox[i][0] = cluster.aabbox[i][1] = cluster.phits.front()->X_truth[i];

      for ( auto& phit : cluster.phits ) {
	for (int i=0; i<3; i++) {
	  if ( phit->X_truth[i] < cluster.aabbox[i][0] )
	    cluster.aabbox[i][0] = phit->X_truth[i];
	  if ( phit->X_truth[i] > cluster.aabbox[i][1] )
	    cluster.aabbox[i][1] = phit->X_truth[i];
	}
      }
    }
    
    // append notruthhits cluster to end of cluster vector
    output.emplace_back( std::move(noclusterhits) );
    
    return output;
  }


  RecoCluster::Cluster_t RecoCluster::assignUnmatchedToClusters( const std::vector<const larlite::larflow3dhit*>& unmatchedhit_v, std::vector<Cluster_t>& cluster_v ) {

    Cluster_t unassigned;
    
    // N^2 algorithm ... hopefully we save some time axis aligned checking
    for ( auto& phit : unmatchedhit_v ) {

      Cluster_t* best_matching_cluster = nullptr;
      float cluster_min_dist = -1;
      
      for ( auto& cluster : cluster_v ) {

	// calc dist to cluster aabb
	bool checkcluster = false;
	float mindist = -1;
	for (int i=0; i<3; i++) {
	  float dist2minedge = fabs( (*phit)[i] - cluster.aabbox[i][0] );
	  float dist2maxedge = fabs( (*phit)[i] - cluster.aabbox[i][1] );
	  
	  bool isinside = ( (*phit)[i] > cluster.aabbox[i][0] && (*phit)[i] < cluster.aabbox[i][1] );
	  if ( isinside || dist2minedge < 2.0 || dist2maxedge > 2.0 )
	    checkcluster = true;

	  if ( checkcluster )
	    break; // we already know we have to
	}

	if ( checkcluster ) {
	  // aabbox checks out, check the internal hits
	  for ( auto pclusterhit : cluster.phits ) {
	    float dist2 = 0.;
	    for ( int i=0; i<3; i++) {
	      dist2 += ((*pclusterhit)[i] - (*phit)[i])*((*pclusterhit)[i] - (*phit)[i]);
	    }
	    dist2 = sqrt(dist2);

	    if ( cluster_min_dist<0 || cluster_min_dist > dist2 ) {
	      cluster_min_dist = dist2;
	      best_matching_cluster = &cluster;
	    }
	  }
	}
      }//end of cluster loop

      // do we add the hit somewhere?
      if ( cluster_min_dist>=0 && cluster_min_dist < 3.0 ) {
	best_matching_cluster->phits.push_back( phit );
      }
      else {
	unassigned.phits.push_back( phit );
      }
    }//end of loop over given umatched clusters

    return unassigned;
  }
  
}
