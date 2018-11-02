#include "TruthCluster.h"

#include <map>
#include <set>

namespace larflow {

  std::vector< std::vector<const larlite::larflow3dhit*> > TruthCluster::clusterHits( const std::vector<larlite::larflow3dhit>& hits,
										      const std::vector<larlite::mctrack>&  mctrack_v,
										      const std::vector<larlite::mcshower>& mcshower_v,
										      bool use_ancestor_id,
										      bool return_unassigned ) {

    std::vector<Cluster_t> cluster_info = createClustersByTrackID( hits, mctrack_v, mcshower_v, use_ancestor_id );

    Cluster_t unmatched = cluster_info.back(); // last entry is the unmatched hits cluster (a copy)
    cluster_info.pop_back(); // remove last element from list

    std::cout << "Created " << cluster_info.size() << " clusters using truth trackid" << std::endl;
    std::cout << "Number of hits without a truth-match: " << unmatched.phits.size() << " of " << hits.size() << std::endl;

    
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
	bool hitok = true;
	for ( int i=0; i<3; i++) {
	  if ( std::isnan((*phit)[i]) ) hitok = false;
	}
	if ( (*phit)[0]==-1 && (*phit)[1]==-1 && (*phit)[2]==-1 ) hitok = false;
	if (hitok )
	  output[icluster].push_back( phit );
      }
    }

    if ( return_unassigned ) {
      output.at(nclusters-1) = unassigned.phits;
    }
    
    return output;
  }
  
  std::vector<TruthCluster::Cluster_t> TruthCluster::createClustersByTrackID( const std::vector<larlite::larflow3dhit>& hit_v,
									      const std::vector<larlite::mctrack>&  mctrack_v,
									      const std::vector<larlite::mcshower>& mcshower_v,
									      bool use_ancestor_id ) 
  {

    std::vector<TruthCluster::Cluster_t> output;
    output.reserve(50);

    std::map<int,int> trackid2index;

    std::map<int,int> id2ancestor;
    std::set<int> neutrino;
    if ( use_ancestor_id ) {
      for ( auto const& mct : mctrack_v ) {
	int tid = mct.TrackID();
	int aid = mct.AncestorTrackID();
	if ( mct.Origin()==2 )	
	  id2ancestor[tid] = aid;
	else {
	  id2ancestor[tid] = -2; // neutrino!
	  neutrino.insert(tid);
	}
      }
      for ( auto const& mcs : mcshower_v ) {
	int sid = mcs.TrackID();
	int aid = mcs.AncestorTrackID();
	if ( mcs.Origin()==2 )	
	  id2ancestor[sid] = aid;
	else {
	  id2ancestor[sid] = -2; // neutrino!
	  neutrino.insert(sid);
	}
      }
    }

    Cluster_t notruthhits;
    notruthhits.trackid = -1;

    for (auto const& hit : hit_v ) {
      if ( hit.truthflag>0 ) {
	int clusterid = hit.trackid;
	if ( use_ancestor_id && clusterid>=0 ) {
	  auto it_aid = id2ancestor.find( clusterid );
	  // replace trackid with ancestorid
	  if ( it_aid != id2ancestor.end() )
	    clusterid = it_aid->second;
	}
	auto it = trackid2index.find( clusterid );
	
	if ( it==trackid2index.end() ) {
	  // create new cluster
	  Cluster_t info;
	  info.trackid = clusterid;
	  info.phits.push_back( &hit ); // store address
	  // put into vector
	  output.emplace_back( std::move(info) );
	  int idx = output.size()-1;
	  trackid2index[clusterid] = idx;
	}
	else {
	  // already exists!
	  output[ it->second ].phits.push_back( &hit );
	}
	
      }
      else {
	// no truth match
	notruthhits.phits.push_back( &hit );
      }
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
    output.emplace_back( std::move(notruthhits) );
    
    return output;
  }
  

  TruthCluster::Cluster_t TruthCluster::assignUnmatchedToClusters( const std::vector<const larlite::larflow3dhit*>& unmatchedhit_v, std::vector<Cluster_t>& cluster_v ) {

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
  
  float TruthCluster::distToCluster( const larlite::larflow3dhit* phit, const Cluster_t* pcluster ) {
    return 0.;
  }

  
  void  TruthCluster::assignToClosestCluster( const larlite::larflow3dhit* phit, std::vector<Cluster_t>& clusters ) {
    
  }

}
