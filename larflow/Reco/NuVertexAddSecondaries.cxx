#include "NuVertexAddSecondaries.h"
#include "larflow/RecoUtils/geofuncs.h"
#include "NuVertexShowerReco.h"
#include "larflow/LArFlowConstants/LArFlowConstants.h"

namespace larflow {
namespace reco {

  void NuVertexAddSecondaries::init_trackbuilder_for_event( larcv::IOManager& iolcv, larlite::storage_manager& ioll  )
  {
    
    // dont want this to be created every time. N2 graph builder...
    //_nu_track_builder.set_verbosity( larcv::msg::kNORMAL );
    _nu_track_builder.clear();
    _nu_track_builder.set_verbosity( logger().level() );    
    _nu_track_builder.loadClustersAndConnections( iolcv, ioll );
    
  }
  
  void NuVertexAddSecondaries::process( larflow::reco::NuVertexCandidate& nuvtx,
					larflow::reco::ClusterBookKeeper& nuclusterbook,
					larcv::IOManager& iolcv,
					larlite::storage_manager& ioll )
  {

    LARCV_INFO() << "start" << std::endl;
    // core loop
    // - make list of unused clusters
    // - attach them to vtx, track ends, track middle
    // - build out tracks (using nutrackbuilder)
    // - build out showers (using nuvertexshower)
    //
    // - storing info
    // - each track/shower in nuvertexcandidate has 'level index'
    // - each track/shower in nuvertexcandidate has mother cluster

    // std::vector<std::string> cluster_sources =
    //   { "trackprojsplit_wcfilter",
    //     "cosmicproton",
    //     "showerkp",
    //     "showergoodhit" };
    // std::vector<int> shower_or_track = 
    //   { 0, //track
    //     0, //track
    //     1, //shower
    //     1  //shower
    //   };
    
    struct SecondaryCandidate_t {
      larlite::larflowcluster* pcluster;
      float dist; // distance between start of cluster and attachment point to interaction particle trajectory
      int attached;
      int trackorshower;
      std::string producername;
      int bookidx;
      int cluster_type;
      int container_idx;
      int npts;
      std::vector<float> attach_pos; // pt on existing track
      std::vector<float> attach_dir; // from pt to startpt of secondary track or shower cluster we are adding
      std::vector<float> seedpos;    // location of startpt of secondary track or shower cluster
      SecondaryCandidate_t( larlite::larflowcluster* pc, int ts )
      : pcluster(pc),
        dist(9999.0),
        attached(0),
        trackorshower(ts),
	producername(""),
	bookidx(-1),
	cluster_type(-1),
	container_idx(-1),
	npts(0)
      {};
      bool operator<( const SecondaryCandidate_t& rhs ) const {
	// sort by distance from attachment point
	if ( dist<rhs.dist )
	  return true;
	return false;
      };
    };

    std::vector< SecondaryCandidate_t > candidates_v;

    // loop through the cluster book keeper to add clusters to event.
    // we use the book keeper in order to exluce clusters added to the vertex
    for (int icluster=0; icluster<(int)nuclusterbook.cluster_producer_v.size(); icluster++) {

      if ( nuclusterbook.cluster_status_v.at(icluster)!=0 )
	continue; // used to vetoed
      
      std::string producername = nuclusterbook.cluster_producer_v.at(icluster);
      int container_index = nuclusterbook.cluster_container_index_v.at(icluster);
      int ictype = nuclusterbook.cluster_type_v.at(icluster);
      if ( ictype<0 || ictype>2 )
	continue; // weird value
      
      larflow::reco::NuVertexCandidate::ClusterType_t ctype = (larflow::reco::NuVertexCandidate::ClusterType_t)ictype;
      
      LARCV_INFO() << "try to attach to cluster[" << producername << ", con_idx=" << container_index << ", ctype=" << ictype << "]" << std::endl;
      
      larlite::event_larflowcluster* ev_cluster =
      	(larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, producername );
      larlite::event_pcaxis* ev_cluster_pca =
      	(larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis, producername );

      // not in container
      if ( container_index<0 && container_index >= (int)ev_cluster->size() )
	continue;      

      auto& cluster = ev_cluster->at(container_index);
      auto& clusterpca = ev_cluster_pca->at(container_index);	

      // now loop through all the current tracks in the nucandidate
      // determine if should attach this cluster
      // look for proximity to exiting track
      for (size_t itrack=0; itrack<nuvtx.track_v.size(); itrack++) {
	LARCV_DEBUG() << "Test cluster(" << producername << "," << icluster << "," << cluster.matchedflash_idx << ")"
		      << " with nutrack[" << itrack << "]" << std::endl;
	auto& track = nuvtx.track_v.at(itrack);
	std::vector<float> attach_pos(3,0);
	std::vector<float> attach_dir(3,0);
	std::vector<float> seedpos(3,0);
	float mindist = 999999;
	if ( ctype==larflow::reco::NuVertexCandidate::kTrack ) {
	  // intersection test if track
	  mindist = testTrackTrackIntersection( track, clusterpca, 2.0,
						attach_pos, attach_dir, seedpos );
	}
	else {
	  // intersection test if shower
	  mindist = testShowerTrackIntersection( track, clusterpca, 5.0,
						 attach_pos, attach_dir, seedpos );
	}

	float mindist_threshold = (ctype==larflow::reco::NuVertexCandidate::kTrack) ? 2.0 : 100.0;
	if (mindist<mindist_threshold) {
	  // register as potential new seed point	  
	  int track_or_shower = (ctype==larflow::reco::NuVertexCandidate::kTrack) ? 0 : 1;
	  
	  SecondaryCandidate_t cand( &cluster, track_or_shower );	    
	  cand.dist = mindist;
	  cand.producername = producername; // name of cluster container
	  cand.bookidx = icluster; // book index
	  cand.container_idx = container_index; // index in cluster container
	  cand.cluster_type = ctype;
	  cand.attach_pos = attach_pos;
	  cand.attach_dir = attach_dir;
	  cand.seedpos = seedpos;
	  cand.npts = cluster.size();
	  candidates_v.emplace_back( std::move(cand) );

	}
      }//end of loop over tracks in the nuvertexcandidate
    }//end of cluster loop

    // sort the candidate additions from closest to furthest
    std::sort( candidates_v.begin(), candidates_v.end() );
    
    LARCV_INFO() << "Number of candidate additions: " << candidates_v.size() << std::endl;
    for (auto& candidate : candidates_v) {
      if ( candidate.trackorshower==1 )
	LARCV_INFO() << "  shower[" << candidate.producername
		     << ", conidx=" << candidate.container_idx
		     << ", bookidx=" << candidate.bookidx << "] "
		     << " dist=" << candidate.dist << std::endl;
      else
	LARCV_INFO() << "  track[" << candidate.producername
		     << ", conidx=" << candidate.container_idx
		     << ", bookidx=" << candidate.bookidx << "] "
		     << " dist=" << candidate.dist << std::endl;	
    }
    
    LARCV_DEBUG() << "Now extend tracks using NuTrackBuilder and NuVertexShowerReco" << std::endl;

    // dont want this to be created every time. N2 graph builder...
    //_nu_track_builder.set_verbosity( larcv::msg::kNORMAL );
    // _nu_track_builder.set_verbosity( logger().level() );    
    // _nu_track_builder.loadClustersAndConnections( iolcv, ioll );
    // //_nu_track_builder.set_verbosity( larcv::msg::kDEBUG );
    // _nu_track_builder.set_verbosity( logger().level() );

    // Set secondary flags for previously added tracks & showers if this hasn't been done
    if ( nuvtx.track_isSecondary_v.size() < nuvtx.track_v.size() ) {
      nuvtx.track_isSecondary_v.resize(nuvtx.track_v.size(),0);
    }
    if ( nuvtx.shower_isSecondary_v.size() < nuvtx.shower_v.size() ) {
      nuvtx.shower_isSecondary_v.resize(nuvtx.shower_v.size(),0);
    }

    // loop over secondary candidates
    for ( auto& candidate : candidates_v ) {

      if ( nuclusterbook.cluster_status_v.at( candidate.bookidx )!=0 ) {
	continue; // claimed, so move on.
      }

      // we will treat the secondary intersection point as a new vertex
      // do build tracks and showers without adding clusters we've already
      // assigned to intersection, need to prepare clusterbook
      ClusterBookKeeper book2 = nuclusterbook; // copy the current cluster book
      bool foundmark = false;
      int nused = 0;
      for (int i=0; i<book2.cluster_status_v.size(); i++) {
	if ( book2.cluster_status_v[i]>0 ) {
	  book2.cluster_status_v[i] = 1; //mark as used (so skip)
	  nused++;
	}
	if ( candidate.trackorshower==1 ) {
	  // if shower, we mark a pre-determined prong seed to build a shower from
	  if ( book2.cluster_producer_v[i]==candidate.producername
	       && book2.cluster_container_index_v[i]==candidate.container_idx ) {
	    book2.cluster_status_v[i] = 2; //mark as pre-determined prong seed
	    foundmark = true;
	  }
	}
      }
      if ( candidate.trackorshower==1 && !foundmark ) {
	LARCV_ERROR() << "Did not find shower cluster to mark in copied clusterbook" << std::endl;
      }
      LARCV_INFO() << " prepared candidate secondary vertex clusterbook: number used=" << nused << std::endl;
      std::vector< ClusterBookKeeper > book_v;
      book_v.push_back( book2 );
      
      if ( candidate.trackorshower==0 ) {
	// track-like: extend with nutrackbuilder
	
	// make a fake nuvtx candididate for the secondary attach point
	NuVertexCandidate nuvtx2;
	nuvtx2.pos = candidate.attach_pos;

	// must provide the seed cluster
	NuVertexCandidate::VtxCluster_t vtxcluster;
	vtxcluster.producer = candidate.producername;
	vtxcluster.type = NuVertexCandidate::kTrack;
	vtxcluster.index = candidate.container_idx;
	vtxcluster.pos = candidate.seedpos;
	vtxcluster.npts = candidate.npts;	
	nuvtx2.cluster_v.push_back( vtxcluster );
	
	std::vector< NuVertexCandidate > nuvtx2_v;
	nuvtx2_v.push_back( nuvtx2 );
	
	_nu_track_builder.set_verbosity( logger().level() );
	_nu_track_builder.clear_track_proposals();
	bool reload_clusters = false;
	_nu_track_builder.process( iolcv, ioll, nuvtx2_v, book_v, reload_clusters );
	LARCV_DEBUG() << "tracks made from this seed: " << nuvtx2_v.at(0).track_v.size() << std::endl;
	if ( nuvtx2_v.at(0).track_v.size()>0 ) {
	  nuvtx.track_v.push_back( nuvtx2_v.at(0).track_v.at(0) );
	  nuvtx.track_hitcluster_v.push_back( nuvtx2_v.at(0).track_hitcluster_v.at(0) );
          nuvtx.track_isSecondary_v.push_back(1);
	}
      }
      else {
	// shower
	NuVertexShowerReco _nuvertex_shower_reco;
	//_nuvertex_shower_reco.set_verbosity( larcv::msg::kINFO );
	_nuvertex_shower_reco.set_verbosity( logger().level() );
	//_nuvertex_shower_reco.set_seed_with_existing_clusters( false );
	_nuvertex_shower_reco.add_cluster_producer("trackprojsplit_wcfilter", NuVertexCandidate::kTrack );
	_nuvertex_shower_reco.add_cluster_producer("showergoodhit", NuVertexCandidate::kShower );	
	_nuvertex_shower_reco.loadClusters(ioll); // load clusters
	
	// make a fake nuvtx candididate for the secondary attach point
	NuVertexCandidate nuvtx2;
	nuvtx2.pos = candidate.seedpos;
	LARCV_INFO() << "  Build secondary shower vertex at ("
		     << nuvtx2.pos[0] << ","
		     << nuvtx2.pos[1] << ","
		     << nuvtx2.pos[2] << ")"
		     << " using cluster[" << nuclusterbook.cluster_producer_v.at( candidate.pcluster->matchedflash_idx )
		     << ", " << candidate.pcluster->matchedflash_idx << "]"
		     << std::endl;

	// must provide the seed cluster
	NuVertexCandidate::VtxCluster_t vtxcluster;
	vtxcluster.producer = candidate.producername;
	vtxcluster.type = NuVertexCandidate::kShower;
	vtxcluster.index = candidate.container_idx;
	vtxcluster.pos = candidate.seedpos;
	vtxcluster.npts = candidate.npts;
	nuvtx2.cluster_v.push_back( vtxcluster );

	_nuvertex_shower_reco.build_vertex_showers( nuvtx2,
						    book2,
						    iolcv, 
						    ioll );
	LARCV_INFO() << "  secondary showers made from this seed: " << nuvtx2.shower_v.size() << std::endl;
	for (size_t ishower=0; ishower<nuvtx2.shower_v.size(); ishower++) {
	  nuvtx.shower_v.push_back( nuvtx2.shower_v.at(ishower) );
	  nuvtx.shower_trunk_v.push_back( nuvtx2.shower_trunk_v.at(ishower) );
	  nuvtx.shower_pcaxis_v.push_back( nuvtx2.shower_pcaxis_v.at(ishower) );
          nuvtx.shower_isSecondary_v.push_back(1);
	  // update the nucluster book
	  for (int ic=0; ic<(int)book2.cluster_status_v.size(); ic++) {
	    //
	  }
	}
	
      }//end of else if shower candidate
    }//end of loop over secondary candidates
    
  }

  float NuVertexAddSecondaries::testTrackTrackIntersection( larlite::track& track,
							    larlite::pcaxis& cluster_pca,
							    const float _max_line_dist,
							    std::vector<float>& attach_pos,
							    std::vector<float>& attach_dir,
							    std::vector<float>& seedpos )
  {
    
    std::vector< float > cluster_start(3,0);
    std::vector< float > cluster_end(3,0);
    std::vector< float > cluster_dir(3,0);

    seedpos.resize(3,0);
    attach_dir.resize(3,0);
    attach_pos.resize(3,0);

    float len = 0;
    for (int i=0; i<3; i++) {
      cluster_start[i] = cluster_pca.getEigenVectors()[3][i];
      cluster_end[i] = cluster_pca.getEigenVectors()[4][i];
      cluster_dir[i] = cluster_end[i]-cluster_start[i];
      len += cluster_dir[i]*cluster_dir[i];
    }
    len = sqrt(len);

    if ( len>0.0 ) {
      for (int i=0; i<3; i++)
	cluster_dir[i] /= len;
    }

    int npts = track.NumberTrajectoryPoints();

    float min_seg_dist = 999999;
    std::vector<float> min_seg_pos(3,0);
    
    for (int ipt=0; ipt<npts-1; ipt++) {
      std::vector<float> pt1(3,0);
      std::vector<float> pt2(3,0);
      std::vector<float> segdir(3,0);
      float seglen = 0.;
      for (int i=0; i<3; i++) {
	pt1[i] = track.LocationAtPoint(ipt)[i];
	pt2[i] = track.LocationAtPoint(ipt+1)[i];
	segdir[i] = pt2[i]-pt1[i];
	seglen += segdir[i]*segdir[i];
      }
      seglen = sqrt(seglen);
      if (seglen>0) {
	for (int i=0; i<3; i++)
	  segdir[i] /= seglen;
      }
      else {
        continue;
      }

      float d = larflow::recoutils::lineLineDistance3f( cluster_start, cluster_end, pt1, pt2 );
      //std::cout << "ipt=" << ipt << " d=" << d << std::endl;
      
      if ( d>_max_line_dist )
	continue;

      float s1 = larflow::recoutils::pointRayProjection3f( pt1, segdir, cluster_start );
      float s2 = larflow::recoutils::pointRayProjection3f( pt1, segdir, cluster_end );

      float ptdist1 = 0.;
      float ptdist2 = 0.;      
      
      if ( s1>-_max_line_dist && s1<seglen ) {
	for (int i=0; i<3; i++) {
	  float dx = (pt1[i] + segdir[i]*s1)-cluster_start[i];
	  ptdist1 += dx*dx;
	}
      }
      else {
	ptdist1 = 999999;
      }
      
      if ( s2>-_max_line_dist && s2<seglen ) {
	for (int i=0; i<3; i++) {
	  float dx = (pt1[i] + segdir[i]*s2)-cluster_end[i];
	  ptdist2 += dx*dx;
	}
      }
      else {
	ptdist2 = 999999;
      }

      // std::cout << "ipt=" << ipt
      // 		<< " s1=" << s1
      // 		<< " s2=" << s2
      // 		<< " seglen=" << seglen
      // 		<< " ptdist1=" << ptdist1
      // 		<< " ptdist2=" << ptdist2
      // 		<< std::endl;

      // candidate intersection point on this track      
      if ( ptdist1<ptdist2 && ptdist1<_max_line_dist && ptdist1<min_seg_dist) {
	for (int i=0; i<3; i++) {
	  min_seg_pos[i] = pt1[i] + segdir[i]*s1;
	  attach_dir[i] = cluster_dir[i];
	  seedpos[i] = cluster_start[i];
	}
	min_seg_dist = ptdist1;
      }
      else if ( ptdist2<ptdist1 && ptdist2<_max_line_dist && ptdist2<min_seg_dist ) {
	for (int i=0; i<3; i++) {
	  min_seg_pos[i] = pt1[i] + segdir[i]*s2;
	  attach_dir[i] = -cluster_dir[i];
	  seedpos[i] = cluster_end[i];	  
	}
	min_seg_dist = ptdist2;
      }
      
    }//end of loop over points along the track line

    attach_pos = min_seg_pos;
    
    return min_seg_dist;
  }

  float NuVertexAddSecondaries::testShowerTrackIntersection( larlite::track& track,
							     larlite::pcaxis& shower_trunk,
							     const float _max_line_dist,
							     std::vector<float>& attach_pos,
							     std::vector<float>& attach_dir,
							     std::vector<float>& seedpos )
  {

    std::vector< float > cluster_start(3,0);
    std::vector< float > cluster_end(3,0);
    std::vector< float > cluster_dir(3,0);

    seedpos.resize(3,0);
    attach_dir.resize(3,0);
    attach_pos.resize(3,0);

    float len = 0;
    for (int i=0; i<3; i++) {
      cluster_start[i] = shower_trunk.getEigenVectors()[3][i];
      cluster_end[i]   = shower_trunk.getEigenVectors()[4][i];      
      // cluster_start[i] = shower_trunk.LocationAtPoint(0)[i];
      // cluster_end[i]   = shower_trunk.LocationAtPoint(1)[i];
      cluster_dir[i]   = cluster_end[i]-cluster_start[i];
      len += cluster_dir[i]*cluster_dir[i];
    }
    len = sqrt(len);

    if ( len>0.0 ) {
      for (int i=0; i<3; i++)
	cluster_dir[i] /= len;
    }

    int npts = track.NumberTrajectoryPoints();

    float min_seg_dist = 999999;
    std::vector<float> min_seg_pos(3,0);
    
    for (int ipt=0; ipt<npts-1; ipt++) {
      std::vector<float> pt1(3,0);
      std::vector<float> pt2(3,0);
      std::vector<float> segdir(3,0);
      float seglen = 0.;
      for (int i=0; i<3; i++) {
	pt1[i] = track.LocationAtPoint(ipt)[i];
	pt2[i] = track.LocationAtPoint(ipt+1)[i];
	segdir[i] = pt2[i]-pt1[i];
	seglen += segdir[i]*segdir[i];
      }
      seglen = sqrt(seglen);
      if (seglen>0) {
	for (int i=0; i<3; i++)
	  segdir[i] /= seglen;
      }
      else {
        continue;
      }

      std::vector<float> closest1(3,0);
      std::vector<float> closest2(3,0);
      //float d = larflow::recoutils::lineLineDistance3f( cluster_start, cluster_end, pt1, pt2 );
      float d_claude = larflow::recoutils::lineLineDistance3f_claude( pt1, segdir,
								      cluster_start, cluster_dir,
								      closest1, closest2 );
      
      //LARCV_DEBUG() << " (geofunc)d=" << d << " (claude)d=" << d_claude << std::endl;
      if ( d_claude>_max_line_dist )
	continue;
      
      // which is the closest point? use point-line distance to tell
      float s1 = larflow::recoutils::pointLineDistance3f( pt1, pt2, closest1 );
      float s2 = larflow::recoutils::pointLineDistance3f( pt1, pt2, closest2 );

      std::vector<float> intersectpt(3,0);
      if ( s1<s2 )
	intersectpt = closest1;
      else
	intersectpt = closest2;

      // is it on the line segment we tested?
      float t1 = larflow::recoutils::pointRayProjection3f( pt1, segdir, intersectpt );
      
      if ( t1>-0.3 && t1<seglen+0.3 ) {
	// on segment, test if its the best intersection pt we've seen
	if ( d_claude < min_seg_dist ) {
	  min_seg_dist = d_claude;
	  min_seg_pos = intersectpt;
	}
      }
      
    }//end of loop over points along the track line

    attach_pos = min_seg_pos;
    float attachlen = 0.;
    for (int v=0; v<3; v++) {
      attach_dir[v] = cluster_start[v]-attach_pos[v];
      attachlen += attach_dir[v]*attach_dir[v];
    }
    attachlen = sqrt(attachlen);
    if ( attachlen>1.0e-10 ) {
      for (int v=0; v<3; v++)
	attach_dir[v] /= attachlen;
    }

    seedpos = cluster_start;
    
    return min_seg_dist;

  }
  
}
}
