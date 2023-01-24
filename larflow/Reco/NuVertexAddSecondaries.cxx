#include "NuVertexAddSecondaries.h"
#include "geofuncs.h"
#include "NuTrackBuilder.h"
#include "NuVertexShowerReco.h"

namespace larflow {
namespace reco {

  void NuVertexAddSecondaries::process( larflow::reco::NuVertexCandidate& nuvtx,
					larflow::reco::ClusterBookKeeper& nuclusterbook,
					larcv::IOManager& iolcv,
					larlite::storage_manager& ioll )
  {
    // core loop
    // - make list of unused clusters
    // - attach them to vtx, track ends, track middle
    // - build out tracks (using nutrackbuilder)
    // - build out showers (using nuvertexshower)
    //
    // - storing info
    // - each track/shower in nuvertexcandidate has 'level index'
    // - each track/shower in nuvertexcandidate has mother cluster
    
    std::vector<std::string> cluster_sources =
      { "trackprojsplit_wcfilter",
	"cosmicproton",
	"showerkp",
	"showergoodhit" };
    std::vector<int> shower_or_track = 
      { 0, //track
	0, //track
	1, //shower
	1  //shower
      };

    struct SecondaryCandidate_t {
      larlite::larflowcluster* pcluster;
      float dist;
      int attached;
      int trackorshower;
      std::string producername;
      int clusteridx;
      std::vector<float> attach_pos;
      std::vector<float> attach_dir;
      std::vector<float> seedpos;            
      SecondaryCandidate_t( larlite::larflowcluster* pc, int ts )
	: pcluster(pc),
	  dist(999.0),
	  attached(0),
	  trackorshower(ts)
      {};
      bool operator<( SecondaryCandidate_t& rhs ) const {
	if ( dist<rhs.dist )
	  return true;
	return false;
      };
    };

    std::vector< SecondaryCandidate_t > candidates_v;

    for ( size_t iproducer=0; iproducer<cluster_sources.size(); iproducer++ ) {
      auto& producername = cluster_sources[iproducer];
      larlite::event_larflowcluster* ev_cluster =
	(larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, producername );
      larlite::event_pcaxis* ev_cluster_pca =
	(larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis, producername );

      for (size_t icluster=0; icluster<ev_cluster->size(); icluster++) {
	auto& cluster = ev_cluster->at(icluster);
	auto& clusterpca = ev_cluster_pca->at(icluster);	
	int status = nuclusterbook.cluster_status_v.at(cluster.matchedflash_idx);
	if ( status>0 )
	  continue; // already used

	// no loop through all the primary tracks
	// should it attach
	for (size_t itrack=0; itrack<nuvtx.track_v.size(); itrack++) {
	  LARCV_DEBUG() << "Test cluster(" << producername << "," << icluster << "," << cluster.matchedflash_idx << ")"
			  << " with nutrack[" << itrack << "]" << std::endl;
	  auto& track = nuvtx.track_v.at(itrack);
	  std::vector<float> attach_pos(3,0);
	  std::vector<float> attach_dir(3,0);
	  std::vector<float> seedpos(3,0);
	  float mindist = 999999;
	  if ( shower_or_track[ iproducer ]==0 ) {	  
	    mindist = testTrackTrackIntersection( track, clusterpca, 2.0,
						  attach_pos, attach_dir, seedpos );
	  }
	  else {
	    // larlite::event_track* ev_lltrunk
	    //   = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, producername);
	    // auto& showertrunk = ev_lltrunk->at(icluster);
	    mindist = testShowerTrackIntersection( track, clusterpca, 5.0,
						   attach_pos, attach_dir, seedpos );
	  }

	  float mindist_threshold = (shower_or_track[iproducer]==0 ) ? 2.0 : 100.0;
	  LARCV_DEBUG() << "  mindist=" << mindist << std::endl;
	  if (mindist<mindist_threshold) {
	    // register as potential new seed point
	    SecondaryCandidate_t cand( &cluster, shower_or_track[ iproducer ] );
	    cand.dist = mindist;
	    cand.producername = producername;
	    cand.clusteridx = icluster;
	    cand.attach_pos = attach_pos;
	    //cand.attach_pos = seedpos;
	    cand.attach_dir = attach_dir;
	    cand.seedpos = seedpos;
	    candidates_v.emplace_back( std::move(cand) );	      
	  }
	}//end of loop over tracks in the nuvertexcandidate
      }//end of cluster loop
    }//end of producer name

    // sort the candidate additions
    std::sort( candidates_v.begin(), candidates_v.end() );
    
    LARCV_DEBUG() << "Number of candidate additions: " << candidates_v.size() << std::endl;
    for (auto& candidate : candidates_v) {
      LARCV_DEBUG() << "  shower=" << candidate.trackorshower << " mindist=" << candidate.dist << std::endl;
    }
    
    LARCV_DEBUG() << "Now extend tracks using NuTrackBuilder" << std::endl;
    NuTrackBuilder _nu_track_builder;
    _nu_track_builder.set_verbosity( larcv::msg::kNORMAL );    
    _nu_track_builder.loadClustersAndConnections( iolcv, ioll );
    _nu_track_builder.set_verbosity( larcv::msg::kDEBUG );

    // Set secondary flags for previously added tracks & showers if this hasn't been done
    if ( nuvtx.track_isSecondary_v.size() == 0 ) {
      for (int i = 0; i < nuvtx.track_v.size(); i++) {
        nuvtx.track_isSecondary_v.push_back(0);
      }
    }
    if ( nuvtx.shower_isSecondary_v.size() == 0 ) {
      for (int i = 0; i < nuvtx.shower_v.size(); i++) {
        nuvtx.shower_isSecondary_v.push_back(0);
      }
    }
    
    for ( auto& candidate : candidates_v ) {

      if ( candidate.pcluster->matchedflash_idx>=0
	   && candidate.pcluster->matchedflash_idx<nuclusterbook.cluster_status_v.size() ) {
	if ( nuclusterbook.cluster_status_v.at( candidate.pcluster->matchedflash_idx )!=0 )
	  continue; // claimed, so move on.
      }
      
      if ( candidate.trackorshower==0 ) {
	// track
	// make a fake nuvtx candididate for the secondary attach point
	NuVertexCandidate nuvtx2;
	nuvtx2.pos = candidate.seedpos;

	// must provide the seed cluster
	NuVertexCandidate::VtxCluster_t vtxcluster;
	vtxcluster.producer = candidate.producername;
	vtxcluster.type = NuVertexCandidate::kTrack;
	vtxcluster.index = candidate.clusteridx;
	vtxcluster.pos = candidate.seedpos;
	nuvtx2.cluster_v.push_back( vtxcluster );
	
	std::vector< NuVertexCandidate > nuvtx2_v;
	std::vector< ClusterBookKeeper > book_v;
	nuvtx2_v.push_back( nuvtx2 );
	book_v.push_back( nuclusterbook );	
	_nu_track_builder.set_verbosity( larcv::msg::kDEBUG );
	_nu_track_builder.clear_track_proposals();
	_nu_track_builder.process( iolcv, ioll, nuvtx2_v, book_v, false );
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
	_nuvertex_shower_reco.set_verbosity( larcv::msg::kINFO );    
	_nuvertex_shower_reco.add_cluster_producer("trackprojsplit_wcfilter", NuVertexCandidate::kTrack );
	_nuvertex_shower_reco.add_cluster_producer("showerkp", NuVertexCandidate::kShowerKP );
	_nuvertex_shower_reco.add_cluster_producer("showergoodhit", NuVertexCandidate::kShower );    

	// make a fake nuvtx candididate for the secondary attach point
	NuVertexCandidate nuvtx2;
	nuvtx2.pos = candidate.seedpos;

	// must provide the seed cluster
	NuVertexCandidate::VtxCluster_t vtxcluster;
	vtxcluster.producer = candidate.producername;
	vtxcluster.type = NuVertexCandidate::kShowerKP;
	vtxcluster.index = candidate.clusteridx;
	vtxcluster.pos = candidate.attach_pos;
	nuvtx2.cluster_v.push_back( vtxcluster );
	_nuvertex_shower_reco.loadClusters(ioll);
	_nuvertex_shower_reco.build_vertex_showers( nuvtx2,
						    nuclusterbook,
						    iolcv, 
						    ioll );
	LARCV_DEBUG() << "tracks made from this seed: " << nuvtx2.shower_v.size() << std::endl;
	for (size_t ishower=0; ishower<nuvtx2.shower_v.size(); ishower++) {
	  nuvtx.shower_v.push_back( nuvtx2.shower_v.at(ishower) );
	  nuvtx.shower_trunk_v.push_back( nuvtx2.shower_trunk_v.at(ishower) );
	  nuvtx.shower_pcaxis_v.push_back( nuvtx2.shower_pcaxis_v.at(ishower) );
          nuvtx.shower_isSecondary_v.push_back(1);
	}
	
      }
    }
    
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

      float d = larflow::reco::lineLineDistance3f( cluster_start, cluster_end, pt1, pt2 );
      //std::cout << "ipt=" << ipt << " d=" << d << std::endl;
      
      if ( d>_max_line_dist )
	continue;

      float s1 = larflow::reco::pointRayProjection3f( pt1, segdir, cluster_start );
      float s2 = larflow::reco::pointRayProjection3f( pt1, segdir, cluster_end );

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

      float d = larflow::reco::lineLineDistance3f( cluster_start, cluster_end, pt1, pt2 );
      std::cout << "ipt=" << ipt << " d=" << d << std::endl;
      
      if ( d>_max_line_dist )
	continue;

      float s1 = larflow::reco::pointRayProjection3f( pt1, segdir, cluster_start );
      float s2 = larflow::reco::pointRayProjection3f( pt1, segdir, cluster_end );

      float ptdist1 = 0.;
      float ptdist2 = 0.;      
      
      if ( s1>-_max_line_dist && s1<seglen+_max_line_dist ) {
	for (int i=0; i<3; i++) {
	  float dx = (pt1[i] + segdir[i]*s1)-cluster_start[i];
	  ptdist1 += dx*dx;
	}
      }
      else {
	ptdist1 = 999999;
      }
      
      if ( s2>-_max_line_dist && s2<seglen+_max_line_dist ) {
	for (int i=0; i<3; i++) {
	  float dx = (pt1[i] + segdir[i]*s2)-cluster_end[i];
	  ptdist2 += dx*dx;
	}
      }
      else {
	ptdist2 = 999999;
      }

      std::cout << "ipt=" << ipt
		<< " s1=" << s1
		<< " s2=" << s2
		<< " seglen=" << seglen
		<< " ptdist1=" << ptdist1
		<< " ptdist2=" << ptdist2
		<< std::endl;

      // candidate intersection point on this track      
      if ( ptdist1<ptdist2 && ptdist1<min_seg_dist) {
	for (int i=0; i<3; i++) {
	  min_seg_pos[i] = pt1[i] + segdir[i]*s1;
	  attach_dir[i] = cluster_dir[i];
	  seedpos[i] = cluster_start[i];
	}
	min_seg_dist = ptdist1;
      }
      else if ( ptdist2<ptdist1 && ptdist2<min_seg_dist ) {
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
  
}
}
