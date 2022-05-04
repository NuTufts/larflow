#include "NuVertexAddSecondaries.h"

#include "geofuncs.h"

namespace larflow {
namespace reco {

  void NuVertexAddSecondaries::process( larflow::reco::NuVertexCandidate& nuvtx,
					larflow::reco::ClusterBookKeeper& nuclusterbook,
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
	2  //shower
      };

    struct SecondaryCandidate_t {
      larlite::larflowcluster* pcluster;
      float dist;
      int attached;
      int trackorshower;
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
	if ( shower_or_track[ iproducer ]==0 ) {
	  for (size_t itrack=0; itrack<nuvtx.track_v.size(); itrack++) {
	    // LARCV_DEBUG() << "Test cluster(" << producername << "," << icluster << "," << cluster.matchedflash_idx << ")"
	    // 		  << " with nutrack[" << itrack << "]" << std::endl;
	    auto& track = nuvtx.track_v.at(itrack);
	    float mindist = testTrackTrackIntersection( track, clusterpca, 2.0 );
	    //LARCV_DEBUG() << "  mindist=" << mindist << std::endl;
	    if (mindist<2.0) {
	      // register as potential new seed point
	      SecondaryCandidate_t cand( &cluster, shower_or_track[ iproducer ] );
	      cand.dist = mindist;
	      candidates_v.emplace_back( std::move(cand) );
	    }
	  }
	}
      }//end of cluster loop
    }//end of producer name

    // sort the candidate additions
    std::sort( candidates_v.begin(), candidates_v.end() );
    
    LARCV_DEBUG() << "Number of candidate additions: " << candidates_v.size() << std::endl;
    
  }

  float NuVertexAddSecondaries::testTrackTrackIntersection( larlite::track& track,
							    larlite::pcaxis& cluster_pca,
							    const float _max_line_dist )
  {
    
    std::vector< float > cluster_start(3,0);
    std::vector< float > cluster_end(3,0);
    std::vector< float > cluster_dir(3,0);

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
	for (int i=0; i<3; i++)
	  min_seg_pos[i] = pt1[i] + segdir[i]*s1;
	min_seg_dist = ptdist1;
      }
      else if ( ptdist2<ptdist1 && ptdist2<_max_line_dist && ptdist2<min_seg_dist ) {
	for (int i=0; i<3; i++)
	  min_seg_pos[i] = pt1[i] + segdir[i]*s2;
	min_seg_dist = ptdist2;
      }
      
    }//end of loop over points along the track line

    return min_seg_dist;
  }

}
}
