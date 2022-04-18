#include "VetoHitClustering.h"
#include "larlite/DataFormat/larflow3dhit.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/track.h"
#include "geofuncs.h"

namespace larflow {
namespace reco {

  void VetoHitClustering::process( larlite::storage_manager& io,
				   larflow::reco::NuVertexCandidate& nuvtx )
  {

    const float veto_radius = 10.;
    const float min_veto_hits = 5;
    
    // get veto'd hits
    larlite::event_larflow3dhit* evout_veto =
      (larlite::event_larflow3dhit*)io.get_data( larlite::data::kLArFlow3DHit, "projsplitvetoed" );

    // get hits around the 
    std::vector<int> close_hit_indices_v;
    close_hit_indices_v.reserve( evout_veto->size() );

    float dist = 0.;
    for ( size_t idx=0; idx<evout_veto->size(); idx++ ) {
      auto const& hit = evout_veto->at(idx);
      dist=0.;
      for (int v=0; v<3; v++)
	dist += ( hit[v]-nuvtx.pos[v] )*( hit[v]-nuvtx.pos[v] );
      dist = sqrt(dist);

      if ( dist < veto_radius ) {
	close_hit_indices_v.push_back( (int)idx );
      }
    }

    LARCV_INFO() << "Vetoed hits close to the radius: " << close_hit_indices_v.size() << std::endl;

    if ( close_hit_indices_v.size()<min_veto_hits )
      return;

    _merge_hits_into_prongs( *evout_veto, close_hit_indices_v, nuvtx );

    std::vector<larflow::reco::cluster_t> output_cluster_v;
    _findVetoClusters( *evout_veto, close_hit_indices_v, nuvtx, output_cluster_v );
    
  }

  void VetoHitClustering::_merge_hits_into_prongs( const larlite::event_larflow3dhit& inputhits,
						   std::vector<int>& close_hits_v,
						   larflow::reco::NuVertexCandidate& nuvtx )
  {

    LARCV_DEBUG() << "start: vtx=(" << nuvtx.pos[0] << "," << nuvtx.pos[1] << "," << nuvtx.pos[2] << ")" << std::endl;
    
    // collect veto hits into a shorter list
    std::vector< std::vector<float> > veto_pts_v;
    std::vector< std::vector<float> > veto_pts_orig_v;
    std::vector< int > hitidx_v; // map into original list

    veto_pts_v.reserve( close_hits_v.size() );
    hitidx_v.reserve( close_hits_v.size() );
    
    for (size_t ihit=0; ihit<close_hits_v.size(); ihit++) {
      int idx = close_hits_v[ihit];
      auto const& hit = inputhits[idx];
      
      std::vector<float> pt = { hit[0], hit[1], hit[2] };
      
      std::vector<float> raddir(3,0);
      float lenr = 0.;
      for (int i=0; i<3; i++) {
	raddir[i] = pt[i]-nuvtx.pos[i];
	lenr += raddir[i]*raddir[i];
      }
      lenr = sqrt(lenr);

      if ( lenr<=0.01 )
	continue;

      veto_pts_orig_v.push_back( pt );      
      
      for (int i=0; i<3; i++)
	raddir[i] /= lenr;

      // std::cout << "vtx hit[" << ihit << "] "
      // 		<< "pt-orig=(" << pt[0] << "," << pt[1] << "," << pt[2] << ") "
      // 		<< "lenr=" << lenr
      // 		<< " dir=(" << raddir[0] << "," << raddir[1] << "," << raddir[2] << ")"
      // 		<< std::endl;
      
      if ( lenr<3.0 ) {
	// we push the point away from the keypoint to prevent overclustering
	for (int i=0; i<3; i++)
	  pt[i] += (3.0-lenr+0.25)*raddir[i];
      }
      
      veto_pts_v.push_back( pt );
      //hitidx_v.push_back(ihit);
      hitidx_v.push_back(idx);
    }

    int nvetohits = veto_pts_v.size();

    
    // for each hit, we decide which track prong to assign the hit to (if any)
    // we do this by defining a line between the track-prong and vertex
    // define the direction between the vertex and track prong starts
    int nprongs = nuvtx.track_v.size();    
    std::vector< std::vector<float> > prong_dir_v;
    std::vector< std::vector<float> > prong_start_v;
    for (int itrack=0; itrack<nprongs; itrack++) {
      auto& track = nuvtx.track_v.at(itrack);
      std::vector<float> prong_dir(3,0);
      std::vector<float> prong_start(3,0);
      float len = 0.;
      for (int v=0; v<3; v++) {
	prong_start[v] = track.LocationAtPoint(0)[v];
	prong_dir[v] = prong_start[v]-nuvtx.pos[v];
	len += prong_dir[v]*prong_dir[v];
      }
      for (int v=0; v<3; v++)
	prong_dir[v] /= len;
      prong_dir_v.push_back(prong_dir);
      prong_start_v.push_back(prong_start);
    }
    
    // we record for each hit, the closest prong and the distance to it
    std::vector<float> min_prong_dist( nvetohits, 99.0 );
    std::vector<float> min_prong_s(    nvetohits, 99.0 );    
    std::vector<int>   min_prong_index( nvetohits, -1 );

    LARCV_DEBUG() << "match " << nvetohits << " to " << nprongs << " track prongs" << std::endl;
    
    for (int ihit=0; ihit<nvetohits; ihit++) {
      std::vector<float>& pt      = veto_pts_v[ihit];
      std::vector<float>& pt_orig = veto_pts_orig_v[ihit];      
      int origidx = hitidx_v[ihit];      
      auto const& hit = inputhits[origidx];
      
      for (int itrack=0; itrack<nprongs; itrack++) {
	// calculate distance from line segment to hit
	float r = larflow::reco::pointLineDistance3f( nuvtx.pos, prong_start_v[itrack], pt );
	float r_orig = larflow::reco::pointLineDistance3f( nuvtx.pos, prong_start_v[itrack], pt_orig );	
	// calculate projection of hit onto line-segment
	float s = larflow::reco::pointRayProjection3f( nuvtx.pos, prong_dir_v[itrack], pt );
	float s_orig = larflow::reco::pointRayProjection3f( nuvtx.pos, prong_dir_v[itrack], pt_orig );	
	// std::cout << "ihit=" << ihit << " itrack=" << itrack
	// 	  << " r=" << r << " s=" << s
	// 	  << " ro=" << r_orig << " so=" << s_orig
	// 	  << " (" << pt[0] << "," << pt[1] << "," << pt[2] << ") "
	// 	  << " orig=(" << pt_orig[0] << "," << pt_orig[1] << "," << pt_orig[2] << ")"
	// 	  << std::endl;
	if ( s>0 && r<min_prong_dist[ihit] ) {
	  // newest closest prong
	  min_prong_dist[ihit] = r;
	  min_prong_s[ihit] = s;
	  min_prong_index[ihit] = itrack;
	}
      }
    }

    // now decide on assignment to prongs
    std::vector<int> prong_modified(nprongs,0);

    // we store info for assigned prongs using this struct.
    // it allows us to sort the hits by projection distance, s.
    // this way we can add the points to the larflowhit cluster vector
    // in order of lowest s to highest s, preserving the order of the hits
    // from the vertex out to the end of the track
    struct ProngAddition_t {
      float s;
      float r;
      int origindex;
      ProngAddition_t(float ss, float rr, float ii )
	: s(ss), r(rr), origindex(ii)
      {};
      bool operator<( const ProngAddition_t& rhs ) {
	if ( s<rhs.s) return true;
	return false;
      };
    };

    LARCV_DEBUG() << "Collect hits matched to prongs" << std::endl;
    std::vector< std::vector<ProngAddition_t> > added_hits_v(nprongs);
    for (int ihit=0; ihit<nvetohits; ihit++) {
      int origidx = hitidx_v[ihit];
      if ( min_prong_index[ihit]>=0 && min_prong_dist[ihit]<2.0 ) {
	int modprong = min_prong_index[ihit];
	// we add the hit cluster
	// we have to get the true s since we pushed the position away from the keypoint

	auto const& hit = inputhits[ origidx ];	
	std::vector<float> pt = { hit[0], hit[1], hit[2] };
	float s = larflow::reco::pointRayProjection3f( nuvtx.pos, prong_dir_v[modprong], pt );
	
	ProngAddition_t addme( s, min_prong_dist[ihit], origidx );
	added_hits_v[modprong].push_back( addme );
	prong_modified[modprong] = 1;
	close_hits_v[ihit] = -1;
      }
    }
    
    // sort the added hits
    LARCV_DEBUG() << "Sort hits matched to prongs" << std::endl;    
    for (int iprong=0; iprong<nprongs; iprong++) {
      if ( added_hits_v[iprong].size()>1 ) {
	std::sort( added_hits_v[iprong].begin(), added_hits_v[iprong].end() );
      }
    }

    // ok, now we modify the prong clusters and tracks
    LARCV_DEBUG() << "Modify the track prongs to include the matched veto hits" << std::endl;        
    int num_modified = 0;        
    for (int iprong=0; iprong<nprongs; iprong++) {

      if ( prong_modified[iprong]==0 )
	continue;

      num_modified++;

      LARCV_DEBUG() << "Modifying track-prong[" << iprong << "] num hits to add = " << added_hits_v[iprong].size() << std::endl;
      
      // unfortunately, since we are prepending, we have to create replace copies of the hit cluster and
      // track object
      
      // orig track
      larlite::track& origtrack = nuvtx.track_v[iprong];
      
      // modify the track
      larlite::track modtrack;
      modtrack.reserve( origtrack.NumberTrajectoryPoints()+1 );

      TVector3 newstartpt;
      TVector3 newstartdir;
      for (int v=0; v<3; v++) {
	newstartpt[v]  = nuvtx.pos[v];
	newstartdir[v] = prong_dir_v[iprong][v];
      }
      
      modtrack.add_vertex( newstartpt );
      modtrack.add_direction( newstartdir );

      for (int i=0; i<(int)origtrack.NumberTrajectoryPoints(); i++) {
	modtrack.add_vertex( origtrack.LocationAtPoint(i) );
	modtrack.add_direction( origtrack.DirectionAtPoint(i) );
      }

      // modify the cluster
      larlite::larflowcluster& origcluster = nuvtx.track_hitcluster_v[iprong];
      larlite::larflowcluster modcluster;
      modcluster.reserve( origcluster.size() + added_hits_v[iprong].size() );
      for (int i=0; i<(int)added_hits_v[iprong].size(); i++) {
	auto const& info = added_hits_v[iprong][i];
	auto const& addedhit = inputhits[info.origindex];
	std::cout << "  added hit index=" << info.origindex << " "
		  << "(" << addedhit[0] << "," << addedhit[1] << "," << addedhit[2] << ")"
		  << std::endl;
	modcluster.push_back(addedhit);
      }
      for ( auto const& hit : origcluster ) {
	modcluster.push_back( hit );
      }

      LARCV_DEBUG() << "orig nhits=" << origcluster.size() << " vs. mod nhits=" << modcluster.size() << std::endl;

      // now swap out modified objects for new objects
      std::swap( nuvtx.track_v[iprong], modtrack );
      std::swap( nuvtx.track_hitcluster_v[iprong], modcluster );
    }// prong loop where we modify the track prongs

    LARCV_INFO() << "Number of modified track prongs: " << num_modified << std::endl;
    
  }
  
  void VetoHitClustering::_findVetoClusters( const larlite::event_larflow3dhit& inputhits,
					     const std::vector<int>& close_hits_v,
					     larflow::reco::NuVertexCandidate& nuvtx,
					     std::vector<larflow::reco::cluster_t>& output_cluster_v )
  {

    std::vector< std::vector<float> > veto_pts_v;
    std::vector< int > hitidx_v;

    veto_pts_v.reserve( close_hits_v.size() );
    hitidx_v.reserve( close_hits_v.size() );
    
    for (size_t ihit=0; ihit<close_hits_v.size(); ihit++) {

      int idx = close_hits_v[ihit];
      if ( idx<0 )
	continue;
      
      auto const& hit = inputhits[idx];
      
      std::vector<float> pt = { hit[0], hit[1], hit[2] };
      std::vector<float> pt_orig = { hit[0], hit[1], hit[2] };      
      std::vector<float> raddir(3,0);
      float lenr = 0.;
      for (int i=0; i<3; i++) {
	raddir[i] = pt[i]-nuvtx.pos[i];
	lenr += raddir[i]*raddir[i];
      }
      lenr = sqrt(lenr);
      for (int i=0; i<3; i++)
	raddir[i] /= lenr;

      if ( lenr<3.0 ) {
	// we push the point away from the keypoint to prevent overclustering
	for (int i=0; i<3; i++)
	  pt[i] += (3.0-lenr+0.25)*raddir[i];
      }
      std::cout << "  veto clust hit[" << ihit << "] "
		<< "pt=(" << pt[0] << "," << pt[1] << "," << pt[2] << ") "	
		<< "pt-orig=(" << pt_orig[0] << "," << pt_orig[1] << "," << pt_orig[2] << ") "
       		<< "lenr=" << lenr
       		<< " dir=(" << raddir[0] << "," << raddir[1] << "," << raddir[2] << ")"
      		<< std::endl;
      
      veto_pts_v.push_back( pt );
      hitidx_v.push_back(ihit);
      
    }

    LARCV_DEBUG() << "look for veto point clusters within " << veto_pts_v.size()
		  << " unclaimed hits (" << close_hits_v.size() << ")" << std::endl;
    
    int _maxkd = 100;
    int minsize = 5;
    float maxdist = 1.0;
    std::vector< larflow::reco::cluster_t > veto_clusters_v;
    larflow::reco::cluster_sdbscan_spacepoints( veto_pts_v, veto_clusters_v,
						maxdist, minsize, _maxkd ); // external implementation, seems best

    //larflow::reco::cluster_runpca( veto_clusters_v );

    LARCV_DEBUG() << "number of clusters returned: " << veto_clusters_v.size() << std::endl;
    
    for (int icluster=0; icluster<(int)veto_clusters_v.size(); icluster++) {
      auto& cluster = veto_clusters_v[icluster];

      // we have to unpush the points and calc the pca to evaluate
      for (size_t ichit=0; ichit<cluster.hitidx_v.size(); ichit++) {
	int orig_ihit = hitidx_v[ cluster.hitidx_v[ichit] ];
	int orig_idx  = close_hits_v[orig_ihit];
	auto const& hit = inputhits[orig_idx];
	for (int i=0; i<3; i++)
	  cluster.points_v[ ichit ][i] = hit[i];
      }

      larflow::reco::cluster_pca( cluster );

      LARCV_DEBUG() << "  veto cluster: "
		    << " nhits=" << cluster.points_v.size()
		    << " length=" << cluster.pca_len
		    << " max-radius=" << cluster.pca_max_r
		    << std::endl;      
      
      if ( cluster.pca_len>1.0 && cluster.pca_max_r<2.0 ) {
	for (size_t ichit=0; ichit<cluster.hitidx_v.size(); ichit++) {
	  int orig_idx = hitidx_v[ cluster.hitidx_v[ichit] ];
	  cluster.hitidx_v[ichit] = orig_idx;
	}
	//output_cluster_v.emplace_back( std::move(cluster) );

	// make track and cluster
	larlite::track track;
	track.reserve(2); // start and end pca
	larlite::larflowcluster trackcluster;
	trackcluster.reserve( cluster.hitidx_v.size() );

	int close_end = larflow::reco::cluster_closest_pcaend( cluster, nuvtx.pos );
	TVector3 start;
	TVector3 end;
	TVector3 vdir;
	if ( close_end==0 ) {
	  for (int i=0; i<3; i++) {
	    start[i] = cluster.pca_ends_v[0][i];
	    end[i]   = cluster.pca_ends_v[1][i];
	  }	  
	}
	else {
	  for (int i=0; i<3; i++) {
	    start[i] = cluster.pca_ends_v[1][i];
	    end[i]   = cluster.pca_ends_v[0][i];
	  }	  
	}
	vdir = start-end;
	float len = vdir.Mag();
	for (int i=0; i<3; i++)
	  vdir[i] /= len;
	track.add_vertex( start );
	track.add_vertex( end );
	track.add_direction( vdir );
	track.add_direction( vdir );

	for (size_t ichit=0; ichit<cluster.hitidx_v.size(); ichit++) {
	  int orig_ihit = cluster.hitidx_v[ichit];
	  int orig_idx  = close_hits_v[orig_ihit];
	  trackcluster.push_back( inputhits.at( orig_idx ) );
	}
	
	nuvtx.track_v.emplace_back( std::move(track) );
	nuvtx.track_hitcluster_v.emplace_back( std::move(trackcluster) );
	
      }//if cluster passes quality selection
    }
  }
  
  
}
}
