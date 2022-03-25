#include "PostNuCheckShowerTrunkOverlap.h"

#include "geofuncs.h"

namespace larflow {
namespace reco {

  /**
   * @brief remove tracks that overlap with showers
   *
   * The overlap occurs when small clusters of spacepoints are mislabeled as track.
   * Spurious tracks are characterized by a high-amount of zig-zagging.
   * Spurious tracks are also contained almost entirely in the shower cone.
   * This is how we will will try to identify overlapping tracks.
   *
   * 
   */
  void PostNuCheckShowerTrunkOverlap::process( std::vector<larflow::reco::NuVertexCandidate>& nu_v )
  {

    for (size_t inu=0; inu<nu_v.size(); inu++) {
      auto& nucand = nu_v[inu];

      size_t ntracks = nucand.track_v.size();
      size_t nshowers = nucand.shower_v.size();

      // no need to check for track/shower overlaps
      // if we have zero of either
      if ( ntracks==0 || nshowers==0 )
	continue;

      std::vector<int> remove_track(ntracks,0);
      
      for (size_t itrk=0; itrk<ntracks; itrk++) {
	// these are larlite::track objects and represent tracks as a sequence of line-segments
	auto& track = nucand.track_v.at(itrk);

	for (size_t ishr=0; ishr<nshowers; ishr++) {
	  // the shower representation we use is the trunk.
	  // this is also a larlite::track object that represents
	  // the shower trunk direction as a single line segment
	  auto const& showertrunk = nucand.shower_trunk_v.at(ishr);

	  // we also look at the 1st principle component of the shower
	  // spacepoints. this is a larlite::pcaxis object, which stores
	  // the first 2 principle component axes
	  auto const& showerpca = nucand.shower_pcaxis_v.at(ishr);

	  float shr_dist_to_vertex = 0.;
	  for (int v=0; v<3; v++) {
	    shr_dist_to_vertex += ( showertrunk.LocationAtPoint(0)[v] - nucand.pos[v] )*( showertrunk.LocationAtPoint(0)[v] - nucand.pos[v] );
	  }
	  shr_dist_to_vertex = sqrt( shr_dist_to_vertex );

	  // only consider showers close to the vertex
	  if ( shr_dist_to_vertex>10.0 )
	    continue;

	  // we need to get the start point and dir for the
	  // two shower representations as vector<float> objects
	  std::vector<float> vshowertrunk_start(3,0);
	  std::vector<float> vshowertrunk_end(3,0);	  
	  std::vector<float> vshowertrunk_dir(3,0);
	  std::vector<float> vshowerpca_start(3,0);
	  std::vector<float> vshowerpca_start1(3,0);
	  std::vector<float> vshowerpca_start2(3,0);	  	  
	  std::vector<float> vshowerpca_end(3,0);	  
	  std::vector<float> vshowerpca_dir(3,0);

	  float startdist[2] = { 0,0 };
	  float pcalen = 0.;
	  for (int v=0; v<3; v++) {
	    vshowertrunk_start[v] = showertrunk.LocationAtPoint(0)[v];
	    vshowertrunk_dir[v]   = showertrunk.DirectionAtPoint(0)[v]/showertrunk.DirectionAtPoint(0).Mag();

	    vshowerpca_start1[v]   = showerpca.getEigenVectors()[3][v];
	    vshowerpca_start2[v]   = showerpca.getEigenVectors()[4][v];
	    float dx1 = vshowerpca_start1[v]-nucand.pos[v];
	    float dx2 = vshowerpca_start2[v]-nucand.pos[v];
	    startdist[0] += dx1*dx1;
	    startdist[1] += dx2*dx2;
	    
	    vshowertrunk_end[v]   = vshowertrunk_start[v] + vshowertrunk_dir[v];
	    pcalen += (vshowerpca_start1[v]-vshowerpca_start2[v])*(vshowerpca_start1[v]-vshowerpca_start2[v]);
	  }
	  pcalen = sqrt(pcalen);
	  if ( startdist[0]<startdist[1] ) {
	    for (int v=0; v<3; v++) {
	      vshowerpca_start[v] = vshowerpca_start1[v];
	      vshowerpca_end[v]   = vshowerpca_start2[v];	      
	      vshowerpca_dir[v] = (vshowerpca_start2[v]-vshowerpca_start1[v])/pcalen;
	    }
	  }
	  else {
	    for (int v=0; v<3; v++) {
	      vshowerpca_start[v] = vshowerpca_start2[v];
	      vshowerpca_end[v]   = vshowerpca_start1[v]; 	      
	      vshowerpca_dir[v] = (vshowerpca_start1[v]-vshowerpca_start2[v])/pcalen;
	    }	    
	  }

	  // now for each point along the track line segment, we calculate the radii from
	  // the two lines. We use these radaii as one set of decision variables
	  int npts = track.NumberTrajectoryPoints();	  
	  std::vector< float > s_pca_v(npts,0);
	  std::vector< float > r_pca_v(npts,0);
	  std::vector< float > s_trunk_v(npts,0);
	  std::vector< float > r_trunk_v(npts,0);
	  std::vector< float > d_v(npts,0);
	  std::vector< float> pt_cos_v;
	  pt_cos_v.reserve(npts);

	  for (int ipt=0; ipt<(int)npts; ipt++) {

	    if (ipt>0) {
	      float pt_cos = 0;
	      float mag = track.DirectionAtPoint(ipt).Mag();
	      float last_mag = track.DirectionAtPoint(ipt-1).Mag();
	      
	      if ( mag>0 && last_mag>0 ) {		
		for (int v=0; v<3; v++)
		  pt_cos += track.DirectionAtPoint(ipt-1)[v]*track.DirectionAtPoint(ipt)[v]/(mag*last_mag);
	      }
	      pt_cos_v.push_back(pt_cos);
	    }
	    
	    std::vector<float> pt(3,0);
	    for (int v=0; v<3; v++) {
	      pt[v] = track.LocationAtPoint(ipt)[v];
	      d_v[ipt] += (pt[v]-nucand.pos[v])*(pt[v]-nucand.pos[v]);
	    }
	    float s_trunk = larflow::reco::pointRayProjection3f( vshowertrunk_start, vshowertrunk_dir, pt );
	    float r_trunk = larflow::reco::pointLineDistance3f(  vshowertrunk_start, vshowertrunk_end, pt );
	    float s_pca   = larflow::reco::pointRayProjection3f( vshowerpca_start, vshowerpca_dir, pt );
	    float r_pca   = larflow::reco::pointLineDistance3f(  vshowerpca_start, vshowerpca_end, pt );
	    s_trunk_v[ipt] = s_trunk;
	    r_trunk_v[ipt] = r_trunk;
	    s_pca_v[ipt] = s_pca;
	    r_pca_v[ipt] = r_pca;
	    d_v[ipt] = sqrt(d_v[ipt]);
	  }//end of track point loop


	  // decision variables
	  float frac_inside_cone_pca = 0;
	  float frac_inside_cone_trunk = 0;
	  for (int ipt=0; ipt<(int)npts; ipt++) {
	    // if point outside the end of the cone, it's not inside
	    if ( s_pca_v[ipt]>pcalen )
	      continue;

	    if ( s_pca_v[ipt]<0.0 ) {
	      if ( d_v[ipt]<3.0 )
		frac_inside_cone_pca += 1.0;
	    }
	    else if ( s_pca_v[ipt]>=0.0 && s_pca_v[ipt]<3.0 ) {
	      if ( r_pca_v[ipt]<2.0 ) {
		frac_inside_cone_pca += 1.0;
	      }
	    }
	    else {
	      float r_pca_cone   = 0.577*s_pca_v[ipt];
	      //LARCV_DEBUG() << "    ipt[" << ipt << "] s=" << s_pca_v[ipt] << " r=" << r_pca_v[ipt] << "r_cone=" << r_pca_cone << std::endl;
	      float r_trunk_cone = 0.577*s_trunk_v[ipt];
	      if ( r_pca_v[ipt]<r_pca_cone )
		frac_inside_cone_pca += 1.0;
	      if ( r_trunk_v[ipt]<r_trunk_cone )
		frac_inside_cone_trunk += 1.0;
	    }
	  }
	  frac_inside_cone_pca /= (float)npts;
	  frac_inside_cone_trunk /= (float)npts;
	  
	  float pt_cos_mean = 0;
	  float pt_cos_var  = 0;
	  int pt_n = 0;
	  for (auto const& pt_cos : pt_cos_v ) {
	    pt_n++;
	    pt_cos_mean += pt_cos;
	    pt_cos_var += pt_cos*pt_cos;
	  }
	  if ( pt_n>0 ) {
	    pt_cos_mean /= (float)pt_n;
	    pt_cos_var = pt_cos_var/(float)pt_n - pt_cos_mean*pt_cos_mean;
	  }

	  LARCV_DEBUG() << "-- decision metrics: nu[" << inu << "]-track[" << itrk << "]-shower[" << ishr << "] ---------" << std::endl;
	  LARCV_DEBUG() << "  vertex: (" << nucand.pos[0] << "," << nucand.pos[1] << "," << nucand.pos[2] << ")" << std::endl;
	  LARCV_DEBUG() << "  shower distance to vertex: " << shr_dist_to_vertex << " cm" << std::endl;
	  LARCV_DEBUG() << "  shower vertex: "
			<< "(" << vshowertrunk_start[0] << "," << vshowertrunk_start[1] << "," << vshowertrunk_start[2] << ")" << std::endl;
	  LARCV_DEBUG() << "  shower dir: "
			<< "(" << vshowertrunk_dir[0] << "," << vshowertrunk_dir[1] << "," << vshowertrunk_dir[2] << ")" << std::endl;
	  LARCV_DEBUG() << "  frac inside trunk cone: " << frac_inside_cone_trunk << std::endl;
	  LARCV_DEBUG() << "  frac inside pca cone:   " << frac_inside_cone_pca << std::endl;
	  LARCV_DEBUG() << "  pt cosine mean:   " << pt_cos_mean << std::endl;
	  LARCV_DEBUG() << "  pt cosine variance:   " << pt_cos_var << std::endl;

	  if ( frac_inside_cone_pca>0.8 )
	    remove_track[itrk] = 1;
	}//end of shower loop
      }//end of track loop

      std::vector<larlite::track>  track_v;     ///< track candidates
      std::vector<larlite::larflowcluster>  track_hitcluster_v;  ///< track candidates
      std::vector<float>           track_len_v;       ///< length of track
      std::vector< std::vector<float> > track_dir_v;  ///< direction of track, using points near vertex

      int numremove = 0;
      for (int itrk=0; itrk<(int)ntracks; itrk++) {
     	if ( remove_track[itrk]==1 ) {
	  numremove++;
	}
      }
      LARCV_INFO() << "nu[" << inu << "] num tracks removed: " << numremove << std::endl;
      if ( numremove>0 ) {
	LARCV_INFO() << "nu[" << inu << "] before ntracks=" << nucand.track_v.size() << " nshowers=" << nucand.shower_v.size() << std::endl;
	
	for (int itrk=0; itrk<(int)ntracks; itrk++) {
	  if ( remove_track[itrk]==0 ) {
	    track_v.emplace_back( std::move(nucand.track_v[itrk]) );
	    track_hitcluster_v.emplace_back( std::move(nucand.track_hitcluster_v[itrk]) );
	    if ( nucand.track_len_v.size()>itrk )
	    track_len_v.push_back( nucand.track_len_v[itrk] );
	    if ( nucand.track_dir_v.size()>itrk )
	      track_dir_v.emplace_back( std::move(nucand.track_dir_v[itrk]) );
	  }
	}
	std::swap( nucand.track_v, track_v );
	std::swap( nucand.track_hitcluster_v, track_hitcluster_v );
	if ( nucand.track_len_v.size()>0 )
	  std::swap( nucand.track_len_v, track_len_v );
	if ( nucand.track_dir_v.size()>0 )
	  std::swap( nucand.track_dir_v, track_dir_v );
	LARCV_INFO() << "nu[" << inu << "] after tracks=" << nucand.track_v.size() << " showers=" << nucand.shower_v.size() << std::endl;
      }
      
    }//end of nu-vertex candidate
    
  }
    
    
}
}
