#include "TrackdQdx.h"

#include "TMatrixD.h"
#include "larlite/LArUtil/Geometry.h"
#include "ublarcvapp/UBImageMod/TrackImageMask.h"

#include "geofuncs.h"

namespace larflow {
namespace reco {

  /**
   * @brief calculate dqdx along the 3d track using space points and charge on the plane
   *
   * @param[in] lltrack We get the track path here
   * @param[in] lfcluster Spacepoints associated to the track
   * @param[in] adc_v Wireplane images to get the charge from
   * @return larlite::track with points corresponding to locations along original track that space points projected onto
   */
  larlite::track TrackdQdx::calculatedQdx( const larlite::track& lltrack,
                                           const larlite::larflowcluster& lfcluster,
                                           const std::vector<const larcv::Image2D*>& padc_v ) const
  {

    auto const geom = larlite::larutil::Geometry::GetME();
    // check the planes are right
    int cluster_tpcid  = lfcluster[0].targetwire[4];
    int cluster_cryoid = lfcluster[0].targetwire[5];
    bool allfound = true;
    for (int iplane=0; iplane<(int)geom->Nplanes(cluster_tpcid,cluster_cryoid); iplane++) {
      int simpleindex = geom->GetSimplePlaneIndexFromCTP( cluster_cryoid, cluster_tpcid, iplane );
      bool found = false;
      for ( auto const& pimg : padc_v ) {
	if ( pimg->meta().id()==simpleindex ) {
	  found = true;
	  break;
	}
      }
      if (!found)
	allfound = false;
    }
    if ( !allfound ) {
      LARCV_CRITICAL() << "TPC images and cluster TPC do not match" << std::endl;
      throw std::runtime_error("TPC images and cluster TPC do not match");
    }
    
    // store the distance from the track line for each cluster hit
    std::vector< float > hit_rad_v( lfcluster.size(), -1.0 );

    // the track line not stored in the larlite form
    std::vector< std::vector<float> > detpath;

    // transfer the point locations from the larlite track object
    for ( int istep=0; istep<(int)lltrack.NumberTrajectoryPoints(); istep++ )  {
      TVector3 pt = lltrack.LocationAtPoint(istep);
      std::vector<float> fpt = { (float)pt[0], (float)pt[1], (float)pt[2] };
      detpath.push_back( fpt );
    }
        
    // collect hit position and img coord for each cluster spacepoint
    std::vector<int> search_index_v;
    std::vector< std::vector<float> > point_v;
    std::vector< std::vector<int> > imgcoord_v;
    search_index_v.reserve( lfcluster.size() );
    point_v.reserve( lfcluster.size() );
    imgcoord_v.reserve( lfcluster.size() );
    
    for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
      auto const& hit = lfcluster[ihit];
      search_index_v.push_back( ihit );
      std::vector<float> pt = { hit[0], hit[1], hit[2] };
      point_v.push_back( pt );
      std::vector<int> imgcoord = { hit.targetwire[0], hit.targetwire[1], hit.targetwire[2], hit.tick };
      imgcoord_v.push_back( imgcoord );
    }
      
    // Fill TrackPtList for all of the cluster hits
    float current_len = 0.;
    TrackPtList_t trackpt_v;
    trackpt_v.reserve( lfcluster.size() );
    std::vector<int> used_pt_v( lfcluster.size(), 0 );
        
    for ( int istep=0; istep<(int)detpath.size()-1; istep++ ) {
      std::vector<float>& start = detpath[istep];
      std::vector<float>& end   = detpath[istep+1];
      std::vector<float> dir(3,0);
      std::vector<float> truedir(3,0);          
      float len = 0.;
      float truelen = 0.;
      for (int dim=0; dim<3; dim++) {
        dir[dim] += end[dim]-start[dim];
        len += dir[dim]*dir[dim];
        
        truedir[dim] = end[dim]-start[dim];
        truelen += truedir[dim]*truedir[dim];
      }
      len = sqrt(len);
      truelen = sqrt(truelen);
      
      if (len<=0.1 ) {
        current_len += len;
        continue;
      }
      
      for (int i=0; i<3; i++ ) {
        dir[i] /= len;
        truedir[i] /= truelen;
      }
      
      for (int ii=0; ii<(int)point_v.size(); ii++) {

        if ( used_pt_v[ii]==1 )
          continue;
        
        auto const& pt = point_v[ii];
        auto const& imgcoord = imgcoord_v[ii];
        int hitidx = search_index_v[ii];
        float r = larflow::reco::pointLineDistance3f( start, end, pt );
        float s = larflow::reco::pointRayProjection3f( start, dir, pt );
        //std::cout << "  point: r=" << r << " s=" << s << std::endl;
          
        if ( r>5.0 || s<0 || s>len ) {
          continue;
        }

        float current_s = s+current_len;
        float lm = lfcluster.at(hitidx).track_score;

        TrackPt_t trkpt;
        _makeTrackPtInfo( start, end, pt, imgcoord, padc_v,
                          hitidx,
                          r, s, current_s, lm,
			  cluster_tpcid, cluster_cryoid,
			  trkpt );        
                
        if ( hit_rad_v[trkpt.hitidx]<0 || trkpt.r<hit_rad_v[trkpt.hitidx] )
          hit_rad_v[trkpt.hitidx] = trkpt.r;
        
        trackpt_v.push_back( trkpt );
	
        used_pt_v[ii] = 1; // mark point as used
        
      }//end of point loop
      
      current_len += len;
    }//end of loop over detpath steps

    // it is possible that there are points that do not match to any line segment.
    // example are points near where two line-segments meet. here if a large change in direction means
    // that points on one side of the line do not project into either.
    // we handle these points by assigning the point to the closest line segment end
    for ( size_t ii=0; ii<used_pt_v.size(); ii++  ) {
      std::vector<float> closest_pt(3,0);
      std::vector<float> closest_dir(3,0);
      for ( int istep=0; istep<(int)detpath.size()-1; istep++ ) {
      }
    }
      
    LARCV_INFO() << "Number of hits assigned to track: " << trackpt_v.size() << std::endl;
    LARCV_INFO() << "Total length of track: " << current_len << " cm" << std::endl;
    std::sort( trackpt_v.begin(), trackpt_v.end() );

    // make a new larlite track with dq/dx values stored
    larlite::track dqdx_track;
    dqdx_track.reserve( trackpt_v.size() );
    int npts = trackpt_v.size();

    // we loop in reverse because the trackpt_v is sorted from end of track to vertex.
    for (int ipt=npts-1; ipt>=0; ipt--) {
      
      auto const& trkpt = trackpt_v[ipt];      
      // fill vertex, direction, dqdx, line to spacepoint vector
      // vertex
      TVector3 vtx( trkpt.linept[0],  trkpt.linept[1],  trkpt.linept[2] );
      // direction
      TVector3 dir( trkpt.dir[0], trkpt.dir[1], trkpt.dir[2] );
      // dqdx: (u,v,y,median of {u,v,y})
      std::vector<double> dqdx_v = { trkpt.dqdx_v[0], trkpt.dqdx_v[1], trkpt.dqdx_v[2], trkpt.dqdx_med };
      // linept to spacepoint: abuse of class, we store it in the covariance data member
      TMatrixD m(3,3);
      for  (int i=0; i<3; i++) {
        m(i,i) = trkpt.err_v[i];
      }
      dqdx_track.add_vertex( vtx );
      dqdx_track.add_direction( dir );
      dqdx_track.add_dqdx( dqdx_v );
      dqdx_track.add_covariance( m );
    }

    return dqdx_track;       
    
  }

  larlite::track TrackdQdx::calculatedQdx( const larlite::track& lltrack,
                                           const larlite::larflowcluster& lfcluster,
                                           const std::vector<larcv::Image2D>& adc_v ) const
  {
    std::vector< const larcv::Image2D* > padc_v;
    for ( auto& img : adc_v ) {
      padc_v.push_back( &img );
    }
    return calculatedQdx( lltrack, lfcluster, padc_v );
  }  

  /**
   * @brief fill out data in TrackPt_t struct using track line segment and a spacepoint
   *
   * @param[in] start Starting end point of line segment
   * @param[in] end   End end point of line segment
   * @param[in] pt    Space point 3D coordinate, not t0-corrected
   * @param[in] imgcoord  (u,v,y,tick) coordinates
   * @param[in] adc_v  Vector of wire plane images for all three planes
   * @param[in] hitidx Index of space point in source cluster container
   * @param[in] r     distance of space point to line segment
   * @param[in] s     distance of projected point of space point onto track line-segment, from start of entire track
   * @param[in] lm_score larmatch score
   * @param[out] trkpt Instance of TrackPt_t whose value we will fill
   */
  void TrackdQdx::_makeTrackPtInfo( const std::vector<float>& start,
                                    const std::vector<float>& end,
                                    const std::vector<float>& pt,
                                    const std::vector<int>& imgcoord,
                                    const std::vector<const larcv::Image2D*>& padc_v,
                                    const int hitidx, 
                                    const float r,
                                    const float local_s,
                                    const float global_s,
                                    const float lm_score,
				    const int tpcid,
				    const int cryoid,
                                    TrackdQdx::TrackPt_t& trkpt ) const
  {

    // direction component of the planes in the Y- and Z- direction
    auto const geom = larlite::larutil::Geometry::GetME();
    
    // we're going to use this struct to sort the q on the wire planes and pick the median value
    struct PtQ_t {
      float q;
      float dqdx;
      bool operator<( const PtQ_t& rhs ) const
      {
        if ( q<rhs.q ) return true;
        return false;
      };
    };    
    
    // get direction of between points
    std::vector<float> dir(3,0);
    float len = 0.;
    for (int dim=0; dim<3; dim++) {
      dir[dim] += end[dim]-start[dim];
      len += dir[dim]*dir[dim];
    }
    if ( len>0 ) {
      len = sqrt(len);
      for (int dim=0; dim<3; dim++)
        dir[dim] /= len;
    }
    else {
      throw std::runtime_error("[TrackdQdx::_makeTrackPtInfo] error. length of line segment is zero");
    }

    std::vector<float> linept(3,0); // point on the line segment
    std::vector<float> rad_v(3,0); // vector from line pt to space point
    for (int i=0; i<3; i++) {
      linept[i] = start[i] + local_s*dir[i];
      rad_v[i] = pt[i]-linept[i];
    }
    
    // on segment
    trkpt.linept = linept;
    trkpt.pt  = pt;
    trkpt.dir = dir;
    trkpt.err_v = rad_v;
    trkpt.hitidx = hitidx;
    trkpt.pid = 0;
    trkpt.r = r;
    //trkpt.s = s+current_len;
    trkpt.s = global_s;
    trkpt.q = 0.;            
    trkpt.dqdx = 0.;
    trkpt.q_med = 0.;
    trkpt.dqdx_med = 0.;
    trkpt.lm = lm_score;
    trkpt.dqdx_v.resize(3,0.); // (u,v,y)

    // get the median charge inside the image
    int row = padc_v.front()->meta().row( imgcoord[3] );

    std::vector< PtQ_t > pixq_v(3);

    for ( int p=0; p<(int)padc_v.size(); p++) {

      const TVector3& wirepitchdir = geom->GetPlane( p, tpcid, cryoid ).fWirePitchDir;
      const TVector3& plane_norm   = geom->GetPlane( p, tpcid, cryoid ).fNormToCathode;
      
      float pixsum = 0.;
      int npix = 0;

      if ( imgcoord[p]>=0 && imgcoord[p]<(int)padc_v[p]->meta().cols() ) {
	for (int dr=-2; dr<=2; dr++ ) {
	  int r = row+dr;
	  if ( r<0 || r>=(int)padc_v.front()->meta().rows() )
	    continue;
	  pixsum += padc_v[p]->pixel( r, imgcoord[p], __FILE__, __LINE__ );
	  npix++;
	}
	if ( npix>0 )
	  pixq_v[p].q = pixsum/float(npix);
	else
	  pixq_v[p].q = 0;
      }
      else {
	pixq_v[p].q = 0.;
      }
      
      float dcos_yz = fabs(dir[1]*wirepitchdir[1] + dir[2]*wirepitchdir[2]);
      float dcos_x  = fabs(dir[0]*plane_norm[0]);
      float dx = 3.0;
      if ( dcos_yz>0.785 )
        dx = 3.0/dcos_yz;
      else
        dx = 3.0/dcos_x;
      pixq_v[p].dqdx = pixsum/dx;
      trkpt.dqdx_v[p] = pixsum/dx;
    }
    // y-plane only
    trkpt.q = pixq_v[2].q;
    trkpt.dqdx = pixq_v[2].dqdx;
    
    // median value
    std::sort( pixq_v.begin(), pixq_v.end() );
    trkpt.q_med    = pixq_v[1].q;
    trkpt.dqdx_med = pixq_v[1].dqdx;

  }

  /**
   * @brief Calculate dq/dx, emphasizing use of data in image
   *
   */
  std::vector< std::vector<float> > TrackdQdx::calculatedQdx2D( const larlite::track& lltrack,
                                                                const std::vector<const larcv::Image2D*>& adc_v,
                                                                const float stepsize ) const
  {

    // we define the points on the line, filling a TrackPtList_t
    TrackPtList_t trkpt_v;
    int npts = (int)lltrack.NumberTrajectoryPoints();

    ublarcvapp::ubimagemod::TrackImageMask  masker;
    masker.set_verbosity( larcv::msg::kDEBUG );

    std::vector< std::vector<float> > plane_dqdx_vv(adc_v.size()*2);
    
    for (int p=0; p<(int)adc_v.size(); p++) {
      auto const& img = *adc_v[p];
      auto const& meta = img.meta();

      // label pixels by distance along track
      larcv::Image2D smin_img(meta);
      larcv::Image2D smax_img(meta);
      int n_core_pixels = masker.labelTrackPath( lltrack, img, smin_img, smax_img, 10.0, 0.05 );
      LARCV_DEBUG() << " plane[" << p << "] num core pixels = " << n_core_pixels << std::endl;

      if ( n_core_pixels<=1 )
        continue;
      
      // split track pixels into regions we'll calculate dq/dx on
      std::vector< std::vector<int> > trackpixel_bounds_v;
      trackpixel_bounds_v.reserve( n_core_pixels );
      plane_dqdx_vv[3+p].reserve( n_core_pixels );
      plane_dqdx_vv[p].reserve( n_core_pixels );

      float current_length = 0;
      float current_s_min = 0;
      float current_s = 0.5*stepsize;
      float current_s_max = stepsize;
      float max_s_seen = 0;
      std::vector<int> current_bounds(2,0);
      for ( size_t idx=0; idx<masker.pixel_v.size(); idx++ ) {
        auto& pix = masker.pixel_v[idx];
        std::pair<int,int> pixcoord( pix[0], pix[1] );
        auto it = masker.pixel_map.find( pixcoord );
        if ( it==masker.pixel_map.end() )
          continue;
        auto& pixdata = it->second;
        std::cout << "test " << pixdata.smax << " > " << current_s_max << std::endl;
        if ( pixdata.smax>=current_s_max ) {
          // hit end of bounds
          current_bounds[1] = (int)idx;
          // store it
          trackpixel_bounds_v.push_back( current_bounds );
          plane_dqdx_vv[3+p].push_back( 0.5*(current_s_min+current_s_max) );
          // update next bounds
          current_bounds[0] = (int)idx;
          current_s_min += stepsize;
          current_s_max += stepsize;
        }
        if ( pixdata.smax>max_s_seen )
          max_s_seen = pixdata.smax;
      }
      // make last one
      current_bounds[1] = (int)masker.pixel_v.size()-1;
      trackpixel_bounds_v.push_back( current_bounds );
      plane_dqdx_vv[3+p].push_back( 0.5*(current_s_min+max_s_seen) );      
      
      LARCV_DEBUG() << " plane[" << p << "] number of segments defined: " << trackpixel_bounds_v.size() << std::endl;
      plane_dqdx_vv[p].resize( trackpixel_bounds_v.size(), 0 );

      for ( int iseg=0; iseg<(int)trackpixel_bounds_v.size(); iseg++) {
        auto& bounds = trackpixel_bounds_v[iseg];
        std::vector< std::vector<int> > pix2sum_v(abs(bounds[1]-bounds[0])+1);
        for (int i=0; i<abs(bounds[1]-bounds[0])+1; i++)
          pix2sum_v[i] = masker.pixel_v[bounds[0]+i];
        masker.set_verbosity(larcv::msg::kDEBUG);
        float pixsum = masker.sumOverPixelList( pix2sum_v, img, 1, 3, 0.01 );
        float s_min, s_max, s_dummy;
        bool ok_min = masker.getExtremaPixelValues( pix2sum_v, smin_img, 1, 3, 0.01, s_min, s_dummy );
        bool ok_max = masker.getExtremaPixelValues( pix2sum_v, smax_img, 1, 3, 0.01, s_dummy, s_max );
        float seg_dqdx = 0.;
        if (  ok_min && ok_max && fabs(s_min-s_max)>0.01 ) {
          seg_dqdx = pixsum/fabs(s_max-s_min);
          std::cout << "bounds=" << bounds[0] << "," << bounds[1]
                    << " pixsum=" << pixsum
                    << " s_min=" << s_min << " s_max=" << s_max << " dx="
                    << fabs(s_max-s_min) << std::endl;
          plane_dqdx_vv[3+p][iseg] = 0.5*(s_min+s_max);
        }
        plane_dqdx_vv[p][iseg] = seg_dqdx;
        
      }
      
    }//end of plane loop
      
      
    // makePixelList( lltrack    
    
    return plane_dqdx_vv;
  }
  
                                   
  

}
}
