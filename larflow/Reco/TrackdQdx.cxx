#include "TrackdQdx.h"

#include "TMatrixD.h"
#include "LArUtil/Geometry.h"

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
                                           const std::vector<larcv::Image2D>& adc_v ) const
  {

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
        _makeTrackPtInfo( start, end, pt, imgcoord, adc_v,
                          hitidx,
                          r, s, current_s, lm, trkpt );        
        
        // std::vector<float> linept(3,0);
        // std::vector<float> rad_v(3,0); // vector from line pt to space point
        // for (int i=0; i<3; i++) {
        //   linept[i] = start[i] + s*dir[i];
        //   rad_v[i] = pt[i]-linept[i];
        // }
        
        // // on segment
        // TrackPt_t trkpt;
        // trkpt.linept = linept;
        // trkpt.pt  = pt;
        // trkpt.dir = dir;
        // trkpt.err_v = rad_v;
        // trkpt.hitidx = 

        // trkpt.pid = 0;
        // trkpt.r = r;
        // trkpt.s = s+current_len;
        // trkpt.q = 0.;            
        // trkpt.dqdx = 0.;
        // trkpt.q_med = 0.;
        // trkpt.dqdx_med = 0.;
        // trkpt.lm = lfcluster.at(trkpt.hitidx).track_score;
        // trkpt.dqdx_v.resize(3,0.); // (u,v,y)

        // // get the median charge inside the image
        // int row = adc_v.front().meta().row( imgcoord[3] );

        // std::vector< PtQ_t > pixq_v(3);
        
        // for ( int p=0; p<3; p++) {
          
        //   float pixsum = 0.;
        //   int npix = 0;
        //   for (int dr=-2; dr<=2; dr++ ) {
        //     int r = row+dr;
        //     if ( r<0 || r>=(int)adc_v.front().meta().rows() )
        //       continue;
        //     pixsum += adc_v[p].pixel( r, imgcoord[p] );
        //     npix++;
        //   }
        //   if ( npix>0 )
        //     pixq_v[p].q = pixsum/float(npix);
        //   else
        //     pixq_v[p].q = 0;
          
        //   float dcos_yz = fabs(truedir[1]*orthy[p] + truedir[2]*orthz[p]);
        //   float dcos_x  = fabs(truedir[0]);
        //   float dx = 3.0;
        //   if ( dcos_yz>0.785 )
        //     dx = 3.0/dcos_yz;
        //   else
        //     dx = 3.0/dcos_x;
        //   pixq_v[p].dqdx = pixsum/dx;
        //   trkpt.dqdx_v[p] = pixsum/dx;
        // }
        // // y-plane only
        // trkpt.q = pixq_v[2].q;
        // trkpt.dqdx = pixq_v[2].dqdx;
        
        // // median value
        // std::sort( pixq_v.begin(), pixq_v.end() );
        // trkpt.q_med    = pixq_v[1].q;
        // trkpt.dqdx_med = pixq_v[1].dqdx;
        
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
    for (auto const& trkpt : trackpt_v ) {
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
    
    // // calculate residual range
    // // calculate likelihood
    // float totw = 0.;
    // float totll = 0.;
    // for ( auto& trkpt : trackpt_v ) {
    //   trkpt.res = current_len - trkpt.s;
      
    //   float mu_dedx = sMuonRange2dEdx->Eval(trkpt.res);
    //   float mu_dedx_birks = q2adc*mu_dedx/(1+mu_dedx*0.0486/0.273/1.38);
    //   float p_dedx = sProtonRange2dEdx->Eval(trkpt.res);
    //   float p_dedx_birks = q2adc*p_dedx/(1+p_dedx*0.0486/0.273/1.38);
      
    //   float dmu = trkpt.dqdx_med-mu_dedx_birks;
    //   float dp  = trkpt.dqdx_med-p_dedx_birks;
      
    //   float llpt = -0.5*dmu*dmu/100.0 + 0.5*dp*dp/100.0;
    //   float w_dedx = (mu_dedx_birks-p_dedx_birks)*(mu_dedx_birks-p_dedx_birks);
    //   trkpt.ll = llpt;
    //   trkpt.llw = w_dedx;
    //   if ( trkpt.dqdx_med>10.0 ) {
    //     totll += llpt*w_dedx;
    //     totw  += w_dedx;
    //   }
    // }
    // if ( totw>0 )
    //   totll /= totw;
    
    // track_len_v[itrack] = current_len;
    // track_ll_v[itrack] = totll;
    
    // trackpt_list_v.emplace_back( std::move(trackpt_v) );    
    
    
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
                                    const std::vector<larcv::Image2D>& adc_v,
                                    const int hitidx, 
                                    const float r,
                                    const float local_s,
                                    const float global_s,
                                    const float lm_score,
                                    TrackdQdx::TrackPt_t& trkpt ) const
  {

    // direction component of the planes in the Y- and Z- direction
    const std::vector<Double_t> orthy = larutil::Geometry::GetME()->GetOrthVectorsY();
    const std::vector<Double_t> orthz = larutil::Geometry::GetME()->GetOrthVectorsZ();
    
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
    int row = adc_v.front().meta().row( imgcoord[3] );

    std::vector< PtQ_t > pixq_v(3);
        
    for ( int p=0; p<3; p++) {      
      float pixsum = 0.;
      int npix = 0;

      if ( imgcoord[p]>=0 && imgcoord[p]<(int)adc_v[p].meta().cols() ) {
	for (int dr=-2; dr<=2; dr++ ) {
	  int r = row+dr;
	  if ( r<0 || r>=(int)adc_v.front().meta().rows() )
	    continue;
	  pixsum += adc_v[p].pixel( r, imgcoord[p], __FILE__, __LINE__ );
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
      
      float dcos_yz = fabs(dir[1]*orthy[p] + dir[2]*orthz[p]);
      float dcos_x  = fabs(dir[0]);
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
                                   
  

}
}
