#include "ClusterImageMask.h"

#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/Geometry.h"

namespace larflow {
namespace reco {

  std::vector< larcv::Image2D >
  ClusterImageMask::makeChargeMask( NuVertexCandidate& nuvtx,
                                    const std::vector<larcv::Image2D>& adc_v )
  {

    std::vector< larcv::Image2D > mask_v;
    for (auto const& adc : adc_v ) {
      larcv::Image2D mask(adc.meta());
      mask.paint(0.0);
      mask_v.emplace_back( std::move(mask) );
    }
    
    _npix = 0;
    
    // loop over tracks
    // for ( auto const& track : nuvtx.track_v ) {
    //   //maskTrack( track, adc_v, mask_v, 10.0, 2, 2, 0.5, 1.0 );
    // }
    for ( auto const& track : nuvtx.track_hitcluster_v ) {
      maskCluster( track, adc_v, mask_v, 10.0, 5 );
    }

    // loop over showers
    for ( auto const& shower : nuvtx.shower_v ) {
      maskCluster( shower, adc_v, mask_v, 10.0, 5 );
    }

    return mask_v;
  }


  void ClusterImageMask::maskCluster( const larlite::larflowcluster& cluster,
                                      const std::vector<larcv::Image2D>& adc_v,
                                      std::vector<larcv::Image2D>& mask_v,
                                      const float thresh,
                                      const int dpix )
  {

    float tick_min = adc_v.front().meta().min_y();
    float tick_max = adc_v.front().meta().max_y();
    int row_min = 0;
    int row_max = (int)adc_v.front().meta().rows();
    
    int nskipped_points = 0;
    for (auto const& sp : cluster ) {

      if ( sp.tick<=tick_min || sp.tick>=tick_max )
        continue;

      int row = adc_v.front().meta().row( sp.tick );
      
      if ( sp.targetwire.size()<adc_v.size() ) {
        nskipped_points++;
        continue;
      }
    
      for ( int dr=-(int)abs(dpix); dr<=(int)abs(dpix); dr++ ) {
        int r = row + dr;
        if (r<row_min || r>=row_max ) continue;

        for (int p=0; p<(int)adc_v.size(); p++) {
          for (int dc=-(int)abs(dpix); dc<=(int)abs(dpix); dc++) {
            int c = sp.targetwire[p] + dc;
            if ( c<0 || c>=(int)adc_v[p].meta().cols() ) continue;
	    float pixvalue = adc_v[p].pixel(r,c,__FILE__,__LINE__);
            if ( pixvalue>thresh
                 && mask_v[p].pixel(r,c)==0 ) {
              _npix++;	      
	      if (!_store_pixel_value)
		mask_v[p].set_pixel(r,c,1.0);
	      else
		mask_v[p].set_pixel(r,c,pixvalue);
            }
          }//end of col loop
        }//end of plane loop
        
      }//end of row loop
    }//end of spacepoint loop
    LARCV_NORMAL() << "_npix labeled=" << _npix << " nskipped=" << nskipped_points << std::endl;
  }

  void ClusterImageMask::maskClusterAndStore( const larlite::larflowcluster& cluster,
					      const std::vector<larcv::Image2D>& adc_v,
					      const float thresh,
					      const int dpix,
					      const bool clear_tracking_image )
  {
    if ( clear_tracking_image ) {
      _npix = 0;
      _cluster_mask_v.clear();
    }

    if ( _cluster_mask_v.size()==0 ) {
      _npix = 0;
      for ( auto const& img : adc_v ) {
	larcv::Image2D newimg( img.meta() );
	_cluster_mask_v.emplace_back( std::move(newimg) );
      }
    }

    if ( _cluster_mask_v.size()!=adc_v.size() ) {
      LARCV_ERROR() <<  "number of mask images and input images does not match" << std::endl;
    }
    
    bool mask_matches = true;
    for ( size_t p=0; p<_cluster_mask_v.size(); p++ ) {
      if ( _cluster_mask_v[p].meta()!=adc_v[p].meta() ) {
	mask_matches = false;
	LARCV_WARNING() << "input and output mask image meta does not match for plane=" << p << std::endl;
	LARCV_WARNING() << "  input: " << adc_v[p].meta().dump() << std::endl;
	LARCV_WARNING() << "  output mask: " << _cluster_mask_v[p].meta().dump() << std::endl;
      }
    }
    if ( !mask_matches ) {
      LARCV_ERROR() << "Metas do not match" << std::endl;
    }

    maskCluster( cluster, adc_v, _cluster_mask_v, thresh, dpix );
    
  }

  float ClusterImageMask::getPlaneMaskSum( int plane )
  {

    float sum = 0;
    if ( plane>=0 && plane<_cluster_mask_v.size() ) {
      auto const& v = _cluster_mask_v[plane].as_vector();
      for (size_t i=0; i<v.size(); i++)
	sum += v[i];
    }
    return sum;    
  }

  
  std::vector<float> ClusterImageMask::getMaskSums()
  {
    std::vector<float> sum_v( _cluster_mask_v.size(), 0.0 );
    for (size_t p=0; p<_cluster_mask_v.size(); p++)
      sum_v[p] = getPlaneMaskSum( p );
    return sum_v;
  }
  

  void ClusterImageMask::maskWithImage( const std::vector<larcv::Image2D>& adc_v,
					std::vector<larcv::Image2D>& mask_v,
					const float thresh,
					const bool invert )    
  {
    bool meta_match = true;
    if ( adc_v.size()!=mask_v.size() ) {
      meta_match = false;
    }
    if ( !meta_match )
      return;
    
    for (size_t p=0; p<adc_v.size(); p++) {
      if ( adc_v[p].meta()!=mask_v[p].meta() )
	meta_match = false;
    }
    if ( !meta_match )
      return;

    for (size_t p=0; p<adc_v.size(); p++) {
      auto const& vimg = adc_v[p].as_vector();
      auto& vmask = mask_v[p].as_mod_vector();

      for (size_t i=0; i<vmask.size(); i++) {
	if ( vmask[i]>=thresh ) {
	  if ( !invert && vimg[i]<thresh )
	    vmask[i] = 0.0;
	  else if ( invert && vimg[i]>=thresh )
	    vmask[i] = 0.0;
	}
      }
    }
  }

  void ClusterImageMask::maskStoredImage( const std::vector<larcv::Image2D>& adc_v,
					  const float thresh,
					  const bool invert )
  {
    maskWithImage( adc_v, _cluster_mask_v, thresh, invert );
  }
  
  void ClusterImageMask::maskTrack( const larlite::track& track,
                                    const std::vector<larcv::Image2D>& adc_v,
                                    std::vector<larcv::Image2D>& mask_v,
                                    const float thresh,                                    
                                    const int dcol,
                                    const int drow,
                                    const float minstepsize,
                                    const float maxstepsize )
  {

    int npts = track.NumberTrajectoryPoints();
    if ( npts<=1 ) {
      LARCV_WARNING() << "No mask generated for track with only 1 point" << std::endl;
      // no mask can be generated in this case
      return;
    }

    const float driftv = larutil::LArProperties::GetME()->DriftVelocity();
    const float usec_per_tick = 0.5;

    float max_tick = adc_v.front().meta().max_y();
    float min_tick = adc_v.front().meta().min_y();

    
    for (int ipt=0; ipt<npts-1; ipt++) {

      TVector3 start = track.LocationAtPoint(ipt);
      TVector3 end   = track.LocationAtPoint(ipt+1);
      TVector3 dir   = end-start;
      
      double segsize = dir.Mag();

      int nsteps = 1;
      if ( segsize>minstepsize ) {
        nsteps = segsize/maxstepsize + 1;
      }
      
      float stepsize = segsize/float(nsteps);

      for (int istep=0; istep<=nsteps; istep++) {
        // get 3d position along track
        TVector3 pos = start + istep*(stepsize/segsize)*dir;
        // project into image
        std::vector<int> imgcoord(4,0); // (u,v,y,tick)

        imgcoord[3] = pos[0]/driftv/usec_per_tick + 3200; // tick

        if ( min_tick>=imgcoord[3] || max_tick<=imgcoord[3] )
          continue;

        int row = adc_v.front().meta().row( imgcoord[3], __FILE__, __LINE__ );
        
        for (int p=0; p<3; p++) {
          imgcoord[p] = larutil::Geometry::GetME()->WireCoordinate( pos, (UInt_t)p );
        }

        //mask around the projected point
        for (int dr=-abs(drow); dr<=abs(drow); dr++) {
          int r = row+dr;
          if ( r<0 || r>=(int)adc_v.front().meta().rows() )
            continue;

          for (int p=0; p<3; p++) {          
            for (int dc=-abs(dcol);dc<=abs(dcol); dc++) {
              int c = imgcoord[p]+dc;
              if (c<=0 || c>=(int)adc_v[p].meta().cols() )
                continue;

              float pixval = adc_v[p].pixel(r,c,__FILE__,__LINE__);
              if ( pixval>thresh && mask_v[p].pixel(r,c)==0) {
                _npix++;
		if ( !_store_pixel_value)
		  mask_v[p].set_pixel( r, c, 1.0 );
		else
		  mask_v[p].set_pixel( r, c, pixval );
              }
            }//end of dc loop
            
          }//end of plane loop

        }//end of dr loop
        
      }//end of segment step loop

    }//end of loop over track segments
    
    LARCV_DEBUG() << "_npix=" << _npix << std::endl;
  }
    


}
}
