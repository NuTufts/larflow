#include "ClusterImageMask.h"

#include "ublarcvapp/RecoTools/DetUtils.h"
#include "larlite/LArUtil/DetectorProperties.h"
#include "larlite/LArUtil/Geometry.h"

namespace larflow {
namespace reco {

  std::vector< larcv::Image2D >
  ClusterImageMask::makeChargeMask( NuVertexCandidate& nuvtx,
                                    const std::vector<larcv::Image2D>& adc_v )
  {

    std::vector<const larcv::Image2D*> padc_v
      = ublarcvapp::recotools::DetUtils::getTPCImages( adc_v, nuvtx.tpcid, nuvtx.cryoid );
    
    std::vector< larcv::Image2D > mask_v;
    for (auto const& adc : padc_v ) {
      larcv::Image2D mask(adc->meta());
      mask.paint(0.0);
      mask_v.emplace_back( std::move(mask) );
    }
    
    _npix = 0;
    
    // loop over tracks
    for ( auto const& track : nuvtx.track_v ) {
      maskTrack( track, padc_v, mask_v, nuvtx.tpcid, nuvtx.cryoid, 10.0, 2, 2, 0.5, 1.0 );
    }

    // loop over showers
    for ( auto const& shower : nuvtx.shower_v ) {
      maskCluster( shower, padc_v, mask_v, 10.0, 2 );
    }

    return mask_v;
  }


  void ClusterImageMask::maskCluster( const larlite::larflowcluster& cluster,
                                      const std::vector<const larcv::Image2D*>& padc_v,
                                      std::vector<larcv::Image2D>& mask_v,
                                      const float thresh,
                                      const int dpix )
  {

    float tick_min = padc_v.front()->meta().min_y();
    float tick_max = padc_v.front()->meta().max_y();
    int row_min = 0;
    int row_max = (int)padc_v.front()->meta().rows();
    
    int nskipped_points = 0;
    for (auto const& sp : cluster ) {

      if ( sp.tick<=tick_min || sp.tick>=tick_max )
        continue;

      int row = padc_v.front()->meta().row( sp.tick );
      
      if ( sp.targetwire.size()!=padc_v.size() ) {
        nskipped_points++;
        continue;
      }
    
      for ( int dr=-(int)abs(dpix); dr<=(int)abs(dpix); dr++ ) {
        int r = row + dr;
        if (r<row_min || r>=row_max ) continue;

        for (int p=0; p<(int)padc_v.size(); p++) {
          for (int dc=-(int)abs(dpix); dc<=(int)abs(dpix); dc++) {
            int c = sp.targetwire[p] + dc;
            if ( c<0 || c>=(int)padc_v[p]->meta().cols() ) continue;

            if ( padc_v[p]->pixel(r,c,__FILE__,__LINE__)>thresh
                 && mask_v[p].pixel(r,c)==0 ) {
              mask_v[p].set_pixel(r,c,1.0);
              _npix++;
            }
          }//end of col loop
        }//end of plane loop
        
      }//end of row loop
    }//end of spacepoint loop
    
  }

  void ClusterImageMask::maskTrack( const larlite::track& track,
                                    const std::vector<const larcv::Image2D*>& padc_v,
                                    std::vector<larcv::Image2D>& mask_v,
				    const int tpcid, const int cryoid,
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

    auto const detp = larutil::DetectorProperties::GetME();
    
    float max_tick = padc_v.front()->meta().max_y();
    float min_tick = padc_v.front()->meta().min_y();

    
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

        imgcoord[3] = detp->ConvertXToTicks( pos[0], 0, tpcid, cryoid );

        if ( min_tick>=imgcoord[3] || max_tick<=imgcoord[3] )
          continue;

        int row = padc_v.front()->meta().row( imgcoord[3], __FILE__, __LINE__ );
        
        for (int p=0; p<3; p++) {
          imgcoord[p] = larlite::larutil::Geometry::GetME()->WireCoordinate( pos, (UInt_t)p, tpcid, cryoid );
        }

        //mask around the projected point
        for (int dr=-abs(drow); dr<=abs(drow); dr++) {
          int r = row+dr;
          if ( r<0 || r>=(int)padc_v.front()->meta().rows() )
            continue;

          for (int p=0; p<3; p++) {          
            for (int dc=-abs(dcol);dc<=abs(dcol); dc++) {
              int c = imgcoord[p]+dc;
              if (c<=0 || c>=(int)padc_v[p]->meta().cols() )
                continue;

              float pixval = padc_v[p]->pixel(r,c,__FILE__,__LINE__);
              if ( pixval>thresh && mask_v[p].pixel(r,c)==0) {
                mask_v[p].set_pixel( r, c, 1.0 );
                _npix++;
              }
            }//end of dc loop
            
          }//end of plane loop

        }//end of dr loop
        
      }//end of segment step loop

    }//end of loop over track segments
    
    LARCV_DEBUG() << "_npix=" << _npix << std::endl;
  }
    
  void ClusterImageMask::maskTrack( const larlite::track& track,
                                    const std::vector<larcv::Image2D>& adc_v,
                                    std::vector<larcv::Image2D>& mask_v,
				    const int tpcid, const int cryoid,				    
                                    const float thresh,                                    
                                    const int dcol,
                                    const int drow,
                                    const float minstepsize,
                                    const float maxstepsize )  
  {
    std::vector< const larcv::Image2D* > padc_v;
    for ( auto const& img : adc_v )
      padc_v.push_back( &img );
    maskTrack( track, padc_v, mask_v, tpcid, cryoid,
	       thresh, dcol, drow, minstepsize, maxstepsize );
  }


}
}
