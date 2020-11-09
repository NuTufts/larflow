#include "TrackdQdx.h"

#include "LArUtil/Geometry.h"

#include "geofuncs.h"

namespace larflow {
namespace reco {

  /**
   * @brief calculate dqdx along the 3d track using space points and charge on the plane
   *
   * @param[inout] track We get the track path here
   * @param[in] trackhits Spacepoints associated to the track
   * @param[in] adc_v Wireplane images to get the charge from
   * @return larlite::track with points corresponding to locations along original track that space points projected onto
   */
  larlite::track TrackdQdx::calculatedQdx( const larlite::track& lltrack,
                                           const larlite::larflowcluster& lfcluster,
                                           const std::vector<larcv::Image2D>& adc_v ) const
  {

    struct TrackPt_t {
      int hitidx;
      int pid;
      float s;
      float res;
      float r;
      float q;
      float dqdx;
      float q_med;
      float dqdx_med;
      float lm;
      float ll;
      float llw;
      std::vector<float> pt;
      std::vector<float> dir;
      std::vector<double> dqdx_v;
      bool operator<( const TrackPt_t& rhs ) const
      {
        if ( s>rhs.s) return true;
        return false;
      };
    };
    
    struct PtQ_t {
      float q;
      float dqdx;
      bool operator<( const PtQ_t& rhs ) const
      {
        if ( q<rhs.q ) return true;
        return false;
      };
    };

    typedef std::vector<TrackPt_t> TrackPtList_t;
    
    const std::vector<Double_t> orthy = larutil::Geometry::GetME()->GetOrthVectorsY();
    const std::vector<Double_t> orthz = larutil::Geometry::GetME()->GetOrthVectorsZ();

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
        auto const& pt = point_v[ii];
        auto const& imgcoord = imgcoord_v[ii];
        float r = larflow::reco::pointLineDistance3f( start, end, pt );
        float s = larflow::reco::pointRayProjection3f( start, dir, pt );
        //std::cout << "  point: r=" << r << " s=" << s << std::endl;
          
        if ( r>5.0 || s<0 || s>len ) {
          continue;
        }
        
        // on segment
        TrackPt_t trkpt;
        trkpt.pt  = pt;
        trkpt.dir = dir;
        trkpt.hitidx = search_index_v[ii];
        trkpt.pid = 0;
        trkpt.r = r;
        trkpt.s = s+current_len;
        trkpt.q = 0.;            
        trkpt.dqdx = 0.;
        trkpt.q_med = 0.;
        trkpt.dqdx_med = 0.;
        trkpt.lm = lfcluster.at(trkpt.hitidx).track_score;
        trkpt.dqdx_v.resize(3,0.); // (u,v,y)

        // get the median charge inside the image
        int row = adc_v.front().meta().row( imgcoord[3] );

        std::vector< PtQ_t > pixq_v(3);
        
        for ( int p=0; p<3; p++) {
          
          float pixsum = 0.;
          int npix = 0;
          for (int dr=-2; dr<=2; dr++ ) {
            int r = row+dr;
            if ( r<0 || r>=(int)adc_v.front().meta().rows() )
              continue;
            pixsum += adc_v[p].pixel( r, imgcoord[p] );
            npix++;
          }
          if ( npix>0 )
            pixq_v[p].q = pixsum/float(npix);
          else
            pixq_v[p].q = 0;
          
          float dcos_yz = fabs(truedir[1]*orthy[p] + truedir[2]*orthz[p]);
          float dcos_x  = fabs(truedir[0]);
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
        
        if ( hit_rad_v[trkpt.hitidx]<0 || trkpt.r<hit_rad_v[trkpt.hitidx] )
          hit_rad_v[trkpt.hitidx] = trkpt.r;
        
        trackpt_v.push_back( trkpt );
        
      }//end of point loop
      
      current_len += len;
    }//end of loop over detpath steps
      
    std::cout << "Number of hits assigned to track: " << trackpt_v.size() << std::endl;
    std::cout << "Total length of track: " << current_len << " cm" << std::endl;
    std::sort( trackpt_v.begin(), trackpt_v.end() );

    // make a new larlite track with dq/dx values stored
    larlite::track dqdx_track;
    dqdx_track.reserve( trackpt_v.size() );
    for (auto const& trkpt : trackpt_v ) {
      // fill vertex, direction, dqdx
      // vertex
      TVector3 vtx( trkpt.pt[0],  trkpt.pt[1],  trkpt.pt[2] );
      // direction
      TVector3 dir( trkpt.dir[0], trkpt.dir[1], trkpt.dir[2] );
      // dqdx: (u,v,y,median of {u,v,y})
      std::vector<double> dqdx_v = { trkpt.dqdx_v[0], trkpt.dqdx_v[1], trkpt.dqdx_v[2], trkpt.dqdx_med };
      dqdx_track.add_vertex( vtx );
      dqdx_track.add_direction( dir );
      dqdx_track.add_dqdx( dqdx_v );
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
  

}
}
