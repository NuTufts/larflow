#include "CRTTrackMatch.h"

#include "larlite/core/LArUtil/LArProperties.h"
#include "larlite/core/LArUtil/Geometry.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/ROOTUtil/ROOTUtils.h"
#include "ublarcvapp/UBWireTool/UBWireTool.h"

#include "TH2D.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TRandom3.h"

namespace larflow {
namespace crtmatch {

  CRTTrackMatch::CRTTrackMatch()
    : _max_iters(20) {
    _sce = new larutil::SpaceChargeMicroBooNE( larutil::SpaceChargeMicroBooNE::kMCC9_Forward );
    _reverse_sce = new larutil::SpaceChargeMicroBooNE( larutil::SpaceChargeMicroBooNE::kMCC9_Backward );
  }

  CRTTrackMatch::~CRTTrackMatch() {
    delete _sce;
    delete _reverse_sce;
  }
  
  void CRTTrackMatch::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll ) {
    larcv::EventImage2D* ev_adc = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    larlite::event_crttrack* ev_crttrack = (larlite::event_crttrack*)ioll.get_data( larlite::data::kCRTTrack, "crttrack" );

    
    // vizualize for debug
    std::vector<TH2D> h2d_v;
    h2d_v = larcv::rootutils::as_th2d_v( ev_adc->Image2DArray(), "crttrack_adc" );
    std::vector< TGraph* > graph_v[3];
    

    std::vector< crttrack_t > data_v;
    for ( size_t i=0; i<ev_crttrack->size(); i++ ) {

      crttrack_t data = _find_optimal_track( ev_crttrack->at(i), ev_adc->Image2DArray() );

      // crttrack_t data = _collect_chargepixels_for_track( ev_crttrack->at(i), ev_adc->Image2DArray() );
      
      // std::cout << "[CRTTrackMatch::process] track collected "
      //           << "npts=" << data.pixelpos_vv.size() << " "
      //           << "plane pixels=("
      //           << data.pixellist_vv[0].size() << ","
      //           << data.pixellist_vv[1].size() << ","
      //           << data.pixellist_vv[2].size() << ")"
      //           << " length=" << data.len_intpc_sce
      //           << " totq=(" << data.totalq_v[0] << "," << data.totalq_v[1] << "," << data.totalq_v[2] << ")"
      //           << std::endl;
      
      if ( data.pixelpos_vv.size()>0 ) {
        for ( size_t p=0; p<3; p++ ) {
          if ( data.pixelcoord_vv[p].size()==0 ) continue;
          TGraph* g = new TGraph( data.pixelcoord_vv[p].size() );
          for ( size_t n=0; n<data.pixelcoord_vv.size(); n++ ) {
            g->SetPoint( n, data.pixelcoord_vv[n][p], data.pixelcoord_vv[n][3] );
          }          
          g->SetLineWidth(1);
          if ( data.pixelpos_vv.front()[0]<80.0 )
            g->SetLineColor(kMagenta);
          else if ( data.pixelpos_vv.front()[0]>160.0 )
            g->SetLineColor(kCyan);
          else
            g->SetLineColor(kOrange);
          
          graph_v[p].push_back( g );
        }
        //break;
      }
        
    }


    TCanvas c("c","c",1200,600);
    c.Divide(1,3);
    for ( size_t p=0; p<3; p++ ) {
      c.cd(p+1);
      h2d_v[p].Draw("colz");
      std::cout << "graphs in plane[" << p << "]: " << graph_v[p].size() << std::endl;
      for ( auto& g : graph_v[p] ) {
        //std::cout << " " << g.N() << std::endl;
        //g->SetMarkerStyle(20);
        g->Draw("Lsame");
      }
    }
    c.Update();
    
    char name[100];
    sprintf( name, "crttrackmatch_debug_run%d_subrun%d_event%d.png",
             (int)iolcv.event_id().run(),
             (int)iolcv.event_id().subrun(),
             (int)iolcv.event_id().event() );
    c.SaveAs(name);

    
  }

  /**
   * we step around the hit, optimizing charge/
   * 
   */
  CRTTrackMatch::crttrack_t CRTTrackMatch::_find_optimal_track( const larlite::crttrack& crt,
                                                                const std::vector<larcv::Image2D>& adc_v ) {
    
    std::cout << "[CRTTrackMatch::_find_optimal_track]" << std::endl;
    std::cout << "  hit1 pos w/ error: ("
              << " " << crt.x1_pos << " +/- " << crt.x1_err << ","
              << " " << crt.y1_pos << " +/- " << crt.y1_err << ","
              << " " << crt.z1_pos << " +/- " << crt.z1_err << ")"
              << std::endl;
    std::cout << "  hit2 pos w/ error: ("
              << " " << crt.x2_pos << " +/- " << crt.x2_err << ","
              << " " << crt.y2_pos << " +/- " << crt.y2_err << ","
              << " " << crt.z2_pos << " +/- " << crt.z2_err << ")"
              << std::endl;

    // walk the point in a 5 cm range
    std::vector< double > hit1_center = { crt.x1_pos, crt.y1_pos, crt.z1_pos };
    std::vector< double > hit2_center = { crt.x2_pos, crt.y2_pos, crt.z2_pos };
    float t0_usec = 0.5*( crt.ts2_ns_h1+crt.ts2_ns_h2 )*0.001;

    const int ntries   = _max_iters; // max iterations

    TRandom3 rand(123);

    float bestfit_q_per_len = 0.;

    int neighborhood = 100;

    // first try, latest iteration of track
    crttrack_t first = _collect_chargepixels_for_track( hit1_center,
                                                        hit2_center,
                                                        t0_usec,
                                                        adc_v,
                                                        1.0, neighborhood );
    // provide additional info, besides projected points found
    first.pcrttrack = &crt;
    first.t0_usec   = t0_usec;

    float old_q_per_cm = 0;
    for (int j=0; j<3; j++ ) {
      if ( first.totalq_v[j]>old_q_per_cm )
        old_q_per_cm = first.totalq_v[j];
    }
    if ( first.len_intpc_sce>0.0 )
      old_q_per_cm /= first.len_intpc_sce;
    
    
    std::cout << "first try: " << _str( first ) << " q/cm=" << old_q_per_cm << std::endl;

    for (int itry=0; itry<ntries; itry++ ) {

      // neighborhood -= 10;
      // if (neighborhood<1 )
      //   neighborhood = 1;
      
      // next use the found points to fit a new track direction
      std::vector<double> hit1_new;
      std::vector<double> hit2_new;
      bool foundproposal = _propose_corrected_track( first, hit1_new, hit2_new );
      if ( !foundproposal )
        break;

      std::cout << "  proposal moves CRT hits to: "
                << "(" << hit1_new[0] << "," << hit1_new[1] << "," << hit1_new[2] << ") "
                << "(" << hit2_new[0] << "," << hit2_new[1] << "," << hit2_new[2] << ") "
                << std::endl;
      std::cout << "  proposal shifts: "
                << "(" << hit1_new[0]-hit1_center[0]
                << "," << hit1_new[1]-hit1_center[1]
                << "," << hit1_new[2]-hit1_center[2] << ") "
                << "(" << hit2_new[0]-hit2_center[0]
                << "," << hit2_new[1]-hit2_center[1]
                << "," << hit2_new[2]-hit2_center[2] << ") "
                << std::endl;

      // make a new track
      crttrack_t sample = _collect_chargepixels_for_track( hit1_new, hit2_new,
                                                           t0_usec,
                                                           adc_v,
                                                           1.0, neighborhood );
      sample.pcrttrack = &crt;
      sample.t0_usec = t0_usec;

      std::cout << " ... [try " << itry << "] " << _str(sample) << std::endl;

      float err_diff = 0.;
      for (size_t p=0; p<3; p++ ) {
        err_diff += (first.toterr_v[p] - sample.toterr_v[p]);
      }
      
      //if ( new_q_per_cm > old_q_per_cm ) {
      if ( err_diff>0.0 ) {      
        std::cout << " ... error improved by " << err_diff << ", replace old" << std::endl;
        // replace the old
        std::swap( sample, first );
        hit1_center = hit1_new;
        hit2_center = hit2_new;
      }
      else {
        std::cout << " ... no improvement ... stop " << std::endl;
        break;
      }
      
    }//end over try loop
    
    // if ( old_q_per_cm==0 ) {
    //   crttrack_t empty(-1,&crt);
    //   return empty;
    // }
    
    std::cout << " making best fit track" << std::endl;
    crttrack_t bestfit_data = _collect_chargepixels_for_track( hit1_center,
                                                               hit2_center,
                                                               t0_usec,
                                                               adc_v,
                                                               0.1, 10 );
    bestfit_data.pcrttrack = &crt;
    bestfit_data.t0_usec = t0_usec;
    std::cout << " track after fit: " << _str(bestfit_data) << std::endl;
    
    return bestfit_data;
    
  }
  
  CRTTrackMatch::crttrack_t
    CRTTrackMatch::_collect_chargepixels_for_track( const std::vector<double>& hit1_pos,
                                                    const std::vector<double>& hit2_pos,
                                                    const float t0_usec,
                                                    const std::vector<larcv::Image2D>& adc_v,
                                                    const float max_step_size,
                                                    const int col_neighborhood ) {

    crttrack_t data( -1, nullptr );

    std::vector<float> dir(3,0);
    float len = 0.;
    for (int i=0; i<3; i++ ) {
      dir[i] = hit2_pos[i]-hit1_pos[i];
      len += dir[i]*dir[i];
    }
    
    len = sqrt( len );
    if ( len==0 )
      return data;
    for ( size_t p=0; p<3; p++ ) dir[p] /= len;
    
    int nsteps = len/max_step_size+1;
    float stepsize = len/float(nsteps);

    std::vector< larcv::Image2D > pix_visited_v;
    for ( auto const& img : adc_v ) {
      larcv::Image2D visited(img.meta());
      visited.paint(0.0);
      pix_visited_v.emplace_back( std::move(visited) );
    }

    std::vector<double> last_pos;

    for (int istep=0; istep<nsteps; istep++) {

      // step position
      std::vector<double> pos(3,0.0);
      for (int i=0; i<3; i++ )
        pos[i] = hit1_pos[i] + istep*stepsize*dir[i];
      
      if ( pos[0]<1.0 || pos[0]>255.0 ) continue;
      if ( pos[1]<-116.0 || pos[1]>116.0  ) continue;
      if ( pos[2]<0.5 || pos[2]>1035.0 ) continue;

      //std::cout << " [" << istep << "] pos=(" << pos[0] << "," << pos[1] << "," << pos[2] << ")" << std::endl;      
      
      std::vector<double> offset = _sce->GetPosOffsets( pos[0], pos[1], pos[2] );
      std::vector<double> pos_sce(3,0);
      pos_sce[0] = pos[0] - offset[0] + 0.6;
      pos_sce[1] = pos[1] + offset[1];
      pos_sce[2] = pos[2] + offset[2];
      
      // space-charge correction
      bool inimage = true;
      std::vector<int> imgcoord_v(4);
      for (size_t p=0; p<adc_v.size(); p++ ) {
        imgcoord_v[p] = (int)(larutil::Geometry::GetME()->WireCoordinate( pos_sce, (UInt_t)p )+0.5);
        if ( imgcoord_v[p]<0 || imgcoord_v[p]>=larutil::Geometry::GetME()->Nwires(p) ) {
          inimage = false;
        }
      }
      imgcoord_v[3] = 3200 + ( pos_sce[0]/larutil::LArProperties::GetME()->DriftVelocity() + t0_usec )/0.5;
      //imgcoord_v[3] = 3200 + ( pos[0]/larutil::LArProperties::GetME()->DriftVelocity() )/0.5;
      if ( imgcoord_v[3]<=adc_v[0].meta().min_y() || imgcoord_v[3]>=adc_v[0].meta().max_y() )
        inimage = false;
      //std::cout << " [" << istep << "] imgcoord=(" << imgcoord_v[0] << "," << imgcoord_v[1] << "," << imgcoord_v[2] << ", tick=" << imgcoord_v[3] << ")" << std::endl;      
      
      if ( !inimage ) continue;
      
      data.pixelcoord_vv.push_back( imgcoord_v );

      pos.resize(6);
      for (int i=0; i<3; i++ ) {
        pos[3+i] = pos[i];      // move original to end
        pos[i]   = pos_sce[i];  // replace with sce-corrected
      }
      data.pixelpos_vv.push_back( pos );

      
      int row = adc_v[0].meta().row( imgcoord_v[3], __FILE__, __LINE__  );

      
      for (int dr=0; dr<=0; dr++ ) { // no row variation for now
        int r = row+dr;
        if ( r<0 || r>=(int)adc_v[0].meta().rows() ) continue;

        for (int p=0; p<3; p++ ) {

          int col = adc_v[p].meta().col( imgcoord_v[p], __FILE__, __LINE__ );
        
          // for one 3D point right now
          // we move through a neighborhood
          float minrad = col_neighborhood;
          float minrad_pixval = 0.;
          std::vector<int> minradpix = { row, col };


          for (int dc=-col_neighborhood; dc<=col_neighborhood; dc++) {
            int c = col+dc;
            if ( c<0 || c>=(int)adc_v[p].meta().cols() ) continue;
      
            if ( pix_visited_v[p].pixel( r, c )>0 ) continue;
            float pixval = adc_v[p].pixel( r, c );
            if ( pixval<10.0 ) continue;

            // store pixel
            std::vector<int> pix = { r, c};
            float rad = sqrt( dr*dr + dc*dc );

            if ( rad < minrad ) {
              minrad = rad;
              minradpix = pix;
              minrad_pixval = pixval;
            }
            
            pix_visited_v[p].set_pixel( r, c, 10.0 );
            
          }//end of dc loop

          // store pixel for same 3D point
          data.pixellist_vv[p].push_back( minradpix );
          data.pixelrad_vv[p].push_back(  minrad );
          data.totalq_v[p] += minrad_pixval;
          data.toterr_v[p] += minrad;

        }//end of plane loop
        
      }//end of dr neighborhood
      

      if ( last_pos.size()>0 ) {
        
        double step_len = 0.;
        for (int i=0; i<3; i++ )
          step_len += (last_pos[i]-pos_sce[i])*(last_pos[i]-pos_sce[i]);
        step_len = sqrt(step_len);
        data.len_intpc_sce += step_len;
      }
      
      last_pos = pos_sce;
      
    }//end of step loop

    for (size_t p=0; p<3; p++ ) {
      if ( data.pixelrad_vv[p].size()>0 )
        data.toterr_v[p] /= float(data.pixelrad_vv[p].size());
    }

    return data;
  }

  bool CRTTrackMatch::_propose_corrected_track( const CRTTrackMatch::crttrack_t& old,
                                                std::vector< double >& hit1_new,
                                                std::vector< double >& hit2_new ) {

    // cannot make new track
    if ( old.pixelpos_vv.size()<5 )
      return false;
    
    // we use points found along old track, along with shifts to make space points
    std::vector< std::vector<double> > candidate_points_vv;
    
    for ( size_t ipt=0; ipt<old.pixelpos_vv.size(); ipt++ ) {

      // make 3d points from the plane hits with good-ol ubwiretool
      std::vector< std::vector<double> > pos_vv; // bank of 3d points from wire combinatinos
      for (int i=0; i<3;i++ ) {
        
        for (int j=i+1; j<3; j++ ) {
          int otherplane = 0;
          int otherwire  = 0;
          std::vector<float> poszy;
          int crosses = 0;

          ublarcvapp::UBWireTool::getMissingWireAndPlane( i, old.pixellist_vv[i][ipt][1],
                                                          j, old.pixellist_vv[j][ipt][1],
                                                          otherplane, otherwire,
                                                          poszy, crosses );

          if ( crosses==1 ) {
            // if valid point, bank this space point
            std::vector<double> newpos = { old.pixelpos_vv[ipt][3], poszy[1], poszy[0] };

            // reverse the space-charge
            std::vector<double> offset_v = _reverse_sce->GetPosOffsets( newpos[0], newpos[1], newpos[2] );
            // std::vector<double> pos_sce(3,0);
            // pos_sce[0] = newpos[0] - offset_v[0] + 0.6;
            // pos_sce[1] = newpos[1] + offset_v[1];
            // pos_sce[2] = newpos[2] + offset_v[2];            
            // pos_vv.push_back( pos_sce );
            pos_vv.push_back( newpos );
          }
        }//end of plane-j loop
      }//end of plane-i loop

      // pick the one closest to the old (non-SCE applied) point
      if ( pos_vv.size()>1 ) {
        int iclosest = 0;
        double mindist = 1.0e9;
        int ii=0; 
        for (auto it_pos : pos_vv ) {

          // non-SCE corrected point in pixelpos_vv[ipt][3:]
          float dist = 0.;
          for (int i=0; i<3; i++ )
            dist += ( it_pos[i]-old.pixelpos_vv[ipt][3+i] )*( it_pos[i]-old.pixelpos_vv[ipt][3+i] );
          
          if ( dist < mindist ) {
            mindist = dist;
            iclosest = ii;
          }
          ii++;
        }
        candidate_points_vv.push_back( pos_vv[iclosest] );
      }
      else if ( pos_vv.size()==1 ) {
        candidate_points_vv.push_back( pos_vv[0] );
      }

      // std::cout << "  old[" << old.pixelpos_vv[ipt][3] << "," <<  old.pixelpos_vv[ipt][4] << "," <<  old.pixelpos_vv[ipt][5] << "]"
      //           << "  -> "
      //           << "  new[" << candidate_points_vv.back()[0] << "," << candidate_points_vv.back()[1] << "," << candidate_points_vv.back()[2] << "]"
      //           << std::endl;
      
    }//end of old point loop

    std::cout << " found " << candidate_points_vv.size() << " new 3D points from " << old.pixelpos_vv.size() << " points" << std::endl;

    // make cluster, and pca
    larflow::reco::cluster_t cluster;
    cluster.points_v.resize( candidate_points_vv.size() );
    int ii=0;
    for ( auto const& pt : candidate_points_vv ) {
      std::vector<float> fpt = { (float)pt[0], (float)pt[1], (float)pt[2] };
      cluster.points_v[ii] = fpt;
      ii++;
    }
    larflow::reco::cluster_pca( cluster );

    // now we use first pca as line-fit
    // we find intersections to crt
    std::vector< std::vector<double> > panel_pos_v;
    std::vector< double > dist2_original_hits;
    bool ok = _crt_intersections( cluster, *old.pcrttrack,
                                  panel_pos_v, dist2_original_hits );
                                  
    if ( !ok )
      return false;

    if ( dist2_original_hits[0]>50.0 || dist2_original_hits[1]>50.0 )
      return false;
    
    hit1_new = panel_pos_v[0];
    hit2_new = panel_pos_v[1];
    
    return true;
    
  }

  std::string CRTTrackMatch::_str( const CRTTrackMatch::crttrack_t& data ) {
    std::stringstream ss;
    ss << "[crttrack_t] ";
    ss << " pixellist_vv.size=("
       << data.pixellist_vv[0].size() << ","
       << data.pixellist_vv[1].size() << ","
       << data.pixellist_vv[2].size() << ") ";
    ss << " pixelpos_vv.size="
       << data.pixelpos_vv.size();
    ss << " totalq=(" << data.totalq_v[0] << ","
       << data.totalq_v[1] << ","
       << data.totalq_v[2] << ") ";
    ss << " toterr=(" << data.toterr_v[0] << ","
       << data.toterr_v[1] << ","
       << data.toterr_v[2] << ") ";
    ss << " len=" << data.len_intpc_sce;

    return ss.str();
  }


  bool CRTTrackMatch::_crt_intersections( larflow::reco::cluster_t& track_cluster, const larlite::crttrack& orig_crttrack,
                                          std::vector< std::vector<double> >& panel_pos_v,
                                          std::vector< double >& dist2_original_hits ) {
    
    int   crt_plane_norm_dim[4] = {        1,        0,       0,      1 }; // Y-axis, X-axis, X-axis, Y-axis

    panel_pos_v.resize(2);
    dist2_original_hits.resize(2,0.0);
    std::vector< int > hitplane_v = { orig_crttrack.plane1,
                                      orig_crttrack.plane2 };
    std::vector< std::vector<double> > crtpos_v(2);
    crtpos_v[0] = { orig_crttrack.x1_pos,
                    orig_crttrack.y1_pos,
                    orig_crttrack.z1_pos };
    crtpos_v[1] = { orig_crttrack.x2_pos,
                    orig_crttrack.y2_pos,
                    orig_crttrack.z2_pos };
    
    std::vector<double> dir(3,0);
    for (int i=0; i<3; i++ )
      dir[i] = track_cluster.pca_axis_v[0][i];

    std::vector<double> center(3);
    for (int i=0; i<3; i++ )
      center[i] = track_cluster.pca_center[i];
    
    for (int p=0; p<2; p++ ) {
      int hitplane = hitplane_v[p];
      if ( dir[ crt_plane_norm_dim[ hitplane ] ]!=0 ) {
        // only evaluate if not parallel to CRT plane
        std::vector<double> crtpos = crtpos_v[p];

        // dist to center
        double s = (crtpos[ crt_plane_norm_dim[ hitplane ] ] - center[ crt_plane_norm_dim[hitplane] ])/dir[ crt_plane_norm_dim[hitplane] ];
        std::vector<double> panel_pos(3);
        for ( int i=0; i<3; i++ ) {
          panel_pos[i] = center[i] + s*dir[i];
        }

        panel_pos_v[p] = panel_pos;
        float dist = 0.;
        for (int i=0; i<3; i++ ) {
          dist += (panel_pos[i]-crtpos_v[p][i])*(panel_pos[i]-crtpos_v[p][i]);
        }
        dist = sqrt(dist);
      }
      else {
        return false;
      }
    }
    
    return true;
    
  }
  
}
}
