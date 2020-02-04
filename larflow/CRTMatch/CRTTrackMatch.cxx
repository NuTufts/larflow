#include "CRTTrackMatch.h"

#include "larlite/core/LArUtil/LArProperties.h"
#include "larlite/core/LArUtil/Geometry.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/ROOTUtil/ROOTUtils.h"
#include "ublarcvapp/UBWireTool/UBWireTool.h"

#include "TH2D.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TStyle.h"

namespace larflow {
namespace crtmatch {

  CRTTrackMatch::CRTTrackMatch()
    : larcv::larcv_base("CRTTrackMatch"),
    _max_iters(25),
    _col_neighborhood(100),
    _max_fit_step_size(1.0),
    _max_last_step_size(0.1),
    _max_dt_flash_crt(2.0),
    _adc_producer("wire"),
    _crttrack_producer("crttrack"),
    _opflash_producer_v( {"simpleFlashBeam","simpleFlashCosmic"} ),
    _make_debug_images(false)
  {
    _sce = new larutil::SpaceChargeMicroBooNE( larutil::SpaceChargeMicroBooNE::kMCC9_Forward );
    _reverse_sce = new larutil::SpaceChargeMicroBooNE( larutil::SpaceChargeMicroBooNE::kMCC9_Backward );
  }

  CRTTrackMatch::~CRTTrackMatch() {
    delete _sce;
    delete _reverse_sce;
  }

  /**
   * process the event data found in the larcv and larlite event data containers
   *
   */
  void CRTTrackMatch::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll ) {

    // clear containers storing output of algorithm    
    clear_output_containers();
    
    // collect input
    // --------------
    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _adc_producer );
    auto const& adc_v = ev_adc->Image2DArray();
    
    larlite::event_crttrack* ev_crttrack
      = (larlite::event_crttrack*)ioll.get_data( larlite::data::kCRTTrack, _crttrack_producer );

    std::vector< const larlite::event_opflash* > opflash_v;
    for ( auto& flashname : _opflash_producer_v ) {
      larlite::event_opflash* ev_opflash
        = (larlite::event_opflash*)ioll.get_data( larlite::data::kOpFlash, flashname );
      opflash_v.push_back( ev_opflash );
    }
    
    // vizualize for debug
    std::vector<TH2D> h2d_v; // for adc image
    std::vector< std::vector< TGraph* > > graph_vv( adc_v.size() ); // for projected crttrack path in image planes
    if ( _make_debug_images ) {
      h2d_v = larcv::rootutils::as_th2d_v( ev_adc->Image2DArray(), "crttrack_adc" );
    }
    
    // try to fit crt track path
    for ( size_t i=0; i<ev_crttrack->size(); i++ ) {

      crttrack_t fit = _find_optimal_track( ev_crttrack->at(i), ev_adc->Image2DArray() );

      // save, only if there are points inside the TPC and in the image
      if ( fit.pixelpos_vv.size()>0 ) {

        LARCV_NORMAL() << "found fitted CRT track: " << _str(fit) << std::endl;
        _fitted_crttrack_v.emplace_back( std::move(fit) );
        
      }//if fitted track has image path with charge
      
    }//end of crttrack loop

    // match opflashes

    _matchOpflashes( opflash_v, _fitted_crttrack_v, _matched_opflash_v );

    for (int i=0; i<_fitted_crttrack_v.size(); i++ ) {
      auto& fitdata = _fitted_crttrack_v[i];

      larlite::crttrack outcrt( *fitdata.pcrttrack );

      // record shift
      outcrt.x1_err = fitdata.hit_pos_vv[0][0] - outcrt.x1_pos;
      outcrt.y1_err = fitdata.hit_pos_vv[0][1] - outcrt.y1_pos;
      outcrt.z1_err = fitdata.hit_pos_vv[0][2] - outcrt.z1_pos;      
      outcrt.x2_err = fitdata.hit_pos_vv[1][0] - outcrt.x2_pos;
      outcrt.y2_err = fitdata.hit_pos_vv[1][1] - outcrt.y2_pos;
      outcrt.z2_err = fitdata.hit_pos_vv[1][2] - outcrt.z2_pos;

      // record new hit positions
      outcrt.x1_pos = fitdata.hit_pos_vv[0][0];
      outcrt.y1_pos = fitdata.hit_pos_vv[0][1];
      outcrt.z1_pos = fitdata.hit_pos_vv[0][2];
      outcrt.x2_pos = fitdata.hit_pos_vv[1][0];
      outcrt.y2_pos = fitdata.hit_pos_vv[1][1];
      outcrt.z2_pos = fitdata.hit_pos_vv[1][2];

      _modified_crttrack_v.emplace_back( std::move(outcrt) );
      
      larlite::larflowcluster cluster = _crttrack2larflowcluster( fitdata );
      _cluster_v.emplace_back( std::move(cluster) );

    }

    if ( _make_debug_images ) {

      gStyle->SetOptStat(0);
      
      int nplanes = (int)adc_v.size();
      
      TCanvas c("c","c",1000,400*nplanes);
      c.Divide(1,nplanes);
      
      for ( int p=0; p<nplanes; p++ ) {
        
        c.cd(p+1);
        h2d_v[p].Draw("colz");

        if ( p<=1 )
          h2d_v[p].GetXaxis()->SetRangeUser(0,2400);

        for ( auto const& fit : _fitted_crttrack_v ) {

          TGraph* g = new TGraph( fit.pixelcoord_vv[p].size() );
          for ( size_t n=0; n<fit.pixelcoord_vv.size(); n++ ) {
            g->SetPoint( n, fit.pixelcoord_vv[n][p], fit.pixelcoord_vv[n][3] );
          }          
          g->SetLineWidth(1);
          if ( fit.pixelpos_vv.front()[0]<80.0 )
            g->SetLineColor(kMagenta);
          else if ( fit.pixelpos_vv.front()[0]>160.0 )
            g->SetLineColor(kCyan);
          else
            g->SetLineColor(kOrange);

          g->Draw("L");
          graph_vv[p].push_back( g );
        }// fitted loop
        
        std::cout << "graphs in plane[" << p << "]: " << graph_vv[p].size() << std::endl;
      }//end of plane loop

      c.Update();

      // save canvas
      char name[100];
      sprintf( name, "crttrackmatch_debug_run%d_subrun%d_event%d.png",
               (int)iolcv.event_id().run(),
               (int)iolcv.event_id().subrun(),
               (int)iolcv.event_id().event() );
      c.SaveAs(name);

      // clean up
      for ( int p=0; p<nplanes; p++ ) {
        for ( auto& pg : graph_vv[p] )
          delete pg;
      }

    }//end of block to make debug image
    
  }

  /**
   * we step around the hit, optimizing charge/
   * 
   */
  CRTTrackMatch::crttrack_t CRTTrackMatch::_find_optimal_track( const larlite::crttrack& crt,
                                                                const std::vector<larcv::Image2D>& adc_v ) {
    
    LARCV_DEBUG() << "  hit1 pos w/ error: ("
                  << " " << crt.x1_pos << " +/- " << crt.x1_err << ","
                  << " " << crt.y1_pos << " +/- " << crt.y1_err << ","
                  << " " << crt.z1_pos << " +/- " << crt.z1_err << ")"
                  << std::endl;
    LARCV_DEBUG() << "  hit2 pos w/ error: ("
                  << " " << crt.x2_pos << " +/- " << crt.x2_err << ","
                  << " " << crt.y2_pos << " +/- " << crt.y2_err << ","
                  << " " << crt.z2_pos << " +/- " << crt.z2_err << ")"
                  << std::endl;

    // walk the point in a 5 cm range
    std::vector< double > hit1_center = { crt.x1_pos, crt.y1_pos, crt.z1_pos };
    std::vector< double > hit2_center = { crt.x2_pos, crt.y2_pos, crt.z2_pos };
    float t0_usec = 0.5*( crt.ts2_ns_h1+crt.ts2_ns_h2 )*0.001;

    const int ntries   = _max_iters; // max iterations

    // first try, latest iteration of track
    crttrack_t first = _collect_chargepixels_for_track( hit1_center,
                                                        hit2_center,
                                                        t0_usec,
                                                        adc_v,
                                                        _max_fit_step_size,
                                                        _col_neighborhood );
    
    // provide additional info, besides projected points found
    first.pcrttrack = &crt;
    first.t0_usec   = t0_usec;

    LARCV_INFO() << "first try: " << _str( first ) << std::endl;
    
    for (int itry=0; itry<ntries; itry++ ) {

      // next use the found points to fit a new track direction
      std::vector<double> hit1_new;
      std::vector<double> hit2_new;
      bool foundproposal = _propose_corrected_track( first, hit1_new, hit2_new );
      if ( !foundproposal )
        break;

      LARCV_DEBUG() << "  proposal moves CRT hits to: "
                    << "(" << hit1_new[0] << "," << hit1_new[1] << "," << hit1_new[2] << ") "
                    << "(" << hit2_new[0] << "," << hit2_new[1] << "," << hit2_new[2] << ") "
                    << std::endl;
      LARCV_DEBUG() << "  proposal shifts: "
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
                                                           _max_fit_step_size,
                                                           _col_neighborhood );
      sample.pcrttrack = &crt;
      sample.t0_usec = t0_usec;
      sample.hit_pos_vv.clear();
      sample.hit_pos_vv.push_back( hit1_new );
      sample.hit_pos_vv.push_back( hit2_new );

      LARCV_DEBUG() << " ... [try " << itry << "] " << _str(sample) << std::endl;

      float err_diff = 0.;
      for (size_t p=0; p<adc_v.size(); p++ ) {
        err_diff += (first.toterr_v[p] - sample.toterr_v[p]);
      }
      
      if ( err_diff>0.0 ) {      
        LARCV_DEBUG() << " ... error improved by " << err_diff << ", replace old" << std::endl;
        // replace the old
        std::swap( sample, first );
        hit1_center = hit1_new;
        hit2_center = hit2_new;
      }
      else {
        LARCV_INFO() << " ... no improvement after " << itry << " iterations" << std::endl;
        break;
      }
      
    }//end over try loop
        
    crttrack_t bestfit_data = _collect_chargepixels_for_track( hit1_center,
                                                               hit2_center,
                                                               t0_usec,
                                                               adc_v,
                                                               _max_last_step_size,
                                                               10 );
    bestfit_data.pcrttrack = &crt;
    bestfit_data.t0_usec = t0_usec;
    bestfit_data.hit_pos_vv.push_back( hit1_center );
    bestfit_data.hit_pos_vv.push_back( hit2_center );    
    LARCV_INFO() << " track after fit: " << _str(bestfit_data) << std::endl;
    
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

    //std::cout << " found " << candidate_points_vv.size() << " new 3D points from " << old.pixelpos_vv.size() << " points" << std::endl;

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

  /**
   * match opflashes to crttrack_t using closest time
   *
   * ties are broken based on closest centroids based 
   *   on pca-center of tracks and charge-weighted mean of flashes
   * the crttrack_t objects are assumed to have been made in _find_optimal_track(...)
   */
  void CRTTrackMatch::_matchOpflashes( std::vector< const larlite::event_opflash* > flash_vv,
                                       const std::vector<CRTTrackMatch::crttrack_t>& tracks_v,
                                       std::vector< larlite::opflash >& matched_opflash_v ) {

    std::vector< std::vector<int> > used_flash_v( flash_vv.size() );
    for ( size_t i=0; i<flash_vv.size(); i++ )
      used_flash_v[i].resize( flash_vv[i]->size(), 0 );
    
    for ( auto const& trackdata : tracks_v ) {

      std::vector< const larlite::opflash* > matched_in_time_v;
      std::vector<float> dt_usec_v;
      std::vector< std::pair<int,int> > matched_index_v;

      float closest_time = 10e9;

      for ( size_t i=0; i<flash_vv.size(); i++ ) {
        for ( size_t j=0; j<flash_vv[i]->size(); j++ ) {
          if ( used_flash_v[i][j]>0 ) continue;
          
          float dt_usec = trackdata.t0_usec - flash_vv[i]->at(j).Time();

          //std::cout <<" ... compare flash and crttrack time: " << flash_vv[i]->at(j).Time() << " vs. " << trackdata.t0_usec << " dt=" << dt_usec << std::endl;
          
          if ( fabs(dt_usec) < closest_time )
            closest_time = fabs(dt_usec);

          if ( fabs(dt_usec)<_max_dt_flash_crt ) {
            matched_in_time_v.push_back( &(flash_vv[i]->at(j)) );
            dt_usec_v.push_back( dt_usec );
            matched_index_v.push_back( std::pair<int,int>( i, j ) );
            //std::cout << "[CRTTrackMatch]  ...  crt-track and opflash matched. dt_usec=" << dt_usec << std::endl;
          }
          
        }
      }

      LARCV_NORMAL() << "crt-track has " << matched_index_v.size() << " flash matches. closest time=" << closest_time  << std::endl;

      
      if ( matched_in_time_v.size()==0 ) {
        // make empty opflash at the time of the crttrack
        std::vector< double > PEperOpDet(32,0.0);
        larlite::opflash blank_flash( trackdata.t0_usec, 0.0, trackdata.t0_usec, 0,
                                      PEperOpDet );
        matched_opflash_v.emplace_back( std::move(blank_flash) );
      }
      else if ( matched_in_time_v.size()==1 ) {
        // store a copy of the opflash
        matched_opflash_v.push_back( *matched_in_time_v[0] );
        used_flash_v[ matched_index_v[0].first ][ matched_index_v[0].second ] = 1;
      }
      else {
      
        float smallest_dist = 1.0e9;
        const larlite::opflash* closest_opflash = nullptr;

        // we have a loose initial standard. if more than one, we use a tighter standard
        bool have_tight_match = false;
        for ( auto& dt_usec : dt_usec_v ) {
          if ( fabs(dt_usec)<1.5 ) have_tight_match = true;
        }

        int flashindex[2] = { 0, 0 };
        for ( size_t i=0; i<matched_in_time_v.size(); i++ ) {

          // ignore loose match if we know we have at least one good one
          if ( have_tight_match && dt_usec_v[i]>1.5 ) continue;

          // get middle of 3d cluster
          std::vector< float > ave_pos_v(3,0);
          int npoints = 0;

          for ( auto const& pixpos : trackdata.pixelpos_vv ) {
            for (int v=0; v<3; v++ ) ave_pos_v[v] += pixpos[v];
            npoints++;
          }

          if (npoints>0) {
            for (int v=0; v<3; v++ ) ave_pos_v[v] /= float(npoints);
          }

          // get mean of opflash
          std::vector<float> flashcenter = { 0.0,
                                             (float)matched_in_time_v[i]->YCenter(),
                                             (float)matched_in_time_v[i]->ZCenter() };

          float dist = 0.;
          for (int v=1; v<3; v++ ) {
            dist += ( flashcenter[v]-ave_pos_v[v] )*( flashcenter[v]-ave_pos_v[v] );
          }
          dist = sqrt(dist);

          if ( dist<smallest_dist ) {
            smallest_dist = dist;
            closest_opflash = matched_in_time_v[i];
            flashindex[0]   = matched_index_v[i].first;
            flashindex[1]   = matched_index_v[i].second;
          }
          std::cout << "[CRTTrackMatch]  ... distance between opflash and track center: " << dist << " cm" << std::endl;
        }//end of candidate opflash match

        // store best match
        if ( closest_opflash ) {
          matched_opflash_v.push_back( *closest_opflash );
          used_flash_v[ flashindex[0] ][ flashindex[1] ] = 1;
        }
        else {
          // shouldnt get here
          throw std::runtime_error( "should not get here" );
        }
      }//else more than 1 match
          
    }//end of track data loop

    // check we have the right amount of flashes
    if ( matched_opflash_v.size()!=tracks_v.size() ) {
      throw std::runtime_error( "different amount of input crttrack data and opflashes");
    }
  }

  
  /**
   * store crttrack_t data in a cluster of larflow3dhits
   *
   */
  larlite::larflowcluster CRTTrackMatch::_crttrack2larflowcluster( const CRTTrackMatch::crttrack_t& fitdata ) {
    
    larlite::larflowcluster cluster;

    // loop over hits, store crttrack_t data in a cluster of larflow3dhits
    for ( size_t i=0; i<fitdata.pixelpos_vv.size(); i++ ) {

      larlite::larflow3dhit lfhit;
      lfhit.resize(3,0);
      lfhit.targetwire.resize(3,0);
      lfhit.X_truth.resize(3,0); // store pre-sce position, CRT hit1, CRT hit2
      for (int v=0; v<3; v++ ) {
        lfhit[v]            = fitdata.pixelpos_vv[i][v];
        lfhit.targetwire[v] = fitdata.pixelcoord_vv[i][v];
        lfhit.X_truth[v]    = fitdata.pixelpos_vv[i][3+v];
      }
      lfhit.tick = fitdata.pixelcoord_vv[i][3];
      lfhit.srcwire = fitdata.pixelcoord_vv[i][2];
      lfhit.idxhit = i;
      
      cluster.emplace_back( std::move(lfhit) );
    }

    return cluster;
  }

  /**
   * save information into larlite::storage_manager
   *
   */
  void CRTTrackMatch::save_to_file( larlite::storage_manager& ioll, bool remove_if_no_flash ) {

    // now store data
    // --------------
    // (1) new crt object with updated hit positions (and old hit positions as well)
    // (2) matched opflash objects
    // (3) larflow3dhit clusters which store 3d pos and corresponding imgcoord locations for each track
    larlite::event_crttrack* out_crttrack
      = (larlite::event_crttrack*)ioll.get_data( larlite::data::kCRTTrack, "fitcrttrack" );
    larlite::event_opflash* out_opflash
      = (larlite::event_opflash*)ioll.get_data( larlite::data::kOpFlash, "fitcrttrack" );
    larlite::event_larflowcluster* out_lfcluster
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "fitcrttrack" );

    for (size_t i=0; i<_modified_crttrack_v.size(); i++ ) {
      auto& crttrack = _modified_crttrack_v[i];
      auto& opflash  = _matched_opflash_v[i];
      auto& cluster  = _cluster_v[i];
      
      if ( remove_if_no_flash && _matched_opflash_v[i].TotalPE()==0.0 ) {
        LARCV_INFO() << "no matching flash for fitted CRT track[" << i << "], not saving" << std::endl;
        continue;
      }
      
      float petot = opflash.TotalPE();
      LARCV_NORMAL() << "saving track with opflash pe=" << petot << " nopdets=" << opflash.nOpDets() << std::endl;

      out_crttrack->push_back(crttrack);            
      out_opflash->push_back(opflash);
      out_lfcluster->push_back( cluster );
      
    }

  }

  void CRTTrackMatch::clear_output_containers() {
    _fitted_crttrack_v.clear();   //< result of crt track fitter
    _matched_opflash_v.clear();   //< opflashes matched to tracks in _fitted_crttrack_v
    _modified_crttrack_v.clear(); //< crttrack object with modified end points
    _cluster_v.clear();
  }
  
}
}
