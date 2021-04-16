#include "TrackForwardBackwardLL.h"

namespace larflow {
namespace reco {


  TrackForwardBackwardLL::TrackForwardBackwardLL()
    : larcv::larcv_base("TrackForwardBackwardLL")
  {
    _load_data();
  }

 
  TrackForwardBackwardLL::~TrackForwardBackwardLL()
  {
    delete _sMuonRange2dEdx;
    delete _sProtonRange2dEdx;
    _splinefile_rootfile->Close();
  }
    


  /**
   * @brief analyze tracks in neutrino candidate. provide variables to cut out decay muon events.
   *
   *
   */
  void TrackForwardBackwardLL::analyze( larflow::reco::NuVertexCandidate& nuvtx,
                                        larflow::reco::NuSelectionVariables& nusel )
  {

    const int nplanes = 3;

    graph_vv.clear();
    graph_vv.resize(3);
    proton_v.clear();
    muon_v.clear();
    muon_bestfit_plane_v.clear();
    proton_bestfit_plane_v.clear();
    muon_chi2_v.clear();
    proton_chi2_v.clear();
    
    for ( int itrack=0; itrack<(int)nuvtx.track_v.size(); itrack++ ) {
      auto& track_segments = nuvtx.track_v[itrack];
      auto& track_hits     = nuvtx.track_hitcluster_v[itrack];


      std::vector< Track_t > dqdx_v(nplanes);           
      int npts = track_segments.NumberTrajectoryPoints();

      for (int p=0; p<nplanes; p++)
        dqdx_v[p].reserve( npts );
      
      float tracklen = 0;
      for (int ipt=npts-1; ipt>=0; ipt--) {
        const TVector3& start = track_segments.LocationAtPoint(ipt);
        float steplen = 0.;

        if ( ipt>=1 ) {
          const TVector3& end   = track_segments.LocationAtPoint(ipt-1);
          steplen = (end-start).Mag();
          if ( steplen==0 )
            continue;
        }

        for (int p=0; p<nplanes; p++) {
          Pt_t pt;
          pt.x = tracklen;
          pt.dqdx = track_segments.DQdxAtPoint( ipt, (larlite::geo::View_t)p );
          pt.ipt = npts-ipt-1;
          dqdx_v[p].push_back(pt);
        }

        tracklen += steplen;
        
      }//end of point loop

      // // make corresponding expectation curves, forward proton, forward muon, backward muon
      // Track_t dedx_forward_proton;
      // dedx_forward_proton.reserve( dqdx_v[0].size() );

      // for (int ipt=0; ipt<(int)dqdx_v[0].size(); ipt++) {
      //   auto& pt = dqdx_v[0][ipt];
      //   float range = tracklen - pt.x; // if forward
      //   float proton_dedx = _sProtonRange2dEdx->Eval(range); // forward proton assumption

      //   Pt_t p_pt;
      //   p_pt.ipt = ipt;
      //   p_pt.x   = pt.x;
      //   p_pt.dqdx = proton_dedx*(93.2/2.2);
      //   dedx_forward_proton.push_back( p_pt );
      // }

      std::vector< Track_t > smoothed_v;
      smoothed_v.reserve(3);
      for (int p=0; p<3; p++) {
        Track_t smoothed = _smooth( dqdx_v[p], 1.0, 10 );
        smoothed_v.emplace_back( std::move(smoothed) );
      }
      
      // scan for best muon curve
      Track_t dedx_backward_muon;
      float min_chi2 = 1e9;
      int min_plane = -1;
      float best_xshift = 0;
      float best_yscale = 1;
      for (int p=0; p<3; p++) {

        float plane_chi2;
        float plane_xshift;
        float plane_yscale;

        Track_t plane_mu_track = _scan_muon_comparison( smoothed_v[p],
                                                        10, 0.5,
                                                        5, 0.1,
                                                        plane_xshift,
                                                        plane_yscale,
                                                        plane_chi2 );
        
        if ( plane_chi2>0 && plane_chi2<min_chi2 ) {
          std::swap( dedx_backward_muon, plane_mu_track );
          min_chi2 = plane_chi2;
          min_plane = p;
          best_xshift = plane_xshift;
          best_yscale = plane_yscale;
        }
      }
      muon_bestfit_plane_v.push_back( min_plane );
      muon_chi2_v.push_back( min_chi2 );
      
      LARCV_DEBUG() << "track[" << itrack << "] muon min-chi2=" << min_chi2
                    << " best plane=" << min_plane
                    << " x-shift=" << best_xshift
                    << " y-scale=" << best_yscale
                    << std::endl;

      // scan for best proton curve
      Track_t dedx_forward_proton;
      dedx_forward_proton.reserve( dqdx_v[0].size() );      
      float p_min_chi2 = 1e9;
      int p_min_plane = -1;
      float p_best_xshift = 0;
      float p_best_yscale = 1;
      for (int p=0; p<3; p++) {

        float plane_chi2;
        float plane_xshift;
        float plane_yscale;

        Track_t plane_p_track = _scan_proton_comparison( smoothed_v[p],
                                                         10, 0.5,
                                                         5, 0.1,
                                                         plane_xshift,
                                                         plane_yscale,
                                                         plane_chi2 );
        
        if ( plane_chi2>0 && plane_chi2<p_min_chi2 ) {
          std::swap( dedx_forward_proton, plane_p_track );
          p_min_chi2 = plane_chi2;
          p_min_plane = p;
          p_best_xshift = plane_xshift;
          p_best_yscale = plane_yscale;
        }
      }
      proton_bestfit_plane_v.push_back( p_min_plane );
      proton_chi2_v.push_back( p_min_chi2 );      
      LARCV_DEBUG() << "track[" << itrack << "] proton min-chi2=" << p_min_chi2
                    << " best plane=" << p_min_plane
                    << " x-shift=" << p_best_xshift
                    << " y-scale=" << p_best_yscale
                    << std::endl;
      

      // for debug
      for (int p=0; p<nplanes; p++) {
        auto const& smoothed = smoothed_v[p];
        TGraphErrors g(smoothed.size());
        for (int i=0; i<(int)smoothed.size(); i++) {
          g.SetPoint(i,smoothed[i].x,smoothed[i].dqdx);
          g.SetPointError(i,0,smoothed[i].var);
        }
        graph_vv[p].emplace_back( std::move(g) );
      }

      TGraph gproton( dqdx_v[0].size());
      TGraph gmuon( dqdx_v[0].size() );
      for (int ipt=0; ipt<(int)dqdx_v[0].size(); ipt++) {
        gproton.SetPoint( ipt ,dedx_forward_proton[ipt].x, dedx_forward_proton[ipt].dqdx );
        gmuon.SetPoint(   ipt ,dedx_backward_muon[ipt].x,  dedx_backward_muon[ipt].dqdx );        
      }
      proton_v.emplace_back( std::move(gproton) );
      muon_v.emplace_back( std::move(gmuon) );
      
    }//end of track loop
    
  }
  
  /**
   * @brief compare forward proton and backward muon likelihoods
   *
   * meant to help discriminate against nu vertex being placed onto muon decay electron.
   *
   */
  // float TrackForwardBackwardLL::calcForwardProtonBackwardMuonLikelihood( larlite::track& track_w_dedx )
  // {
    
  // }


  /**
   * @brief smooth de/dx vs. position
   *
   */
  TrackForwardBackwardLL::Track_t
  TrackForwardBackwardLL::_smooth( TrackForwardBackwardLL::Track_t& track,
                                   const float maxdist, const int nn )
  {

    Track_t smoothed(track.size());
    int npts = track.size();

    for (int i=0; i<npts; i++) {
      int n = 0;
      float ave = 0;
      float ave2 = 0.;

      for (int di=-abs(nn); di<=abs(nn); di++) {
        int j = i+di;
        if ( i<0 || i>=npts ) continue;

        float dist = fabs( track[j].x - track[i].x );
        if( dist>maxdist ) continue;
        
        ave  += track[j].dqdx;
        ave2 += track[j].dqdx*track[j].dqdx;
        n++;
      }
      if ( n>0 ) {
        ave  /= (float)n;
        ave2 /= (float)n;
      }
    
      float var = sqrt( ave2 - ave*ave );
      smoothed[i] = track[i];
      smoothed[i].dqdx = ave;
      smoothed[i].var = var;
    }

    return smoothed;
  }

  void TrackForwardBackwardLL::_load_data()
  {    
    
    std::string larflowdir = std::getenv("LARFLOW_BASEDIR");
    std::string splinefile = larflowdir + "/larflow/Reco/data/Proton_Muon_Range_dEdx_LAr_TSplines.root";

    _splinefile_rootfile = new TFile( splinefile.c_str(), "open" );
    _sMuonRange2dEdx = (TSpline3*)_splinefile_rootfile->Get("sMuonRange2dEdx");
    _sProtonRange2dEdx = (TSpline3*)_splinefile_rootfile->Get("sProtonRange2dEdx");      
  }


  TrackForwardBackwardLL::Track_t
  TrackForwardBackwardLL::_generate_muon_expectation( const TrackForwardBackwardLL::Track_t& data_track,
                                                      float x_shift, float y_scale )
  {
    
    Track_t dedx_backward_muon;
    dedx_backward_muon.reserve( data_track.size() );

    const float q2adc = (93.2/2.2);

    for (int ipt=0; ipt<(int)data_track.size(); ipt++) {
      auto const& pt = data_track[ipt];
      float x = pt.x + x_shift;
      Pt_t mu_pt;
      mu_pt.ipt = ipt;
      mu_pt.x   = pt.x;
      
      if ( x>=0 ) {
        float mu_dedx = _sMuonRange2dEdx->Eval(x); // backward muon assumption
        float mu_dqdx_birks = (y_scale*q2adc)*mu_dedx/(1+mu_dedx*0.0486/0.273/1.38); //modified box model
        mu_pt.dqdx = mu_dqdx_birks;
      }
      else {
        mu_pt.dqdx = -1.0;
      }
      //LARCV_DEBUG() << " pt[" << ipt << "] x=" << mu_pt.x << " dqdx=" << mu_pt.dqdx << std::endl;
      dedx_backward_muon.push_back( mu_pt );
    }

    return dedx_backward_muon;
  }

  TrackForwardBackwardLL::Track_t
  TrackForwardBackwardLL::_generate_proton_expectation( const TrackForwardBackwardLL::Track_t& data_track,
                                                        float x_shift, float y_scale )
  {

    const float q2adc = (93.2/2.2);
    
    Track_t dedx_forward_proton;
    dedx_forward_proton.reserve( data_track.size() );

    float tracklen = data_track.back().x;
    
    for (int ipt=0; ipt<(int)data_track.size(); ipt++) {
      auto& pt = data_track[ipt];
      float range = tracklen - pt.x + x_shift; // if forward
      Pt_t p_pt;

      p_pt.ipt = ipt;
      p_pt.x   = pt.x;
      
      if ( range>=0 ) {
        float proton_dedx = _sProtonRange2dEdx->Eval(range); // forward proton assumption
        float proton_dqdx_birks = (y_scale*q2adc)*proton_dedx/(1+proton_dedx*0.0486/0.273/1.38); //modified box model        
        p_pt.dqdx = proton_dqdx_birks;
      }
      else {
        p_pt.dqdx = -1;
      }
      dedx_forward_proton.push_back( p_pt );      
    }

    return dedx_forward_proton;
  }

  float TrackForwardBackwardLL::_calc_chi2( const TrackForwardBackwardLL::Track_t& data_track,
                                            const TrackForwardBackwardLL::Track_t& expect_track )
  {
    float chi2 = 0.;
    int npts = 0;
    
    if ( data_track.size()!=expect_track.size() ) {
      LARCV_CRITICAL() << "calculating chi2 for different number of points!"
                       << " ndata=" << data_track.size()
                       << " nexpect=" << expect_track.size()
                       << std::endl;
      throw std::runtime_error( "calculating chi2 for different number of points!" );      
    }

    int npt_badexpect = 0;
    int npt_badvar    = 0;
    int npt_nan       = 0;
    
    for (size_t i=0; i<data_track.size(); i++) {
      auto const& data_pt   = data_track[i];
      auto const& expect_pt = expect_track[i];

      if ( expect_pt.dqdx<0 ) {
        npt_badexpect++;
        continue;
      }

      if ( data_pt.var<=0 ) {
        npt_badvar++;
        continue;
      }

      if ( std::isnan(data_pt.var) ) {
        npt_nan++;
        continue;
      }
      
      npts++;
      float diff = data_pt.dqdx - expect_pt.dqdx;

      
      chi2 += (diff*diff)/(data_pt.var*data_pt.var);
    }

    // LARCV_DEBUG() << " npts=" << npts
    //               << " badexpect=" << npt_badexpect
    //               << " badvar=" << npt_badvar
    //               << " badnan=" << npt_nan
    //               <<  " chi2=" << chi2 << std::endl;
    
    if ( npts>0 )
      chi2 /= (float)npts;
    else
      return -1;
    
    return chi2;
  }

  TrackForwardBackwardLL::Track_t  
  TrackForwardBackwardLL::_scan_muon_comparison( const TrackForwardBackwardLL::Track_t& data_track,
                                                 const int nx, const float xstep,
                                                 const int ny, const float ystep,
                                                 float& best_xshift, float& best_yscale, float& min_chi2 )
  {

    best_xshift = 0;
    best_yscale = 0;
    min_chi2 = 1e9;
    
    TrackForwardBackwardLL::Track_t best_expectation;
    
    for (int ix=-abs(nx); ix<=abs(nx); ix++) {
      for (int iy=-abs(ny); iy<=abs(ny); iy++) {

        float xshift = ix*xstep;
        float yscale = 1.0 + iy*ystep;

        TrackForwardBackwardLL::Track_t mu_track
          = _generate_muon_expectation( data_track, xshift, yscale );

        //LARCV_DEBUG() << " mu_track npts=" << mu_track.size() << " data_track npts=" << data_track.size() << std::endl;
        
        float chi2 = _calc_chi2( data_track, mu_track );

        //LARCV_DEBUG() << "[" << xshift << "," << yscale << "] chi2=" << chi2 << std::endl;
        
        if ( chi2<0 )
          continue;

        if ( chi2<min_chi2 ) {
          // update best
          best_xshift = xshift;
          best_yscale = yscale;
          min_chi2 = chi2;
          std::swap( best_expectation, mu_track );
        }
      }
    }
    
    return best_expectation;
  }


  TrackForwardBackwardLL::Track_t  
  TrackForwardBackwardLL::_scan_proton_comparison( const TrackForwardBackwardLL::Track_t& data_track,
                                                   const int nx, const float xstep,
                                                   const int ny, const float ystep,
                                                   float& best_xshift, float& best_yscale, float& min_chi2 )
  {

    best_xshift = 0;
    best_yscale = 0;
    min_chi2 = 1e9;
    
    TrackForwardBackwardLL::Track_t best_expectation;
    
    for (int ix=-abs(nx); ix<=abs(nx); ix++) {
      for (int iy=-abs(ny); iy<=abs(ny); iy++) {

        float xshift = ix*xstep;
        float yscale = 1.0 + iy*ystep;

        TrackForwardBackwardLL::Track_t p_track
          = _generate_proton_expectation( data_track, xshift, yscale );

        //LARCV_DEBUG() << " mu_track npts=" << mu_track.size() << " data_track npts=" << data_track.size() << std::endl;
        
        float chi2 = _calc_chi2( data_track, p_track );

        //LARCV_DEBUG() << "[" << xshift << "," << yscale << "] chi2=" << chi2 << std::endl;
        
        if ( chi2<0 )
          continue;

        if ( chi2<min_chi2 ) {
          // update best
          best_xshift = xshift;
          best_yscale = yscale;
          min_chi2 = chi2;
          std::swap( best_expectation, p_track );
        }
      }
    }
    
    return best_expectation;
  }
  
}
}
