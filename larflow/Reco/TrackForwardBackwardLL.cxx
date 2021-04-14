#include "TrackForwardBackwardLL.h"

namespace larflow {
namespace reco {


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

      for (int p=0; p<nplanes; p++) {
        Track_t smoothed = _smooth( dqdx_v[p], 1.0, 10 );
        TGraphErrors g(smoothed.size());
        for (int i=0; i<(int)smoothed.size(); i++) {
          g.SetPoint(i,smoothed[i].x,smoothed[i].dqdx);
          g.SetPointError(i,0,smoothed[i].var);
        }
        graph_vv[p].emplace_back( std::move(g) );
      }
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
}
}
