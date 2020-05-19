#include "TrackReco2KP.h"

#include <algorithm>

#include "ublarcvapp/ubdllee/dwall.h"
#include "DataFormat/track.h"

#include "geofuncs.h"

namespace larflow {
namespace reco {

  void TrackReco2KP::process( larcv::IOManager& iolcv,
                              larlite::storage_manager& ioll,
                              const std::vector<KPCluster>& kpcluster_v )
  {

    // rank KP by dwall
    std::cout << "Make KPInfo_t list" << std::endl;
    std::vector< KPInfo_t > sorted_kp_v(kpcluster_v.size());
    for (int idx=0; idx<(int)kpcluster_v.size(); idx++) {
      KPInfo_t& info = sorted_kp_v[idx];
      info.idx = idx;
      info.dwall = ublarcvapp::dwall( kpcluster_v[idx].max_pt_v, info.boundary_type );
    }

    std::sort( sorted_kp_v.begin(), sorted_kp_v.end() );
    
    // make pairs
    std::cout << "Make KPPair_t list" << std::endl;    
    std::vector< KPPair_t > sorted_pair_v;
    for (int i=0; i<(int)sorted_kp_v.size(); i++) {
      auto const& kp_i = kpcluster_v[i];
      for (int j=0; j<(int)sorted_kp_v.size(); j++) {
        if ( i==j ) continue;
        auto const& kp_j = kpcluster_v[j];

        // get distance of point-j onto pca-line
        std::vector<float> line_end(3);
        float dist = 0.;
        for (int v=0; v<3; v++) {
          line_end[v] = kp_i.max_pt_v[v] + 10.0*kp_i.pca_axis_v[0][v];
          dist += ( kp_i.max_pt_v[v] - kp_j.max_pt_v[v] )*( kp_i.max_pt_v[v] - kp_j.max_pt_v[v] );
        }
        dist = sqrt(dist);

        float rad_dist = pointLineDistance( kp_i.max_pt_v, line_end, kp_j.max_pt_v );

        // skip trying to connect points if larger than this distance
        if ( rad_dist>20.0 )
          continue;
        
        KPPair_t kpp;
        kpp.start_idx = i;
        kpp.end_idx = j;
        kpp.dist2axis = rad_dist;
        kpp.dist2pts  = dist;

        sorted_pair_v.push_back( kpp );
        
      }
    }

    std::sort( sorted_pair_v.begin(), sorted_pair_v.end() );
    std::cout << "Number of KPPair_t: " << sorted_pair_v.size() << std::endl;
    
    // pair up, reconstruct 3D points

    
    // get triplets
    larlite::event_larflow3dhit* ev_larflowhits 
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit,"larmatch");

    std::cout << "Number of larflow spacepoints: " << ev_larflowhits->size() << std::endl;

    // flag for keypoints we used up
    std::vector<int> kp_used_v( sorted_kp_v.size(), 0 ); //< track which variables have been used

    // flag for spacepoints we used up
    std::vector<int> sp_used_v( ev_larflowhits->size(), 0 );

    larlite::event_track* evout_track
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "track2kp" );
    
    for ( int ipair=0; ipair<(int)sorted_pair_v.size(); ipair++ ) {
      auto const& apair = sorted_pair_v[ipair];
      if ( kp_used_v[ apair.start_idx ]!=0 || kp_used_v[ apair.end_idx ]!=0 ) continue;

      // attempt to reconstruct points
      std::cout << "[pair " << ipair << "] "
                << " dist2axis=" << apair.dist2axis
                << " dist2pts=" << apair.dist2pts
                << std::endl;
      
      std::vector<int> trackpoints_v = makeTrack( kpcluster_v[apair.start_idx].max_pt_v,
                                                  kpcluster_v[apair.end_idx].max_pt_v,
                                                  *ev_larflowhits, sp_used_v );
      std::cout << "[pair " << ipair << "] returns track of " << trackpoints_v.size() << " spacepoints" << std::endl;
      if ( trackpoints_v.size()==0 ) continue;
      
      larlite::track track;
      track.reserve(trackpoints_v.size());
      int ipt=0;
      for ( auto& idx : trackpoints_v ) {
        TVector3 pt = { (*ev_larflowhits)[idx][0], (*ev_larflowhits)[idx][1], (*ev_larflowhits)[idx][2] };
        TVector3 ptdir = { 0, 0, 0};
        track.add_vertex( pt );
        track.add_direction( ptdir );
        ipt++;
      }
      evout_track->emplace_back( std::move(track) );
    }

  }
  

  std::vector<int> TrackReco2KP::makeTrack( const std::vector<float>& startpt,
                                            const std::vector<float>& endpt,
                                            const larlite::event_larflow3dhit& lfhits,
                                            std::vector<int>& sp_used_v )
  {

    std::cout << "make track between: (" << startpt[0] << "," << startpt[1] << "," << startpt[2] << ") and "
              << "(" << endpt[0] << "," << endpt[1] << "," << endpt[2] << ")"
              << std::endl;

    // make bounding box for fast comparison
    float bounds[3][2];
    for (int i=0; i<3; i++) {
      bounds[i][0] = (startpt[i]<endpt[i]) ? startpt[i] : endpt[i];
      bounds[i][1] = (startpt[i]<endpt[i]) ? endpt[i]   : startpt[i];
    }
    
    // collect the points
    struct Point_t {
      float s; // projection on line between start and endpt
      int idx; // index in lfhits array
      std::vector<float> pos; // 3d point
      bool operator<( const Point_t& rhs ) {
        if (s<rhs.s) return true;
        return false;
      };
    };

    std::vector< Point_t > point_v;

    std::vector<float> linedir(3,0);
    float dist = 0.;
    for (int i=0; i<3; i++) {
      linedir[i] = endpt[i]-startpt[i];
      dist += linedir[i]*linedir[i];
    }
    dist = sqrt(dist);
    if (dist>0) {
      for (int i=0; i<3; i++)
        linedir[i] /= dist;
    }

    for (int i=0; i<(int)lfhits.size(); i++) {

      if ( sp_used_v[i]!=0 ) continue;

      // bounding box test
      bool inbox = true;
      for (int v=0; v<3; v++) {
        if ( lfhits[i][v]<bounds[v][0] || lfhits[i][v]>bounds[v][1] ) inbox=false;
      }
      if ( !inbox ) continue;
      
      std::vector<float> pos = { lfhits[i][0], lfhits[i][1], lfhits[i][2] };
      float rad = pointLineDistance( startpt, endpt, pos );

      if ( rad<20.0 ) {

        Point_t pt;
        pt.idx = i;
        pt.pos = pos;
        pt.s   = 0.;
        for (int v=0; v<3; v++)
          pt.s += linedir[v]*(pos[v]-startpt[v]);        
        point_v.push_back(pt);
        sp_used_v[i] = 1;
      }
    }

    sort( point_v.begin(), point_v.end() );
    
    // graph-based reconstruction
    std::vector<int> trackpoints_v;
    for ( auto& pt : point_v )
      trackpoints_v.push_back( pt.idx );
    
    return trackpoints_v;
  }
  
}
}
