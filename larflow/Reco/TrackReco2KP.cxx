#include "TrackReco2KP.h"

#include <algorithm>

#include "ublarcvapp/ubdllee/dwall.h"

#include "geofuncs.h"

namespace larflow {
namespace reco {

  void TrackReco2KP::process( larcv::IOManager& iolcv,
                              larlite::storage_manager& ioll,
                              const std::vector<KPCluster>& kpcluster_v )
  {

    // rank KP by dwall
    std::vector< KPInfo_t > sorted_kp_v(kpcluster_v.size());
    for (int idx=0; idx<(int)kpcluster_v.size(); idx++) {
      KPInfo_t& info = sorted_kp_v[idx];
      info.idx = idx;
      info.dwall = ublarcvapp::dwall( kpcluster_v[idx].max_pt_v, info.boundary_type );
    }

    std::sort( sorted_kp_v.begin(), sorted_kp_v.end() );
    
    // make pairs
    std::vector< KPPair_t > sorted_pair_v;
    for (int i=0; i<(int)sorted_kp_v.size(); i++) {
      auto const& kp_i = kpcluster_v[i];
      for (int j=0; j<(int)sorted_kp_v.size(); j++) {
        if ( i==j ) continue;
        auto const& kp_j = kpcluster_v[j];

        // get distance of point-j onto pca-line
        std::vector<float> line_end(3);
        for (int v=0; v<3; v++) line_end[v] = kp_i.max_pt_v[i] + 10.0*kp_i.pca_axis_v[0][i];

        float rad_dist = pointLineDistance( kp_i.max_pt_v, line_end, kp_j.max_pt_v );

        // skip trying to connect points if larger than this distance
        if ( rad_dist>100.0 )
          continue;
        
        KPPair_t kpp;
        kpp.start_idx = i;
        kpp.end_idx = j;
        kpp.dist2axis = rad_dist;

        sorted_pair_v.push_back( kpp );
        
      }
    }

    std::sort( sorted_pair_v.begin(), sorted_pair_v.end() );

    // pair up, reconstruct 3D points
    
    // get triplets
    larlite::event_larflow3dhit* ev_larflowhits 
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit,"larmatch");

    // flag for keypoints we used up
    std::vector<int> kp_used_v( sorted_kp_v.size(), 0 ); //< track which variables have been used

    // flag for spacepoints we used up
    std::vector<int> sp_used_v( ev_larflowhits->size(), 0 );

    for ( int ipair=0; ipair<(int)sorted_pair_v.size(); ipair++ ) {
      auto const& apair = sorted_pair_v[ipair];
      if ( kp_used_v[ apair.start_idx ]!=0 || kp_used_v[ apair.end_idx ]!=0 ) continue;

      // attempt to reconstruct points
      std::vector<int> trackpoints_v = makeTrack( kpcluster_v[apair.start_idx].max_pt_v,
                                                  kpcluster_v[apair.end_idx].max_pt_v,
                                                  *ev_larflowhits, sp_used_v );
    }
    
  }
  

  std::vector<int> TrackReco2KP::makeTrack( const std::vector<float>& startpt,
                                            const std::vector<float>& endpt,
                                            const larlite::event_larflow3dhit& lfhits,
                                            std::vector<int>& sp_used_v )
  {

    // collect the points
    std::vector< int > point_idx_v;
    std::vector< std::vector<float> > point_v;

    for (int i=0; i<(int)lfhits.size(); i++) {

      if ( sp_used_v[i]!=0 ) continue;
      
      std::vector<float> pos = { lfhits[i][0], lfhits[i][1], lfhits[i][2] };

      float rad = pointLineDistance( startpt, endpt, pos );

      if ( rad<100.0 ) {
        point_v.push_back(pos);
        point_idx_v.push_back(i);
      }
    }

    // graph-based reconstruction
    
  }
  
}
}
