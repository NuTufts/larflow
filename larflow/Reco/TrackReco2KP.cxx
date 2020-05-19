#include "TrackReco2KP.h"

#include <algorithm>

#include "ublarcvapp/ubdllee/dwall.h"
#include "DataFormat/track.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "geofuncs.h"
#include "KPTrackFit.h"

namespace larflow {
namespace reco {

  void TrackReco2KP::process( larcv::IOManager& iolcv,
                              larlite::storage_manager& ioll,
                              const std::vector<KPCluster>& kpcluster_v )
  {

    // get bad channel image
    larcv::EventImage2D* ev_badch
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "badch" );
    auto const& badch_v = ev_badch->Image2DArray();
    
    // rank KP by dwall
    std::cout << "Make KPInfo_t list" << std::endl;
    std::vector< KPInfo_t > sorted_kp_v(kpcluster_v.size());
    for (int idx=0; idx<(int)kpcluster_v.size(); idx++) {
      KPInfo_t& info = sorted_kp_v[idx];
      info.idx = idx;
      info.dwall = ublarcvapp::dwall( kpcluster_v[idx].max_pt_v, info.boundary_type );
    }

    std::sort( sorted_kp_v.begin(), sorted_kp_v.end() );

    // copy of keypoints considered
    larlite::event_larflow3dhit* evout_kpcopy
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit,"keypoint");
    for (auto const& kpc : kpcluster_v ) {
      larlite::larflow3dhit hit;
      hit.resize(3,0);
      for (int i=0; i<3; i++ ) hit[i] = kpc.max_pt_v[i];
      evout_kpcopy->emplace_back( std::move(hit) );
    }
    
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
        if ( rad_dist>100.0 )
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
      std::cout << "-------------------------------------" << std::endl;
      std::cout << "[pair " << ipair << "] "
                << " KP[" << apair.start_idx << "] -> KP[" << apair.end_idx << "]"
                << " dist2axis=" << apair.dist2axis
                << " dist2pts=" << apair.dist2pts
                << std::endl;
      
      std::vector<int> trackpoints_v = makeTrack( kpcluster_v[apair.start_idx].max_pt_v,
                                                  kpcluster_v[apair.end_idx].max_pt_v,
                                                  *ev_larflowhits, badch_v, sp_used_v );
      std::cout << "[pair " << ipair << "] returns track of " << trackpoints_v.size() << " spacepoints" << std::endl;
      if ( trackpoints_v.size()<=2 ) continue;
      std::cout << "[pair " << ipair << "] store track." << std::endl;

      larlite::track track;
      track.reserve(trackpoints_v.size());
      int ipt=0;
      for ( auto& idx : trackpoints_v ) {
        if ( idx==-1 ) {
          TVector3 pt = { kpcluster_v[apair.start_idx].max_pt_v[0],
                          kpcluster_v[apair.start_idx].max_pt_v[1],
                          kpcluster_v[apair.start_idx].max_pt_v[2] };
          TVector3 ptdir = {0,0,0};
          track.add_vertex( pt );
          track.add_direction( ptdir );
        }
        else if ( idx==-2 ) {

          TVector3 pt = { kpcluster_v[apair.end_idx].max_pt_v[0],
                          kpcluster_v[apair.end_idx].max_pt_v[1],
                          kpcluster_v[apair.end_idx].max_pt_v[2] };
          TVector3 ptdir = {0,0,0};
          track.add_vertex( pt );
          track.add_direction( ptdir );
          
        }
        else {
          TVector3 pt = { (*ev_larflowhits)[idx][0], (*ev_larflowhits)[idx][1], (*ev_larflowhits)[idx][2] };
          TVector3 ptdir = { 0, 0, 0};
          track.add_vertex( pt );
          track.add_direction( ptdir );
        }
        ipt++;
      }
      evout_track->emplace_back( std::move(track) );

      //break;
    }//end of pair loop

  }
  

  std::vector<int> TrackReco2KP::makeTrack( const std::vector<float>& startpt,
                                            const std::vector<float>& endpt,
                                            const larlite::event_larflow3dhit& lfhits,
                                            const std::vector<larcv::Image2D>& badch_v,
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

    int start_idx = -1;
    int end_idx = -1;

    // make sure the start point is first
    Point_t start;
    start.idx = -1;      // assign later
    start.pos = startpt; // set 3d pos
    start.s = 0.;        // by definition
    point_v.push_back( start );

    // make the end point object, we will add it last
    Point_t  end;
    end.idx = -2;    // assign later
    end.pos = endpt; // set 3d pos
    end.s = dist;    // by definition
    
    for (int i=0; i<(int)lfhits.size(); i++) {

      if ( sp_used_v[i]!=0 ) continue;

      // bounding box test
      bool inbox = true;
      for (int v=0; v<3; v++) {
        if ( lfhits[i][v]<bounds[v][0] || lfhits[i][v]>bounds[v][1] ) inbox=false;
      }
      if ( !inbox ) continue;

      // check if its the start and end point      
      std::vector<float> pos = { lfhits[i][0], lfhits[i][1], lfhits[i][2] };

      if ( start_idx==-1 && startpt==pos ) {
        std::cout << "found start point" << std::endl;
        point_v[0].idx = i;
        start_idx = i;
        continue;
      }

      if ( end_idx==-1 && endpt==pos ) {
        std::cout << "found end point" << std::endl;
        end.idx = i;
        end_idx = i;
        continue;
      }
      
      float rad = pointLineDistance( startpt, endpt, pos );

      if ( rad<20.0 ) {

        Point_t pt;
        pt.idx = i;
        pt.pos = pos;
        pt.s   = 0.;
        for (int v=0; v<3; v++)
          pt.s += linedir[v]*(pos[v]-startpt[v]);        
        point_v.push_back(pt);
        //sp_used_v[i] = 1;
      }
    }
    
    // add end point
    point_v.push_back( end );

    if ( point_v.size()<10 ) {
      std::vector<int> empty;
      return empty;
    }
      

    std::vector< std::vector<float> > pos3d_v(point_v.size());
    for ( int i=0; i<(int)point_v.size(); i++ ) {
      pos3d_v[i] = point_v[i].pos;
    }

    std::cout << "RUN KPTrackFit w/ " << point_v.size() << " points in graph" << std::endl;
    KPTrackFit tracker;
    tracker.set_verbosity(larcv::msg::kINFO);
    std::vector<int> trackpoints_v = tracker.fit( pos3d_v, badch_v, 0, (int)pos3d_v.size()-1 );

    std::vector<int> lfhit_index_v;
    for ( auto& idx : trackpoints_v ) {
      // std::cout << "  point_v[" << idx << "] lfhit[" << point_v[idx].idx << "] "
      //           << " (" << point_v[idx].pos[0] << "," << point_v[idx].pos[1] << "," << point_v[idx].pos[2] << ")";
      // if ( point_v[idx].idx==-1 )
      //   std::cout << "(" << startpt[0] << "," << startpt[1] << "," << startpt[2] << ")";
      // else if ( point_v[idx].idx==-2 )
      //   std::cout << "(" << endpt[0] << "," << endpt[1] << "," << endpt[2] << ")";
      // else
      //   std::cout << " (" << lfhits[ point_v[idx].idx ][0] << "," << lfhits[ point_v[idx].idx ][1] << "," << lfhits[ point_v[idx].idx ][2] << ")";
      // std::cout << std::endl;
      lfhit_index_v.push_back( point_v[idx].idx );
    }
    
    return lfhit_index_v;
  }
  
}
}
