#include "TrackReco2KP.h"

#include <algorithm>

#include "ublarcvapp/ubdllee/dwall.h"
#include "DataFormat/track.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/pcaxis.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "geofuncs.h"
#include "KPTrackFit.h"

namespace larflow {
namespace reco {

  void TrackReco2KP::process( larcv::IOManager& iolcv,
                              larlite::storage_manager& ioll,
                              const std::vector<KPCluster>& kpcluster_v )
  {

    std::vector< std::vector<float> > kppos_v;
    std::vector< std::vector<float> > kpaxis_v;

    for ( auto& kpc : kpcluster_v ) {
      kppos_v.push_back( kpc.max_pt_v );
      kpaxis_v.push_back( kpc.pca_axis_v[0] );
    }
    
    process( iolcv, ioll, kppos_v, kpaxis_v );
    
  }

  void TrackReco2KP::process( larcv::IOManager& iolcv,
                              larlite::storage_manager& ioll )
  {

    larlite::event_larflow3dhit* ev_keypoint =
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _keypoint_treename );
    larlite::event_pcaxis* ev_pcaxis =
      (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, _keypoint_treename );

    LARCV_DEBUG() << "number of keypoints from " << _keypoint_treename << ": " << ev_keypoint->size() << std::endl;
    LARCV_DEBUG() << "number of pca-axis from  " << _keypoint_treename << ": " << ev_pcaxis->size() << std::endl;    
    
    std::vector< std::vector<float> > kppos_v;
    std::vector< std::vector<float> > kpaxis_v;

    for ( size_t ipt=0; ipt<ev_keypoint->size(); ipt++ ) {
      std::vector<float> pos(3,0);
      std::vector<float> axis(3,0);

      for (int v=0; v<3; v++ ) {
        pos[v] = ev_keypoint->at(ipt)[v];
        axis[v] = ev_pcaxis->at(ipt).getEigenVectors()[0][v];
      }
      
      kppos_v.push_back( pos );
      kpaxis_v.push_back( axis );
    }
    
    process( iolcv, ioll, kppos_v, kpaxis_v );
    
  }
  
  
  void TrackReco2KP::process( larcv::IOManager& iolcv,
                              larlite::storage_manager& ioll,
                              const std::vector< std::vector<float> >& kppos_v,
                              const std::vector< std::vector<float> >& kpaxis_v )                              
  {
    
    // get bad channel image
    larcv::EventImage2D* ev_badch
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "badch" );
    auto const& badch_v = ev_badch->Image2DArray();
    
    // rank KP by dwall
    std::cout << "Make KPInfo_t list" << std::endl;
    std::vector< KPInfo_t > sorted_kp_v(kppos_v.size());
    for (int idx=0; idx<(int)kppos_v.size(); idx++) {
      KPInfo_t& info = sorted_kp_v[idx];
      info.idx = idx;
      info.dwall = ublarcvapp::dwall( kppos_v[idx], info.boundary_type );
    }

    std::sort( sorted_kp_v.begin(), sorted_kp_v.end() );
    
    // make pairs
    std::cout << "Make KPPair_t list" << std::endl;    
    std::vector< KPPair_t > sorted_pair_v;
    for (int i=0; i<(int)sorted_kp_v.size(); i++) {
      auto const& kp_i = kppos_v[i];
      auto const& kp_i_pca = kpaxis_v[i];
      for (int j=0; j<(int)sorted_kp_v.size(); j++) {
        if ( i==j ) continue;
        auto const& kp_j = kppos_v[j];
        auto const& kp_j_pca = kpaxis_v[j];        

        // get distance of point-j onto pca-line
        std::vector<float> line_end(3);
        float dist = 0.;
        float cospca = 0.;
        float pca1 = 0.;
        float pca2 = 0.;
        for (int v=0; v<3; v++) {
          line_end[v] = kp_i[v] + 10.0*kp_i_pca[v];
          dist += ( kp_i[v] - kp_j[v] )*( kp_i[v] - kp_j[v] );
          pca1 += (kp_i_pca[v]*kp_i_pca[v]);
          pca2 += (kp_j_pca[v]*kp_j_pca[v]);
          cospca += kp_i_pca[v]*kp_j_pca[v];
        }
        dist = sqrt(dist);
        cospca /= (pca1*pca2);

        cospca = fabs(cospca);

        float rad_dist = pointLineDistance( kp_i, line_end, kp_j );

        LARCV_DEBUG() << "pair proposed [" << i << "," << j << "] rad_dist" << rad_dist << " cosine=" << cospca << std::endl;

        // skip trying to connect points if larger than this distance
        if ( rad_dist>100.0 )
          continue;
        
        KPPair_t kpp;
        kpp.start_idx = i;
        kpp.end_idx = j;
        kpp.dist2axis = rad_dist;
        kpp.dist2pts  = dist;
        kpp.cospca    = cospca;

        sorted_pair_v.push_back( kpp );
        
      }
    }

    std::sort( sorted_pair_v.begin(), sorted_pair_v.end() );
    std::cout << "Number of KPPair_t: " << sorted_pair_v.size() << std::endl;
    
    // pair up, reconstruct 3D points

    
    // get triplets
    larlite::event_larflow3dhit* ev_larflowhits 
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, _larflow_hit_treename );

    std::cout << "Number of larflow spacepoints (from " << _larflow_hit_treename << "): "
              << ev_larflowhits->size() << std::endl;

    // flag for keypoints we used up
    std::vector<int> kp_used_v( sorted_kp_v.size(), 0 ); //< track which variables have been used

    // flag for spacepoints we used up
    std::vector<int> sp_used_v( ev_larflowhits->size(), 0 );

    larlite::event_track* evout_track
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "track2kp" );

    larlite::event_larflowcluster* evout_cluster
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, "track2kp" );


    std::set< std::pair<int,int> > completed_pairs;
    
    for ( int ipair=0; ipair<(int)sorted_pair_v.size(); ipair++ ) {

      auto const& apair = sorted_pair_v[ipair];
      if ( kp_used_v[ apair.start_idx ]!=0 || kp_used_v[ apair.end_idx ]!=0 ) continue;

      std::pair<int,int> iipair(apair.start_idx,apair.end_idx);
      if ( completed_pairs.find( iipair )!=completed_pairs.end() )
        continue;
           
      // attempt to reconstruct points
      std::cout << "-------------------------------------" << std::endl;
      std::cout << "[pair " << ipair << "] "
                << " KP[" << apair.start_idx << "] -> KP[" << apair.end_idx << "]"
                << " dist2axis=" << apair.dist2axis
                << " dist2pts=" << apair.dist2pts
                << std::endl;

      if ( (apair.start_idx==27 && apair.end_idx==22 )
           || (apair.start_idx==22 && apair.end_idx==27 ) )
        _tracker.set_verbosity(larcv::msg::kDEBUG);
      else
        _tracker.set_verbosity(larcv::msg::kINFO);
      
      larlite::track trackout;
      larlite::larflowcluster lfcluster;
      std::vector<int> trackpoints_v = makeTrack( kppos_v[apair.start_idx],
                                                  kppos_v[apair.end_idx],
                                                  *ev_larflowhits, badch_v,
                                                  trackout,
                                                  lfcluster,
                                                  sp_used_v );
      
      std::cout << "[pair " << ipair << "] returns track of " << trackpoints_v.size() << " spacepoints" << std::endl;
      if ( trackpoints_v.size()<=2 ) continue;
      std::cout << "[pair " << ipair << "] store track w/ " << lfcluster.size() << " hits" << std::endl;
      // block reverse KP direction
      completed_pairs.insert( iipair );

      evout_track->emplace_back( std::move(trackout) );
      evout_cluster->emplace_back( std::move(lfcluster) );

      //break;
    }//end of pair loop

    larlite::event_larflow3dhit* evout_unused
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, "track2kpunused");
    for ( size_t i=0; i<ev_larflowhits->size(); i++) {
      if ( sp_used_v[i]==0 )
        evout_unused->push_back( ev_larflowhits->at(i) );
    }
    LARCV_INFO() << "remaining track hits: " << evout_unused->size() << std::endl;
  }
  

  /**
   * Try to form a track following a path from the start to the end point
   *
   * @param[in]  startpt      Starting 3D point
   * @param[in]  endpt        Ending 3D point
   * @param[in]  lfhits       Container of larflow hits to use
   * @param[in]  badch_v      Image marking location of bad channels
   * @param[out] trackout     Track instance the method will populate
   * @param[out] lfclusterout LArFlowCluster instance to be filled
   * @param[out] sp_used_v    Flags indicating if larflow3dhit in lfhits was used
   */
  std::vector<int> TrackReco2KP::makeTrack( const std::vector<float>& startpt,
                                            const std::vector<float>& endpt,
                                            const larlite::event_larflow3dhit& lfhits,
                                            const std::vector<larcv::Image2D>& badch_v,
                                            larlite::track& trackout,
                                            larlite::larflowcluster& lfclusterout,
                                            std::vector<int>& sp_used_v )
  {

    LARCV_INFO() << "make track between: (" << startpt[0] << "," << startpt[1] << "," << startpt[2] << ") and "
                 << "(" << endpt[0] << "," << endpt[1] << "," << endpt[2] << ")"
                 << std::endl;
    
    // make bounding box for fast comparison
    float bounds[3][2];
    for (int i=0; i<3; i++) {
      bounds[i][0] = (startpt[i]<endpt[i]) ? startpt[i] : endpt[i];
      bounds[i][1] = (startpt[i]<endpt[i]) ? endpt[i]   : startpt[i];
      bounds[i][0] -= 30.0;
      bounds[i][1] += 30.0;
    }
    

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

    LARCV_DEBUG() << "RUN KPTrackFit w/ " << point_v.size() << " points in graph" << std::endl;

    //tracker.set_verbosity(larcv::msg::kDEBUG);
    std::vector<int> trackpoints_v = _tracker.fit( pos3d_v, badch_v, 0, (int)pos3d_v.size()-1 );
    LARCV_DEBUG() << "KPTrackFit return track with " << trackpoints_v.size() << std::endl;

    float tracklen = 0.;
    bool isgood = _isTrackGood( trackpoints_v, point_v, tracklen );
    if ( !isgood ) {
      std::vector<int> empty;
      return empty;
    }

    std::vector<int> sp_in_track_v( sp_used_v.size(),0);
    _prepareTrack( trackpoints_v, point_v, lfhits,
                   startpt, endpt,
                   trackout, lfclusterout,
                   sp_in_track_v );

    int npts = 0;
    for (auto& inout : sp_in_track_v )
      npts += inout;
    // last check on density
    LARCV_INFO() << "Track sp/len = " << float(npts)/tracklen << std::endl;

    // last chance to ditch track
    if ( float(npts)/tracklen < 10.0 ) {
      LARCV_INFO() << "track is too sparse" << std::endl;
      std::vector<int> empty;
      return empty;
    }

    for ( size_t i=0; i<sp_in_track_v.size(); i++ )
      if ( sp_in_track_v[i]==1 )
        sp_used_v[i] = 1;
    
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

  /**
   * 
   * prepare track and collect points
   *
   * @param[in]  trackpoints_v  Container with index of larflow hits, in lfhit_v, returned by the tracker
   * @param[in]  subset_v       Container of Point_t info for points bounded by the start and end points
   * @param[in]  lfhit_v        Container with all the larflow hits
   * @param[in]  start          3D starting point
   * @param[in]  end            3D ending point
   * @param[out] track          Output track whose info the method will populate
   * @param[out] lfcluster      Output larflow cluster which will be populated by method
   *
   */
  void TrackReco2KP::_prepareTrack( const std::vector<int>& trackpoints_v,
                                    const std::vector<Point_t>& subset_v,
                                    const larlite::event_larflow3dhit& lfhit_v,
                                    const std::vector<float>& start,
                                    const std::vector<float>& end,
                                    larlite::track& track,
                                    larlite::larflowcluster& lfcluster,
                                    std::vector<int>& sp_used_v )
  {

    lfcluster.clear();
    lfcluster.reserve( trackpoints_v.size() );
    track.reserve(trackpoints_v.size());

    const float boxsize = 3.0;
    
    std::set<int> points_saved;

    for ( int ipt=0; ipt<(int)trackpoints_v.size(); ipt++ ) {
      
      int subidx = trackpoints_v[ (int)trackpoints_v.size()-ipt-1 ];
      int idx = subset_v[subidx].idx;
      //LARCV_DEBUG() << " subidx=" << subidx << " start" << std::endl;
      
      if ( idx==-1 ) {
        TVector3 pt = { start[0], start[1], start[2] };
        track.add_vertex( pt );
      }
      else if ( idx==-2 ) {
        TVector3 pt = { end[0], end[1], end[2] };
        track.add_vertex( pt );
      }
      else {
        TVector3 pt = { lfhit_v[idx][0], lfhit_v[idx][1], lfhit_v[idx][2] };
        track.add_vertex( pt );
      }

      // collect points along track
      TVector3 ptdir = { 0, 0, 0 };

      if ( ipt>0 ) {

        //LARCV_DEBUG() << " collect points @pt=" << ipt << std::endl;
        
        ptdir = track.LocationAtPoint(ipt)-track.LocationAtPoint(ipt-1);
        float dist = ptdir.Mag();
        ptdir *= 1.0/dist;

        // collect points near here
        int nsteps = (int)(dist/boxsize)+1;
        float step = dist/(float)nsteps;

        // if last point, check at current (i.e. end) point
        if ( ipt+1==(int)trackpoints_v.size() )
          nsteps++;

        // check for points in steps
        for (int istep=0; istep<nsteps; istep++) {
          std::vector<float> center(3);
          for (int v=0; v<3; v++)
            center[v] = track.LocationAtPoint(ipt-1)[v] + step*ptdir[v];
        
          float bounds[3][2];
          for (int v=0; v<3; v++) {
            bounds[v][0] = center[v]-boxsize/2.0;
            bounds[v][1] = center[v]+boxsize/2.0;
          }
        
          for ( auto const& pt : subset_v ) {

            // check if we've already save the point
            if ( points_saved.find(pt.idx)!=points_saved.end() )
              continue;
            
            bool inbox = true;
            for (int v=0; v<3; v++) {
              if ( pt.pos[v]<bounds[v][0] || pt.pos[v]>bounds[v][1] )
                inbox = false;
              if ( !inbox ) break;
            }

            bool incylinder = false;
            if ( inbox ) {
              float axis_dist = pointLineDistance( start, end, pt.pos );
              if ( axis_dist<30 )
                incylinder = true;
            }

            if ( inbox && incylinder ) {
              //LARCV_DEBUG() << "save points idx=" << pt.idx << std::endl;
              points_saved.insert(pt.idx);
              if ( pt.idx==-1 ) {
                larlite::larflow3dhit hit;
                hit.resize(4,1.0);
                for (int v=0; v<3; v++ ) hit[v] = start[v];
                lfcluster.emplace_back( std::move(hit) );
              }
              else if ( pt.idx==-2 ) {
                larlite::larflow3dhit hit;
                hit.resize(4,1.0);
                for (int v=0; v<3; v++ ) hit[v] = end[v];
                lfcluster.emplace_back( std::move(hit) );                
              }
              else {
                lfcluster.push_back( lfhit_v.at(pt.idx) );
                sp_used_v[pt.idx] = 1;                
              }
            }
          }//end of subset point loop
        }//end of step loop
        
        track.add_direction( ptdir );
        if ( ipt==1 ) {
          // we had skipped the first point, so add additional direction
          track.add_direction( ptdir );
        }
        
      } //end if not the first point

    }//end of loop over points returned by the track fitter

  }

  /**
   * evaluate if track is good. 
   * we look for large, sharp turns which is likely unphysical
   *
   * @param[in] track_idx_v Index of points that composes a track as made the KPTrackFit algo
   * @param[in] point_v     Struct of points we tried to fit. Made by the `makeTrack` method.
   * @return    bool        True if passes tests; False if does not.
   */
  bool TrackReco2KP::_isTrackGood( const std::vector<int>& track_idx_v,
                                   const std::vector<Point_t>& point_v,
                                   float& totaldist )
  {

    // we step through and find large jumps
    float maxdist = 0.;
    totaldist = 0.;

    std::vector<int> check_at_idx;
    
    for (int i=0; i<(int)track_idx_v.size()-1; i++) {

      const std::vector<float>& pos  = point_v[ track_idx_v[i] ].pos;
      const std::vector<float>& next = point_v[ track_idx_v[i+1] ].pos;

      float dist = 0.;
      for (int v=0; v<3; v++) dist += (pos[v]-next[v])*(pos[v]-next[v]);
      dist = sqrt(dist);

      if ( maxdist<dist ) maxdist = dist;
      if ( dist>10.0 ) {
        check_at_idx.push_back( i );
      }

      totaldist += dist;
    }

    float gapfrac = 1.;
    if ( totaldist>0 )
      gapfrac = maxdist/totaldist;
    
    // if we have a gap large enough, we try to check the angle between the track and the large step
    float min_cosdir = 1.0;
    for (int icheck=0; icheck<(int)check_at_idx.size(); icheck++) {
      // we step from icheck+1 and go 5 cm. then we compute the angle
      float distpast = 0.;
      int itrack=check_at_idx[icheck]+1;
      std::vector<float> current_pos(3,0);
      while (distpast<10.0 && itrack+1<(int)track_idx_v.size()) {
        int idx1 = track_idx_v[itrack];
        int idx2 = track_idx_v[itrack+1];
        float segdist = 0.;
        for (int v=0; v<3; v++) {
          segdist += ( point_v[idx2].pos[v]-point_v[idx1].pos[v] )*( point_v[idx2].pos[v]-point_v[idx1].pos[v] );
        }
        current_pos = point_v[idx2].pos;
        segdist = sqrt(segdist);
        itrack++;
        distpast += segdist;
      }

      // hopefully we've gotten a new point, get the angle
      std::vector<float> dir1(3,0);
      std::vector<float> dir2(3,0);
      float cosdir = 0.;
      float len1 = 0;
      float len2 = 0.;
      int idx0 = track_idx_v[check_at_idx[icheck]];
      int idx1 = track_idx_v[check_at_idx[icheck]+1];
      for (int v=0; v<3; v++) {
        dir1[v] = point_v[idx1].pos[v]-point_v[idx0].pos[v];
        dir2[v] = current_pos[v] - point_v[idx1].pos[v];
        len1 += dir1[v]*dir1[v];
        len2 += dir2[v]*dir2[v];
        cosdir += dir1[v]*dir2[v];
      }
      len1 = sqrt(len1);
      len2 = sqrt(len2);
      if ( len1>0 && len2>0 ) {
        cosdir /= (len1*len2);
        if ( min_cosdir>cosdir )
          min_cosdir = cosdir;
      }
    }

    float pts_per_len = 0;
    if ( totaldist>0 )
      pts_per_len = float(track_idx_v.size())/totaldist;

    
    LARCV_INFO() << " max-gap=" << maxdist << " cm,"
                 << " totaldist=" << totaldist << " cm,"
                 << " gapfrac=" << gapfrac
                 << " npts/dist=" << pts_per_len
                 << " and min-cos=" << min_cosdir << std::endl;
    if ( min_cosdir<0.1 || maxdist>50.0 || gapfrac>0.5 ) {
      LARCV_INFO() << "track is bad" << std::endl;
      return false;
    }
    
    LARCV_INFO() << "track is good" << std::endl;
    return true;
    
  }
  
}
}
