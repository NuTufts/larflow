#include "NuVertexShowerReco.h"

#include "geofuncs.h"

namespace larflow {
namespace reco {

  /**
   * @brief process data from one event
   *
   * @param[in] iolcv LArCV IOManager containing event data
   * @param[in] ioll  larlite storage_manager containing event data
   * @param[inout] nu_candidate_v List of neutrino vertex candidates to which we will append shower objects
   * 
   */
  void NuVertexShowerReco::process( larcv::IOManager& iolcv,
                                    larlite::storage_manager& ioll,
                                    std::vector<NuVertexCandidate>& nu_candidate_v )
  {

    // load up the clusters
    LARCV_INFO() << "Number of cluster producers: " << _cluster_producers.size() << std::endl;
    for ( auto it=_cluster_producers.begin(); it!=_cluster_producers.end(); it++ ) {
      LARCV_INFO() << "Load cluster data with tree name[" << it->first << "]" << std::endl;
      it->second = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, it->first );
      auto it_pca = _cluster_pca_producers.find( it->first );
      if ( it_pca==_cluster_pca_producers.end() ) {
        _cluster_pca_producers[it->first] = nullptr;
        it_pca = _cluster_pca_producers.find( it->first );
      }
      it_pca->second = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, it->first );
      LARCV_INFO() << "clusters from [" << it->first << "]: " << it->second->size() << " clusters" << std::endl;
    }
    
    for ( auto& nuvtx : nu_candidate_v ) {
      LARCV_DEBUG() << "Build Vertex Showers: (" << nuvtx.pos[0] << "," << nuvtx.pos[1] << "," << nuvtx.pos[2] << ")" << std::endl;
      _build_vertex_showers( nuvtx, iolcv, ioll );
    }
    
  }

  /**
   * @brief [internal] build showers for given neutrino candidate vertex
   *
   * we loop through shower prongs associated to the vertex.
   * first is to look for mislabeled track components along line between vertex and shower start.
   * then we absorb nearby fragments within a cone of the closest fragment
   * 
   * @param[in] nuvtx Neutrino candidate vertex
   * @param[in] iolcv LArCV Event data
   * @param[in] ioll  larlite event data
   *
   */
  void NuVertexShowerReco::_build_vertex_showers( NuVertexCandidate& nuvtx,
                                                  larcv::IOManager& iolcv, 
                                                  larlite::storage_manager& ioll ) 
  {

    // we want to sort seeding priority
    struct ProngRank_t {
      std::string producer;
      int prong_idx;
      int container_idx;
      float score;
      float dist2vtx;
      std::vector<float> axis;
      std::vector<float> axis_start;
      std::vector<float> axis_end;
      ProngRank_t( std::string p, int pi, int ci, float s )
        : producer(p), prong_idx(pi), container_idx(ci), score(s)
      {};
      bool operator<( const ProngRank_t& rhs ) {
        // threshold on hits, else rank on hits        
        if ( score<rhs.score ) return true;
        return false;
      };
    };

    const float r_mollier = 9.04; // cm, liquid argon
    const float tau_startpt = 3.0; // cm
    
    std::vector<ProngRank_t> seed_rank_v; //< rank how we will seed the hits
    
    for ( int iprong=0; iprong<(int)nuvtx.cluster_v.size(); iprong++) {
        
      auto const& vtxcluster = nuvtx.cluster_v[iprong];

      // -log(exp[-r/tau]) = r/tau
      // only deal with showers
      if ( vtxcluster.type!=NuVertexCandidate::kShower && vtxcluster.type!=NuVertexCandidate::kShowerKP ) {
        continue;
      }

      bool found = false;
      std::cout << "check seed: " << vtxcluster.producer << " " << vtxcluster.index << std::endl;
      for ( auto& seed : seed_rank_v ) {
        std::cout << " past seed: " << seed.producer << " " << seed.container_idx << std::endl;
        if ( seed.producer==vtxcluster.producer && seed.container_idx==vtxcluster.index )
          found = true;
        if ( found )
          break;
      }

      if ( found ) {
        std::cout << " cluster duplicate." << std::endl;
        continue;
      }

      const larlite::larflowcluster& lfcluster =
        ( (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, vtxcluster.producer))->at( vtxcluster.index );


      // define shower axis -- start point to vertex
      std::vector<float> axis(3,0);
      float dist = 0.;
      for (int i=0; i<3; i++) {
        axis[i] = vtxcluster.pos[i]-nuvtx.pos[i];
        dist += axis[i]*axis[i];
      }
      if ( dist>0 ) {
        dist = sqrt(dist);
        for (int i=0; i<3; i++)
          axis[i] /= dist;
      }

      std::vector<float> axis_start(3,0);
      std::vector<float> axis_end(3,0);      
      for (int i=0; i<3; i++) {
        axis_start[i] = vtxcluster.pos[i];
        axis_end[i]   = vtxcluster.pos[i] + 30.0*axis[i];
      }

      float score_ll = 0;
      // -log(P(r)*P(s))
      // P(r): distance from axis falls off as exp
      // P(s): projection along axis. exp penalty for being behind vertex
      // only deal with showers
      
      for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
        auto const& hit = lfcluster[ihit];
        std::vector<float> pt(3,0);
        for (int i=0; i<3; i++)
          pt[i] = hit[i];
        
        float r = pointLineDistance3f( axis_start, axis_end, pt );
        float s = pointRayProjection3f( axis_start, axis, pt );

        score_ll += r/r_mollier;
        if (s<0)
          score_ll += -s/tau_startpt;
      }
      if ( lfcluster.size()>0 )
        score_ll /= float(lfcluster.size());

      if ( lfcluster.size()>10 ) {
        score_ll = dist;
      }
      else {
        score_ll = 30.0 + dist; // blerg
      }

      ProngRank_t rank( vtxcluster.producer, iprong, vtxcluster.index, score_ll );
      rank.dist2vtx = dist;
      rank.axis = axis;
      rank.axis_start = axis_start;
      rank.axis_end   = axis_end;
      seed_rank_v.push_back( rank );
    }

    std::sort( seed_rank_v.begin(), seed_rank_v.end() );

    // now we can begin to build out a shower
    // we need to track which clusters we used up
    std::vector<int> prong_used_v( nuvtx.cluster_v.size(), 0 );
    // the other clusters
    std::map< std::string, std::vector<int> > cluster_used_v;
    for ( auto it=_cluster_producers.begin(); it!=_cluster_producers.end(); it++  ) {
      if ( it->second!=nullptr )
        cluster_used_v[it->first] = std::vector<int>( it->second->size(), 0 );
    }

    // loop through the seeds and start building showers
    for ( auto const& rankedprong : seed_rank_v) {

      int prongidx = rankedprong.prong_idx;
      
      // if prong used skip it
      if ( prong_used_v[prongidx]!=0 )
        continue;

      // use the cluster to seed
      auto const& vtxcluster = nuvtx.cluster_v[prongidx];

      // check we havent already absorbed the cluster already
      if ( cluster_used_v.find( vtxcluster.producer )!=cluster_used_v.end() ) {
        if ( cluster_used_v[vtxcluster.producer].at( vtxcluster.index )!=0 )
          continue;
      }
      // using cluster as seed
      cluster_used_v[vtxcluster.producer][vtxcluster.index] = 1;
      
      const larlite::larflowcluster& lfcluster =
        ( (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, vtxcluster.producer))->at( vtxcluster.index );

      LARCV_DEBUG() << "RankedProng[" << vtxcluster.producer << "," << rankedprong.container_idx << ",prong " << prongidx << "] "
                    << " score=" << rankedprong.score
                    << " npts=" << lfcluster.size()
                    << std::endl;
      
      
      larlite::larflowcluster shower_hit_v;
      for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
        shower_hit_v.push_back( lfcluster[ihit] );
      }

      // loop over trunk clusters, find those along the shower axis.
      // we are assuming this is ssnet mislabeling
      int ntrunk_hits_added = 0;
      std::vector<float> track_s_v;
      larlite::larflowcluster trunk_hit_v;
      for ( auto it=_cluster_producers.begin(); it!=_cluster_producers.end(); it++ ) {
        if ( _cluster_type[it->first]==NuVertexCandidate::kTrack ) {
          // loop over track cluster in this event container
          for ( auto const& track_lfcluster : *it->second ) {
            // track cluster
            int nclose_to_axis = 0;
            for (auto const& trackhit : track_lfcluster ) {
              std::vector<float> trackpt = { (float)trackhit[0], (float)trackhit[1], (float)trackhit[2] };
              float r = pointLineDistance3f(  rankedprong.axis_start, rankedprong.axis_end, trackpt );
              float s = pointRayProjection3f( rankedprong.axis_start, rankedprong.axis, trackpt );
              float vtxdist = 0.;
              for (int i=0; i<3; i++)
                vtxdist += ( trackpt[i]-nuvtx.pos[i] )*( trackpt[i]-nuvtx.pos[i] );

              if ( s>-rankedprong.dist2vtx && s<0 && ((vtxdist<1.0 && r<0.5) || (vtxdist>0.0 && r<2.0)) )  {
                trunk_hit_v.push_back( trackhit );

                float trunk_s = pointRayProjection3f( nuvtx.pos, rankedprong.axis, trackpt );
                track_s_v.push_back( trunk_s );
                
                ntrunk_hits_added++;
              }
              
            }
            
          }          
          
        }
      }
      track_s_v.push_back( 0 );      
      track_s_v.push_back( rankedprong.dist2vtx );
      std::sort( track_s_v.begin(), track_s_v.end() );
      float max_gap_s = 0;
      for (int i=1; i<(int)track_s_v.size(); i++) {
        //std::cout << " gap: " << track_s_v[i] << "-" << track_s_v[i-1] << " = " << fabs(track_s_v[i]-track_s_v[i-1]) << std::endl;
        if ( max_gap_s < fabs(track_s_v[i]-track_s_v[i-1]) )
          max_gap_s = fabs(track_s_v[i]-track_s_v[i-1]);
      }
      LARCV_DEBUG() << "Number of trunk hits found: " << ntrunk_hits_added << " max gap=" << max_gap_s << std::endl;      
      if ( max_gap_s<10.0 ) {
        for (auto& hit : trunk_hit_v )
          shower_hit_v.push_back(hit);
      }
      



      // absorb shower clusters within cone
      for ( auto it=_cluster_producers.begin(); it!=_cluster_producers.end(); it++ ) {

        auto const& cluster_type = _cluster_type[it->first];
        if ( cluster_type==NuVertexCandidate::kShowerKP ||
             cluster_type==NuVertexCandidate::kShower ) {

          for ( int ishower=0; ishower< (*it->second).size(); ishower++ ) {

            // don't absorb points from seeding cluster
            if ( vtxcluster.producer==it->first && vtxcluster.index==ishower )
              continue;

            // don't absorb points from previous used cluster
            if ( cluster_used_v[it->first][ishower]!=0 )
              continue;

            // get the cluster
            auto const& shower_lfcluster = (*it->second).at(ishower);

            // skip zero clusters
            if ( shower_lfcluster.size()==0 )
              continue;
            
            // make sure its not the clsuter we are using as the seed
            int nhits_within_cone = 0;
            for ( auto const& showerhit : shower_lfcluster ) {
              std::vector<float> showerpt = { showerhit[0], showerhit[1], showerhit[2] };
              float r = pointLineDistance3f(  rankedprong.axis_start, rankedprong.axis_end, showerpt );
              float s = pointRayProjection3f( rankedprong.axis_start, rankedprong.axis, showerpt );

              if ( s>0 ) {
                float rovers = r/s;
                if ( rovers < 9.0/14.0 ) {
                  // mollier/radiation length
                  nhits_within_cone++;
                }
              }
            }

            float frac_within_cone = nhits_within_cone/float(shower_lfcluster.size());
            if ( frac_within_cone>0.5 ) {
              // add the shower cluster
              for ( auto const& showerhit : shower_lfcluster )
                shower_hit_v.push_back( showerhit );
              cluster_used_v[it->first][ishower] =  1;
            }
          }//end of shower cluster loop for given producer
        }//end of if cluster is shower type
      }//loop over producers to build showers
      
      larlite::track shower_trunk_dir;
      shower_trunk_dir.add_vertex( TVector3(rankedprong.axis_start[0],
                                            rankedprong.axis_start[1],
                                            rankedprong.axis_start[2]) );
      shower_trunk_dir.add_vertex( TVector3(rankedprong.axis_end[0],
                                            rankedprong.axis_end[1],
                                            rankedprong.axis_end[2]) );
      shower_trunk_dir.add_direction( TVector3(0,0,0) );
      shower_trunk_dir.add_direction( TVector3(0,0,0) );          
      
      // save shower to nuvtx candidate object
      nuvtx.shower_v.emplace_back( std::move(shower_hit_v) );
      nuvtx.shower_trunk_v.emplace_back( std::move(shower_trunk_dir) );

    }//end of seed prong loop      
    
  }
  
}
}
