#include "NuVertexShowerReco.h"

#include "geofuncs.h"
#include "cluster_functions.h"

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
                                    std::vector<NuVertexCandidate>& nu_candidate_v,
				    std::vector<ClusterBookKeeper>& nu_cluster_book_v )
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

    for ( size_t ivtx=0; ivtx<nu_candidate_v.size(); ivtx++) {
      auto& nuvtx = nu_candidate_v.at(ivtx);
      auto& book  = nu_cluster_book_v.at(ivtx);
      LARCV_DEBUG() << "Build Vertex Showers: (" << nuvtx.pos[0] << "," << nuvtx.pos[1] << "," << nuvtx.pos[2] << ")" << std::endl;
      _build_vertex_showers( nuvtx, book, iolcv, ioll );
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
						  ClusterBookKeeper& nuclusterbook,
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
    const float r_trunk   = 3.0;
    const float tau_startpt = 3.0; // cm

    const float max_showerpt_dist = 200.0;
    const float max_showerpt_d2 = max_showerpt_dist*max_showerpt_dist;
    
    std::vector<ProngRank_t> seed_rank_v; //< rank how we will seed the hits
    
    for ( int iprong=0; iprong<(int)nuvtx.cluster_v.size(); iprong++) {
        
      auto const& vtxcluster = nuvtx.cluster_v[iprong];

      // -log(exp[-r/tau]) = r/tau
      // only deal with showers
      if ( vtxcluster.type!=NuVertexCandidate::kShower && vtxcluster.type!=NuVertexCandidate::kShowerKP ) {
        continue;
      }

      bool found = false;
      //std::cout << "check seed: " << vtxcluster.producer << " " << vtxcluster.index << std::endl;
      for ( auto& seed : seed_rank_v ) {
        //std::cout << " past seed: " << seed.producer << " " << seed.container_idx << std::endl;
        if ( seed.producer==vtxcluster.producer && seed.container_idx==vtxcluster.index )
          found = true;
        if ( found )
          break;
      }

      if ( found ) {
        //std::cout << " cluster duplicate." << std::endl;
        continue;
      }

      const larlite::larflowcluster& lfcluster =
        ( (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, vtxcluster.producer))->at( vtxcluster.index );


      // define shower start, dir, ll-score
      std::vector<float> shower_start;
      std::vector<float> shower_dir;
      float shower_ll;
      _make_trunk_cand( nuvtx.pos,
                        lfcluster,
                        shower_start,
                        shower_dir,
                        shower_ll );

      std::vector<float> shower_end(3,0);
      for (int i=0; i<3; i++)
        shower_end[i] = shower_start[i] + 10.0*shower_dir[i];
      
      // // define shower axis -- start point to vertex
      std::vector<float> axis(3,0);
      float dist = 0.;
      for (int i=0; i<3; i++) {
        axis[i] = shower_start[i]-nuvtx.pos[i];
        dist += axis[i]*axis[i];
      }
      if ( dist>0 ) {
        dist = sqrt(dist);
        for (int i=0; i<3; i++)
          axis[i] /= dist;
      }

      // std::vector<float> axis_start(3,0);
      // std::vector<float> axis_end(3,0);      
      // for (int i=0; i<3; i++) {
      //   axis_start[i] = vtxcluster.pos[i];
      //   axis_end[i]   = vtxcluster.pos[i] + 30.0*axis[i];
      // }

      // float score_ll = 0;
      // // -log(P(r)*P(s))
      // // P(r): distance from axis falls off as exp
      // // P(s): projection along axis. exp penalty for being behind vertex
      // // only deal with showers
      
      // for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
      //   auto const& hit = lfcluster[ihit];
      //   std::vector<float> pt(3,0);
      //   for (int i=0; i<3; i++)
      //     pt[i] = hit[i];
        
      //   float r = pointLineDistance3f( axis_start, axis_end, pt );
      //   float s = pointRayProjection3f( axis_start, axis, pt );

      //   score_ll += r/r_mollier;
      //   if (s<0)
      //     score_ll += -s/tau_startpt;
      // }
      // if ( lfcluster.size()>0 )
      //   score_ll /= float(lfcluster.size());

      float score_ll = 0;
      if ( lfcluster.size()>10 ) {
        score_ll = dist;
      }
      else {
        score_ll = 100.0 + dist; // blerg
      }

      ProngRank_t rank( vtxcluster.producer, iprong, vtxcluster.index, score_ll );
      rank.dist2vtx = dist;
      rank.axis = shower_dir;
      rank.axis_start = shower_start;
      rank.axis_end   = shower_end;
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
      
      // absorb hits into shower_hit_v
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
      if ( max_gap_s<3.0 ) {
        for (auto& hit : trunk_hit_v )
          shower_hit_v.push_back(hit);
      }
      
      // absorb shower clusters within cone. we sample from all producers given, not just within vertex.
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

              // set max distance from prong start to the point in question
              float d2 = 0.;
              for (int i=0; i<3; i++)
                d2 += ( rankedprong.axis_start[i]-showerpt[i] )*( rankedprong.axis_start[i]-showerpt[i] );

              if ( s>0.0 && d2<max_showerpt_d2 ) {
                float rovers = r/s;
                //if ( rovers < 9.0/14.0 ) {
                if ( (s<5.0 && r<r_trunk) || (s>=5.0 && r<r_mollier ) ) {
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


      // get pca of shower
      larflow::reco::cluster_t shower_cluster_t = larflow::reco::cluster_from_larflowcluster(shower_hit_v);
      larflow::reco::cluster_pca( shower_cluster_t );
      larlite::pcaxis shower_hit_pca = larflow::reco::cluster_make_pcaxis( shower_cluster_t );
      
      larlite::track shower_trunk = larflow::reco::cluster_make_trunk( shower_cluster_t, nuvtx.pos );
    
      // larlite::track shower_trunk_dir;
      // shower_trunk_dir.add_vertex( TVector3(rankedprong.axis_start[0],
      //                                       rankedprong.axis_start[1],
      //                                       rankedprong.axis_start[2]) );
      // shower_trunk_dir.add_vertex( TVector3(rankedprong.axis_end[0],
      //                                       rankedprong.axis_end[1],
      //                                       rankedprong.axis_end[2]) );
      // double trunkDir[3] = { rankedprong.axis_end[0] - rankedprong.axis_start[0],
      //                        rankedprong.axis_end[1] - rankedprong.axis_start[1],
      //                        rankedprong.axis_end[2] - rankedprong.axis_start[2] };
      // double trunkMag = sqrt( pow(trunkDir[0],2) + pow(trunkDir[1],2) + pow(trunkDir[2],2) );
      // shower_trunk_dir.add_direction( TVector3(trunkDir[0]/trunkMag,
      //                                          trunkDir[1]/trunkMag,
      //                                          trunkDir[2]/trunkMag) );
      // shower_trunk_dir.add_direction( TVector3(trunkDir[0]/trunkMag,
      //                                          trunkDir[1]/trunkMag,
      //                                          trunkDir[2]/trunkMag) );

      
      // save shower to nuvtx candidate object
      nuvtx.shower_v.emplace_back( std::move(shower_hit_v) );
      nuvtx.shower_trunk_v.emplace_back( std::move(shower_trunk) );
      nuvtx.shower_pcaxis_v.emplace_back( std::move(shower_hit_pca) );

    }//end of seed prong loop

    // book the clusters we used
    // loop over pairs of (producer, used vector)
    for ( auto itc=cluster_used_v.begin(); itc!=cluster_used_v.end(); itc++ ) {
      for (size_t idx=0; idx<itc->second.size(); idx++) {
	if ( itc->second[idx]>0 ) {
	  // this cluster was used by this vertex
	  const larlite::larflowcluster& lfcluster =
	    ( (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, itc->first))->at( idx );
	  if ( lfcluster.matchedflash_idx>=0 && lfcluster.matchedflash_idx<nuclusterbook.cluster_status_v.size() ) {
	    nuclusterbook.cluster_status_v[ lfcluster.matchedflash_idx ] = 1; // book it!
	  }
	  else {
	    LARCV_WARNING() << "Used cluster index outside the cluster book range!" << std::endl;
	  }
	}
      }
    }
    
  }

  /**
   * @brief Define the shower trunk
   *
   * @param[in] pos Position of vertex.
   * @param[in] lfcluster Cluster to find trunk for.
   * @param[out] shower_start Start of defined shower trunk.
   * @param[out] shower_dir   Direction of defined shower trunk.
   * @param[out] shower_ll    Score for choosing best trunk for shower cluster.
   */
  void NuVertexShowerReco::_make_trunk_cand( const std::vector<float>& pos,
                                             const larlite::larflowcluster& lfcluster,
                                             std::vector<float>& shower_start,
                                             std::vector<float>& shower_dir,
                                             float& shower_ll )
  {

    // calculate distance to vertex for every hit in the cluster
    std::vector<float> dist2vertex(lfcluster.size(),0);
    float min_dist = 1e9; // the 
    for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
      float dist = 0.;
      for (int i=0; i<3; i++) {
        dist += (lfcluster[ihit][i]-pos[i])*(lfcluster[ihit][i]-pos[i]);
      }
      dist2vertex[ihit] = sqrt(dist);
      if ( min_dist>dist2vertex[ihit] )
        min_dist = dist2vertex[ihit];
    }

    std::vector< std::vector<float> > close_hit_v;
    close_hit_v.reserve( lfcluster.size() );
    
    for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
      if ( dist2vertex[ihit]-min_dist < 3.5 ) {
        std::vector<float> pt = { lfcluster[ihit][0], lfcluster[ihit][1], lfcluster[ihit][2] };
        close_hit_v.push_back( pt );
      }
    }

    std::vector<cluster_t> trunk_cand_v;
    larflow::reco::cluster_spacepoint_v( close_hit_v, trunk_cand_v );
    

    struct CandRank_t {
      int idx;
      float llscore;
      std::vector<float> start;
      std::vector<float> dir;
      CandRank_t( int ii, float ll )
        : idx(ii), llscore(ll)
      {};
      bool operator<( const CandRank_t& rhs ) {
        if ( llscore<rhs.llscore )
          return true;
        return false;
      }
    };

    std::vector< CandRank_t > rank_v;

    for ( int icluster=0; icluster<(int)trunk_cand_v.size(); icluster++) {
      
      auto& trunk = trunk_cand_v[icluster];
      if ( trunk.points_v.size()<5 ) {
        CandRank_t rank( icluster, 1e9 );
        rank_v.push_back( rank );
        continue;
      }

      larflow::reco::cluster_pca( trunk );

      // determine direction
      // we want to use the pca axis, but we can switch to vertex->centroid if the trunk is bad
      std::vector<float> vtx2centroid(3,0);
      std::vector<float> pca1(3,0);
      float lenv2c = 0.;
      float lenpca = 0.;
      float cos_pca_v2c = 0.;
      for (int i=0; i<3; i++) {
        vtx2centroid[i] = trunk.pca_center[i]-pos[i];
        lenv2c += vtx2centroid[i]*vtx2centroid[i];
        pca1[i] = trunk.pca_axis_v[0][i];
        lenpca += pca1[i]*pca1[i];
        cos_pca_v2c += pca1[i]*vtx2centroid[i];
      }
      lenv2c = sqrt(lenv2c);
      lenpca = sqrt(lenpca);
      if ( lenv2c>0 && lenpca>0 ) {
        for (int i=0; i<3; i++)
          vtx2centroid[i] /= lenv2c;
        for (int i=0; i<3; i++)
          pca1[i] /= lenpca;
        cos_pca_v2c /= (lenpca*lenv2c);
      }

      // get start point of pca line
      int pca_start;
      int pca_end;
      float pcaend_dist[2] = {0,0};
      for (int i=0; i<3; i++) {
        pcaend_dist[0] += (trunk.pca_ends_v[0][i]-pos[i])*(trunk.pca_ends_v[0][i]-pos[i]);
        pcaend_dist[1] += (trunk.pca_ends_v[1][i]-pos[i])*(trunk.pca_ends_v[1][i]-pos[i]);
      }
      if ( pcaend_dist[0]<pcaend_dist[1] ) {
        pca_start = 0;
        pca_end = 1;
      }
      else {
        pca_start = 1;
        pca_end = 0;
      }
      std::vector<float> pcadir(3,0);
      float len_pcadir = 0.;
      for (int i=0; i<3; i++) {        
        pcadir[i] = trunk.pca_ends_v[pca_end][i]-trunk.pca_ends_v[pca_start][i];
        len_pcadir += pcadir[i]*pcadir[i];
      }
      len_pcadir = sqrt(len_pcadir);
      for (int i=0; i<3; i++)
        pcadir[i] /= len_pcadir;
             

      float score_pca = 0.;
      float score_v2c = 0.;
      std::vector<float> v2c_start(3,0);
      float max_s_v2c = 1e9;
      for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
        std::vector<float> pt = { lfcluster[ihit][0], lfcluster[ihit][1], lfcluster[ihit][2] };

        // pca score
        float r_pca = larflow::reco::pointLineDistance3f( trunk.pca_ends_v[pca_start], trunk.pca_ends_v[pca_end], pt );
        float s_pca = larflow::reco::pointRayProjection3f( trunk.pca_ends_v[pca_start], pcadir, pt );
        if ( s_pca>3.0 )
          score_pca += r_pca/( (s_pca/14.0)*9.0 );
        else if (s_pca>0.0 && s_pca<3.0 )
          score_pca += r_pca/1.0;
        else
          score_pca += -s_pca/1.0;

        // v2c score
        float r_v2c = larflow::reco::pointLineDistance3f( pos, trunk.pca_center, pt );
        float s_v2c = larflow::reco::pointRayProjection3f( pos, vtx2centroid, pt )-min_dist;

        if ( s_v2c<max_s_v2c ) {
          max_s_v2c = s_v2c;
          v2c_start = pt;
        }
        
        if ( s_v2c>3.0 )
          score_v2c += r_v2c/( (s_v2c/14.0)*9.0 );
        else if (s_v2c>0.0 && s_v2c<3.0 )
          score_v2c += r_v2c/1.0;
        else
          score_v2c += -s_v2c/1.0;
        
      }

      if ( fabs(cos_pca_v2c)>0.7 ) {
        // use the pca score
        CandRank_t rank( icluster, score_pca );
        rank.start = trunk.pca_ends_v[pca_start];
        rank.dir   = pcadir;
        rank_v.push_back( rank );
      }
      else {
        CandRank_t rank( icluster, score_v2c );
        rank.start = v2c_start;
        rank.dir   = vtx2centroid;
        rank_v.push_back( rank );
      }
      
      
    }//loop over trunk candidates


    std::sort( rank_v.begin(), rank_v.end() );

    shower_start = rank_v.front().start;
    shower_dir   = rank_v.front().dir;
    shower_ll    = rank_v.front().llscore;

  }

  
}
}
