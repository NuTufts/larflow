#include "NuVertexActivityReco.h"
#include "LArUtil/LArProperties.h"
#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "larflow/Reco/geofuncs.h"

namespace larflow {
namespace reco {

  /**
   * @brief process event
   * 
   * @param[in] iolcv larcv IO manager to get data from and to store event output
   * @param[in] ioll  larlite IO manager to get data  from and to store event output
   */
  void NuVertexActivityReco::process( larcv::IOManager& iolcv,
                                      larlite::storage_manager& ioll )
  {

    clear_ana_variables();
    vtxact_v.clear();
    _input_hit_v.clear();
    _input_hit_origin_v.clear();
    _event_cluster_v.clear();
    
    // cluster
    makeClusters( ioll, _event_cluster_v, 0.7 );
    
    // find hot points on cluster ends
    float va_threshold = 1000.0;    
    vtxact_v  = findVertexActivityCandidates( ioll, iolcv, _event_cluster_v, va_threshold );

    // split between points on shower ends and track ends
    larlite::event_larflow3dhit* evout_vacand
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _output_treename );
    larlite::event_pcaxis* evout_pca
      = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, _output_treename );
    
    for (size_t iv=0; iv<vtxact_v.size(); iv++ ) {
      auto& va = vtxact_v[iv];
      checkWireCellCosmicMask( va, iolcv );
      analyzeVertexActivityCandidates( va, _event_cluster_v, ioll, iolcv, 10.0 );
      analyzeAttachedCluster( va, _event_cluster_v, ioll, iolcv );

      // hack: jam direction into lfhit definition
      int n = (int)va.lfhit.size();
      va.lfhit.resize(n+3,0);
      for (int i=0; i<3; i++)
        va.lfhit[n+i] =  va.va_dir[i];
      va.lfhit[3] = (int)larflow::kVertexActivity;
        
      // store pca
      const larflow::reco::cluster_t* cluster = va.pattached;
      larlite::pcaxis va_pca = larflow::reco::cluster_make_pcaxis( *cluster, iv );
      
      evout_vacand->push_back( va.lfhit );
      evout_pca->push_back( va_pca );
    }
    if ( _va_ana_tree && _kown_tree )
      _va_ana_tree->Fill();
          
  }

  /**
   * @brief clear variable containers to be stored in ROOT tree
   */
  void NuVertexActivityReco::clear_ana_variables()
  {
    pca_dir_vv.clear();
    nbackwards_shower_pts.clear();
    nbackwards_track_pts.clear();
    nforwards_shower_pts.clear();
    nforwards_track_pts.clear();
    npix_on_cosmic_v.clear();
    attcluster_nall_v.clear();
    attcluster_nshower_v.clear();
    attcluster_ntrack_v.clear();
    ntrue_nupix_v.clear();    
    dist_closest_forwardshower.clear();
    shower_likelihood.clear();
    dist2truescevtx.clear();
    min_dist2truescevtx = 1000.0;
  }

  /**
   * @brief make clusters to search for nu vertex activity
   *
   * Collects larflow3dhits from trees use names are stored in _input_hittree_list.
   * Hits are clustered using larflow::reco::cluster_sdbscan_larflow3dhits.
   *
   * @param[in] ioll larlite::storage_manager containing event data.
   * @param[out] cluster_v Output container to be filled by function.
   * @param[in] larmatch_threshold Minimum larmatch score to include hit.
   */
  void NuVertexActivityReco::makeClusters( larlite::storage_manager& ioll,
                                           std::vector<larflow::reco::cluster_t>& cluster_v,
                                           const float larmatch_threshold )
  {

    // we have to collect the hits into a single container
    // we record a map back to the original hit container, if needed
    _input_hit_v.clear();
    _input_hit_origin_v.clear();
    
    for ( size_t ilist=0; ilist<_input_hittree_list.size(); ilist++ ) {
      
      auto const& hit_tree_name = _input_hittree_list[ilist];
      
      larlite::event_larflow3dhit* ev_lm
        = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit,hit_tree_name);
      LARCV_INFO() << "Number of input larflow hits from " << hit_tree_name << ": " << ev_lm->size() << std::endl;
    
      _input_hit_v.reserve( _input_hit_v.size() + ev_lm->size() );
      for ( size_t idx=0; idx<ev_lm->size(); idx++ ) {
        auto const& hit = (*ev_lm)[idx];
        if ( hit[9]>larmatch_threshold ) {
          _input_hit_v.push_back( hit );
          _input_hit_v.back().idxhit = (int)idx; // refers back to original hit vectr
          _input_hit_origin_v[ (int)_input_hit_v.size()-1 ] = (int)ilist;
        }
      }
    }
    
    LARCV_INFO() << "Number of collected larmatch hits: " << _input_hit_v.size() << std::endl;

    cluster_v.clear();
    larflow::reco::cluster_sdbscan_larflow3dhits( _input_hit_v, cluster_v, 2.5, 5, 20 );
    larflow::reco::cluster_runpca( cluster_v );
    // reindex back to original hit vector
    for ( size_t c=0; c<cluster_v.size(); c++ ) {
      auto& cluster = cluster_v[c];
      for ( auto& ih : cluster.hitidx_v ) {
        _input_hit_v.at(ih).trackid = (int)c; // assign hit to index of cluster vector
      }
    }
    
    LARCV_INFO() << "Number of clusters made from input hits: " << cluster_v.size() << std::endl;

    // collect input clusters
    for (size_t ic=0; ic<(int)_input_clustertree_list.size(); ic++) {
      std::string cluster_tree_name = _input_clustertree_list[ic];
      larlite::event_larflowcluster* ev_in_cluster
        = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, cluster_tree_name );
      // re-constitute cluster objects, add hits to input collection
      for ( auto const& lfcluster : *ev_in_cluster ) {
        larflow::reco::cluster_t clust = larflow::reco::cluster_from_larflowcluster( lfcluster );
        for ( size_t ihit=0; ihit<clust.hitidx_v.size(); ihit++ ) {
          larlite::larflow3dhit chit = lfcluster.at(ihit);
          chit.trackid = cluster_v.size();
          chit.idxhit = -1;
          clust.hitidx_v[ihit] = _input_hit_v.size();
          _input_hit_v.push_back( chit );
        }
        cluster_v.push_back( clust );
      }
    }

    LARCV_INFO() << "Added clusters from input list. Num clusters=" << cluster_v.size() << " Num Hits=" << _input_hit_v.size() << std::endl;
    
  }

  /** 
   * @brief Search spacepoints to create Vertex Activity Candidates
   *
   * Inputs used from larlite and larcv storage managers:
   * @verbatim embed:rst:leading-asterisk
   *  * Image2D containing pixel values (e.g. "wire")
   *  * No larlite products.
   * @endverbatim
   * 
   * To get charge of each spacepoint, projects down into image.
   * High charge spacepoints are only accepted if they are within 3 cm of end of clusters.
   * There is also a 20% consistancey requirement between the two planes with the highest pixel sum.
   * 
   * @param[in] ioll larlite::storage_manager with event data
   * @param[in] iolcv larcv::IOManager with event data
   * @param[in] cluster_v Clusters of space points
   * @param[in] va_threshold Minimum pixel sum threshold around vertex to create vertex activity candidate.
   * @return Container of Vertex Activity candidates.
   */
  std::vector<larflow::reco::NuVertexActivityReco::VACandidate_t>
  NuVertexActivityReco::findVertexActivityCandidates( larlite::storage_manager& ioll,
                                                      larcv::IOManager& iolcv,
                                                      std::vector<larflow::reco::cluster_t>& cluster_v,
                                                      const float va_threshold )
  {

    // for each cluster, look for high energy deposit regions
    // if find one, calculate its position on the pca-axis. is it at the end?

    larcv::EventImage2D* ev_img
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wire");
    auto const& adc_v = ev_img->as_vector();
    int nplanes = adc_v.size();

    std::vector<VACandidate_t> va_candidate_v;
    
    for ( size_t idx_cluster=0; idx_cluster<cluster_v.size(); idx_cluster++ ) {

      auto& cluster = cluster_v[idx_cluster];
      
      // loop over hits, find high charge regions
      std::vector<int> index_v;
      
      for ( size_t ihit=0; ihit<cluster.hitidx_v.size(); ihit++ ) {
        // get larflow3dhit
        auto& lfhit = _input_hit_v.at( cluster.hitidx_v[ihit] );

        std::vector<float> pixsum = calcPlanePixSum( lfhit, adc_v );

        // now we can decide
        float med_q = pixsum[nplanes+1];
        float consistency = pixsum[nplanes];
        if ( consistency>0 && consistency<0.2 && med_q>va_threshold ) {
          index_v.push_back( cluster.hitidx_v[ihit] );
        }
      }//end of hit loop

      LARCV_DEBUG() << "num high-q regions on cluster = " << index_v.size() << std::endl;
      
      // for high charge regions, we check if its on the edge of the pca.
      int min_start_idx = -1;
      int min_end_idx   = -1;
      float min_start_dist = 1e9;
      float min_end_dist   = 1e9;
      for (auto& idx : index_v ) {
        auto& lfhit = _input_hit_v.at(idx);

        float start_dist = 0.;
        float end_dist   = 0.;
        for (int i=0; i<3; i++) {
          start_dist += ( lfhit[i]-cluster.pca_ends_v[0][i] )*( lfhit[i]-cluster.pca_ends_v[0][i] );
          end_dist   += ( lfhit[i]-cluster.pca_ends_v[1][i] )*( lfhit[i]-cluster.pca_ends_v[1][i] );
        }

        start_dist = sqrt(start_dist);
        end_dist   = sqrt(end_dist);

        if ( start_dist<min_start_dist ) {
          min_start_dist = start_dist;
          min_start_idx  = idx;
        }
        if ( end_dist<min_end_dist ) {
          min_end_dist = end_dist;
          min_end_idx  = idx;
        }
        
      }//end of loop over high charge locations

      if ( min_start_idx>=0 )
        LARCV_DEBUG() << "closest distance high-q region to pca-end[0]=" << min_start_dist << std::endl;
      if ( min_end_idx>=0 )
        LARCV_DEBUG() << "closest distance high-q region to pca-end[1]=" << min_end_dist << std::endl;

      std::vector<float> va_dir(3,0);
      VACandidate_t va;
      va.pattached = &cluster;
      va.attached_cluster_index = (int)idx_cluster;
      
      if ( min_start_idx>=0 && min_start_dist<3.0 ) {
        float valen = 0.;
        for (int i=0; i<3; i++) {
          va_dir[i] = ( cluster.pca_ends_v[1][i] - cluster.pca_ends_v[0][i] );
          valen += va_dir[i]*va_dir[i];
        }
        valen = sqrt(valen);
        for (int i=0; i<3; i++)
          va_dir[i] /= valen;
        //vtxact_dir_v.push_back( va_dir );
        //va_candidate_v.push_back( (*ev_lm).at(min_start_idx) );

        va.va_dir = va_dir;        
        va.lfhit = _input_hit_v.at(min_start_idx);
        va.hit_index = (int)min_start_idx;
        va_candidate_v.push_back( va );
      }
      
      if ( min_end_idx>=0 && min_end_dist<3.0 ) {
        std::vector<float> va_dir(3,0);
        float valen = 0.;
        for (int i=0; i<3; i++) {
          va_dir[i] = ( cluster.pca_ends_v[0][i] - cluster.pca_ends_v[1][i] );
          valen += va_dir[i]*va_dir[i];
        }
        valen = sqrt(valen);
        for (int i=0; i<3; i++)
          va_dir[i] /= valen;

        va.va_dir = va_dir;        
        va.lfhit = _input_hit_v.at(min_end_idx);
        va.hit_index = (int)min_end_idx;
        va_candidate_v.push_back( va );
      }
      
    }//end of cluster loop

    LARCV_INFO() << "Number of high-charge regions on cluster ends: " << va_candidate_v.size() << std::endl;
    
    return va_candidate_v;
  }

  /**
   * @brief return the pixel sum and consistency measures
   *
   * The vector returned by the function contains the follow:
   * @verbatim embed:rst:leading-asterisk
   *  * [0] plane U sum
   *  * [1] plane V sum
   *  * [2] plane Y sum
   *  * [3] consistency: |difference|/average of two larges sum
   *  * [4] average of two largest pixel sums from planes
   * @endverbatim
   *
   * The projected pixel in the plane for the spacepoint is found
   * using larlite::larflow3dhit::targetwire and larlite::larflow3dhit::tick.
   *
   * @param[in] hit Spacepoint to test.
   * @param[in] adc_v Container with wire signal Image2D.
   * @return vector containing (plane U sum, plane V sum, plane Y sum, consistency, average).
   */
  std::vector<float> NuVertexActivityReco::calcPlanePixSum( const larlite::larflow3dhit& hit,
                                                            const std::vector<larcv::Image2D>& adc_v )
  {

    std::vector<float> pixelsum(adc_v.size(),0.0);
    
    std::vector<int> imgcoord(4); // (c1, c2, c3, row)
    for (int p=0; p<3; p++) {
      imgcoord[p] = hit.targetwire[p];
      pixelsum[p] = 0.; // clear pixel sum
    }
    imgcoord[3] = adc_v[0].meta().row( hit.tick );

    // pixelsum loop
    for (int dr=-3; dr<=3; dr++) {
      int row = imgcoord[3]+dr;
      if ( row<0 || row>=adc_v[0].meta().rows() )
        continue;
      
      for (int p=0; p<3; p++) {
        for (int dc=-3; dc<=3; dc++) {
          int col = imgcoord[p]+dc;
          if ( col<0 || col>=adc_v[p].meta().cols() )
            continue;
          
          if ( adc_v[p].pixel(row,col)>10.0 )
            pixelsum[p] += adc_v[p].pixel(row,col,__FILE__,__LINE__);
        }
      }//end of plane loop
    }//end of row loop
    
    std::vector<float> q_v = pixelsum; // copy
    std::sort(q_v.begin(),q_v.end());

    float sumconsistency = -1.0;
    if ( q_v[2]+q_v[1]>0 )
      sumconsistency = std::fabs((q_v[2]-q_v[1])/(0.5*(q_v[2]+q_v[1])));
    
    pixelsum.resize(adc_v.size()+2,0);
    pixelsum[(int)adc_v.size()]   = sumconsistency;
    pixelsum[(int)adc_v.size()+1] = 0.5*(q_v[1]+q_v[2]);
    return pixelsum;
  }

  /**
   * @brief Calculate selection variables for finding true Vertex Activity candidates
   *
   * @param[in] vacand The vertex activity candidates.
   * @param[in] cluster_v Spacepoint clusters. Must be the same ones used to find vertex candidates.
   * @param[in] ioll larlite::storage_manager with event data.
   * @param[in] iolcv larcv::IOManager with event data.
   * @param[in] min_dist2cluster Minimum distance to nearby clusters to include in calculations.
   */
  void NuVertexActivityReco::analyzeVertexActivityCandidates( larflow::reco::NuVertexActivityReco::VACandidate_t& vacand,
                                                              std::vector<larflow::reco::cluster_t>& cluster_v,
                                                              larlite::storage_manager& ioll,
                                                              larcv::IOManager& iolcv,
                                                              const float min_dist2cluster )
  {

    // track/shower images
    std::vector< larcv::EventImage2D* > ev_spuburn_v(3,0);
    std::vector< const larcv::Image2D* > ssnet_v(3,0);
    for (int p=0; p<3; p++ ) {
      char tname[50];
      sprintf( tname, "ubspurn_plane%d", p );
      ev_spuburn_v[p] = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, tname );
      ssnet_v[p] = &(ev_spuburn_v[p]->as_vector()[0]);
    }
    
    
    // for each vertex candidate, we calculate the following metrics
    int nback_shower = 0; // number of shower pixels in backward direction
    int nback_track  = 0; // number of track pixels in backward direction
    int nforw_shower = 0; // number of shower pixels in forward direction
    int nforw_track  = 0; // number of track pixels in forward direction
    float d_closest_forw_shower = 1e9; // closest forward shower spacepoint
    float shw_ll = 0; // shower likelihood
    
    float min_d2 = min_dist2cluster*min_dist2cluster;
    std::vector< float > vapos = { vacand.lfhit[0],
                                   vacand.lfhit[1],
                                   vacand.lfhit[2] };
    
    for ( size_t icluster=0; icluster<cluster_v.size(); icluster++ ) {
      auto& cluster = cluster_v[icluster];
      std::vector<float> dist(3,0);

      for (int i=0; i<3; i++) {
        dist[0] += (cluster.pca_center[i]-vacand.lfhit[i])*(cluster.pca_center[i]-vacand.lfhit[i]);
        dist[1] += (cluster.pca_ends_v[0][i]-vacand.lfhit[i])*(cluster.pca_ends_v[0][i]-vacand.lfhit[i]);
        dist[2] += (cluster.pca_ends_v[1][i]-vacand.lfhit[i])*(cluster.pca_ends_v[1][i]-vacand.lfhit[i]);
      }

      std::sort( dist.begin(), dist.end() );

      if ( dist[0]>min_d2 ) {
        continue;
      }

      // analyze this cluster
      for ( size_t ihit=0; ihit<cluster.hitidx_v.size(); ihit++ ) {
        auto& hitidx = cluster.hitidx_v[ihit];
        auto& lmhit = _input_hit_v.at(hitidx);

        std::vector< float > hitpos(3,0);
        std::vector< float > vaend(3,0);
        for (int i=0; i<3; i++) {
          hitpos[i] = lmhit[i];
          vaend[i] = vapos[i] + 10.0*vacand.va_dir[i];
        }        
        float rad = larflow::reco::pointLineDistance3f( vapos, vaend, hitpos );
        float s   = larflow::reco::pointRayProjection3f( vapos, vacand.va_dir, hitpos );

        // accept within 45 degree cone
        if ( s==0 || rad/std::fabs(s)>0.707 )
          continue;
        
        // check if in bounds
        if ( lmhit.tick<=ssnet_v[0]->meta().min_y() || lmhit.tick>=ssnet_v[0]->meta().max_y() )
          continue;

        // get shower prob from ssnet images
        
        std::vector<float> shower_prob(3,0);        
        int row = ssnet_v[0]->meta().row( lmhit.tick, __FILE__, __LINE__ );
        for (int p=0; p<3; p++) {
          if ( lmhit.targetwire[p]>(int)ssnet_v[p]->meta().min_x() && lmhit.targetwire[p]<(int)ssnet_v[p]->meta().max_x() ) {
            int col = ssnet_v[p]->meta().col( lmhit.targetwire[p], __FILE__, __LINE__ );
            shower_prob[p] = ssnet_v[p]->pixel( row, col, __FILE__, __LINE__ );
          }
        }
        std::sort( shower_prob.begin(), shower_prob.end() );
        
        float dist2hit = 0.;
        for (int i=0; i<3; i++) {
          dist2hit += ( lmhit[i]-vacand.lfhit[i] )*( lmhit[i]-vacand.lfhit[i] );
        }

        if ( s>=0 ) {
          // forward direction hit
          if ( shower_prob.back()>0.5 )
            nforw_shower++;
          else
            nforw_track++;
          if ( dist2hit<d_closest_forw_shower )
            d_closest_forw_shower = dist2hit;

          float maxr = s*1.414;
          if ( s>0 ) {
            shw_ll += rad/maxr;
          }
          
        }
        else {
          // backward direction hit
          if ( shower_prob.back()>0.5 )
            nback_shower++;
          else
            nback_track++;
        }              
      }
    }

    if ( nforw_shower>0 )
      shw_ll /= (float)nforw_shower;
    else
      shw_ll = 0.;
    
    // add va-candidate analysis variblaes to tree
    nbackwards_shower_pts.push_back( nback_shower );
    nbackwards_track_pts.push_back( nback_track );
    nforwards_shower_pts.push_back( nforw_shower );
    nforwards_track_pts.push_back( nforw_track );
    dist_closest_forwardshower.push_back( d_closest_forw_shower );
    shower_likelihood.push_back( shw_ll );
    
  }

  /**
   * @brief Calculate truth variables to aid in analysis of candidates
   *
   * @param[in] ioll larlite::storage_manager with event data.
   * @param[in] iolcv larcv::IOManager with event data.
   * @param[in] truedata Object containing truth info for event.
   */
  void NuVertexActivityReco::calcTruthVariables( larlite::storage_manager& ioll,
                                                 larcv::IOManager& iolcv,
                                                 const ublarcvapp::mctools::LArbysMC& truedata )
  {
    // get reconstructed va candidates
    larlite::event_larflow3dhit* evout_vacand
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "vacand");
    
    // get true vertex
    std::vector<float> vtx = { truedata._vtx_detx, truedata._vtx_sce_y, truedata._vtx_sce_z };
    // convert tick to x
    //const float cm_per_tick = larutil::LArProperties::GetME()->DriftVelocity()*0.5;
    //vtx[0] = (vtx[0]-3200.0)*cm_per_tick;

    float min_dist_2_true = -1.0;
    for (size_t iv=0; iv<evout_vacand->size(); iv++ ) {
      const larlite::larflow3dhit& lmhit = evout_vacand->at(iv);
      
      float dist2truth = 0;
      for (int i=0; i<3; i++) {
        dist2truth += ( lmhit[i]-vtx[i] )*( lmhit[i]-vtx[i] );
      }
      dist2truth = sqrt(dist2truth);
      LARCV_DEBUG() << "reco vtx activity [" << iv << "] dist-to-true-vtx=" << dist2truth << std::endl;
      if ( min_dist_2_true<0 || dist2truth<min_dist_2_true ) {
	min_dist_2_true = dist2truth;
      }
      dist2truescevtx.push_back( dist2truth );
    }
    min_dist2truescevtx = min_dist_2_true;
    LARCV_INFO() <<  "min distance to true vtx: " << min_dist2truescevtx << std::endl;

    calcTruthNeutrinoPixels( vtxact_v, iolcv );
    
  }

  /**
   * @brief Get the fraction of pixels near the vertex activity candidates that are on true neutrino pixels
   *
   * The function will look for the "segment" tree containing larcv::Image2D.
   * The content of the segment image tells which pixels are from neutrino pixels.
   *
   * @param[in] valist_v Container of vertex acitivity candidates.
   * @param[in] iolcv larcv::IOManager with event data.
   */
  void NuVertexActivityReco::calcTruthNeutrinoPixels( std::vector<VACandidate_t>& valist_v,
                                                      larcv::IOManager& iolcv )
  {

    const int dpix = 3;
    
    larcv::EventImage2D* ev_instance =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "segment" );
    auto const& seg_v = ev_instance->as_vector();

    for ( auto& va : valist_v ) {

      for (int i=0; i<4; i++)
        va.truth_num_nupix[i] = 0;
      
      int hitrow = seg_v[0].meta().row( va.lfhit.tick );
    
      for (int dr=-dpix; dr<=dpix; dr++) {
        int row = hitrow+dr;
        if ( row<0 || row>=(int)seg_v[0].meta().rows() )
          continue;

        for (int p=0; p<3; p++) {
        
          for (int dc=-dpix; dc<=dpix; dc++) {
            int col = va.lfhit.targetwire[p]+dc;
            if ( col<0 || col>=(int)seg_v[p].meta().cols() )
              continue;

            float segvalue = seg_v[p].pixel(row,col,__FILE__,__LINE__);

            if ( segvalue>0 ) {
              va.truth_num_nupix[p]++;
              va.truth_num_nupix[3]++;
            }
          }//end of col loop
        }//end of plane loop
      }//end of row loop
      
      ntrue_nupix_v.push_back( va.truth_num_nupix[3] );
    }//end of vertex activity list
  }
  

  /**
   * @brief check WireCell cosmics mask and tag
   *
   * Finds the number of pixels on each plane that are above ADC threshold of 10 and
   * are also tagged by the Wire-Cell out-of-time image.
   * Wire signal image tree name is assumed to be 'wire'.
   * WireCell out-of-time image is assumed to be 'thrumu'.
   * 
   * @param[in] va The vertex candidate.
   * @param[in] iolcv larcv::IOManager with event data.
   */
  void NuVertexActivityReco::checkWireCellCosmicMask( NuVertexActivityReco::VACandidate_t& va,
                                                      larcv::IOManager& iolcv ) {
    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    larcv::EventImage2D* ev_thrumu
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "thrumu" );

    auto const& adc_v    = ev_adc->as_vector();
    auto const& thrumu_v = ev_thrumu->as_vector();

    int npix_w_charge[3] = {0,0,0};
    int npix_on_cosmic[3] = {0,0,0};

    const int dpix = 3;

    int hitrow = adc_v[0].meta().row( va.lfhit.tick );
    
    for (int dr=-dpix; dr<=dpix; dr++) {
      int row = hitrow+dr;
      if ( row<0 || row>=(int)adc_v[0].meta().rows() )
        continue;

      for (int p=0; p<3; p++) {
        
        for (int dc=-dpix; dc<=dpix; dc++) {
          int col = va.lfhit.targetwire[p]+dc;
          if ( col<0 || col>=(int)adc_v[p].meta().cols() )
            continue;

          float pixadc = adc_v[p].pixel(row,col,__FILE__,__LINE__);
          float pixthrumu = thrumu_v[p].pixel(row,col,__FILE__,__LINE__);

          if ( pixadc>10.0 ) {
            npix_w_charge[p]++;
            if ( pixthrumu>10.0 )
              npix_on_cosmic[p]++;
          }
        }
      }
    }

    va.num_pix_on_thrumu[3] = 0;
    for (int p=0; p<3; p++) {
      va.num_pix_on_thrumu[p] = npix_on_cosmic[p];
      va.num_pix_on_thrumu[3] += npix_on_cosmic[p];
    }
    npix_on_cosmic_v.push_back( va.num_pix_on_thrumu[3] );
  }

  /**
   * @brief Make the ROOT tree that will store analysis variables for vertex activity candidates
   *
   * Will create a tree named "vtxactivityana" inside the active ROOT file.
   * Calls `bind_to_tree` to create branches.
   * Assumes ownership of the tree and will delete it when the destructor is called.
   */
  void NuVertexActivityReco::make_tree()
  {
    _va_ana_tree = new TTree("vtxactivityana", "Vertex Activity Analysis Tree");
    _kown_tree = true;
    bind_to_tree( _va_ana_tree );
  }

  /**
   * @brief Calculate metrics pertaining to the cluster the vertex activity candidate is within
   *
   * Fills variables within VACandidate_t. 
   * Also fills variable containers stored in the analysis tree.
   * 
   * @param[in] vacand The vertex activity candidate.
   * @param[in] cluster_v All clusters in the event.
   * @param[in] ioll larlite::storage_manager with event data.
   * @param[in] iolcv larcv::IOManager with event data.
   */
  void NuVertexActivityReco::analyzeAttachedCluster( larflow::reco::NuVertexActivityReco::VACandidate_t& vacand,
                                                     std::vector<larflow::reco::cluster_t>& cluster_v,
                                                     larlite::storage_manager& ioll,
                                                     larcv::IOManager& iolcv )
  {

    // track/shower images
    std::vector< larcv::EventImage2D* > ev_spuburn_v(3,0);
    std::vector< const larcv::Image2D* > ssnet_v(3,0);
    for (int p=0; p<3; p++ ) {
      char tname[50];
      sprintf( tname, "ubspurn_plane%d", p );
      ev_spuburn_v[p] = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, tname );
      ssnet_v[p] = &(ev_spuburn_v[p]->as_vector()[0]);
    }

    vacand.attclust_nallhits = 0;
    vacand.attclust_ntrackhits = 0;
    vacand.attclust_nshowerhits = 0;
    
    auto const& cluster = cluster_v[ vacand.attached_cluster_index ];
    for (int i=0; i<(int)cluster.hitidx_v.size(); i++) {
      auto const& lmhit = _input_hit_v[ cluster.hitidx_v[i] ];

      // check if in bounds
      if ( lmhit.tick<=ssnet_v[0]->meta().min_y() || lmhit.tick>=ssnet_v[0]->meta().max_y() )
        continue;

      // get shower prob from ssnet images
        
      std::vector<float> shower_prob(3,0);        
      int row = ssnet_v[0]->meta().row( lmhit.tick, __FILE__, __LINE__ );
      for (int p=0; p<3; p++) {
        if ( lmhit.targetwire[p]>(int)ssnet_v[p]->meta().min_x() && lmhit.targetwire[p]<(int)ssnet_v[p]->meta().max_x() ) {
          int col = ssnet_v[p]->meta().col( lmhit.targetwire[p], __FILE__, __LINE__ );
          shower_prob[p] = ssnet_v[p]->pixel( row, col, __FILE__, __LINE__ );
        }
      }
      std::sort( shower_prob.begin(), shower_prob.end() );

      vacand.attclust_nallhits++;      
      if( shower_prob.back()>=0.5 )
        vacand.attclust_nshowerhits++;
      else
        vacand.attclust_ntrackhits++;
      
    }//end of hit loop

    attcluster_nall_v.push_back( vacand.attclust_nallhits );
    attcluster_nshower_v.push_back( vacand.attclust_nshowerhits );
    attcluster_ntrack_v.push_back(  vacand.attclust_ntrackhits );    
    
  }
  

  /**
   * @brief Create branches that will store Vertex activity candidate metrics
   *
   */
  void NuVertexActivityReco::bind_to_tree( TTree* tree )
  {

    if ( !_va_ana_tree ) {
      _va_ana_tree = tree;
    }
    else {
      LARCV_CRITICAL() << "Pointer to Vertex Acticity already defined!" << std::endl;
      throw std::runtime_error( "[NuVertexActivityReco::bind_to_tree] Pointer to Vertex Acticity Ana tree already defined" );
    }

    _va_ana_tree->Branch( "pca_dir_vv", &pca_dir_vv );
    _va_ana_tree->Branch( "nbackwards_shower_pts_v", &nbackwards_shower_pts );
    _va_ana_tree->Branch( "nbackwards_track_pts_v", &nbackwards_track_pts );
    _va_ana_tree->Branch( "nforwards_shower_pts_v", &nforwards_shower_pts );
    _va_ana_tree->Branch( "nforwards_track_pts_v", &nforwards_track_pts );
    _va_ana_tree->Branch( "npix_on_cosmic_v", &npix_on_cosmic_v );
    _va_ana_tree->Branch( "attcluster_nall_v",    &attcluster_nall_v );
    _va_ana_tree->Branch( "attcluster_nshower_v", &attcluster_nshower_v );
    _va_ana_tree->Branch( "attcluster_ntrack_v",  &attcluster_ntrack_v );
    _va_ana_tree->Branch( "ntrue_nupix_v", &ntrue_nupix_v );
    _va_ana_tree->Branch( "dist_closest_forwardshower_v", &dist_closest_forwardshower );
    _va_ana_tree->Branch( "shower_likelihood_v", &shower_likelihood );
    _va_ana_tree->Branch( "dist2truescevtx_v", &dist2truescevtx );
    _va_ana_tree->Branch( "min_dist2truescevtx", &min_dist2truescevtx, "min_dist2truescevtx/F" );
    
  }


  
}
}
