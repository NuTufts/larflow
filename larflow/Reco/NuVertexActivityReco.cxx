#include "NuVertexActivityReco.h"

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

    // cluster
    std::vector<larflow::reco::cluster_t> cluster_v;
    makeClusters( ioll, cluster_v, 0.7 );
    
    // find hot points on cluster ends
    float va_threshold = 1000.0;
    std::vector<larlite::larflow3dhit> vtxact_v
      = findVertexActivityCandidates( ioll, iolcv, cluster_v, va_threshold );

    // split between points on shower ends and track ends
    larlite::event_larflow3dhit* evout_vacand
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "vacand");
    for (auto& va : vtxact_v )
      evout_vacand->emplace_back( std::move(va) );
  }

  void NuVertexActivityReco::makeClusters( larlite::storage_manager& ioll,
                                           std::vector<larflow::reco::cluster_t>& cluster_v,
                                           const float larmatch_threshold )
  {

    larlite::event_larflow3dhit* ev_lm
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit,"larmatch");
    LARCV_INFO() << "Number of input larmatch hits: " << ev_lm->size() << std::endl;
    
    std::vector< larlite::larflow3dhit > hit_v;
    hit_v.reserve( ev_lm->size() );
    for ( size_t idx=0; idx<ev_lm->size(); idx++ ) {
      auto const& hit = (*ev_lm)[idx];
      if ( hit[9]>larmatch_threshold ) {
        hit_v.push_back( hit );
        hit_v.back().idxhit = (int)idx; // refers back to original hit vectr
      }
    }
    LARCV_INFO() << "Number of selected larmatch hits: " << hit_v.size() << std::endl;

    cluster_v.clear();
    larflow::reco::cluster_sdbscan_larflow3dhits( hit_v, cluster_v, 2.5, 5, 20 );
    larflow::reco::cluster_runpca( cluster_v );
    // reindex back to original hit vector
    for ( size_t c=0; c<cluster_v.size(); c++ ) {
      auto& cluster = cluster_v[c];
      for ( auto& ih : cluster.hitidx_v ) {
        ih = hit_v[ih].idxhit;          // assign hitidx_v elements to point back to original vector index
        ev_lm->at(ih).trackid = (int)c; // assign hit to index of cluster vector
      }
    }
    
    LARCV_INFO() << "Number of clusters: " << cluster_v.size() << std::endl;
    
  }

  std::vector<larlite::larflow3dhit>
  NuVertexActivityReco::findVertexActivityCandidates( larlite::storage_manager& ioll,
                                                      larcv::IOManager& iolcv,
                                                      std::vector<larflow::reco::cluster_t>& cluster_v,
                                                      const float va_threshold )
  {

    // for each cluster, look for high energy deposit regions
    // if find one, calculate its position on the pca-axis. is it at the end?

    larlite::event_larflow3dhit* ev_lm
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit,"larmatch");

    larcv::EventImage2D* ev_img
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wire");
    auto const& adc_v = ev_img->as_vector();
    int nplanes = adc_v.size();

    std::vector<larlite::larflow3dhit> va_candidate_v;
    
    for ( auto& cluster: cluster_v ) {
      // loop over hits, find high charge regions
      std::vector<int> index_v;
      
      for ( size_t ihit=0; ihit<cluster.hitidx_v.size(); ihit++ ) {
        // get larflow3dhit
        auto& lfhit = (*ev_lm).at( cluster.hitidx_v[ihit] );

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
        auto& lfhit = (*ev_lm).at(idx);

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
      
      if ( min_start_idx>=0 && min_start_dist<3.0 )
        va_candidate_v.push_back( (*ev_lm).at(min_start_idx) );
      
      if ( min_end_idx>=0 && min_end_dist<3.0 )
        va_candidate_v.push_back( (*ev_lm).at(min_end_idx) );
      
    }//end of cluster loop

    LARCV_INFO() << "Number of high-charge regions on cluster ends: " << va_candidate_v.size() << std::endl;
    
    return va_candidate_v;
  }

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
            pixelsum[p] += adc_v[p].pixel(row,col);
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
  
}
}
