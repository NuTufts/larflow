#include "SplitHitsBySSNet.h"

#include "larcv/core/DataFormat/EventImage2D.h"

#include <ctime>

namespace larflow {
namespace reco {

  /**
   * @brief split-up container of larflow3dhit using ssnet output images
   *
   * @param[in] ssnet_score_v            SSNet shower score images for each plane
   * @param[in] larmatch_hit_v           LArMatch hits
   * @param[in] ssnet_score_threshold    Threshold shower score
   * @param[in] larmatch_score_threshold Threshold larmatch score
   * @param[out] accept_v                Hits above threshold
   * @param[out] reject_v                Hits below threshold
   */
  void SplitHitsBySSNet::split( const std::vector<larcv::Image2D>& ssnet_score_v,
                                const larlite::event_larflow3dhit& larmatch_hit_v,
                                const float ssnet_score_threshold,
                                const float larmatch_score_threshold,
                                std::vector<larlite::larflow3dhit>& accept_v,
                                std::vector<larlite::larflow3dhit>& reject_v )
  {

    clock_t begin = clock();
    
    accept_v.clear();
    reject_v.clear();
    accept_v.reserve( larmatch_hit_v.size() );
    reject_v.reserve( larmatch_hit_v.size() );

    std::vector< const larcv::ImageMeta* > meta_v( ssnet_score_v.size(),0);
    for ( size_t p=0; p<ssnet_score_v.size(); p++ )
      meta_v[p] = &(ssnet_score_v[p].meta());

    int below_threshold = 0;
    
    for ( auto const & hit : larmatch_hit_v ) {

      //std::cout << "hit[9]=" << hit[9] << std::endl;
      if ( larmatch_score_threshold>0 && hit.size()>=10 && hit[9]<larmatch_score_threshold ) {
        below_threshold++;
        continue;
      }
      
      std::vector<float> scores(3,0);
      scores[0] = ssnet_score_v[0].pixel( meta_v[0]->row( hit.tick, __FILE__, __LINE__ ), hit.targetwire[0], __FILE__, __LINE__ );
      scores[1] = ssnet_score_v[1].pixel( meta_v[1]->row( hit.tick, __FILE__, __LINE__ ), hit.targetwire[1], __FILE__, __LINE__ );
      scores[2] = ssnet_score_v[2].pixel( meta_v[2]->row( hit.tick, __FILE__, __LINE__ ), hit.srcwire,       __FILE__, __LINE__ );

      // condition ... gather metrics
      int n_w_score = 0;
      float tot_score = 0.;
      float max_score = 0.;
      float min_non_zero = 1.;
      for ( auto s : scores ) {
        if ( s>0 ) n_w_score++;
        tot_score += s;
        if ( max_score<s )
          max_score = s;
        if ( s>1 && s<min_non_zero )
          min_non_zero = 0;
      }
      // we form a weighted average of the score

      if ( n_w_score>0 && tot_score/float(n_w_score)>ssnet_score_threshold ) {
        accept_v.push_back( hit );
      }
      else
        reject_v.push_back( hit );
    }

    clock_t end = clock();
    double elapsed = double(end-begin)/CLOCKS_PER_SEC;
    
    LARCV_INFO() << "original=" << larmatch_hit_v.size()
                 << " accepted=" << accept_v.size()
                 << " and rejected=" << reject_v.size()
                 << " below-threshold=" << below_threshold
                 << " elasped=" << elapsed << " secs"
                 << std::endl;
    
  }

  /**
   * @brief Process event data in the larcv and larlite IO managers
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void SplitHitsBySSNet::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {

    larcv::EventImage2D* ev_ssnet_v[3] = {nullptr};
    for ( size_t p=0; p<3; p++ ) {
      char prodname[20];
      sprintf( prodname, "%s%d", _ssnet_stem_name.c_str(), (int)p );
      ev_ssnet_v[p] = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, prodname );
    }

    larcv::EventImage2D* ev_adc_v = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _adc_name );
    const std::vector<larcv::Image2D>& adc_v = ev_adc_v->Image2DArray();

    // collect track images
    std::vector<larcv::Image2D> ssnet_trackimg_v;
    for ( size_t p=0; p<3; p++ )
      ssnet_trackimg_v.push_back(ev_ssnet_v[p]->Image2DArray()[1]);

    // collect shower images
    std::vector<larcv::Image2D> ssnet_showerimg_v;
    for ( size_t p=0; p<3; p++ )
      ssnet_showerimg_v.push_back(ev_ssnet_v[p]->Image2DArray()[0]);
    

    // larflow hits
    larlite::event_larflow3dhit* ev_lfhit
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, _input_larmatch_hit_tree_name );

    _shower_hit_v.clear();
    _track_hit_v.clear();
    split( ssnet_showerimg_v, *ev_lfhit, _score_threshold, _larmatch_threshold, _shower_hit_v, _track_hit_v );

    larlite::event_larflow3dhit* evout_shower_hit
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, _output_larmatch_hit_stem_name+"_showerhit" );

    larlite::event_larflow3dhit* evout_track_hit
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, _output_larmatch_hit_stem_name+"_trackhit" );

    for ( auto& hit : _shower_hit_v )
      evout_shower_hit->push_back( hit );

    for ( auto& hit : _track_hit_v )
      evout_track_hit->push_back( hit );

    LARCV_NORMAL() << "Split hits into " << _track_hit_v.size() << " trackhit and " << _shower_hit_v.size() << " showerhit" << std::endl;
    
  }
 
  
}
}
