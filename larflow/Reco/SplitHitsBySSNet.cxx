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
  void SplitHitsBySSNet::label_and_split( const std::vector<larcv::Image2D>& ssnet_score_v,
                                          const larlite::event_larflow3dhit& larmatch_hit_v,
                                          const float ssnet_score_threshold,
                                          const float larmatch_score_threshold,
                                          std::vector<larlite::larflow3dhit>& accept_v,
                                          std::vector<larlite::larflow3dhit>& reject_v )
  {
    
    larlite::event_larflow3dhit hitcopy_v = larmatch_hit_v;
    label( ssnet_score_v, hitcopy_v );
    split( hitcopy_v, ssnet_score_threshold, larmatch_score_threshold,
           accept_v, reject_v );
    
  }
  
  /**
   * @brief split-up container of larflow3dhit using ssnet output images
   *
   * @param[in] larmatch_hit_v           LArMatch hits, already run through `SplitHitsBySSNet::label`.
   * @param[in] ssnet_score_threshold    Threshold shower score
   * @param[in] larmatch_score_threshold Threshold larmatch score
   * @param[out] accept_v                Hits above threshold
   * @param[out] reject_v                Hits below threshold
   */
  void SplitHitsBySSNet::split( larlite::event_larflow3dhit& larmatch_hit_v,
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

    int below_threshold = 0;
    
    for ( auto& hit : larmatch_hit_v ) {

      //std::cout << "hit[9]=" << hit[9] << std::endl;
      if ( larmatch_score_threshold>0 && hit.size()>=10 && hit[9]<larmatch_score_threshold ) {
        below_threshold++;
        continue;
      }
      
      if ( hit.renormed_shower_score>ssnet_score_threshold ) {           
        accept_v.emplace_back( std::move(hit) );
      }
      else {
        reject_v.emplace_back( std::move(hit) );
      }
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
   * @brief split-up container of larflow3dhit using ssnet output images
   *
   * @param[in] larmatch_hit_v           LArMatch hits, already run through `SplitHitsBySSNet::label`.
   * @param[in] ssnet_score_threshold    Threshold shower score
   * @param[in] larmatch_score_threshold Threshold larmatch score
   * @param[out] accept_v                Hits above threshold
   * @param[out] reject_v                Hits below threshold
   */
  void SplitHitsBySSNet::split_constinput( const larlite::event_larflow3dhit& larmatch_hit_v,
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

    int below_threshold = 0;
    
    for ( auto const& hit : larmatch_hit_v ) {
      
      //std::cout << "hit[9]=" << hit[9] << std::endl;
      if ( larmatch_score_threshold>0 && hit.size()>=10 && hit[9]<larmatch_score_threshold ) {
        below_threshold++;
        continue;
      }
      
      if ( hit.renormed_shower_score>ssnet_score_threshold ) {           
        accept_v.push_back( hit );
      }
      else {
        reject_v.push_back( hit );
      }
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
   * @brief label container of larflow3dhit using 2D track/shower ssnet output images
   *
   * calculates weighted ssnet score and modifies hit to carry value.
   * the weighted ssnet score for the space point is in `larlite::larflow3dhit::renormed_shower_score`
   *
   * @param[in] ssnet_score_v            SSNet shower score images for each plane
   * @param[inout] larmatch_hit_v        LArMatch hits, modified
   */
  void SplitHitsBySSNet::label( const std::vector<larcv::Image2D>& ssnet_score_v,
                                larlite::event_larflow3dhit& larmatch_hit_v )
  {
    
    clock_t begin = clock();
    
    std::vector< const larcv::ImageMeta* > meta_v( ssnet_score_v.size(),0);
    for ( size_t p=0; p<ssnet_score_v.size(); p++ )
      meta_v[p] = &(ssnet_score_v[p].meta());

    int below_threshold = 0;
    
    for ( auto & hit : larmatch_hit_v ) {
      
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

      float weighted_score = tot_score/float(n_w_score);
      if ( n_w_score>0 )
        hit.renormed_shower_score = weighted_score;
      else
        hit.renormed_shower_score = 0.;
    }//end of hit loop
    
    clock_t end = clock();
    double elapsed = double(end-begin)/CLOCKS_PER_SEC;
    
    LARCV_INFO() << " elasped=" << elapsed << " secs" << std::endl;
    
  }
  
  
  /**
   * @brief Process event data in the larcv and larlite IO managers
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void SplitHitsBySSNet::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {

    larcv::EventImage2D* ev_adc_v = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _adc_name );
    const std::vector<larcv::Image2D>& adc_v = ev_adc_v->Image2DArray();
    
    larcv::EventImage2D* ev_ssnet_v[3] = {nullptr};
    for ( size_t p=0; p<3; p++ ) {
      char prodname[20];
      sprintf( prodname, "%s%d", _ssnet_stem_name.c_str(), (int)p );
      ev_ssnet_v[p] = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, prodname );
      int nimages = ev_ssnet_v[p]->Image2DArray().size();
      if (nimages==0) {
	LARCV_NORMAL() << "Missing " << prodname << " images. Need to make them from raw SSNet output" << std::endl;
	_make_trackshower_images_from_sparse_uresnet( p, adc_v.at(p), iolcv, *(ev_ssnet_v[p]) );
      }
      else {
	LARCV_NORMAL() << "  number of images: " << nimages << std::endl;
      }
    }

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
    label( ssnet_showerimg_v, *ev_lfhit );
    split_constinput( *ev_lfhit, _score_threshold, _larmatch_threshold, _shower_hit_v, _track_hit_v );

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

  /**
   * @brief Process hits through labeler
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void SplitHitsBySSNet::process_labelonly( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {

    larcv::EventImage2D* ev_adc_v = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _adc_name );
    const std::vector<larcv::Image2D>& adc_v = ev_adc_v->Image2DArray();
    
    larcv::EventImage2D* ev_ssnet_v[3] = {nullptr};
    for ( size_t p=0; p<3; p++ ) {
      char prodname[20];
      sprintf( prodname, "%s%d", _ssnet_stem_name.c_str(), (int)p );
      ev_ssnet_v[p] = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, prodname );
      int nimages = ev_ssnet_v[p]->Image2DArray().size();	
      if (nimages==0) {
	LARCV_NORMAL() << "Missing " << prodname << " images. Need to make them from raw SSNet output" << std::endl;
	_make_trackshower_images_from_sparse_uresnet( p, adc_v.at(p), iolcv, *(ev_ssnet_v[p]) );
      }
      else {
	LARCV_NORMAL() << "  number of images: " << nimages << std::endl;
      }      
    }
    

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
    label( ssnet_showerimg_v, *ev_lfhit );
    
  }

  /**
   * @brief Process event data only through splitter
   *
   * assumes the labeling function has been run already on the input larflow spacepoints
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void SplitHitsBySSNet::process_splitonly( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {

    // larflow hits
    larlite::event_larflow3dhit* ev_lfhit
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, _input_larmatch_hit_tree_name );

    _shower_hit_v.clear();
    _track_hit_v.clear();
    split_constinput( *ev_lfhit, _score_threshold, _larmatch_threshold, _shower_hit_v, _track_hit_v );

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

  /**
   * @brief make shower and track image from Sparse UResnet output
   *
   * will fill container with first the shower image, then the track image.
   * We look for the UResnet output in the larcv tree: sparseimg_sparseuresnetout_tree
   *
   */
  void SplitHitsBySSNet::_make_trackshower_images_from_sparse_uresnet( const int plane,
								       const larcv::Image2D& adc,
								       larcv::IOManager& iolcv,
								       larcv::EventImage2D& container )
  {

    /*
        if(pdg_code == 2212 or pdg_code == -2212): category = 0
        elif not pdg_code in [11,-11,22]: category = 1
        elif pdg_code == 22: category = 2
        else:
            if process in ['primary','nCapture','conv','compt']: category = 2
            elif process in ['muIoni','hIoni']: category = 3
            elif process in ['muMinusCaptureAtRest','muPlusCaptureAtRest','Decay']: category = 4
    */
    
    std::string producer_uresenet = "sparseuresnetout";
    larcv::EventSparseImage* ev_sparseimg
      = (larcv::EventSparseImage*)iolcv.get_data( larcv::kProductSparseImage, producer_uresenet );

    // make new images
    larcv::Image2D shower_adc( adc.meta() );    
    larcv::Image2D track_adc( adc.meta() );

    LARCV_NORMAL() << "Number of uresnet SparseImages: " << ev_sparseimg->SparseImageArray().size() << std::endl;
    for (int i=0; i<(int)ev_sparseimg->SparseImageArray().size(); i++) {
      auto const& img = ev_sparseimg->SparseImageArray().at(i);
      LARCV_NORMAL() << " image[" << i << "] len=" << img.len() << " nfeatures=" << img.nfeatures() << std::endl;
      // for (int j=0; j<(int)img.len(); j++) {
      // 	std::cout << " [" << j << "]";
      // 	for (int f=0; f<(int)img.nfeatures()+2; f++) {
      // 	  std::cout << " " << img.getfeature(j,f);
      // 	}
      // 	std::cout << std::endl;
      // }
      // features for each entry in the sparse tensor [row] [col] [proton score] [muon score] [electron score] [delta] [michel]
      // scores should be normalized 
      for (int j=0; j<(int)img.len(); j++) {     
	int row = img.getfeature(j,0);
	int col = img.getfeature(j,1);
	float track_score  = img.getfeature(j,2) + img.getfeature(j,3);
	float shower_score = img.getfeature(j,4) + img.getfeature(j,5) + img.getfeature(j,6);
	track_adc.set_pixel( row, col, track_score );
	shower_adc.set_pixel( row, col, shower_score );
      }
    }

    container.Emplace( std::move(shower_adc) );
    container.Emplace( std::move(track_adc) );
    
    return;
  }
  
}
}
