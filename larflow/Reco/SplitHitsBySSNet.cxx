#include "SplitHitsBySSNet.h"

#include "larlite/LArUtil/LArUtilConfig.h"
#include "larlite/LArUtil/Geometry.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventSparseImage.h"

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
    label( ssnet_score_v, hitcopy_v, 0, 0 );
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
                                larlite::event_larflow3dhit& larmatch_hit_v,
				const int tpcid, const int cryoid )
  {

    clock_t begin = clock();
    LARCV_INFO() << "Label hits coming from TPCID=" << tpcid << " CryoID=" << cryoid << std::endl;
    
    std::vector< const larcv::ImageMeta* > meta_v( ssnet_score_v.size(),0);
    for ( size_t p=0; p<ssnet_score_v.size(); p++ )
      meta_v[p] = &(ssnet_score_v[p].meta());

    int below_threshold = 0;
    int n_w_labels = 0;
    int n_in_tpc = 0;
    for ( auto & hit : larmatch_hit_v ) {

      if ( hit[3]!=tpcid || hit[4]!=cryoid ) {
	//std::cout << "not right tpc: tpcid=" << hit[3] << " cryoid=" << hit[4] << std::endl;	
	continue;
      }

      n_in_tpc++;
      
      std::vector<float> scores(3,0);
      scores[0] = ssnet_score_v[0].pixel( hit.targetwire[3], hit.targetwire[0], __FILE__, __LINE__ );
      scores[1] = ssnet_score_v[1].pixel( hit.targetwire[3], hit.targetwire[1], __FILE__, __LINE__ );
      scores[2] = ssnet_score_v[2].pixel( hit.targetwire[3], hit.targetwire[2], __FILE__, __LINE__ );

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
      if ( n_w_score>0 ) {
        hit.renormed_shower_score = weighted_score;
	n_w_labels++;
      }
      else
        hit.renormed_shower_score = 0.;
    }//end of hit loop
    
    clock_t end = clock();
    double elapsed = double(end-begin)/CLOCKS_PER_SEC;
    
    LARCV_NORMAL() << " elasped=" << elapsed << " secs. num w labels=" << n_w_labels << " num in tpc=" << n_in_tpc << std::endl;
    
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
    label( ssnet_showerimg_v, *ev_lfhit, 0, 0 );
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

    if ( larutil::LArUtilConfig::Detector()==larlite::geo::kMicroBooNE ) {
      // MICROBOONE LABELING: HAS DATA MODEL FOR SHOWER IMAGES THATS NOT EASILY EXTENDABLE
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
      label( ssnet_showerimg_v, *ev_lfhit, 0, 0  );

    }
    else {
      // we use sparsessnet SparseImage instead
      // we use it to create a shower image
      auto const geom = larlite::larutil::Geometry::GetME();
      larcv::EventSparseImage* ev_sparsessnet =
	(larcv::EventSparseImage*)iolcv.get_data( larcv::kProductSparseImage, "sparsessnet" );
      larcv::EventImage2D* ev_ssnet2d =
	(larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "ssnetshower" );
      larcv::EventImage2D* ev_image2d = 
	(larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
      auto const& adc_v = ev_image2d->as_vector();

      int nlabeled = 0;
      for ( int icryo=0; icryo<(int)geom->Ncryostats(); icryo++) {
	for (int itpc=0; itpc<(int)geom->NTPCs(icryo); itpc++) {

	  // the first index number of cryo and tpc IDs
	  std::vector<larcv::Image2D> ssnet_showerimg_v;

	  for (int iplane=0; iplane<(int)geom->Nplanes(itpc,icryo); iplane++) {
	    int isimpleplaneid = geom->GetSimplePlaneIndexFromCTP( icryo, itpc, iplane );

	    // need a meta for this plane
	    const larcv::ImageMeta* meta = nullptr;
	    for (int i=0; i<(int)adc_v.size(); i++) {
	      if ( adc_v.at(i).meta().id()==isimpleplaneid )
		meta = &adc_v.at(i).meta();
	    }
	    if ( !meta ) {
	      LARCV_DEBUG() << "No image for (cryo,tpc,plane)=(" << icryo << "," << itpc << "," << iplane << ")" << std::endl;
	      continue;
	    }
	    else {
	      LARCV_DEBUG() << "Process image for (cryo,tpc,plane)=(" << icryo << "," << itpc << "," << iplane << ")" << std::endl;
	      LARCV_DEBUG() << "Input ADC image meta: " << meta->dump() << std::endl;
	    }
	    
	    // find the plane data
	    bool found_plane = false;
	    LARCV_DEBUG() << "search " << ev_sparsessnet->SparseImageArray().size() << " images" << std::endl;
	    for (int ii=0; ii<(int)ev_sparsessnet->SparseImageArray().size(); ii++) {
	      int spimg_planeid = ev_sparsessnet->at(ii).meta_v()[0].id();
	      LARCV_DEBUG() << "sparse image index: " << spimg_planeid << std::endl;
	      
	      if ( spimg_planeid==isimpleplaneid ) {

		auto const& spimg = ev_sparsessnet->at(ii);
		auto const& spmeta = ev_sparsessnet->at(ii).meta_v()[0];
		// make the image
		larcv::Image2D ssnet( *meta );
		ssnet.paint(0.0);

		int stride = spimg.stride();
		for (size_t ipix=0; ipix<spimg.len(); ipix++) {
		  float showerscore = 0;
		  for (size_t iclass=0; iclass<spimg.nfeatures(); iclass++) {
		    float pixscore = spimg.pixellist().at( ipix*stride + 2 + iclass );
		    if (iclass>1)
		      showerscore += pixscore;
		  }
		  int irow = int(spimg.pixellist().at( ipix*stride ));
		  int icol = int(spimg.pixellist().at( ipix*stride+1));

		  if ( meta->pixel_height()<spmeta.pixel_height() ) {		  
		    float ssnet_tick = spmeta.pos_y(irow);
		    if ( ssnet_tick>meta->min_y() && ssnet_tick<meta->max_y() ) {
		      int meta_row = meta->row(ssnet_tick);
		      int upscale_factor = (int)spmeta.pixel_height()/meta->pixel_height();
		      for (int uppix=0; uppix<upscale_factor; uppix++) {
			ssnet.set_pixel(meta_row+uppix,icol,showerscore);
		      }
		      nlabeled++;		      
		    }
		  }
		  else if ( meta->pixel_height()==spmeta.pixel_height() ) {  
		    ssnet.set_pixel(irow,icol,showerscore);
		    nlabeled++;		      
		  }
		  else {
		    LARCV_CRITICAL() << "downsampling ssnet image not supported yet" << std::endl;
		    throw std::runtime_error("downsampling ssnet image not supported yet");
		  }
		}
		found_plane = true;
		ssnet_showerimg_v.emplace_back( std::move(ssnet) );
		break;
	      }//if plane has the id we're looking for
	    }//end of loop over sparsessnet vector

	    if ( !found_plane ) {
	      LARCV_CRITICAL() << "Could not find sparseimage for (simple) planeid=" << isimpleplaneid << std::endl;
	      throw std::runtime_error("Could not find sparseimage for (simple) planeid");
	    }
	  }//end of plane in TPC loop

	  if ( ssnet_showerimg_v.size()>0 && ssnet_showerimg_v.size()!=(int)geom->Nplanes(itpc,icryo) ) {
	    LARCV_CRITICAL() << "The number of shower images (" << ssnet_showerimg_v.size() << ") "
			     << "does not match the number of planes in the TPC (" << geom->Nplanes(itpc,icryo)  << ")"
			     << std::endl;
	    throw std::runtime_error("The number of shower images does not match the number of planes in the TPC");
	  }

	  if ( ssnet_showerimg_v.size()==0 ) {
	    // no info for this TPC
	    continue;
	  }
	  
	  // do the labeling for this tpc and cryostat
	  larlite::event_larflow3dhit* ev_lfhit
	    = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, _input_larmatch_hit_tree_name );
	  label( ssnet_showerimg_v, *ev_lfhit, itpc, icryo  );

	  for ( auto showerimg : ssnet_showerimg_v )
	    ev_ssnet2d->Emplace( std::move(showerimg) );
	  
	}//end of tpc loop
      }// end of cryo loop

      LARCV_INFO() << "number pixels labeled: " << nlabeled << std::endl;
    }//end of if detector is not MicroBooNE
    
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
  
  
}
}
