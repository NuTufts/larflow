#include "FilterByWCTagger.h"

#include "larlite/LArUtil/DetectorProperties.h"
#include "larlite/LArUtil/Geometry.h"
#include "larlite/DataFormat/pcaxis.h"

namespace larflow {
namespace reco {

  /** @brief default constructor */
  FilterByWCTagger::FilterByWCTagger()
    : larcv::larcv_base("FilterByWCTagger")
  {
    set_defaults();
  }


  /**
   * @brief Filter hits and keypoints from the event containers
   *
   * @param[in] iolcv LArCV IO manager with the data we need
   * @param[in] ioll  larlite storage_manager with the data we need
   *
   */
  void FilterByWCTagger::process( larcv::IOManager& iolcv,
                                          larlite::storage_manager& ioll )
  {

    process_hits( iolcv, ioll );
    process_keypoints( iolcv, ioll );
    
  }

  /**
   * @brief filter larmatch hits using wc tagged image
   *
   * @param[in] iolcv LArCV event data container
   * @param[in] ioll storage_manager event data container
   */
  void FilterByWCTagger::process_hits( larcv::IOManager& iolcv,
				       larlite::storage_manager& ioll )
  {

    // get adc images
    larcv::EventImage2D* ev_adc_v =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _input_adc_tree_name );
    const std::vector<larcv::Image2D>& adc_v = ev_adc_v->Image2DArray();    
    LARCV_INFO() << "Input wire images [" << _input_adc_tree_name << "]: " << adc_v.size() << std::endl;
    
    // get cosmic tagged image
    larcv::EventImage2D* ev_tagger =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _input_taggerimg_tree_name );
    const std::vector<larcv::Image2D>& tagged_v = ev_tagger->Image2DArray();
    LARCV_INFO() << "Input tagged images [" << _input_taggerimg_tree_name << "]: " << tagged_v.size() << std::endl;

    // get larmatch hits to filter
    LARCV_INFO() << "Input larmatch hit tree: " << _input_larmatch_tree_name << std::endl;
    larlite::event_larflow3dhit* ev_larmatch =
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _input_larmatch_tree_name );
    
    // get ssnet shower images
    larcv::EventImage2D* ev_ssnet =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _ssnet_stem_name );
    auto const& ssnet_showerimg_v = ev_ssnet->as_vector();
    std::vector< const larcv::Image2D* > pssnet_shower_v;
    for ( auto const& img : ssnet_showerimg_v )
      pssnet_shower_v.push_back( &img );

    std::vector<int> kept_hit_v( ev_larmatch->size(), 0 );
    filter_larmatchhits_using_tagged_image( adc_v, tagged_v, pssnet_shower_v, *ev_larmatch, kept_hit_v );    

    larlite::event_larflow3dhit* ev_filteredhits_output = 
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _output_filteredhits_tree_name );

    larlite::event_larflow3dhit* ev_rejectedhits_output = nullptr;
    if ( _save_rejected_hits ) {
      ev_rejectedhits_output =
        (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _output_rejectedhits_tree_name );
    }

    for ( size_t ihit=0; ihit<ev_larmatch->size(); ihit++ ) {
      if ( kept_hit_v[ihit]==1 ) 
        ev_filteredhits_output->push_back( ev_larmatch->at(ihit) );
      else if ( _save_rejected_hits && kept_hit_v[ihit]==0 )
        ev_rejectedhits_output->push_back( ev_larmatch->at(ihit) );
    }

    LARCV_INFO() << "num of filtered hits: " << ev_filteredhits_output->size() << " of " << ev_larmatch->size() << std::endl;
    if ( _save_rejected_hits ) {
      LARCV_INFO() << "num of rejected (i.e. on off-beam charge) hits: " << ev_rejectedhits_output->size() << " of " << ev_larmatch->size() << std::endl;
    }
    
  }

  /**
   * @brief filter keypoints hits using wc tagged image
   *
   * @param[in] iolcv LArCV event data container
   * @param[in] ioll storage_manager event data container
   */
  void FilterByWCTagger::process_keypoints( larcv::IOManager& iolcv,
                                                    larlite::storage_manager& ioll )
  {

    // get adc images
    larcv::EventImage2D* ev_adc_v =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _input_adc_tree_name );
    const std::vector<larcv::Image2D>& adc_v = ev_adc_v->Image2DArray();    
    LARCV_INFO() << "Input wire images [" << _input_adc_tree_name << "]: " << adc_v.size() << std::endl;
    
    // get cosmic tagged image
    larcv::EventImage2D* ev_tagger =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _input_taggerimg_tree_name );
    const std::vector<larcv::Image2D>& tagged_v = ev_tagger->Image2DArray();
    LARCV_INFO() << "Input tagged images [" << _input_taggerimg_tree_name << "]: " << tagged_v.size() << std::endl;

    // get keypoint network image
    LARCV_INFO() << "Input keypoint tree: " << _input_keypoint_tree_name << std::endl;    
    larlite::event_larflow3dhit* ev_keypoint =
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _input_keypoint_tree_name );
    larlite::event_pcaxis* ev_keypoint_pcaxis =
      (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, _input_keypoint_tree_name );
    
    // get ssnet shower images
    larcv::EventImage2D* ev_ssnet_v[3] = {nullptr};
    for ( size_t p=0; p<3; p++ ) {
      char prodname[20];
      sprintf( prodname, "%s%d", _ssnet_stem_name.c_str(), (int)p );
      ev_ssnet_v[p] = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, prodname );
    }

    // collect shower images
    std::vector<const larcv::Image2D*> ssnet_showerimg_v;
    for ( size_t p=0; p<3; p++ )
      ssnet_showerimg_v.push_back(&(ev_ssnet_v[p]->Image2DArray()[0]));


    std::vector<int> kept_keypoint_v( ev_keypoint->size(), 0 );
    filter_keypoint_using_tagged_image( adc_v, tagged_v, ssnet_showerimg_v, *ev_keypoint, kept_keypoint_v );
    

    larlite::event_larflow3dhit* ev_keypoint_output = 
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _output_keypoint_tree_name );
    larlite::event_pcaxis* ev_kpaxis_output = 
      (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, _output_keypoint_tree_name );

    for ( size_t ikp=0; ikp<ev_keypoint->size(); ikp++ ) {
      if ( kept_keypoint_v[ikp]==1 )  {
        ev_keypoint_output->push_back( ev_keypoint->at(ikp) );
        ev_kpaxis_output->push_back( ev_keypoint_pcaxis->at(ikp) );
      }
    }

    LARCV_INFO() << "num of keypoint hits: " << ev_keypoint_output->size() << " of " << ev_keypoint->size() << std::endl;
    
  }
  

  /**
   * @brief filter larmatch hits using wirecell cosmic-tagged image
   *
   * The goal is to only remove cosmic muon pixels. 
   * This means we are trying to let cosmic-tagged shower pixels
   * pass. Effectively passing all shower pixels is done in order to
   * maintain electron neutrino efficiency.
   * 
   * Check the wirecell cosmic tag and SSNet (2D) score to determine
   * if the keep the hit.
   * Each hit already contains the row and column in each plane that it corresponds to.
   * For each pixel, we see if the pixel has a tag >5 (is cosmic) in the wirecell image.
   * If it is tagged as cosmic, we check the ssnet shower score.
   * If the score is below 0.5, we decide it must be a cosmic muon and tag the pixel.
   * We check a larmatch space point 3 times, one for each plane. 
   * If the space point is tagged in at least 2 out of 3 planes, the
   * larmatch point is rejected.
   * 
   * @param[in] adc_v Wire plane images
   * @param[in] tagged_v Wirecell cosmic-tagged pixels (tree usually named thrumu)
   * @param[in] shower_ssnet_v Shower SSNet score images
   * @param[in] larmatch_v LArMatch larflow3d hits containing larmatch score
   * @param[in] kept_v Vector set to be the same size as larmatch_v. If 1, it has been kept.
   *
   */
  void FilterByWCTagger::filter_larmatchhits_using_tagged_image( const std::vector<larcv::Image2D>& adc_v,
								 const std::vector<larcv::Image2D>& tagged_v,
								 const std::vector< const larcv::Image2D* >& shower_ssnet_v,
								 const std::vector<larlite::larflow3dhit>& larmatch_v,
								 std::vector<int>& kept_v )
  {

    kept_v.clear();
    kept_v.resize( larmatch_v.size(), 1 );

    // loop over tpcs
    auto const geom = larlite::larutil::Geometry::GetME();
    for (int icryo=0; icryo<(int)geom->Ncryostats(); icryo++) {
      for (int itpc=0; itpc<(int)geom->NTPCs(icryo); itpc++) {

	int startplaneindex = geom->GetSimplePlaneIndexFromCTP( icryo, itpc, 0 );

	// look for it in the adc_v list
	bool found = false;
	for ( auto const& img : adc_v ) {
	  if ( img.meta().id()==startplaneindex ) {
	    found = true;
	    break;
	  }
	}

	if ( !found )
	  continue; // to next tpc

	// ok, we have images for this tpc
	// we get the tagger images and ssnet images for the tpc
	std::vector< const larcv::Image2D* > padc_v;
	std::vector< const larcv::Image2D* > ptagged_v;
	std::vector< const larcv::Image2D* > pshower_v;

	const int nplanes = geom->Nplanes( itpc, icryo );
	
	for (int iplane=0; iplane<nplanes; iplane++) {
	  int simpleindex = startplaneindex + iplane;

	  for ( auto const& tagged : tagged_v ) {
	    if (tagged.meta().id()==simpleindex) {
	      ptagged_v.push_back( &tagged );
	      break;
	    }
	  }

	  
	  for ( auto const& adc : adc_v ) {
	    if (adc.meta().id()==simpleindex) {
	      padc_v.push_back( &adc );
	      break;
	    }
	  }
	  for (auto const& pshower : shower_ssnet_v ) {
	    if ( pshower->meta().id()==simpleindex ) {
	      pshower_v.push_back( pshower );
	      break;
	    }
	  }
	}//loop over the TPC planes

	if ( ptagged_v.size()==0 )
	  continue; // no wirecell tagger info
	
	if ( ptagged_v.size()>0 && ptagged_v.size()!=nplanes ) {
	  LARCV_CRITICAL() << "Wrong number of cosimc-tagged images (" << ptagged_v.size() << ") versus number of TPC planes (" << geom->Nplanes(itpc,icryo) << ")"  << std::endl;
	}
	if ( padc_v.size()!=nplanes ) {
	  LARCV_CRITICAL() << "Wrong number of wire signal images (" << padc_v.size() << ") versus number of TPC planes (" << geom->Nplanes(itpc,icryo) << ")"  << std::endl;
	}

	// ok loop over the hits	
	for ( size_t ihit=0; ihit<larmatch_v.size(); ihit++ ) {

	  auto const& hit = larmatch_v[ihit];
	  //LARCV_DEBUG() << "hit[" << ihit << "] tick=" << hit.tick+1 << std::endl;
	  //int tick = hit[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5 + 3200;
	  if ( hit.tick<=padc_v[0]->meta().min_y() || hit.tick>=padc_v[0]->meta().max_y() )
	    continue;
	  if ( hit[3]!=itpc || hit[4]!=icryo )
	    continue;

	  
	  int row = padc_v[0]->meta().row( hit.tick );
	  int nplanes_tagged = 0;
	  for ( size_t p=0; p<nplanes; p++ ) {
	    int col = padc_v[p]->meta().col( hit.targetwire[p], __FILE__, __LINE__ );
	    int tagged = ptagged_v[p]->pixel( row, col );
	    //LARCV_DEBUG() << " tagged[col " << col << "]=" << tagged;        
	    if ( tagged>5) {
	      // check if shower
	      float shower_score = 0.0;
	      if ( pshower_v.size()>0 )
		shower_score = pshower_v[p]->pixel(row,col);
	      
	      //LARCV_DEBUG() << " showerscore=" << shower_score;        
	      if ( shower_score<0.5 ) {
		nplanes_tagged++;
	      }
	      
	    }        
	  }
	  if ( nplanes_tagged>=2 ) {
	    kept_v[ihit] = 0;
	  }
	  else {
	    kept_v[ihit] = 1;
	  }
	}//end of loop over hits
      }//end of loop over TPC      
    }//end of loop over cryostats
    
  }
  

  /**
   * @brief filter keypoints using wirecell cosmic-tagged image
   *
   * The goal is to only remove vertices laying on cosmic muon pixels. 
   * This means we are trying to let keypoints on cosmic-tagged shower pixels
   * pass. Effectively passing all shower pixels is done in order to
   * maintain electron neutrino efficiency.
   * 
   * First the 3d keypoint position is projected into the image and a (row,col)
   * is found for each plane.
   *
   * Next, a 11x11 window is searched on each plane around the projected point.
   * For each pixel in this window, we check to see if the pixel is tagged
   * as cosmic and has an ssnet shower score below 0.25. 
   * if at least one pixel is cosmic-tagged and no more than 2 pixels are 
   * shower pixels, the keypoint is considered tagged on that plane.
   * The keypoint must be tagged at a max of one plane to pass. Else it is filtered.
   * 
   * @param[in] adc_v Wire plane images
   * @param[in] tagged_v Wirecell cosmic-tagged pixels (tree usually named thrumu)
   * @param[in] shower_ssnet_v Shower SSNet score images
   * @param[in] keypoint_v LArMatch larflow3d hits containing larmatch score
   * @param[in] kept_v Vector set to be the same size as larmatch_v. If 1, it has been kept.
   *
   */
  void FilterByWCTagger::filter_keypoint_using_tagged_image( const std::vector<larcv::Image2D>& adc_v,
							     const std::vector<larcv::Image2D>& tagged_v,
							     const std::vector< const larcv::Image2D* >& shower_ssnet_v,
							     const std::vector<larlite::larflow3dhit>& keypoint_v,
							     std::vector<int>& kept_v )
  {

    auto const geom = larlite::larutil::Geometry::GetME();
    auto const detp = larutil::DetectorProperties::GetME();
    
    kept_v.clear();
    kept_v.resize( keypoint_v.size(), 1 );
    
    for (int icryo=0; icryo<(int)geom->Ncryostats(); icryo++) {
      for (int itpc=0; itpc<(int)geom->NTPCs(icryo); itpc++) {

	int startplaneindex = geom->GetSimplePlaneIndexFromCTP( icryo, itpc, 0 );
	
	// look for it in the adc_v list
	bool found = false;
	for ( auto const& img : adc_v ) {
	  if ( img.meta().id()==startplaneindex ) {
	    found = true;
	    break;
	  }
	}

	if ( !found )
	  continue; // to next tpc

	// ok, we have images for this tpc
	// we get the tagger images and ssnet images for the tpc
	std::vector< const larcv::Image2D* > padc_v;
	std::vector< const larcv::Image2D* > ptagged_v;
	std::vector< const larcv::Image2D* > pshower_v;

	const int nplanes = geom->Nplanes( itpc, icryo );
	
	for (int iplane=0; iplane<nplanes; iplane++) {
	  int simpleindex = startplaneindex + iplane;

	  for ( auto const& tagged : tagged_v ) {
	    if (tagged.meta().id()==simpleindex) {
	      ptagged_v.push_back( &tagged );
	      break;
	    }
	  }
	  
	  for ( auto const& adc : adc_v ) {
	    if (adc.meta().id()==simpleindex) {
	      padc_v.push_back( &adc );
	      break;
	    }
	  }
	  
	  for (auto const& pshower : shower_ssnet_v ) {
	    if ( pshower->meta().id()==simpleindex ) {
	      pshower_v.push_back( pshower );
	      break;
	    }
	  }
	}//loop over the TPC planes

	if ( ptagged_v.size()==0 )
	  continue; // no wirecell tagger info
	
	if ( ptagged_v.size()>0 && ptagged_v.size()!=nplanes ) {
	  LARCV_CRITICAL() << "Wrong number of cosimc-tagged images (" << ptagged_v.size() << ") versus number of TPC planes (" << geom->Nplanes(itpc,icryo) << ")"  << std::endl;
	}
	if ( padc_v.size()!=nplanes ) {
	  LARCV_CRITICAL() << "Wrong number of wire signal images (" << padc_v.size() << ") versus number of TPC planes (" << geom->Nplanes(itpc,icryo) << ")"  << std::endl;
	}

	auto const& zerometa = padc_v[0]->meta();
	
	for ( size_t ihit=0; ihit<keypoint_v.size(); ihit++ ) {

	  auto const& hit = keypoint_v[ihit];

	  // silly that this isnt the same as the larmatch points!
	  int tpcid  = hit.targetwire[4];
	  int cryoid = hit.targetwire[5];

	  if ( tpcid!=itpc || cryoid!=icryo )
	    continue; // didn't plane
	  
	  float tick = hit.tick;
	  if ( tick<=zerometa.min_y() || tick>=zerometa.max_y() )
	    continue;
      
	  int row = zerometa.row( tick, __FILE__, __LINE__ );
      
	  int nplanes_tagged = 0;
	  for ( int p=0; p<nplanes; p++ ) {
	    
	    auto const& meta = padc_v[p]->meta();	
	    int col = meta.col( hit.targetwire[p] );

	    int nshower = 0;
	    int ntagged = 0;

	    for (int dr=-5; dr<=5; dr++) {
	      int r = row+dr;
	      if ( r<0 || r>=(int)meta.rows() ) continue;
	      for (int dc=-5; dc<=5; dc++) {
		int c = col+dc;
		if ( c<0 || c>=(int)meta.cols() ) continue;

		int tagged = tagged_v[p].pixel( r, c );
		if ( tagged>5 ) {
		  ntagged++;
		  // check if shower
		  float shower_score = pshower_v[p]->pixel(r,c);

		  if ( shower_score<0.25 )
		    nshower++;
		}

	      }//col neighborhood    
	    }//row neighborhood

	    if (ntagged>0 && nshower<=2 )
	      nplanes_tagged++;
	  }//end of plane loop
	  
	  if ( nplanes_tagged>=2 ) {
	    kept_v[ihit] = 0;
	  }
	  else {
	    kept_v[ihit] = 1;
	  }
	}//end of hit loop
      }//end of tpc loop
    }//end of cryo loop
    
  }
  
  
}
}


