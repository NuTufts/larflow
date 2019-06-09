#include "makecontourflowmatches.h"

#include "larcv/core/Base/larcv_logger.h"

namespace larflow {

  /**
   * extract flow for pixels in the cropped image.
   *
   * we compile the relationships between pixels and the different contour-clusters
   * the goal is to start to see what contours on source and target imager are paired together
   * through flow predictions
   *
   * @param[in] contour_data contour list for the wholeview image
   * @param[in] flow_img flow data for a cropped images
   * @param[in] src_adc_full  adc values for the whole source image
   * @param[in] tar_adc_full  adc values for the whole target image
   * @param[in] src_adc_crop  adc values for the cropped source image
   * @param[in] tar_adc_crop  adc values for the cropped target image
   * @param[in] threshold adc threshold
   * @param[inout] matchdict container storing src-to-target contour pair information
   *
   */
  void createMatchData( const ublarcvapp::ContourClusterAlgo& contour_data,
                        const larcv::Image2D& src_adc_full,
                        const larcv::Image2D& tar_adc_full,
                        const larcv::Image2D& flow_img_crop,
                        const larcv::Image2D& src_adc_crop,
                        const larcv::Image2D& tar_adc_crop,
                        ContourFlowMatchDict_t& matchdict,
                        const float threshold,
                        const float max_dist_to_target_contour )
  //const larcv::msg::level_t msg_level )
  {
                        

    larcv::logger log("createMatchData");
    
    int src_planeid = src_adc_crop.meta().plane();
    int tar_planeid = tar_adc_crop.meta().plane();

    const larcv::ImageMeta& srcfullmeta = src_adc_full.meta();
    const larcv::ImageMeta& tarfullmeta = tar_adc_full.meta();    
    const larcv::ImageMeta& srcmeta = src_adc_crop.meta();
    const larcv::ImageMeta& tarmeta = tar_adc_crop.meta();
    
    // allocate arrays for image pixel to contour index lookup
    if ( !matchdict.index_map_initialized ) {
      matchdict.src_ctr_index_map.resize( src_adc_full.meta().rows()*src_adc_full.meta().cols(), -1 );
      matchdict.tar_ctr_index_map.resize( tar_adc_full.meta().rows()*tar_adc_full.meta().cols(), -1 );
      matchdict.index_map_initialized = true;
    }
    if ( !matchdict.src_ctr_pixel_v_initialized )
      matchdict.src_ctr_pixel_v.resize( contour_data.m_plane_atomics_v.at(src_planeid).size() );
      
    int nsrcpix_in_ctr = 0; // number of source image pixels inside a contour
    int ntarpix_in_ctr = 0; // number of target image pixels inside a contour
    int nflow_into_ctr = 0; // number of pixels that flow into a contour. unlikely zero, so here for a check
    for ( int r=0; r<(int)srcmeta.rows(); r++) {
      
      // for each row, we find the available contours on the images.
      // saves us search each time

      std::set< int >  tar_ctr_ids;     // (crop) index of target contours in this row
      std::vector<int> src_cols_in_ctr; // columns on the source (crop) image that have ctrs
      std::map<int,int> src_cols2ctrid; // map from (crop) src col to src ctr
      src_cols_in_ctr.reserve(20);

      // std::cout << "------------------------------------------" << std::endl;
      // std::cout << "Find row=" << r << " contours" << std::endl;

      // Find contours on source image in this row
      // std::cout << "source: ";      
      for ( int c=0; c<(int)srcmeta.cols(); c++) {
	if ( src_adc_crop.pixel(r,c)<threshold )
	  continue;

        float full_row = srcfullmeta.row( srcmeta.pos_y(r) );
        float full_col = srcfullmeta.col( srcmeta.pos_x(c) );
	cv::Point pt( full_col, full_row ); // point in the full image

        // search the contours
        for ( size_t ictr=0; ictr<contour_data.m_plane_atomics_v[src_planeid].size(); ictr++ ) {
          auto const& ctr     = contour_data.m_plane_atomics_v[src_planeid][ictr];
          auto const& ctrmeta = contour_data.m_plane_atomicmeta_v[src_planeid][ictr];

          // bbox check
          if ( pt.x < ctrmeta.getMinX() || pt.x > ctrmeta.getMaxX() ) continue;
          if ( pt.y < ctrmeta.getMinY() || pt.y > ctrmeta.getMaxY() ) continue;
          
	  double result =  cv::pointPolygonTest( ctr, pt, false );
	  if ( result>=0 ) {
	    src_cols_in_ctr.push_back( c );
	    src_cols2ctrid[c] = ictr;
	    //std::cout << " " << ictr;
            // store in lookup map for crop
            int pixindex = full_row*srcfullmeta.cols() + full_col;
            matchdict.src_ctr_pixel_v.at( ictr ).push_back( pixindex );
	    matchdict.src_ctr_index_map[ pixindex ] = ictr;
            nsrcpix_in_ctr++;;
	    break;
	  }
	}
      }//end of cropped column loop
      //std::cout << std::endl;

      // Find Contours on the target image in this row
      //std::cout << "target: ";      
      for ( int c=0; c<(int)tarmeta.cols(); c++) {
	if ( tar_adc_crop.pixel(r,c)<threshold )
	  continue;

        float full_row = tarfullmeta.row( tarmeta.pos_y(r) );
        float full_col = tarfullmeta.col( tarmeta.pos_x(c) );
        
	cv::Point pt( full_col, full_row );
        // search the contours
        for ( size_t ictr=0; ictr<contour_data.m_plane_atomics_v[tar_planeid].size(); ictr++ ) {
          auto const& ctr     = contour_data.m_plane_atomics_v[tar_planeid][ictr];
          auto const& ctrmeta = contour_data.m_plane_atomicmeta_v[tar_planeid][ictr];

          // bbox check
          if ( pt.x < ctrmeta.getMinX() || pt.x > ctrmeta.getMaxX() ) continue;
          if ( pt.y < ctrmeta.getMinY() || pt.y > ctrmeta.getMaxY() ) continue;          
	
	  double result =  cv::pointPolygonTest( ctr, pt, false );
	  if ( result>=0 ) {
	    tar_ctr_ids.insert( ictr );
            matchdict.tar_ctr_index_map[ full_row*tarfullmeta.cols() + full_col ] = ictr;
	    //tar_img2ctrindex[ r*tarmeta.cols() + c ] = ictr;
            ntarpix_in_ctr++;
	    break;
	  }
	}//end of target contour loop
      }//end of target col loop
      //std::cout << std::endl;
      
      // Nothing in this row, move on to the next row
      if ( src_cols_in_ctr.size()==0 || tar_ctr_ids.size()==0 ) {
	//std::cout << "nothing to match" << std::endl;
	continue;
      }

      // now loop over source columns in contours and make matches to target contours
      for ( auto const& source_col : src_cols_in_ctr ) {

	float flow     = flow_img_crop.pixel(r,source_col);
	int target_col = source_col+flow;
        if ( target_col>= tarmeta.cols() )
          continue; // flow out of bounds
        if ( target_col<0 )
          continue; // flow out of bounds
        float tar_row_full = tarfullmeta.row( tarmeta.pos_y(r) );
        float tar_col_full = tarfullmeta.col( tarmeta.pos_x(target_col) );
        cv::Point tar_pt( tar_col_full, tar_row_full );
	
	// retrieve the contour we're in
	int src_ctr_id = src_cols2ctrid[source_col];
	
	// loop through the target contours, find the contour closest to the flowed-to target wire
        int closest_contour_idx = -1;
        float closest_dist = -50;
	for ( auto const& ctrid : tar_ctr_ids ) {
          // have to move point back to full image coordinates
          
	  float dist = cv::pointPolygonTest( contour_data.m_plane_atomics_v[tar_planeid][ctrid], tar_pt, true );
          // if dist is negatice, its outside the contour
          // if dist is positive, its inside the contour

          if ( dist>=0 ) {
            // inside, no need to search further
            closest_contour_idx = ctrid;
            closest_dist        = 0.;
            nflow_into_ctr++;
            break;
          }
          else if ( dist>-1*max_dist_to_target_contour ) {
            // enforce a max distance to a contour
            if ( closest_contour_idx<0 || closest_dist < dist ) {
              closest_contour_idx = ctrid;
              closest_dist        = dist;
            }
          }
        }//end of contour loop

        // if no contour found, move on
        if ( closest_contour_idx<0 ) continue;

        // else we store/append a src-tar contour match for this source pixel
        
        // store the match data
        // --------------------

        // create a pair key
        SrcTarPair_t idpair = { src_ctr_id, closest_contour_idx };

        // check if we have one for this pair already
        auto it_indexmap = matchdict.find( idpair );
        if ( it_indexmap==matchdict.end() ) {
	  // if the map doesn't have the pair we're looking for, we create the data
          ContourFlowMatch_t x( src_ctr_id,  closest_contour_idx);
          matchdict.insert( std::pair<SrcTarPair_t,ContourFlowMatch_t>(idpair,x) );
          it_indexmap = matchdict.find(idpair);
        }
        
        
        // store the specific pixel flow information
        float src_full_wire = srcmeta.pos_x(source_col);
        float tar_full_wire = tarmeta.pos_x(target_col);
        float in_contour_wire = tar_full_wire;

        if ( closest_dist<0 ) {
          // if we missed a contour, we find the offset that moves us into it
          const ublarcvapp::Contour_t& tar_ctr = contour_data.m_plane_atomics_v[tar_planeid][closest_contour_idx];
          float shifted_dist = -50;
          float shifted_pos  = 0;
          for (int ioffset=-1; ioffset<=1; ioffset+=2) {
            float tar_col_full_shifted = tar_col_full + float(ioffset)*fabs(closest_dist);
            float testdist = cv::pointPolygonTest( tar_ctr, cv::Point( tar_col_full_shifted, tar_row_full ), true );
            if ( testdist>shifted_dist ) {
              shifted_dist = testdist;
              shifted_pos = tar_col_full_shifted;
            }
          }
          if ( shifted_dist>-50 )
            in_contour_wire = shifted_pos;
        }

          
        ContourFlowMatch_t::FlowPixel_t flowpix;
        flowpix.src_wire = src_full_wire;
        flowpix.tar_wire = in_contour_wire;
        flowpix.tar_orig = tar_full_wire;
        flowpix.row      = tar_row_full;
        flowpix.tick     = tarmeta.pos_y( tar_row_full );
        flowpix.pred_miss = std::fabs(closest_dist);
        flowpix.dist2cropcenter = std::fabs(source_col - (float)srcmeta.cols()/2);
        it_indexmap->second.matchingflow_v.push_back( flowpix );
        // log.send( larcv::msg::kDEBUG,__FUNCTION__,__LINE__)
        //   << "creating a src-tar datapoint: "
        //   << "[src=" << src_full_wire << " -> "
        //   << "tar(orig)=" << tar_full_wire << " tar(shift)=" << in_contour_wire << "] "
        //   << "dist2ctr=" << closest_dist << "(orig) " << 
        //   << std::endl;
        
      }//end of loop over source cols
    }//end of loop over rows
    log.send( larcv::msg::kINFO,__FUNCTION__,__LINE__ )
      << "Number of pixels inside crop that flowed into a contour: "
      << nflow_into_ctr
      << std::endl;
  }

}
