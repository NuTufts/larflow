#include "FlowContourMatch.h"

// larcv
#include "larcv/core/Base/LArCVBaseUtilFunc.h"
#include "larcv/core/Base/larcv_logger.h"
#include "larcv/core/DataFormat/ROI.h"

// ublarcvapp
#include "ublarcvapp/ContourTools/ContourShapeMeta.h"
#include "ublarcvapp/ContourTools/ContourClusterAlgo.h"
#include "ublarcvapp/UBImageMod/UBSplitDetector.h"
#include "ublarcvapp/UBImageMod/UBCropLArFlow.h"

#include "makehitsfrompixels.h"
#include "makecontourflowmatches.h"
#include "makesimpleflowhits.h"
#include "ContourFlowMatch.h"

#include "larcv/core/ROOTUtil/ROOTUtils.h"
#include "TH2D.h"

namespace larflow {

  /**
   * create 3D spacepoints using the LArFlow network outputs
   *
   * @param[in] adc_crop_v vector of Image2D containing ADC values. Whole view.
   * @param[in] flowy2u vector of SparseImage crops containing Y->U flow predictions
   * @param[in] flowy2v vector of SparseImage crops containing Y->V flow predictions
   * @param[in] threshold ignore information where pixel value is below threshold
   * @param[in] cropcfg path to config file for UBSplitDetector. default can be found in repo as 'ubcrop.cfg'.
   * @param[in] verbosity Control amount of output: [0] DEBUG [1] INFO [2] NORMAL [3] QUIET. Default [2].
   * @param[in] visualize If true, we display the flow and save an png.  Advisable to only one run event in this mode.
   *
   * return vector of larflow3dhit
   */
  std::vector<larlite::larflow3dhit> makeFlowHitsFromSparseCrops( const std::vector<larcv::Image2D>& adc_v,
                                                                  const std::vector<larcv::SparseImage>& flowdata,
                                                                  const float threshold,
                                                                  const std::string cropcfg,
                                                                  const larcv::msg::Level_t verbosity,
                                                                  const bool visualize ) 
  {

    larcv::logger log("larflow::makeFlowHitsFromSparseCrops");
    log.set(verbosity);

    //std::cout << "make pixels from images" << std::endl;
    if ( log.debug() ) log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__) << "make pixels from images" << std::endl;

    /// step: make hits from pixels in image, for each plane
    std::vector<larlite::event_hit> hit_vv;
    for ( auto const& adc : adc_v ) {
      auto hit_v = makeHitsFromWholeImagePixels( adc, threshold );
      if ( log.debug() ) log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__)
        << "made " << hit_v.size() << " pixel hits in plane[" << adc.meta().plane() << "]" << std::endl;
      hit_vv.emplace_back( std::move(hit_v) );
    }

    /// step: make contours on whole view images.
    ublarcvapp::ContourClusterAlgo contours;
    contours.analyzeImages( adc_v );

    if ( log.debug() ) {
      log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__) << "ContourClusterAlgo results:" << std::endl;
      for ( size_t p=0; p<3; p++ ) {
        log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__) << "  atomic clusters in plane[" << p << "]: "
                                                           << contours.m_plane_atomicmeta_v[p].size() << std::endl;
      }
    }

    // step: crop image
    larcv::PSet cropper_cfg = larcv::CreatePSetFromFile( cropcfg, "ubcropcfg" );
    ublarcvapp::UBSplitDetector cropper_algo;
    cropper_algo.configure( cropper_cfg.get<larcv::PSet>("UBSplitDetector") );
    std::vector<larcv::Image2D> cropped_v;
    std::vector<larcv::ROI> roi_v;
    cropper_algo.process( adc_v, cropped_v, roi_v );

    // loop over flow crops, for each crop,
    // in each crop, for each source pixel,
    //   accumulate number of flows between unique (src/tar) index pairs
    // repeat this for all flow directions
    std::vector< ContourFlowMatchDict_t > matchdict_v(2);
    int src_index[2] = { 2, 2 }; // Y, Y
    int tar_index[2] = { 0, 1 }; // U, V
    for ( int iflowdir=0; iflowdir<2; iflowdir++ ) {

      const larcv::Image2D& srcimg = adc_v.at( src_index[iflowdir] );
      const larcv::Image2D& tarimg = adc_v.at( tar_index[iflowdir] );

      // loop through flow data
      for ( size_t iflowdata=0; iflowdata<flowdata.size()/2; iflowdata++ ) {
        const larcv::SparseImage& sparseimg = flowdata.at( 2*iflowdata + iflowdir );

        // find the proper cropped meta
        int cropped_index = -1;
        for ( size_t icropset=0; icropset<cropped_v.size()/3; icropset++ ) {
          const larcv::Image2D& crop_y = cropped_v.at( 3*icropset+2 );
          // do the meta's match
          if ( crop_y.meta()==sparseimg.meta_v().front() ) {
            if ( log.debug() ) {
              log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__)
                << "Meta matched between cropped_adc[" << icropset << "] "
                << "and sparse flowimage[" << iflowdata << "] "
                << std::endl;
            }
            cropped_index = (int)icropset;
            break;
          }
        }
        if ( cropped_index==-1 ) {
          for ( size_t icropset=0; icropset<cropped_v.size()/3; icropset++ ) {
            const larcv::Image2D& crop_y = cropped_v.at( 3*icropset+2 );
            log.send(larcv::msg::kDEBUG) << crop_y.meta().dump() << std::endl;
          }
          log.send( larcv::msg::kCRITICAL, __FUNCTION__, __LINE__ ) << "Could not find matching crop.\n"
                                                                    << "       sparsemeta=" << sparseimg.meta_v().front().dump()
                                                                    << std::endl;
          throw std::runtime_error("Could not find matching crop");
        }
        const larcv::Image2D& src_adc_crop = cropped_v.at( 3*cropped_index+2 );
        const larcv::Image2D& tar_adc_crop = cropped_v.at( 3*cropped_index+tar_index[iflowdir] );

        createMatchData(  contours, srcimg, tarimg,
                          sparseimg.as_Image2D().at(iflowdir),
                          src_adc_crop, tar_adc_crop,
                          matchdict_v.at(iflowdir),
                          threshold, 30.0, verbosity, visualize );
        
      }//end of flow data loop
    }//end of flow direction loop
    
    // make this from this compiled information
    std::vector<larlite::larflow3dhit> flowhits_v
      = makeSimpleFlowHits( adc_v, contours, matchdict_v );

    if ( log.info() )
      log.send(larcv::msg::kINFO,__FUNCTION__,__LINE__) << "Made " << flowhits_v.size() << " flowhits" << std::endl;
    
    return flowhits_v;
  }
  
  // ==============================================================================
  //  MCTRUTH FUNCTIONS
  // --------------------
  
  /**
   * create 3D spacepoints using true flow information
   *
   * @param[in] adc_v vector of Image2D containing ADC values. Whole view.
   * @param[in] chstatus EventChStatus object containing good/bad channel list
   * @param[in] larflow_v vector of Image2D containing true flow invalues. Whole view.
   * @param[in] threshold ignore information where pixel value is below threshold
   * @param[in] cropcfg path to config file for UBSplitDetector. default can be found in repo as 'ubcroptrueflow.cfg'.
   * @param[in] verbosity verbosity Control amount of output: [0] kDEBUG [1] kINFO [2] kNORMAL. Default [2].
   * @param[in] visualize If true, we display the flow and save an png.  Advisable to only one run event in this mode.
   *
   * return vector of larflow3dhit
   */
  std::vector<larlite::larflow3dhit> makeTrueFlowHitsFromWholeImage( const std::vector<larcv::Image2D>& adc_v,
                                                                     const larcv::EventChStatus& chstatus,
                                                                     const std::vector<larcv::Image2D>& larflow_v,
                                                                     const float threshold,
                                                                     const std::string cropcfg,
                                                                     const larcv::msg::Level_t verbosity,
                                                                     const bool visualize ) 
  {
    
    larcv::logger log("larflow::makeTrueFlowHitsFromWholeImage");
    log.set(verbosity);
    
    //std::cout << "make pixels from images" << std::endl;
    if ( log.debug() ) log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__) << "make true flow pixels from whole images" << std::endl;
    
    /// step: make hits from pixels in image, for each plane
    std::vector<larlite::event_hit> hit_vv;
    for ( auto const& adc : adc_v ) {
      auto hit_v = makeHitsFromWholeImagePixels( adc, threshold );
      if ( log.debug() ) {
        log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__)
          << "made " << hit_v.size() << " pixel hits in plane[" << adc.meta().plane() << "]" << std::endl;
      }
      hit_vv.emplace_back( std::move(hit_v) );
    }
    
    /// step: make contours on whole view images.
    ublarcvapp::ContourClusterAlgo contours;
    contours.analyzeImages( adc_v );

    if ( log.debug() ) {
      log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__) << "ContourClusterAlgo results:" << std::endl;
      for ( size_t p=0; p<3; p++ ) {
        log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__) << "  atomic clusters in plane[" << p << "]: "
                                                         << contours.m_plane_atomicmeta_v[p].size() << std::endl;
      }
    }
    
    // step: splitter and cropper
    larcv::PSet cropper_cfg = larcv::CreatePSetFromFile( cropcfg, "ubcropcfg" );
    
    ublarcvapp::UBSplitDetector split_algo;
    split_algo.configure( cropper_cfg.get<larcv::PSet>("UBSplitDetector") );
    
    ublarcvapp::UBCropLArFlow   crop_algo;
    crop_algo.configure(  cropper_cfg.get<larcv::PSet>("UBCropLArFlow") );
    
    // run splitter
    std::vector<larcv::Image2D> cropped_v;
    std::vector<larcv::ROI> roi_v;
    split_algo.process( adc_v, cropped_v, roi_v );
    
    // run larflow cropper
    std::vector<larcv::Image2D> cropped_flow_v;
    std::vector<float> threshold_v( threshold, 3 );
    for ( auto const& roi : roi_v ) {
      std::vector<larcv::Image2D> crop_from_roi_v;
      crop_algo.make_cropped_flow_images( 2, roi, adc_v,
                                          chstatus, larflow_v, threshold_v, 
                                          crop_from_roi_v );
      for ( auto& img : crop_from_roi_v ) {
        cropped_flow_v.emplace_back( std::move(img) );
      }
    }
    
    // check crop alignment
    if ( log.debug() )
      log.send( larcv::msg::kDEBUG,__FUNCTION__,__LINE__)
        << " ncropped=" << cropped_v.size() << " nflow=" << cropped_flow_v.size() << std::endl;
    
    if ( cropped_flow_v.size()/2!=cropped_v.size()/3 ) {
      log.send( larcv::msg::kCRITICAL,__FUNCTION__,__LINE__ )
        << "number of flow and ADC crops are inconsistent" << std::endl;
      throw std::runtime_error("number of flow and ADC crops are inconsistent");
    }

    // loop over flow crops, for each crop,
    // in each crop, for each source pixel,
    //   accumulate number of flows between unique (src/tar) index pairs
    // repeat this for all flow directions
    std::vector< ContourFlowMatchDict_t > matchdict_v(2);
    int src_index[2] = { 2, 2 }; // Y, Y
    int tar_index[2] = { 0, 1 }; // U, V
    for ( int iflowdir=0; iflowdir<2; iflowdir++ ) {

      const larcv::Image2D& srcimg = adc_v.at( src_index[iflowdir] );
      const larcv::Image2D& tarimg = adc_v.at( tar_index[iflowdir] );

      // loop through flow data
      for ( size_t iflowdata=0; iflowdata<cropped_flow_v.size()/2; iflowdata++ ) {
        const larcv::Image2D& truthflow = cropped_flow_v.at( 2*iflowdata + iflowdir );

        // find the proper cropped meta
        int cropped_index = -1;
        for ( size_t icropset=0; icropset<cropped_v.size()/3; icropset++ ) {
          const larcv::Image2D& crop_y = cropped_v.at( 3*icropset+2 );
          // do the meta's match
          if ( crop_y.meta()==truthflow.meta() ) {
            if ( log.debug() ) {
              log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__)
                << "Meta matched between cropped_adc[" << icropset << "] "
                << "and sparse flowimage[" << iflowdata << "] "
                << std::endl;
            }
            cropped_index = (int)icropset;
            break;
          }
        }
        if ( cropped_index==-1 ) {
          for ( size_t icropset=0; icropset<cropped_v.size()/3; icropset++ ) {
            const larcv::Image2D& crop_y = cropped_v.at( 3*icropset+2 );
            log.send(larcv::msg::kDEBUG) << crop_y.meta().dump() << std::endl;
          }
          log.send( larcv::msg::kCRITICAL, __FUNCTION__, __LINE__ ) << "Could not find matching crop.\n"
                                                                    << "       truthmeta=" << truthflow.meta().dump()
                                                                    << std::endl;
          throw std::runtime_error("Could not find matching crop");
        }
        const larcv::Image2D& src_adc_crop = cropped_v.at( 3*cropped_index+2 );
        const larcv::Image2D& tar_adc_crop = cropped_v.at( 3*cropped_index+tar_index[iflowdir] );
        
        createMatchData(  contours, srcimg, tarimg,
                          truthflow,
                          src_adc_crop, tar_adc_crop,
                          matchdict_v.at(iflowdir),
                          threshold, 30.0, verbosity, visualize );
                          
        
      }//end of flow data loop
    }//end of flow direction loop
    
    // make this from this compiled information
    std::vector<larlite::larflow3dhit> flowhits_v
      = makeSimpleFlowHits( adc_v, contours, matchdict_v );

    if ( log.info() ) log.send(larcv::msg::kINFO,__FUNCTION__,__LINE__) << "Made " << flowhits_v.size() << " flowhits" << std::endl;
    return flowhits_v;
  }
  
  // same as above, but with EventChStatus pointer, better for python?
}
