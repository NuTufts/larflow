#include "FlowContourMatch.h"

// larcv
#include "larcv/core/Base/LArCVBaseUtilFunc.h"
#include "larcv/core/Base/larcv_logger.h"
#include "larcv/core/DataFormat/ROI.h"

// ublarcvapp
#include "ublarcvapp/ContourTools/ContourShapeMeta.h"
#include "ublarcvapp/ContourTools/ContourClusterAlgo.h"
#include "ublarcvapp/UBImageMod/UBSplitDetector.h"

#include "makehitsfrompixels.h"
#include "makecontourflowmatches.h"
#include "makesimpleflowhits.h"
#include "ContourFlowMatch.h"


namespace larflow {

  /**
   * create 3D spacepoints using the LArFlow network outputs
   *
   * @param[in] adc_crop_v vector of Image2D containing ADC values. Whole view.
   * @param[in] flowy2u vector of SparseImage crops containing Y->U flow predictions
   * @param[in] flowy2v vector of SparseImage crops containing Y->V flow predictions
   * @param[in] threshold ignore information where pixel value is below threshold
   * @param[in] cropcfg path to config file for UBSplitDetector. default can be found in repo as 'ubcrop.cfg'.
   *
   * return vector of larflow3dhit
   */
  std::vector<larlite::larflow3dhit> makeFlowHitsFromSparseCrops( const std::vector<larcv::Image2D>& adc_v,
                                                                  const std::vector<larcv::SparseImage>& flowdata,
                                                                  const float threshold,
                                                                  const std::string cropcfg ) 
  {

    larcv::logger log("larflow::makeFlowPixels");

    //std::cout << "make pixels from images" << std::endl;
    log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__) << "make pixels from images" << std::endl;

    /// step: make hits from pixels in image, for each plane
    std::vector<larlite::event_hit> hit_vv;
    for ( auto const& adc : adc_v ) {
      auto hit_v = makeHitsFromWholeImagePixels( adc, threshold );
      log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__)
        << "made " << hit_v.size() << " pixel hits in plane[" << adc.meta().plane() << "]" << std::endl;
      hit_vv.emplace_back( std::move(hit_v) );
    }

    /// step: make contours on whole view images.
    ublarcvapp::ContourClusterAlgo contours;
    contours.analyzeImages( adc_v );

    log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__) << "ContourClusterAlgo results:" << std::endl;
    for ( size_t p=0; p<3; p++ ) {
      log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__) << "  atomic clusters in plane[" << p << "]: "
                                                         << contours.m_plane_atomicmeta_v[p].size() << std::endl;
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
            log.send(larcv::msg::kDEBUG,__FUNCTION__,__LINE__)
              << "Meta matched between cropped_adc[" << icropset << "] "
              << "and sparse flowimage[" << iflowdata << "] "
              << std::endl;
            cropped_index = (int)icropset;
            break;
          }
        }
        if ( cropped_index==-1 ) {
          throw std::runtime_error("Could not find matching crop");
        }
        const larcv::Image2D& src_adc_crop = cropped_v.at( 3*cropped_index+2 );
        const larcv::Image2D& tar_adc_crop = cropped_v.at( 3*cropped_index+tar_index[iflowdir] );
        
        createMatchData(  contours, srcimg, tarimg,
                          sparseimg.as_Image2D().at(iflowdir),
                          src_adc_crop, tar_adc_crop,
                          matchdict_v.at(iflowdir),
                          threshold );
                          
        
      }//end of flow data loop
    }//end of flow direction loop
    
    // make this from this compiled information
    std::vector<larlite::larflow3dhit> flowhits_v
      = makeSimpleFlowHits( adc_v, contours, matchdict_v );

    log.send(larcv::msg::kINFO,__FUNCTION__,__LINE__) << "Made " << flowhits_v.size() << " flowhits" << std::endl;
    return flowhits_v;
  }
  

}
