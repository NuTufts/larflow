#include "FillPixelMask.h"

#include <sstream>

#include "ContourTools/ContourCluster.h"

namespace larflow {

  /** 
   * recover untagged pixels within a cluster and its pixelmask
   *
   * We first find contours around the pixels. 
   * Next, we perform defect analysis to find convex components.
   * We then find the contours onto which the original pixelmasks fall.
   * Finally, we collect the unmasked pixels within the overlapping contours.
   * Unmasked pixels and original pixels are collected into the new mask.
   * This is done for each plane.
   *
   * @param[in] adc_v ADC images for each plane
   * @param[in] cluster3d LArFlowCluster of 3D hits
   * @param[in] origmask_v Original PixelMask created from larflowcluster
   * @return 
   */
  std::vector<larlite::pixelmask> FillPixelMask::fillMask( const std::vector<larcv::Image2D>& adc_v,
							   const larlite::larflowcluster& cluster3d,
							   const std::vector<larlite::pixelmask>& origmask_v ) {
    // first we need to prep the opencv image.
    // we a region around the pixelmask

    const int padding = 10;

    size_t nplanes = adc_v.size();
    if ( nplanes!=origmask_v.size() ) {
      std::stringstream ss;
      ss << "[FillPixelMask::fillMask] the number of planes for the ADC images (" << nplanes << ") "
	 << "does not match the number of planes for the pixelmasks (" << origmask_v.size() << ")"
	 << std::endl;
    }

    // loop over the planes and prep crops around the original mask
    std::vector< larcv::Image2D > cropped_v;
    for ( size_t planeid=0; planeid<adc_v.size(); planeid++ ) {

      auto const& mask = origmask_v.at(planeid);
      const larcv::Image2D& adc = adc_v.at(planeid);
      const larcv::ImageMeta& adcmeta = adc.meta();

      std::vector<float> bbox = mask.as_vector_bbox();

      // pad
      bbox[0] -= padding*adcmeta.pixel_width();
      bbox[1] -= padding*adcmeta.pixel_height();
      bbox[2] += padding*adcmeta.pixel_width();
      bbox[3] += padding*adcmeta.pixel_height();

      // bounds check the new box
      bbox[0] = ( bbox[0]<adcmeta.min_x() ) ? adcmeta.min_x() : bbox[0];
      bbox[1] = ( bbox[1]<adcmeta.min_x() ) ? adcmeta.min_y() : bbox[1];      
      bbox[2] = ( bbox[2]>adcmeta.max_x() ) ? adcmeta.max_x() : bbox[2];
      bbox[3] = ( bbox[3]>adcmeta.max_x() ) ? adcmeta.max_y() : bbox[3];

      int rows = (bbox[3]-bbox[1])/adcmeta.pixel_height();
      int cols = (bbox[2]-bbox[0])/adcmeta.pixel_width();

      // define a new imagemeta
      larcv::ImageMeta cropmeta( bbox[0], bbox[1], bbox[2], bbox[3],
				 (size_t)rows, (size_t)cols,
				 adcmeta.id() );

      // use the new meta to crop the ADC image
      larcv::Image2D crop = adc.crop( cropmeta );

      cropped_v.emplace_back( std::move(crop) );
      
    }//end of plane loop


    // within the crop we generate split-contours
    larlitecv::ContourCluster contouralgo;
    contouralgo.analyzeImages( cropped_v, cropped_v, 10.0, 2 );

    for ( size_t p=0; p<nplanes; p++ ) {
      std::cout << "[FillPixelMask::fillMask] contours on plane " << p
		<< " = " << contouralgo.m_plane_atomicmeta_v.size() << std::endl;
    }
    
  }
  
  /** 
   * apply FillPixelMask::fillMask to entire set of pixelmasks/clusers in an event
   *
   * @param[in] adc_v Whole-view ADC images for each plane
   * @param[in] evcluster_v Event LArFlowClusters
   * @param[in] evmask_vv Event pixelmasks. Vector is for each plane. Each event_pixelmask contains clusters.
   * @return Filled-in pixelmasks for each cluster (each cluster has masks for each plane)
   */
  std::vector< std::vector<larlite::pixelmask> >
  FillPixelMask::fillMasks( const std::vector<larcv::Image2D>& adc_v, const larlite::event_larflowcluster& evcluster_v,
			    const std::vector<const larlite::event_pixelmask*>& evmask_vv) {
    
    std::vector< std::vector<larlite::pixelmask> > outmask_vv;

    // check the assumptions on the structure of the input
    if ( adc_v.size()!=evmask_vv.size() ) {
      std::stringstream ss;
      ss << "[FillFixelMask::fillMasks] number of adc plane images does not match the number of plane masks" << std::endl;
      throw std::runtime_error( ss.str() );
    }
    if ( evmask_vv.front()->size()!=evcluster_v.size() ) {
      std::stringstream ss;
      ss << "[FillFixelMask::fillMasks] number of larflow clusters does not match number of masks" << std::endl;
      throw std::runtime_error( ss.str() );
    }
    size_t nplanes   = adc_v.size();
    size_t nclusters = evcluster_v.size();

    for ( size_t icluster=0; icluster<nclusters; icluster++ ) {
      auto const& cluster = evcluster_v.at(icluster);

      std::vector<larlite::pixelmask> mask_v;
      for (size_t p=0; p<nplanes; p++ ) {
	mask_v.push_back( evmask_vv.at(p)->at(icluster) );
      }

      std::vector< larlite::pixelmask > filledmask_v = fillMask( adc_v, cluster, mask_v );

      outmask_vv.emplace_back( std::move(filledmask_v) );
    }
    
    return outmask_vv;
  }
  

}
