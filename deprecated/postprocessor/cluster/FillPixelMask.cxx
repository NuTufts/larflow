#include "FillPixelMask.h"

#include <sstream>

// ROOT
#include "TH2D.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"

// larcv2
#include "larcv/core/ROOTUtil/ROOTUtils.h"

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

// postprocessor/contourtools
#include "ContourTools/ContourCluster.h"
#include "ContourTools/ContourROOTVisUtils.h"

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
   * @return Filled-out pixelmask for each plane
   */
  std::vector<larlite::pixelmask> FillPixelMask::fillMask( const std::vector<larcv::Image2D>& adc_v,
							   const larlite::larflowcluster& cluster3d,
							   const std::vector<larlite::pixelmask>& origmask_v ) {
    // first we need to prep the opencv image.
    // we a region around the pixelmask

    const int padding = 50;
    const float threshold = 5;

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
      bbox[1] = ( bbox[1]<adcmeta.min_y() ) ? adcmeta.min_y() : bbox[1];      
      bbox[2] = ( bbox[2]>adcmeta.max_x() ) ? adcmeta.max_x() : bbox[2];
      bbox[3] = ( bbox[3]>adcmeta.max_y() ) ? adcmeta.max_y() : bbox[3];

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


    // for each plane:
    // - loop through all the original mask pixels
    // - mark the clusters we land in as active
    // - make a pixelmask with original and active contour pixels
    std::vector< larlite::pixelmask > filledmask_v;
    filledmask_v.reserve( nplanes );
    
    std::vector< std::vector<bool> > orig_activelist_vv( nplanes );
    std::vector< std::vector<bool> > atomic_activelist_vv( nplanes );
    
    for ( size_t p=0; p<nplanes; p++ ) {

      auto const& cropmeta = cropped_v.at(p).meta();
      auto const& origmask = origmask_v.at(p);

      size_t atomic_ncontours = contouralgo.m_plane_atomicmeta_v[p].size();
      std::vector<bool>& atomic_activecontour_v = atomic_activelist_vv[p];
      atomic_activecontour_v.resize( atomic_ncontours, false);
      
      size_t orig_ncontours   = contouralgo.m_plane_contours_v[p].size();
      std::vector<bool>& orig_activecontour_v   = orig_activelist_vv[p];
      orig_activecontour_v.resize( orig_ncontours, false);

      for ( size_t ipt=0; ipt<origmask.len(); ipt++ ) {
	std::vector<float> pos = origmask.point( ipt );
	// act on pixels inside cropped image
	if ( cropmeta.contains( pos[1], pos[0] ) ) {
	  // convert into pixel (row,col)
	  size_t row = cropmeta.row( pos[1] );
	  size_t col = cropmeta.col( pos[0] );

	  // test each inactive contour (both original and atomic)

	  // atomic contours
	  for ( size_t icluster=0; icluster<atomic_ncontours; icluster++ ) {
	    if ( atomic_activecontour_v[icluster] ) {
	      // already on
	      continue;
	    }
	    // else check
	    const larlitecv::Contour_t& contour = contouralgo.m_plane_atomics_v[p].at(icluster);
	    float testresult    = cv::pointPolygonTest( contour, cv::Point2f( (float)col, (float)row ), false );
	    if ( testresult>=0 )
	      atomic_activecontour_v[icluster] = true;
	  }

	  // original contours
	  for ( size_t icluster=0; icluster<orig_ncontours; icluster++ ) {
	    
	    if ( orig_activecontour_v[icluster] ) {
	      // already on
	      continue;
	    }
	    
	    // else check
	    const larlitecv::Contour_t& contour = contouralgo.m_plane_contours_v[p].at(icluster);
	    float testresult = cv::pointPolygonTest( contour, cv::Point2f( (float)col, (float)row ), false );
	    
	    if ( testresult>=0 )
	      orig_activecontour_v[icluster] = true;
	    
	  }
	  
	}//if meta contains mask point
      }//loop over mask points
	
      // ********************************************
      // ** DEBUG/DEV **
      if ( fDebugMode ) {          
	std::cout << "[FillPixelMask::fillMask] contours on plane " << p
		  << " = " << contouralgo.m_plane_atomicmeta_v.size() << std::endl;
	
	TH2D debughist = debugMakeTH2D( cropped_v.at(p),
					&origmask_v.at(p),
					nullptr,
					&(contouralgo.m_plane_atomicmeta_v.at(p) ) );
	
	std::vector< TGraph > g_atomic_contours =
	  larlitecv::ContourROOTVisUtils::contour_as_tgraph( contouralgo.m_plane_atomics_v.at(p), &cropped_v.at(p).meta() );
	std::vector< TGraph > g_orig_contours =
	  larlitecv::ContourROOTVisUtils::contour_as_tgraph( contouralgo.m_plane_contours_v.at(p), &cropped_v.at(p).meta() );
	

	char debugpng[100];
	sprintf(debugpng, "fillpixelmask_debugimg_plane%d.png",(int)p);
	gStyle->SetOptStat(0);
	TCanvas c("c","c",800,600);
	c.Draw();
	debughist.Draw("colz");

	size_t orig_ncontours = contouralgo.m_plane_contours_v[p].size();	
	for ( size_t icluster=0; icluster<orig_ncontours; icluster++ ) {
	  auto& g = g_orig_contours.at(icluster);
	  g.SetLineStyle(2);
	  g.SetLineWidth(2);
	  if ( orig_activelist_vv.at(p).at(icluster) )
	    g.SetLineColor( kBlue );
	  g.Draw("L");
	}

	size_t atomic_ncontours = contouralgo.m_plane_atomicmeta_v[p].size();	
	for ( size_t icluster=0; icluster<atomic_ncontours; icluster++ ) {
	  auto& g = g_atomic_contours.at(icluster);
	  if ( atomic_activelist_vv.at(p).at(icluster) )
	    g.SetLineColor( kRed );
	  g.Draw("L");
	}
	
	c.Update();
	c.SaveAs(debugpng);

      }
      // END OF DEBUG
      // ********************************************

      // now we rebuild the pixel cluster
      // -- we create a copy of the adc window
      // -- we mask pixels above threshold with value 1
      // -- we mark current pixels as 2
      // -- we mark pixels inside active atomic contour 3
      // -- we mark pixels inside active original contour 4

      auto const& cropimg = cropped_v.at(p);
      larcv::Image2D croptag( cropmeta );
      croptag.paint(0.0);

      // use the orig mask to tag pixels
      for ( size_t ipt=0; ipt<origmask.len(); ipt++ ) {
	std::vector<float> pos = origmask.point( ipt );
	// act on pixels inside cropped image
	if ( cropmeta.contains( pos[1], pos[0] ) ) {
	  size_t row = cropmeta.row( pos[1] );
	  size_t col = cropmeta.col( pos[0] );
	  croptag.set_pixel( row, col, 2 );
	}
      }

      // now loop, saving pixels to use in the pixels
      std::vector< std::vector<float> > fillmaskpoint_v;
      
      for ( size_t r=0; r<cropmeta.rows(); r++ ) {
	for ( size_t c=0; c<cropmeta.cols(); c++ ) {

	  if ( cropimg.pixel( r, c )<threshold ) continue;
	  
	  int pixval = 1;

	  // aplly orignal mask tag
	  if ( croptag.pixel(r,c)>0 ) pixval = croptag.pixel(r,c);

	  // if untagged, check if in atmoic contour	  
	  if ( pixval==1 ) {
	    for ( size_t icontour=0; icontour<atomic_ncontours; icontour++ ) {
	      if ( atomic_activecontour_v[icontour] ) {
		const larlitecv::Contour_t& contour = contouralgo.m_plane_atomics_v[p].at(icontour);
		float testresult    = cv::pointPolygonTest( contour, cv::Point2f( (float)c, (float)r ), false );
		if ( testresult>=0 ) {
		  pixval = 3;
		  break;
		}
	      }
	    }
	  }

	  // if still untagged check original contours
	  if ( pixval==1 ) {
	    for ( size_t icontour=0; icontour<orig_ncontours; icontour++ ) {
	      if ( orig_activecontour_v[icontour] ) {
		const larlitecv::Contour_t& contour = contouralgo.m_plane_contours_v[p].at(icontour);
		float testresult    = cv::pointPolygonTest( contour, cv::Point2f( (float)c, (float)r ), false );
		if ( testresult>=0 ) {
		  pixval = 4;
		  break;
		}
	      }
	    }
	  }
	      
	  // set the tag
	  croptag.set_pixel( r, c, (float)pixval );
	  
	  // fill the point list if inside contour
	  if ( pixval>1 ) {
	    std::vector<float> fillpoint = { (float)cropmeta.pos_x(c), (float)cropmeta.pos_y(r), (float)cropimg.pixel(r,c), (float)pixval };
	    fillmaskpoint_v.push_back( fillpoint );
	  }
	  
	}// end of col loop
      }// end of row loop

      // build the pixelmask
      // 0=label
      // 4=dim_per_point (x,y,adc value, tag)
      larlite::pixelmask fillmask( 0, fillmaskpoint_v,
				   cropmeta.min_x(), cropmeta.min_y(), cropmeta.max_x(), cropmeta.max_y(),
				   cropmeta.cols(), cropmeta.rows(), 4 );

      filledmask_v.emplace_back( std::move(fillmask) );
      
    }//end of plane loop
      
    return filledmask_v;
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

  /**
   * for debug and dev, make a th2d where different pixels are tagged
   * 0: no tags
   * 1: pixel above threshold
   * 2: pixel in original mask
   * 3: pixel in filled mask
   * 10+n: contour pixels (n is for ID)
   *
   * @param[in] img ADC image
   * @param[in] origmask original mask
   * @param[in] filledmask filled mask
   * @param[in] contour_v list of contours
   */
  TH2D FillPixelMask::debugMakeTH2D( const larcv::Image2D& img,
				     const larlite::pixelmask* origmask,
				     const larlite::pixelmask* filledmask,
				     const std::vector< larlitecv::ContourShapeMeta >* contour_v ) {
    TH2D hist = larcv::as_th2d( img, "fillpixelmask_debughist" );
    
    // tag above threshold
    for ( size_t r=0; r<img.meta().rows(); r++ ) {
      for ( size_t c=0; c<img.meta().cols(); c++ ) {
	if ( hist.GetBinContent( c+1, r+1 )>10.0 )
	  hist.SetBinContent( c+1, r+1, 1.0 );
	else
	  hist.SetBinContent( c+1, r+1, 0.0 );
      }
    }
    
    // tag original pixels
    if ( origmask ) {

      auto const& meta = img.meta();

      for ( size_t ipt=0; ipt<origmask->len(); ipt++ ) {
	std::vector<float> pt = origmask->point(ipt);
	if ( meta.contains( pt[1], pt[0] ) ) {
	  size_t r = meta.row( pt[1] );
	  size_t c = meta.col( pt[0] );
	  hist.SetBinContent( c+1, r+1, 2 );
	}
      }
      
    }

    // tag filled pixels
    if ( filledmask ) {

      auto const& meta = img.meta();

      for ( size_t ipt=0; ipt<filledmask->len(); ipt++ ) {
	std::vector<float> pt = filledmask->point(ipt);
	if ( meta.contains( pt[1], pt[0] ) ) {
	  size_t r = meta.row( pt[1] );
	  size_t c = meta.col( pt[0] );
	  if ( hist.GetBinContent(c+1,r+1)<=1 )
	    hist.SetBinContent( c+1, r+1, 3 );
	}
      }
      
    }

    // label contours
    if ( contour_v ) {
      size_t icontour=0;
      for ( auto const& contour : *contour_v ) {
	for ( auto const& cvpoint : contour ) {
	  hist.SetBinContent( (int)cvpoint.x+1, (int)cvpoint.y+1, 10+icontour );
	}
	icontour++;
      }
    }


    return hist;
  }//end of debugMakeTH2D

  
  
}
