#include "ContourCluster.h"

#ifdef USE_OPENCV
#include "larcv/core/CVUtil/CVUtil.h"
#endif

// LArLite
#include "BasicTool/FhiclLite/PSet.h"

// LArOpenCV
#include "LArOpenCV/ImageCluster/AlgoClass/DefectBreaker.h"
#include "LArOpenCV/ImageCluster/AlgoData/TrackClusterCompound.h"

#include "TRandom3.h"

namespace larlitecv {


  void ContourCluster::clear() {
    cvimg_stage0_v.clear(); // unchanged images
    cvimg_stage1_v.clear(); // contour points over time scan
    cvimg_stage2_v.clear(); // 3D-matched contour points
    cvimg_stage3_v.clear(); // 3D-matched spacepointso

    m_plane_contours_v.clear();
    m_plane_hulls_v.clear();
    m_plane_defects_v.clear();
    m_plane_atomics_v.clear();
    m_plane_atomicmeta_v.clear();
  }
  
  void ContourCluster::analyzeImages( const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v, const float threshold, const int iterations ) {
    
    TRandom3 rand(1983);
    
    // ------------------------------------------------------------------------
    // NO OPENCV
#ifndef USE_OPENCV
    throw std::runtime_error( "In order to use ContourCluster, you must compile with OpenCV" );
    return sp_v;
#else
    // ------------------------------------------------------------------------
    // HAS OPENCV

    clear();
    
    // first convert the images into cv and binarize
    for ( auto const& img : img_v ) {
      cv::Mat cvimg = larcv::as_gray_mat( img, threshold, 256.0, 1.0 );
      cv::Mat cvrgb = larcv::as_mat_greyscale2bgr( img, threshold, 100.0 );
      cv::Mat thresh( cvimg );
      cv::threshold( cvimg, thresh, 0, 255, cv::THRESH_BINARY );
      cvimg_stage0_v.emplace_back( std::move(cvrgb) );
      cvimg_stage1_v.emplace_back( std::move(thresh) );
    }
    
    for (int p=0; p<3; p++) {
      // dilate image first
      cv::Mat& cvimg = cvimg_stage1_v[p];
      cv::dilate( cvimg, cvimg, cv::Mat(), cv::Point(-1,-1), iterations, 1, 1 );
      
      // find contours
      ContourList_t contour_v;
      cv::findContours( cvimg_stage1_v[p], contour_v, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE );

      std::cout << "Plane " << p << " number of contours: " << contour_v.size() << std::endl;
      
      // for each contour, find convex hull, find defect points
      std::vector< ContourIndices_t > hull_v( contour_v.size() );
      std::vector< Defects_t > defects_v( contour_v.size() );
      for ( int idx=0; idx<(int)contour_v.size(); idx++ ) {

	Contour_t& contour = contour_v[idx];
	if ( contour.size()<10 )
	  continue;
	
	// draw contours
	//cv::drawContours( cvimg_stage0_v[p], contour_v, idx, cv::Scalar( rand.Uniform(10,255),rand.Uniform(10,255),rand.Uniform(10,255),255), 1 );	
	
	// convex hull
	cv::convexHull( cv::Mat( contour ), hull_v[idx], false );

	if ( hull_v[idx].size()<=3 ) {
	  // store 
	  continue; // no defects can be found
	}
	
	// defects
	cv::convexityDefects( contour, hull_v[idx], defects_v[idx] );

	// plot defect point information
	for ( auto& defectpt : defects_v[idx] ) {
	  float depth = defectpt[3]/256;
	  if ( depth>3 ) {
	    int faridx = defectpt[2];
	    cv::Point ptFar( contour[faridx] );
	    cv::circle( cvimg_stage0_v[p], ptFar, 1, cv::Scalar(0,255,0,255), -1 );
	  }
	}

      }
      m_plane_contours_v.emplace_back( std::move(contour_v) );
      m_plane_hulls_v.emplace_back( std::move(hull_v) );
      m_plane_defects_v.emplace_back( std::move(defects_v) );
    }

    splitContour( img_v );
#endif
    
  }// end of findBoundarySpacePoints


  
  void ContourCluster::splitContour( const std::vector<larcv::Image2D>& img_v ) {
    // given a contour with at least one defect point, we split it with the aim of producing the straightest daughter contours
    // setup the contour breaker
    
    fcllite::PSet emptyset("Empty");
    larocv::DefectBreaker defectb;
    defectb.Configure( emptyset );

    TRandom3 rand(time(NULL));    

    m_plane_atomics_v.clear();
    m_plane_atomics_v.resize(3);
    m_plane_atomicmeta_v.clear();
    m_plane_atomicmeta_v.resize(3);
    
    for (int p=0; p<3; p++) {
      // atomic_contours;
      auto& contour_v = m_plane_contours_v[p];
      for ( auto& contour : contour_v ) {
	larocv::data::TrackClusterCompound atomics = defectb.BreakContour( contour ); // returns a vector<AtomicContour>
	for ( auto& ctr : atomics ) {
	  if ( ctr.size()<3 )
	    continue;
	  //larlitecv::ContourShapeMeta ctrinfo( ctr, img_v[p].meta() );
	  larlitecv::ContourShapeMeta ctrinfo( ctr, img_v[p] );	  
	  m_plane_atomics_v[p].push_back( std::move(ctr) );
	  m_plane_atomicmeta_v[p].emplace_back( std::move(ctrinfo) );
	}
      }

      for (int idx=0; idx<(int)m_plane_atomics_v[p].size(); idx++) {
	cv::drawContours( cvimg_stage0_v[p], m_plane_atomics_v[p], idx, cv::Scalar( rand.Uniform(10,255),rand.Uniform(10,255),rand.Uniform(10,255),255), 1 );
	cv::line( cvimg_stage0_v[p], m_plane_atomicmeta_v[p].at(idx).getFitSegmentStart(), m_plane_atomicmeta_v[p].at(idx).getFitSegmentEnd(), cv::Scalar(255,255,255,255), 1 );
      }
    }    

    
  }//splitContour

}
