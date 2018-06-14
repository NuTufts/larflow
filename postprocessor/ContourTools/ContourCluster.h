#ifndef __ContourCluster_H__
#define __ContourCluster_H__

/* -------------------------------------------------------------------------
 * ContourCluster
 * --------------
 *
 * This class does 1-plane segment shape analysis using the contour tools
 * from opencv.
 *
 * -----------------------------------------------------------------------*/

#include <vector>

#include "larcv/core/DataFormat/Image2D.h"

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#endif

#include "ContourShapeMeta.h"

namespace larlitecv {

  typedef std::vector<cv::Point> Contour_t;
  typedef std::vector< Contour_t > ContourList_t;
  typedef std::vector< int > ContourIndices_t;
  typedef std::vector< cv::Vec4i > Defects_t;

  class ContourCluster {
  public:

    ContourCluster(){};
    virtual ~ContourCluster() {};

    void analyzeImages( const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v, const float threshold, const int iterations );
    //void splitContour(  const std::vector<larcv::Image2D>& img_v );    

    void clear();

#ifdef USE_OPENCV
    std::vector<cv::Mat> cvimg_stage0_v; // unchanged images
    std::vector<cv::Mat> cvimg_stage1_v; // contour points over time scan
    std::vector<cv::Mat> cvimg_stage2_v; // 3D-matched contour points
    std::vector<cv::Mat> cvimg_stage3_v; // 3D-matched spacepointso

    std::vector< ContourList_t >                 m_plane_contours_v;
    std::vector< std::vector<ContourIndices_t> > m_plane_hulls_v;
    std::vector< std::vector<Defects_t> >        m_plane_defects_v;
    std::vector< ContourList_t >                 m_plane_atomics_v;
    std::vector< std::vector< larlitecv::ContourShapeMeta > > m_plane_atomicmeta_v;
#endif
  };


}

#endif
