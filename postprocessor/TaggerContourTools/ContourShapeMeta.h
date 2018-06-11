#ifndef __ContourShapeMeta__
#define __ContourShapeMeta__

#include <vector>

#include "DataFormat/ImageMeta.h"
#include "DataFormat/Image2D.h"

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#endif


namespace larlitecv {

 class ContourShapeMeta : public std::vector<cv::Point> {
   // Wrapper around OpenCV contour
   // Stores meta data for contours
   
 public:
   ContourShapeMeta();   
   //ContourShapeMeta( const std::vector<cv::Point>& contour, const larcv::ImageMeta& img );
   ContourShapeMeta( const std::vector<cv::Point>& contour, const larcv::Image2D& img );   
   virtual ~ContourShapeMeta() {};

   const larcv::ImageMeta& meta() const { return m_meta; };    
   const cv::Point& getFitSegmentStart() const { return m_start; };
   const cv::Point& getFitSegmentEnd() const { return m_end; };   
   const cv::Rect&  getBBox() const  { return m_bbox; };

   std::vector<float> getEndDir() const { return m_dir; };
   std::vector<float> getStartDir() const {
     std::vector<float> reverse_dir(m_dir.size(),0);
     for (size_t i=0; i<m_dir.size(); i++) reverse_dir[i] = -1.0*m_dir[i];
     return reverse_dir;
   };
   
   float getMinX() const { return xbounds[0]; };
   float getMaxX() const { return xbounds[1]; };
   float getMinY() const { return ybounds[0]; };
   float getMaxY() const { return ybounds[1]; };

   std::vector<float> getPCAdir( int axis=0 ) const;
   std::vector<float> getPCAStartdir() const;
   std::vector<float> getPCAEnddir() const;
   std::vector<float> getPCAStartPos() const { return m_pca_startpt; };
   std::vector<float> getPCAEndPos() const   { return m_pca_endpt; };

   
 protected:
   
   // ImageMeta
   const larcv::ImageMeta m_meta;
   
   // Line Fit/Projected End
   std::vector<float> m_dir;
   cv::Point m_start;
   cv::Point m_end;
   void _fill_linefit_members();

   // Bounding Box (for collision detection)
   cv::Rect m_bbox;
   void _build_bbox();

   // Bounds
   std::vector<float> ybounds;
   std::vector<float> xbounds;
   void _get_tick_range();

   // Charge core PCA
   void _charge_core_pca( const larcv::Image2D& img );
   cv::Point center;
   std::vector<cv::Point2d> eigen_vecs;
   std::vector<double> eigen_val;
   // start and end points determined by radius from center and either neg or pos on major axis
   std::vector<float> m_pca_startpt; 
   std::vector<float> m_pca_endpt;
   
   
   
 };
 

}

#endif
