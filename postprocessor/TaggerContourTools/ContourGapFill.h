#ifndef CONTOUR_GAP_FILL_H
#define CONTOUR_GAP_FILL_H

#include <vector>

#include "DataFormat/Image2D.h"
#include "DataFormat/ImageMeta.h"
#include "BMTCV.h"
#include "ContourShapeMeta.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


namespace larlitecv {

  class ContourGapFill {
  public:

    ContourGapFill();
    virtual ~ContourGapFill();


    void makeGapFilledImages( const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v,
			      const std::vector< std::vector<larlitecv::ContourShapeMeta> >& plane_contours_v,
			      std::vector<larcv::Image2D>& gapfilled_v );

    //void mergeWireAndGap( const std::vector<larcv::Image2D>& img_v, std::vector<larcv::Image2D>& gapfilled_v );

    void makeDebugImage( bool debug=true );

    std::vector<cv::Mat>& getDebugImages() { return m_cvimg_debug_v; };

    struct CtrInfo_t {
      int col;
      int row_ave;
      int ctridx;
      int npix;
    };
    
    struct BadChSpan {
      int start;
      int end;
      int width;
      int ngood;
      int planeid;
      std::set<int> leftctridx;  // index of contours that touch left side of span
      std::set<int> rightctridx; // index of contours that touch right side of span

      std::map<int,CtrInfo_t> leftinfo;
      std::map<int,CtrInfo_t> rightinfo;
    };
    
  protected:

    std::vector< BadChSpan > findBadChSpans( const larcv::Image2D& badch, int goodchwidth=2 );
    
    bool fMakeDebugImage;

    std::vector<larcv::ImageMeta> m_meta_v;
    std::vector<cv::Mat> m_cvimg_debug_v;
    std::vector< ContourList_t > m_plane_contour_v;

    void createDebugImage( const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v );
    void associateContoursToSpans( const std::vector<larlitecv::ContourShapeMeta>& contour_v,
				   const larcv::ImageMeta& meta,
				   std::vector<BadChSpan>& span_v,
				   const int colwidth );
    int connectSpan( const ContourGapFill::BadChSpan& span,
		     const std::vector<larlitecv::ContourShapeMeta>& contour_v,
		     const larcv::Image2D& img, const larcv::Image2D& badch, larcv::Image2D& fillimg );


    struct match_t {
      float score;
      int rightidx;
      int leftidx;
      std::vector<float> leftpos;
      std::vector<float> rightpos;
      match_t() {
	score = 0;
	rightidx = -1;
	leftidx  = -1;
	leftpos.resize(2,0);
	rightpos.resize(2,0);
      };
    };
    
    static bool compare_match( const match_t& a, const match_t& b ) {
      if ( a.score < b.score ) return true;
      return false;
    };
    
    

  };



}

#endif
