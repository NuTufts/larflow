#include "ContourShapeMeta.h"
#include <sstream>

namespace larlitecv {

  ContourShapeMeta::ContourShapeMeta() {
    std::stringstream msg;
    msg << __FILE__ << ":" << __LINE__ << ": Default construct should not be used. Only defined for dictionary purposes." << std::endl;
    throw std::runtime_error(msg.str());
  }
  
  ContourShapeMeta::ContourShapeMeta( const std::vector<cv::Point>& contour, const larcv::Image2D& img )
    : std::vector<cv::Point>(contour),
      m_meta(img.meta()),
      m_start( cv::Point(0,0) ),
      m_end( cv::Point(0,0) )  {

    eigen_vecs.clear();
    eigen_val.clear();
    
    _fill_linefit_members();
    _build_bbox();
    _get_tick_range();
    _charge_core_pca(img);
  }

  void ContourShapeMeta::_fill_linefit_members() {
    cv::Vec4f out_array;

    cv::fitLine( *this, out_array, cv::DIST_L2, 0, 0.01, 0.01 );
    m_dir.resize(2,0);
    
    // loop through contour points and get min and max on projected line
    if ( out_array[1]!=0 ) {
      float norm  = sqrt(out_array[0]*out_array[0] + out_array[1]*out_array[1]);
      m_dir[0] = out_array[0]/norm;
      m_dir[1] = out_array[1]/norm;
    }
    else {
      // vertical
      float norm  = sqrt(out_array[0]*out_array[0] + out_array[1]*out_array[1]);
      m_dir[0] = out_array[0]/norm;
      m_dir[1] = 0;
    }
    
    cv::Point maxpt(0,0);
    cv::Point minpt(0,0);
    float mincos =  1e6;
    float maxcos = -1e6;
    for ( auto& pt : (*this) ) {
      float dx[2];
      dx[0] = pt.x-out_array[2];
      dx[1] = pt.y-out_array[3];
      float ptcos = 0.;
      for (int i=0; i<2; i++)
	ptcos += dx[i]*m_dir[i];
      if ( ptcos < mincos ) {
	minpt = pt;
	mincos = ptcos;
      }
      if ( ptcos > maxcos ) {
	maxpt = pt;
	maxcos = ptcos;
      }
    }
    m_start = minpt;
    m_end   = maxpt;

    // orient: start is at low y
    if ( m_start.y > m_end.y ) {
      cv::Point temp = m_start;
      m_start = m_end;
      m_end = temp;
      for (int i=0; i<2; i++)
	m_dir[i] *= -1.;
    }
    
  }
  
  void ContourShapeMeta::_build_bbox() {
    m_bbox = cv::boundingRect( *this );
  }

  void ContourShapeMeta::_get_tick_range() {
    float miny = -1;
    float maxy = -1;
    float minx = -1;
    float maxx = -1;
    for (auto& pt : *this ) {
      if ( miny<0 || pt.y<miny )
	miny = pt.y;
      if (maxy<0 || pt.y>maxy )
	maxy = pt.y;
      if ( minx<0 || pt.x<minx )
	minx = pt.x;
      if ( maxx<0 || pt.x>maxx )
	maxx = pt.x;
    }

    ybounds.resize(2,0);
    xbounds.resize(2,0);
    ybounds[0] = miny;
    ybounds[1] = maxy;
    xbounds[0] = minx;
    xbounds[1] = maxx;
  }

  void ContourShapeMeta::_charge_core_pca( const larcv::Image2D& img ) {

    // collect charge pixels
    std::vector< cv::Point > qpixels;
    qpixels.reserve( int( (xbounds[1]-xbounds[0])*(ybounds[1]-ybounds[0]) ) );
    for ( int c=xbounds[0]; c<=xbounds[1]; c++ ) {
      for (int r=ybounds[0]; r<=ybounds[1]; r++ ) {

	if ( img.pixel(r,c)>10.0 ) {
	  cv::Point testpt( c,r );
	  double dist = cv::pointPolygonTest( *this, testpt, false );
	  if ( dist>0 )
	    qpixels.push_back( testpt );
	}
	
      }
    }
    
    if ( qpixels.size()<=0 )
      return;
    
    // do pca
    int sz = static_cast<int>(qpixels.size());
    cv::Mat data_pts = cv::Mat(sz, 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i) {
      data_pts.at<double>(i, 0) = qpixels[i].x;
      data_pts.at<double>(i, 1) = qpixels[i].y;

    }
    
    //Perform PCA analysis
    cv::PCA pca_analysis(data_pts, cv::Mat(), CV_PCA_DATA_AS_ROW);
    //Store the center of the object
    center = cv::Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
		       static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    
    //Store the eigenvalues and eigenvectors
    eigen_vecs.resize(2);
    eigen_val.resize(2);
      
    for (int i = 0; i < 2; ++i)
      {
	eigen_vecs[i] = cv::Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
				    pca_analysis.eigenvectors.at<double>(i, 1));
	eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
      }

    // determine start and end points
    float max_r_pos = 0;
    float max_r_neg = 0;
    m_pca_startpt.resize(2);
    m_pca_endpt.resize(2);
    m_pca_startpt[0] = m_pca_endpt[0] = center.x;
    m_pca_startpt[1] = m_pca_endpt[1] = center.y;    
    for ( auto const& pix : qpixels ) {

      std::vector<float> dir(2);
      dir[0] = pix.x - center.x;
      dir[1] = pix.y - center.y;
      float dist = sqrt( dir[0]*dir[0] + dir[1]*dir[1] );

      float cospca1 = dir[0]*eigen_vecs[0].x + dir[1]*eigen_vecs[0].y;

      if ( cospca1 <= 0 ) {
	if ( dist > max_r_neg ) {
	  max_r_neg = dist;
	  m_pca_startpt[0] = pix.x;
	  m_pca_startpt[1] = pix.y;
	}
      }
      else {
	if( dist > max_r_pos ) {
	  max_r_pos = dist;
	  m_pca_endpt[0] = pix.x;
	  m_pca_endpt[1] = pix.y;
	}
      }
      
    }
    
  }

  std::vector<float> ContourShapeMeta::getPCAdir( int axis ) const {
    std::vector<float> dir(2);
    dir[0] = dir[1] = 0.;

    if ( eigen_vecs.size()==0 )
      return dir;
    
    dir[0] = eigen_vecs[axis].x;
    dir[1] = eigen_vecs[axis].y;

    return dir;
  }

  std::vector<float> ContourShapeMeta::getPCAEnddir() const {
    return getPCAdir(0);
  }

  std::vector<float> ContourShapeMeta::getPCAStartdir() const {
    std::vector<float> dir = getPCAdir(0);
    dir[0] *= -1.0;
    dir[1] *= -1.0;
    return dir;
  }
  
}
