#include "ContourGapFill.h"

#include <algorithm>

// larlite
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"

// larcv
#include "UBWireTool/UBWireTool.h"
#include "CVUtil/CVUtil.h"

// geo2d
#include "Geo2D/Core/Geo2D.h"


namespace larlitecv {

  ContourGapFill::ContourGapFill() {
    fMakeDebugImage = false;
    m_meta_v.clear();
  }

  ContourGapFill::~ContourGapFill() {
  }

  void ContourGapFill::makeDebugImage( bool debug ) {
    fMakeDebugImage = debug;
  }
  
  void ContourGapFill::makeGapFilledImages( const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v,
					    const std::vector< std::vector<larlitecv::ContourShapeMeta> >& plane_contours_v,
					    std::vector<larcv::Image2D>& gapfilled_v ) {

    // Steps
    // ------
    // -- define badch spans
    // -- find contours on the sides of the spans
    // -- match contours across a given span which point to one another
    // -- AStar across the span?
    // -- fill the gap

    // store meta information on the image
    if ( m_meta_v.size()!=img_v.size() ) {
      for ( auto const& img : img_v ) {
	m_meta_v.push_back( img.meta() );
      }
    }

    // allocate a debug image if the flag is on
    if ( fMakeDebugImage ) {
      m_cvimg_debug_v.clear();
      m_plane_contour_v.clear();
      
      createDebugImage( img_v, badch_v );
      // we need to make a list of contours of type contour, not contourshapemeta

      for ( size_t p=0; p<plane_contours_v.size(); p++) {
	std::vector< std::vector<cv::Point> > contour_v;
	for ( auto const& ctr : plane_contours_v[p] )
	  contour_v.push_back( ctr );
	m_plane_contour_v.emplace_back( std::move(contour_v) );
      }
    }

    // define the bad channel spans on each plan
    std::vector< std::vector<BadChSpan> > plane_span_v;
    for ( auto const& badch : badch_v ) {
      std::vector<BadChSpan> span_v = findBadChSpans( badch, 2 );
      //std::cout << "spans on plane=" << badch.meta().plane() << ": " << span_v.size() << std::endl;
      plane_span_v.emplace_back( std::move(span_v) );      
    }

    // find the contours on each plane that touch the spans
    for (size_t p=0; p<m_meta_v.size(); p++) {
      larcv::Image2D gapfillimg( m_meta_v[p] );
      gapfillimg.paint(0);
      
      associateContoursToSpans( plane_contours_v[p], m_meta_v[p], plane_span_v[p], 3 );
      for ( auto const& span : plane_span_v[p] ) {
	// std::cout << "Connecting span of width=" << span.width << ": "
	// 	  << "leftctrs=" << span.leftctridx.size() << " rightctrs=" << span.rightctridx.size() <<  std::endl;
	
	connectSpan( span, plane_contours_v[p], img_v[p], badch_v[p], gapfillimg );
	//std::cout << "Number of matches in this span: " << nmatches << std::endl;
      }

      gapfilled_v.emplace_back( std::move(gapfillimg) );
      //break;
    }
    
  }
  
  void ContourGapFill::createDebugImage( const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v ) {
    // we create a cv::Mat image to draw on
    // we make an rgb image
    
    cv::Mat cvimg = larcv::as_mat_greyscale2bgr( img_v.front(), 0, 255.0 );
    for (size_t p=1; p<img_v.size(); p++) {
      // if ( p!=0 )
      // 	continue;
      for (size_t r=0; r<m_meta_v[p].rows(); r++) {
    	for (size_t c=0; c<m_meta_v[p].cols(); c++) {
    	  if ( img_v[p].pixel(r,c)>5.0 ) {
    	    for (int i=0; i<3; i++)
    	      cvimg.at<cv::Vec3b>(cv::Point(c,r))[i] = img_v[p].pixel(r,c);
    	  }
    	}
      }
    }

    // fill the bad channels
    for ( auto const& badch : badch_v ) {
      auto const& meta = badch.meta();
      int plane = (int)meta.plane();
      //if ( plane!=0 ) continue;
      for ( size_t col=0; col<meta.cols(); col++ ) {
	if ( badch.pixel(0,col)>0 ) {
	  for (size_t row=0; row<meta.rows(); row++) {
	    cv::Point pix(col,row);
	    int val = (int)cvimg.at<cv::Vec3b>(pix)[plane];
	    if ( val<255 ){
	      if ( plane==1 )
		cvimg.at<cv::Vec3b>(pix)[plane] = 10;
	      else
		cvimg.at<cv::Vec3b>(pix)[plane] = 20;
	    }
	  }
	}
      }
    }
    
    m_cvimg_debug_v.emplace_back(std::move(cvimg));
  }
  
  std::vector< ContourGapFill::BadChSpan > ContourGapFill::findBadChSpans( const larcv::Image2D& badch, int goodchwidth ) {
    // we get a list of bad channel spans for the image.
    const larcv::ImageMeta& meta = badch.meta();
    size_t row = meta.rows()/2;

    // simple on-off region finder, where end of region requires goodchwidth of good channels to be observed
    std::vector< BadChSpan > span_v;
    bool inregion = false;
    for ( size_t c=0; c<meta.cols(); c++) {

      if ( !inregion ) {
	// not yet in a badch region
	if ( badch.pixel(row,c)>0 ) {
	  // now we're on. create region

	  BadChSpan newspan;
	  newspan.start = c;
	  newspan.end   = c;
	  newspan.width = 0;
	  newspan.ngood = 0;
	  newspan.planeid = (int)meta.plane();
	  span_v.emplace_back( std::move(newspan) );
	  inregion = true;
	}
      }
      else {
	// currently in a badch region
	BadChSpan& last = span_v.back();
	if ( badch.pixel(row,c)>0 ) {
	  // still in bad region
	  last.end = c;
	  last.ngood = 0; //< reset good ch counter
	}
	else {
	  // now in good region
	  last.ngood++;
	  if ( last.ngood>=goodchwidth ) {
	    // we end the region
	    last.end = (int)c-goodchwidth;
	    last.width = last.end-last.start+1;
	    inregion = false; // reset the inregion marker
	  }
	  else {
	    // do nothing, wait for another good col
	  }
	}//end of if in good
      }//end of if inregion
    }//end of col loop

    if ( inregion ) {
      // if still in a region
      BadChSpan& last = span_v.back();
      last.end = (int)meta.cols()-1;
      last.width = last.end-last.start+1;
    }
    
    return span_v;
  }

  void ContourGapFill::associateContoursToSpans( const std::vector<larlitecv::ContourShapeMeta>& contour_v,
						 const larcv::ImageMeta& meta,
						 std::vector<ContourGapFill::BadChSpan>& span_v,
						 const int colwidth ) {
    
    // inputs
    // ------
    // contour_v: contour list for a plane
    // span_v: bad channel span list for the same plane
    // colwidth: number of columns away from span edges to look for contours
    //
    // outputs
    // -------
    // span_v: spans are updated with the left/right-ctrindx vectors filled

    int nleftctrs  = 0;
    int nrightctrs = 0;
    int ispan = -1;
    for ( auto& span : span_v ) {
      ispan++;
      // we scan down the start and end col and look to see if it touches a contour

      //std::cout << "span #" << ispan << ": [" << span.start << "," << span.end << "]" << std::endl;

      // start
      for ( int c=span.start-colwidth+1; c<=span.start; c++) {
	if ( c<0 || c>=(int)meta.cols() ) continue;
	for (size_t r=0; r<meta.rows(); r++) {

	  for (int idx=0; idx<(int)contour_v.size(); idx++) {
	    auto const& ctr = contour_v[idx];

	    cv::Point testpt( c, (int)r );
	    double dist = cv::pointPolygonTest( ctr, testpt, false );
	    if ( dist>0 ) {
	      // inside contour

	      if ( span.leftinfo.find(idx)==span.leftinfo.end() ) {
		// make a new instance of ctrinfo if idx not yet found
		span.leftinfo.insert( std::make_pair(idx,CtrInfo_t()) );
	      }
	      span.leftctridx.insert(idx);

	      // update span info
	      span.leftinfo[idx].col    = c;
	      span.leftinfo[idx].ctridx = idx;
	      // update rolling row ave
	      span.leftinfo[idx].row_ave = span.leftinfo[idx].row_ave*span.leftinfo[idx].npix + (int)r;
	      span.leftinfo[idx].npix++;
	      span.leftinfo[idx].row_ave /= span.leftinfo[idx].npix;
	      nleftctrs++;
	    }	    
	  }
	}
      }//end of start loop

      // end
      for ( int c=span.end; c<span.end+colwidth; c++) {
	if ( c<0 || c>=(int)meta.cols() ) continue;
	for (size_t r=0; r<meta.rows(); r++) {
	  
	  for (int idx=0; idx<(int)contour_v.size(); idx++) {
	    auto const& ctr = contour_v[idx];
	    
	    cv::Point testpt( c, (int)r );
	    double dist = cv::pointPolygonTest( ctr, testpt, false );
	    if ( dist>0 ) {
	      // inside contour
	      if ( span.rightinfo.find(idx)==span.rightinfo.end() ) {
		// make a new instance of ctrinfo if idx not yet found
		span.rightinfo.insert( std::make_pair(idx,CtrInfo_t()) );
	      }
	      span.rightctridx.insert(idx);
	      // update span info
	      span.rightinfo[idx].col    = c;
	      span.rightinfo[idx].ctridx = idx;
	      // update rolling row ave
	      span.rightinfo[idx].row_ave = span.rightinfo[idx].row_ave*span.rightinfo[idx].npix + (int)r;
	      span.rightinfo[idx].npix++;
	      span.rightinfo[idx].row_ave /= span.rightinfo[idx].npix;
	      nrightctrs++;
	    }	    
	  }
	}
      }//end of end loop
      //std::cout << "ncontours close to this span: left=" << span.leftctridx.size() << " right=" << span.rightctridx.size() << std::endl;
      
      if ( fMakeDebugImage ) {
	//std::cout << "Draw contours on " << m_cvimg_debug_v.size() << " debug images" << std::endl;
	cv::Mat& cvimg = m_cvimg_debug_v.front();
	
      	cv::Scalar contourcolor;
	if ( meta.plane()==0 )
	  contourcolor = cv::Scalar(255,0,0,255);
	else if ( meta.plane()==1 )
	  contourcolor = cv::Scalar(0,255,0,255);
	else if ( meta.plane()==2 )
	  contourcolor = cv::Scalar(0,0,255,255);
	
	for ( auto const& idx : span.leftctridx ) {
	  cv::drawContours( cvimg, m_plane_contour_v[meta.plane()], idx, contourcolor, 1 );
	  cv::circle( cvimg, cv::Point( span.leftinfo[idx].col, span.leftinfo[idx].row_ave ), 3, cv::Scalar( 255, 255, 0, 255 ), 1 );
	}
	for ( auto const& idx : span.rightctridx ) {
	  //std::cout << "  draw ctr idx=" << idx << "(of " << contour_v.size() << ")" << std::endl;
	  cv::drawContours( cvimg, m_plane_contour_v[meta.plane()], idx, contourcolor, 1 );
	  cv::circle( cvimg, cv::Point( span.rightinfo[idx].col, span.rightinfo[idx].row_ave ), 3, cv::Scalar( 255, 255, 0, 255 ), 1 );
	}
      }
    }

    return;
  }
						 

  int ContourGapFill::connectSpan( const ContourGapFill::BadChSpan& span,
				   const std::vector<larlitecv::ContourShapeMeta>& contour_v,
				   const larcv::Image2D& img, const larcv::Image2D& badch, larcv::Image2D& fillimg ) {
    // connect left/right
    if ( span.leftctridx.size()==0 || span.rightctridx.size()==0 )
      return 0;

    // score: take direction from point and get intersection position on other side of span
    //        distance is smallest distance from intersection to other span point

    // one issue with this is that the direction for the segments cna be a bit noisy

    //std::cout << "ContourGapFill::connectSpan" << std::endl;

    int plane = img.meta().plane();
    std::vector<match_t> match_v;
    match_v.reserve( span.leftctridx.size()*span.rightctridx.size() );
    
    for ( auto const& idxl : span.leftctridx ) {
      
      const larlitecv::ContourShapeMeta& ctrleft = contour_v[idxl];
      auto it_l = span.leftinfo.find(idxl);
      if ( it_l==span.leftinfo.end() ) {
	//std::cout << "didnt find matching left info for idx=" << idxl << std::endl;
	continue;
      }
      const CtrInfo_t& infoleft = it_l->second;

      // get direction (start or end) based on closeness
      float leftstart = fabs( ctrleft.getFitSegmentStart().x - infoleft.col );
      float leftend   = fabs( ctrleft.getFitSegmentEnd().x   - infoleft.col );

      geo2d::Vector<float> leftpos;
      leftpos.x = infoleft.col;
      leftpos.y = infoleft.row_ave;
      
      geo2d::Vector<float> leftdir;
      if ( leftstart<leftend ) {
    	//leftdir.x = ctrleft.getStartDir()[0];
	//leftdir.y = ctrleft.getStartDir()[1];
    	leftdir.x = ctrleft.getPCAStartdir()[0];
	leftdir.y = ctrleft.getPCAStartdir()[1];
      }
      else {
    	// leftdir.x = ctrleft.getEndDir()[0];
    	// leftdir.y = ctrleft.getEndDir()[0];
    	leftdir.x = ctrleft.getPCAEnddir()[0];
    	leftdir.y = ctrleft.getPCAEnddir()[1];
      }

      geo2d::Line<float> leftline(  leftpos, leftdir );
      if ( fMakeDebugImage )
	cv::arrowedLine( m_cvimg_debug_v.front(), leftpos, leftpos+10*leftdir, cv::Scalar(255,255,255,255), 1 );

      geo2d::Vector<float> leftcol;
      leftcol.x = infoleft.col;
      leftcol.y = 0;
      geo2d::Vector<float> up;
      up.x = 0;
      up.y = 1.0;
      geo2d::Line<float> leftcolline( leftcol, up );
      
      for ( auto const& idxr : span.rightctridx  ) {

	if ( idxl==idxr ) {
	  //std::cout << "span on the same contour (" << idxl << "," << idxr << ")" << std::endl;
	  continue;
	}
	
    	const larlitecv::ContourShapeMeta& ctrright = contour_v[idxr];
	auto it_r = span.rightinfo.find(idxr);
	if ( it_r==span.rightinfo.end() ) {
	  //std::cout << "didnt find matching right info for idx=" << idxr << std::endl;
	  continue;
	}
	const CtrInfo_t& inforight                  = it_r->second;

    	float rightstart = fabs( ctrright.getFitSegmentStart().x - inforight.col );
    	float rightend   = fabs( ctrright.getFitSegmentEnd().x   - inforight.col );

	geo2d::Vector<float> rightpos; // (x,y)
	rightpos.x = inforight.col;
	rightpos.y = inforight.row_ave;
	geo2d::Vector<float> rightdir;
    	if ( rightstart<rightend ) {
    	  // rightdir.x = ctrright.getStartDir()[0];
	  // rightdir.y = ctrright.getStartDir()[1];
    	  rightdir.x = ctrright.getPCAStartdir()[0];
	  rightdir.y = ctrright.getPCAStartdir()[1];
	}
    	else {
    	  //rightdir.x = ctrright.getEndDir()[0];
    	  //rightdir.y = ctrright.getEndDir()[1];
    	  rightdir.x = ctrright.getPCAEnddir()[0];
    	  rightdir.y = ctrright.getPCAEnddir()[1];
	}

	cv::arrowedLine( m_cvimg_debug_v.front(), rightpos, rightpos+10*rightdir, cv::Scalar(255,255,255,255), 1 );	

	geo2d::Line<float> rightline( rightpos, rightdir );
	geo2d::Vector<float> rightcol;
	rightcol.x = inforight.col;
	rightcol.y = 0;
	geo2d::Line<float> rightcolline( rightcol, up );

	//geo2d::LineSegment<float> spanline( leftpos.x, leftpos.y, rightpos.x, rightpos.y );
	//float spandist = geo2d::length( spanline );

	geo2d::Vector<float> inter_pt_left  = geo2d::IntersectionPoint( leftcolline, rightline );
	geo2d::Vector<float> inter_pt_right = geo2d::IntersectionPoint( rightcolline, leftline );
	float leftdist = fabs(inter_pt_left.y-leftpos.y);
	float rightdist = fabs(inter_pt_right.y-rightpos.y);

	float height = 0;
	geo2d::Vector<float> inter_pt;
	if ( leftdist < rightdist ) {
	  inter_pt = inter_pt_left;
	  height = leftdist;
	}
	else {
	  inter_pt = inter_pt_right;
	  height = rightdist;
	}
	
	match_t match;
	match.leftidx  = idxl;
	match.rightidx = idxr;
	match.score = height;
	match.leftpos[0]  = leftpos.x;
	match.leftpos[1]  = leftpos.y;
	match.rightpos[0] = rightpos.x;
	match.rightpos[1] = rightpos.y;

	if ( !std::isnan(match.score) && !std::isinf(match.score) && height<20.0 ) {
	  if ( fMakeDebugImage )
	    cv::line( m_cvimg_debug_v.front(), leftpos, rightpos, cv::Scalar(255,255,0,255), 2 );	  
	  match_v.emplace_back( std::move(match) );
	}
	
      }
    }

    std::sort( match_v.begin(), match_v.end(), compare_match );


    for ( auto const& match : match_v ) {
      //std::cout << "match (" << match.leftidx << "," << match.rightidx << ") score=" << match.score << std::endl;
      // fill the bad channel file
      std::vector<float> dir(2);
      float norm = 0;
      for (int i=0; i<2; i++) {
	dir[i] = match.rightpos[i]-match.leftpos[i];
	norm += dir[i]*dir[i];
      }
      norm = sqrt(norm);

      for (int i=0; i<2; i++)
	dir[i] /= norm;

      int nsteps = norm/0.3;
      nsteps++;
      float stepsize = norm/nsteps;

      for (int istep=0; istep<=nsteps; istep++) {
	std::vector<int> pix(2);
	for (int i=0; i<2; i++)
	  pix[i] = int( match.leftpos[i] + istep*stepsize*dir[i] );
	if ( pix[0]<0 || pix[0]>=(int)badch.meta().cols())
	  continue;
	for (int dr=-1; dr<=3; dr++) {
	  int r = pix[1]+dr;
	  if ( r<0 || r>=(int)badch.meta().rows() )
	    continue;
	  if ( badch.pixel( r, pix[0] )>0 ) {
	    fillimg.set_pixel( r, pix[0], 10.0 );

	    if ( fMakeDebugImage ) {
	      cv::Mat& debugimg = m_cvimg_debug_v.front();
	      for (int p=0; p<3; p++) {
		if (p==plane)
		  debugimg.at<cv::Vec3b>( cv::Point(pix[0],r) )[p] = 255;
		else
		  debugimg.at<cv::Vec3b>( cv::Point(pix[0],r) )[p] = 0;
	      }
	    }
	    
	  }
	}
      }
      
    }
    
    return match_v.size();
    
  }
  
}
