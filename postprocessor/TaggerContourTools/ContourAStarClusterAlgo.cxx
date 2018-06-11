#include "ContourAStarClusterAlgo.h"

#include <sstream>
#include <stdexcept>

// larlite
#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"

#include "UBWireTool/UBWireTool.h"

// larcv
#include "CVUtil/CVUtil.h"
#include "DataFormat/Pixel2D.h"
#include "Reco3D/AStar3DAlgoConfig.h"
#include "Reco3D/AStar3DAlgo.h"

// LArOpenCV
#include "LArOpenCV/ImageCluster/AlgoClass/DefectBreaker.h"
#include "LArOpenCV/ImageCluster/AlgoData/TrackClusterCompound.h"



namespace larlitecv {

  ContourAStarCluster::~ContourAStarCluster() {
  }
  
  void ContourAStarCluster::setImageMeta( const std::vector<larcv::Image2D>& img_v ) {
    m_nplanes = img_v.size();
    m_bmtcv_indices.resize(m_nplanes);
    m_plane_contours.resize(m_nplanes);
    m_current_min = -1;
    m_current_max = -1;
    m_path3d.clear();

    // we make blank cv images
    m_cvimg_v.clear();
    m_cvpath_v.clear();
    //m_clusterimg_v.clear();
    m_meta_v.clear();
    for ( auto const& img : img_v ) {
      larcv::Image2D blank( img.meta() );
      blank.paint(0.0);
      //cv::Mat cvimg1 = larcv::as_gray_mat( blank, 0, 255.0, 1.0 );
      cv::Mat cvimg2 = larcv::as_gray_mat( blank, 0, 255.0, 1.0 );      
      //m_cvpath_v.push_back( cvimg1 );
      m_cvimg_v.push_back( cvimg2 );
      //m_clusterimg_v.emplace_back( std::move(blank) );
      m_meta_v.push_back( img.meta() );
    }

    if ( fMakeDebugImage ) {
      m_cvimg_debug = larcv::as_mat_greyscale2bgr( img_v.front(), 0, 255.0 );    
      for (int p=1; p<m_nplanes; p++) {
	for (size_t r=0; r<img_v[p].meta().rows(); r++) {
	  for (size_t c=0; c<img_v[p].meta().cols(); c++) {
	    if ( img_v[p].pixel(r,c)>5.0 ) {
	      for (int i=0; i<3; i++)
		m_cvimg_debug.at<cv::Vec3b>(cv::Point(c,r))[i] = img_v[p].pixel(r,c);
	    }
	  }
	}
      }
    }
  }

  void ContourAStarCluster::resetDebugImage( const std::vector<larcv::Image2D>& img_v ) {
    if ( !fMakeDebugImage )
      return;
    
    m_cvimg_debug = larcv::as_mat_greyscale2bgr( img_v.front(), 0, 255.0 );
    for (int p=1; p<(int)img_v.size(); p++) {
      for (size_t r=0; r<img_v[p].meta().rows(); r++) {
	for (size_t c=0; c<img_v[p].meta().cols(); c++) {
	  if ( img_v[p].pixel(r,c)>5.0 ) {
	    for (int i=0; i<3; i++)
	      m_cvimg_debug.at<cv::Vec3b>(cv::Point(c,r))[i] = img_v[p].pixel(r,c);
	  }
	}
      }
    }
  }

  void ContourAStarCluster::addContour( int plane, const larlitecv::ContourShapeMeta* ctr, int idx ) {
    if (ctr==NULL) {
      std::stringstream ss;
      ss << __FILE__ << ":" << __LINE__ << "Adding a NULL contour shape meta";
      throw std::runtime_error( ss.str() );
    }
    m_plane_contours[plane].push_back( ctr );
    if ( idx>=0 ) {
      m_bmtcv_indices[plane].insert( idx );
    }
  }
  
  void ContourAStarCluster::updateCVImage() {
    // try to use this sparingly. it's probably slow.
    
    // we loop over all contours, fill in the image. Recontour.
    for ( int p=0; p<m_nplanes; p++) {
      std::vector< std::vector<cv::Point> > contour_list;
      int i = 0;
      for ( auto const& pctr : m_plane_contours[p] ) {
	contour_list.push_back( *pctr );
	cv::drawContours( m_cvimg_v[p], contour_list, i, cv::Scalar(255,0,0), -1 );

	// debug image
	if ( fMakeDebugImage ) {
	  if ( p==0 )
	    cv::drawContours( m_cvimg_debug, contour_list, i, cv::Scalar(255,0,0), 1 );
	  else if (p==1)
	    cv::drawContours( m_cvimg_debug, contour_list, i, cv::Scalar(0,255,0), 1 );
	  else if ( p==2)
	    cv::drawContours( m_cvimg_debug, contour_list, i, cv::Scalar(0,0,255), 1 );
	}
      }
      // we copy back into the original image (slow-slow)
      // for (size_t r=0; r<m_clusterimg_v[p].meta().rows(); r++) {
      // 	for (size_t c=0; c<m_clusterimg_v[p].meta().cols(); c++) {
      // 	  if ( m_cvimg_v[p].at<uchar>(cv::Point(c,r))>0 )
      // 	    m_clusterimg_v[p].set_pixel(r,c,255.0);
      // 	}
      // }
    }
  }

  void ContourAStarCluster::updateClusterContour() {
    // we cluster the pixels in cvimg_v and form a cluster
    // we use this to perform the other analyses
    m_current_contours.clear();
    m_current_contours.resize(m_nplanes);
    for ( int p=0; p<m_nplanes; p++) {
      std::vector< std::vector<cv::Point> > contour_v;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours( m_cvimg_v[p], contour_v, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0) );
      for ( auto& ctr : contour_v ) {
	//ContourShapeMeta ctrmeta( ctr, m_clusterimg_v[p].meta() );
	ContourShapeMeta ctrmeta( ctr, m_meta_v[p] );
	m_current_contours[p].emplace_back( std::move(ctrmeta) );
      }
    }
  }

  std::vector<int> ContourAStarCluster::getOverlappingRowRange() {
    int min_row = -1;
    int max_row = -1;

    m_plane_rowminmax.resize(m_nplanes);

    
    for (int p=0; p<m_nplanes; p++) {
      m_plane_rowminmax[p].resize(2,-1);
      int planemin = -1;
      int planemax = -1;
      for ( auto& ctr : m_current_contours[p] ) {

	// total maxi-min mini-max to determine overlap
	if ( min_row < ctr.getMinY() || min_row<0 )
	  min_row = ctr.getMinY();

	if ( max_row > ctr.getMaxY() || max_row<0 )
	  max_row = ctr.getMaxY();

	// plane abs min and max to get extent of clusters in this plane
	if ( planemin>ctr.getMinY() || planemin<0 )
	  planemin = ctr.getMinY();
	if ( planemax<ctr.getMaxY() || planemax<0 )
	  planemax = ctr.getMaxY();
      }
      m_plane_rowminmax[p][0] = planemin;
      m_plane_rowminmax[p][1] = planemax;
    }

    std::vector<int> range(2);
    range[0] = min_row;
    range[1] = max_row;
    return range;
  }

  bool ContourAStarCluster::getCluster3DPointAtTimeTick( const int row, const std::vector<larcv::Image2D>& img_v,
							 const std::vector<larcv::Image2D>& badch_v, bool use_badch,
							 std::vector<int>& imgcoords, std::vector<float>& pos3d ) {
    // we get 2D points from each plane. Then we infer 3D point
    struct ctr_pt_t {
      int plane;
      int minc;
      int maxc;
      int maxq;
      float maxqc;
    };

    const float cm_per_tick = ::larutil::LArProperties::GetME()->DriftVelocity()*0.5;

    if ( m_verbosity>1 )
      std::cout << "getCluster3DPointAtTimeTick: row=" << row << std::endl;
    
    std::vector<ctr_pt_t> contour_points;
    
    for (int p=0; p<m_nplanes; p++) {
      const std::vector<ContourShapeMeta>& ctr_v = m_current_contours[p];
      const larcv::Image2D& img                  = img_v[p];
      int nfound = 0;
      for ( auto const& ctr : ctr_v ) {

	// scan across the cols at a certain time and get the min,max and max-q points inside the cluster
	int minc = -1;
	int maxc = -1;
	int maxqc = -1;
	float maxq = -1;
	bool incontour = false;
	for (int c=0; c<(int)img.meta().cols(); c++) {
	  cv::Point testpt( c, row );
	  double dist = cv::pointPolygonTest( ctr, testpt, false );
	  //std::cout << "point (" << c << "," << row << ") dist=" << dist << " incontour=" << incontour << std::endl;
	  if ( dist<0 ) {
	    // not in contour
	    if ( incontour ) {
	      // close out a contour crossing
	      ctr_pt_t ctr_xing;
	      ctr_xing.plane = p;
	      ctr_xing.minc  = minc;
	      ctr_xing.maxc  = maxc;
	      ctr_xing.maxq  = maxq;
	      ctr_xing.maxqc = maxqc;
	      contour_points.emplace_back( std::move(ctr_xing) );
	      incontour = false;
	      minc = -1;
	      maxc = -1;
	      maxqc = -1;
	      maxq = -1;
	      nfound++;
	    }
	    continue; // move to next col
	  }

	  // makes it here, then in contour
	  incontour = true;
	  
	  if ( minc<0 || c<minc ) {
	    minc = c;
	  }
	  if ( maxc<0 || c>maxc )
	    maxc = c;
	  if ( maxq<0 || img.pixel(row,c)>maxq ) {
	    maxq  = img.pixel(row,c);
	    maxqc = c;
	  }
	}//end of col loops
      }//end of loop ctr on the plane
      if ( m_verbosity>0 )
	std::cout << "Number of contour points of plane #" << p << ": " << nfound << std::endl;
    }//end of plane loop


    // We make 3D points based on 2 plane crossings: we check if point inside the other plane
    // (this is for non-horizontal tracks. for that we have to use the edges)
    // we remove close points as well
    int npts = contour_points.size();
    imgcoords.resize(4,0);
    pos3d.resize(3,-1.0e5); // sentinel value
    for (int a=0; a<npts; a++) {
      ctr_pt_t& pta = contour_points[a];
      for (int b=a+1; b<npts; b++) {
	ctr_pt_t& ptb = contour_points[b];
	if ( pta.plane==ptb.plane )
	  continue; // don't match the same planes, bruh

	int otherp;
	int otherw;
	int crosses;
	std::vector<float> xsec_zy(2,0);
	larcv::UBWireTool::getMissingWireAndPlane( pta.plane, pta.maxqc, ptb.plane, ptb.maxqc, otherp, otherw, xsec_zy, crosses );
	if ( m_verbosity>1 )
	  std::cout << "  candidate point: (" << a << "," << b << ") crosses=" << crosses << " pos=(" << xsec_zy[1] << "," << xsec_zy[0] << ")" << std::endl; 
	
	// bad crossing or out of wire range
	if ( crosses==0 || otherw<0 || otherw>=img_v[otherp].meta().max_x() )
	  continue;

	// check for charge or badch
	bool hasgoodpt = false;
	//std::vector<int> goodpix(4,0);
	for (int dr=-2; dr<=2; dr++) {
	  int r=row+dr;
	  if ( r<0 || r>=(int)(img_v.front().meta().rows()) ) continue;
	  for (int dc=-2; dc<=2; dc++) {
	    int c=otherw+dc;
	    if ( c<0 || c>=(int)(img_v.front().meta().cols()) ) continue;	    
	    if ( img_v[otherp].pixel( r, c )>10 || badch_v[otherp].pixel(r, c)>0 ) {
	      hasgoodpt = true;
	    }
	    if (hasgoodpt)
	      break;
	  }
	  if (hasgoodpt)
	    break;
	}
	if ( hasgoodpt ) {
	  // good point
	  imgcoords[0] = row;
	  imgcoords[pta.plane+1] = pta.maxqc;
	  imgcoords[ptb.plane+1] = ptb.maxqc;
	  imgcoords[otherp+1]    = otherw;

	  float tick = img_v.front().meta().pos_y( row ); // tick
	  pos3d[0] = (tick-3200.0)*cm_per_tick;
	  pos3d[1] = xsec_zy[1];
	  pos3d[2] = xsec_zy[0];

	  if ( m_verbosity>1 ) {
	    std::cout << "Produced 3D Point: imgcoords(" << imgcoords[0] << "," << imgcoords[1] << "," << imgcoords[2] << "," << imgcoords[3] << ")"
		      << " tick=" << tick
		      << " pos3d=(" << pos3d[0] << "," << pos3d[1] << "," << pos3d[2] << ")" << std::endl;
	  }
	}
      }
    }

    for (int i=0; i<3; i++) {
      if ( pos3d[i]<-1.0e4 )
	return false;
    }


    return true;
  }
  
  // =================================================================================
  // ALGO METHODS

  ContourAStarCluster ContourAStarClusterAlgo::makeCluster( const std::vector<float>& pos3d, const std::vector<larcv::Image2D>& img_v,
							    const std::vector<larcv::Image2D>& badch_v,
							    const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
							    const float max_dist2cluster, const int maxloopsteps ) {

    // Primary user interface method
    // Goal is to make a cluster of pixels using contour segments on each plane.
    // A 3D point is provided as the seed position
    // AStar is used to verify consistency of straight line path

    // inputs
    // ------
    // pos3d: 3D position
    // img_v: vector images, each element corresponds to a plane
    // badch_v: image marking badch pixels
    // plane_contours_v: a list of contour segments for each plane. outer vector is the plane, the inner vector is the list of contours
    //                   one makes this object from BMTCV::analyzeImages
    // max_dist2cluster: maximum the seed 3d point can be from a cluster in an image
    // maxloopsteps: attempts to extend the initial seed cluster
    // 
    // output
    // -------
    // returns the cluster with a 3D path. if the algorithm fails, the cluster path will be empty.


    if ( pos3d.size()!=3 ) {
      std::stringstream msg;
      msg << __FILE__ << ":" << __LINE__ << ": pos3d should be a vector of length 3" << std::endl;
      throw std::runtime_error( msg.str() );
    }
    
    ContourAStarCluster cluster = makeSeedClustersFrom3DPoint( pos3d, img_v, plane_contours_v, max_dist2cluster );
    cluster.setVerbosity(m_verbosity);
    extendSeedCluster( pos3d, img_v, badch_v, plane_contours_v, maxloopsteps, cluster );
    return cluster;
  }


  void ContourAStarClusterAlgo::extendSeedCluster( const std::vector<float>& pos3d, const std::vector<larcv::Image2D>& img_v,
						   const std::vector<larcv::Image2D>& badch_v,
						   const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
						   const int maxloopsteps, ContourAStarCluster& cluster ) {
    const float cm_per_tick = ::larutil::LArProperties::GetME()->DriftVelocity()*0.5;


    // get the first start/end points
    // first get the overlapping time region
    std::vector<int> timerange = cluster.getOverlappingRowRange();
    if ( abs(timerange[0]-timerange[1])>10 ) {
      timerange[0] += 3;
      timerange[1] -= 3;
    }
    if ( m_verbosity>1 ) 
      std::cout << "Row range: [" << timerange[0] << "," << timerange[1] << "]" << std::endl;
    cluster.m_current_min = timerange[0];
    cluster.m_current_max = timerange[1];


    // img coord of seed point
    std::vector<int> seed_imgcoord = larcv::UBWireTool::getProjectedImagePixel( pos3d, img_v.front().meta(), img_v.size() );
    if ( m_verbosity>1 )
      std::cout << "Seed imgcoord: (" << seed_imgcoord[1] << "," << seed_imgcoord[2] << "," << seed_imgcoord[3] << ") tick="  << seed_imgcoord[0] << std::endl;
    
    // draw line
    if ( fMakeDebugImage ) {
      cv::line( cluster.m_cvimg_debug, cv::Point(0,timerange[0]), cv::Point(3455,timerange[0]), cv::Scalar(255,255,255), 1 );
      cv::line( cluster.m_cvimg_debug, cv::Point(0,timerange[1]), cv::Point(3455,timerange[1]), cv::Scalar(255,255,255), 1 );

      // draw seed
      for (int p=0; p<3; p++) {
	cv::circle( cluster.m_cvimg_debug, cv::Point(seed_imgcoord[p+1], seed_imgcoord[0]), 3, cv::Scalar(0,255,255), 1 );
      }
    }
    
    // we scan at the time range for a good 3D point (or points?)
    std::vector<int> min_imgcoords;
    std::vector<float> min_pos3d;
    std::vector<int> max_imgcoords;
    std::vector<float> max_pos3d;
    //std::cout << "Get min point -----------------------" << std::endl;
    bool foundpoint_min = cluster.getCluster3DPointAtTimeTick( timerange[0], img_v, badch_v, true, min_imgcoords, min_pos3d );
    //std::cout << "Get max point -----------------------" << std::endl;    
    bool foundpoint_max = cluster.getCluster3DPointAtTimeTick( timerange[1], img_v, badch_v, true, max_imgcoords, max_pos3d );

    if ( !foundpoint_min || !foundpoint_max ) {
      // bad seed
      if ( m_verbosity>1 )
	std::cout << "Bad seeds @min=" << foundpoint_min << " @max=" << foundpoint_max << std::endl;
      return;
    }

    // astar config ---------------------------
    larcv::AStar3DAlgoConfig astar_cfg;
    astar_cfg.verbosity = 0;
    astar_cfg.min_nplanes_w_hitpixel = 3;
    astar_cfg.min_nplanes_w_charge = 2;
    astar_cfg.astar_threshold.resize(3,10);
    astar_cfg.astar_neighborhood.resize(3,10);
    astar_cfg.restrict_path = true;
    astar_cfg.path_restriction_radius = 30.0;
    astar_cfg.accept_badch_nodes = true;
    astar_cfg.astar_start_padding = 3;
    astar_cfg.astar_end_padding = 3;
    astar_cfg.lattice_padding = 3;
    // ----------------------------------------


    if ( fMakeDebugImage ) {
      for (int p=0; p<3; p++) {
	cv::circle( cluster.m_cvimg_debug, cv::Point(min_imgcoords[p+1], min_imgcoords[0]), 3, cv::Scalar(255,0,255), 1 );
	cv::circle( cluster.m_cvimg_debug, cv::Point(max_imgcoords[p+1], max_imgcoords[0]), 3, cv::Scalar(255,0,255), 1 );	
      }
      std::cout << "Wrote seed cluster image and time bounds" << std::endl;
      cv::imwrite( "boundaryptimgs/astarcluster_seedcluster.png", cluster.m_cvimg_debug );      
    }
    
    // now we enter the buiding loop
    int iloop = 0;
    while( iloop<maxloopsteps ) {
      if ( m_verbosity>0 ) {
	std::cout << "////// LOOP " << iloop << " /////////" << std::endl;
	std::cout << " Start: (" << min_pos3d[0] << "," << min_pos3d[1] << "," << min_pos3d[2] << ") " << std::endl;
	std::cout << " End:   (" << max_pos3d[0] << "," << max_pos3d[1] << "," << max_pos3d[2] << ")   " << std::endl;
      }
      iloop++;

      cluster.resetDebugImage( img_v );

      const larcv::ImageMeta& meta = img_v.front().meta();
      larcv::AStar3DAlgo algo( astar_cfg );
      std::vector<larcv::AStar3DNode> path;
      std::vector< int > start_cols;
      std::vector< int > end_cols;
      for (int i=0; i<3; i++) {
	start_cols.push_back( min_imgcoords[i+1] );
	end_cols.push_back(   max_imgcoords[i+1] );
      }
      timerange[0] = min_imgcoords[0]; // row
      timerange[1] = max_imgcoords[0]; // row

      if ( fMakeDebugImage ) {
	// plot for debug ----------------------------------------------------------------------------------------------
	cv::circle( cluster.m_cvimg_debug, cv::Point(min_imgcoords[1],timerange[0]), 4, cv::Scalar(0,255,255,255), 1 );
	cv::circle( cluster.m_cvimg_debug, cv::Point(min_imgcoords[2],timerange[0]), 4, cv::Scalar(0,255,255,255), 1 );
	cv::circle( cluster.m_cvimg_debug, cv::Point(min_imgcoords[3],timerange[0]), 4, cv::Scalar(0,255,255,255), 1 );
	
	cv::circle( cluster.m_cvimg_debug, cv::Point(max_imgcoords[1],timerange[1]), 4, cv::Scalar(0,255,255,255), 1 );
	cv::circle( cluster.m_cvimg_debug, cv::Point(max_imgcoords[2],timerange[1]), 4, cv::Scalar(0,255,255,255), 1 );
	cv::circle( cluster.m_cvimg_debug, cv::Point(max_imgcoords[3],timerange[1]), 4, cv::Scalar(0,255,255,255), 1 );
	// -------------------------------------------------------------------------------------------------------------
      }
      
      
      int goalhit = 0;
      path = algo.findpath( img_v, badch_v, badch_v, timerange[0], timerange[1], start_cols, end_cols, goalhit );

      if ( m_verbosity>0 )
	std::cout << "Goal hit: " << goalhit << " pathsize=" << path.size() << std::endl;

      // stop if astar cannot connect
      if ( goalhit==0 )
	break;

      if ( fMakeDebugImage ) {
	for ( auto &node : path ) {
	  std::vector<int> imgcoords = larcv::UBWireTool::getProjectedImagePixel( node.tyz, meta, img_v.size() );
	  imgcoords[0] = meta.row( node.tyz[0] );
	  for (int i=0; i<3; i++) {
	    cv::circle( cluster.m_cvimg_debug, cv::Point( imgcoords[i+1], imgcoords[0] ), 1, cv::Scalar(255,0,255), -1 );
	  }
	}
      }
      
      // evaluate track.
      
      // if good, extend track
      // we do this by getting a 3d fit of the line, extending past the end points, absorbing new clusters on all three planes
      std::vector< std::vector<float> > v3d;
      for (int idx=0; idx<(int)path.size(); idx++) {
	auto &node = path[idx];
	std::vector<float> v3 = node.tyz;
	// convert tick to x
	v3[0] = (v3[0]-3200.0)*cm_per_tick;
	v3d.push_back( v3 );
      }

      std::vector< std::set<int> > plane_overlapping_clusters = extendClusterUsingAStarPath( cluster, v3d, img_v, plane_contours_v, 20.0, 10.0, 1.0 );
      fillInClusterImage( cluster, v3d, img_v, badch_v, plane_overlapping_clusters, plane_contours_v, 0.3, 10.0, 2 );

      // we reset the start and end points
      min_imgcoords = larcv::UBWireTool::getProjectedImagePixel( v3d.front(), meta, img_v.size() );
      max_imgcoords = larcv::UBWireTool::getProjectedImagePixel( v3d.back(),  meta, img_v.size() );

      // go to ticks
      min_imgcoords[0] = v3d.front()[0]/cm_per_tick + 3200.0;
      max_imgcoords[0] = v3d.back()[0]/cm_per_tick  + 3200.0;

      // got to rows
      min_imgcoords[0] = meta.row( min_imgcoords[0] );
      max_imgcoords[0] = meta.row( max_imgcoords[0] );

      min_pos3d = v3d.front();
      max_pos3d = v3d.back();

      // timerange = cluster.getOverlappingRowRange();
      // timerange[0]++;
      // timerange[1]--;

      // cluster.getCluster3DPointAtTimeTick( timerange[0], img_v, badch_v, true, min_imgcoords, min_pos3d );
      // cluster.getCluster3DPointAtTimeTick( timerange[1], img_v, badch_v, true, max_imgcoords, max_pos3d );
      // min_imgcoords[0] = timerange[0];
      // max_imgcoords[0] = timerange[1];

      std::swap( cluster.m_path3d, v3d );

      float dist = 0.;
      for (int i=0; i<3; i++) {
	dist += ( min_pos3d[i]-max_pos3d[i] )*(min_pos3d[i]-max_pos3d[i]);
      }
      // stop: path is good enough
      if ( dist>20.0 )
	break;
    }
    
  }
  
  ContourAStarCluster ContourAStarClusterAlgo::makeSeedClustersFrom3DPoint( const std::vector<float>& pos3d, const std::vector<larcv::Image2D>& img_v,
									    const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
									    const float max_dist2cluster ) {
    // we establish the seed cluster


    // first check if this is in a contour
    std::vector<int> imgcoords;
    try {
      imgcoords = larcv::UBWireTool::getProjectedImagePixel( pos3d, img_v.front().meta(), img_v.size() );
    }
    catch (...) {
      std::stringstream ss;
      ss << __FILE__ << ":" << __LINE__ << " Spacepoint could not project into the image." << std::endl;
      throw std::runtime_error( ss.str() );
    }

    if ( m_verbosity>0 ) {
      std::cout << __FILE__ << ":" << __LINE__ << " Seed Point: "
		<< " pos=" << pos3d[0] << "," << pos3d[1] << "," << pos3d[2]
		<< "  imgcoords=(" << imgcoords[0] << "," << imgcoords[1] << "," << imgcoords[2] << "," << imgcoords[3] << ")"
		<< "  tick=" << img_v.front().meta().pos_y( imgcoords[0] )
		<< std::endl;
    }
    
    
    std::vector<cv::Point> imgpt;
    for (size_t p=0; p<img_v.size(); p++) {
      cv::Point pt( imgcoords[p+1], imgcoords[0] );
      imgpt.emplace_back( std::move(pt) );
    }

    clock_t begin_contourloop = clock();
    std::vector< const ContourShapeMeta* > plane_seedcontours;
    int nplanes_found = 0;
    std::vector<int> seed_idx(3,-1);
    size_t nplanes = plane_contours_v.size();
    for (size_t p=0; p<nplanes; p++) {
      
      bool contains = false;
      int containing_idx = -1;
      float closest_dist = -1;
      for (size_t idx=0; idx<plane_contours_v[p].size(); idx++) {
	// test imgpt
	const ContourShapeMeta& ctr = plane_contours_v[p][idx];
	
	// fast test
	if ( imgpt[p].y<ctr.getMinY() || imgpt[p].y>ctr.getMaxY() )
	  continue;
	
	// more detailed test
	clock_t begin_pointpoly = clock();	
	double dist = cv::pointPolygonTest( ctr, imgpt[p], true );
	//std::cout << " contour (" << p << "," << idx << ") dist=" << dist << std::endl;
	if ( dist>=max_dist2cluster && (closest_dist<0 || closest_dist>fabs(dist) ) ) {
	  contains       = true;
	  containing_idx = (int)idx;
	  closest_dist   = fabs(dist);
	}
	clock_t end_pointpoly = clock();
	m_stage_times[kPointPolyTest] += float(end_pointpoly-begin_pointpoly)/CLOCKS_PER_SEC;
    
	//if ( contains )
	//break;
      }
      
      if ( contains ) {
	plane_seedcontours.push_back( &(plane_contours_v[p][containing_idx]) );
	nplanes_found++;
      }
      else {
	plane_seedcontours.push_back( NULL );
      }
      seed_idx[p] = containing_idx;
    }//end of loop over planes

    clock_t end_contourloop = clock();
    m_stage_times[kContourLoop] += float( end_contourloop-begin_contourloop )/CLOCKS_PER_SEC;
    
    // Make a cluster using the seed clusters
    clock_t begin_createcluster = clock();
    ContourAStarCluster seed( img_v, fMakeDebugImage );
    clock_t end_createcluster = clock();
    m_stage_times[kCreateCluster] += float(end_createcluster-begin_createcluster)/CLOCKS_PER_SEC;

    // make the seed cluster
    clock_t begin_addcontours = clock();
    for (int p=0; p<3; p++) {
      if ( seed_idx[p]>=0 ) {
	seed.addContour(p, plane_seedcontours[p], seed_idx[p] );
      }
    }
    clock_t end_addcontours = clock();
    m_stage_times[kAddContours] += float(end_addcontours-begin_addcontours)/CLOCKS_PER_SEC;
    
    clock_t begin_imageprep = clock();    
    seed.updateCVImage();
    seed.updateClusterContour();
    clock_t end_imageprep = clock();
    m_stage_times[kImagePrep] += float(end_imageprep-begin_imageprep)/CLOCKS_PER_SEC;
    return seed;
  }

  ContourAStarCluster ContourAStarClusterAlgo::buildClusterFromSeed( const std::vector<float>& pos3d, const std::vector<larcv::Image2D>& img_v,
								     const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
								     const float max_dist2cluster ) {
    ContourAStarCluster cluster = makeSeedClustersFrom3DPoint( pos3d, img_v, plane_contours_v, max_dist2cluster );
    std::vector<int> rowrange = cluster.getOverlappingRowRange();
    return cluster;
  }

  std::vector< std::set<int> >  ContourAStarClusterAlgo::extendClusterUsingAStarPath( ContourAStarCluster& cluster, std::vector< std::vector<float> >& path3d,
										      const std::vector<larcv::Image2D>& img_v, const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
										      const float distfromend, const float distextended, const float stepsize ) {
    // extend cluster using astar path to get direction of track
    // (1) get points on the end of the track
    // (2) fit line to points using opencv
    // (3) extend from ends and record clusters it intersects
    // return this list of clusters
    // // (4) also, make cluster image if (a) inside cluster and has charge (b) 2-plane has charge, 1-plane is dead
    // // (5) update the core cluster using the cluster image (contour it)
    // // (6) update the min/max range
    // //      -- first 
    
    const float cm_per_tick = ::larutil::LArProperties::GetME()->DriftVelocity()*0.5;
    const larcv::ImageMeta& meta = img_v.front().meta();

    // (1) get points on the end of the track
    // make a Point array list for opencv
    // we gather points some distance from the ends
    std::vector< cv::Point3f > point3d_start;
    std::vector< cv::Point3f > point3d_end;    

    const std::vector<float>& start = path3d.front();
    const std::vector<float>& end   = path3d.back();

    //std::cout << "extend path w/ size=" << path3d.size() << std::endl;
    
    for (int idx=0; idx<(int)path3d.size(); idx++) {
      const std::vector<float>& v3 = path3d[idx];

      float dist_start = 0;
      float dist_end   = 0;
      for (int i=0; i<3; i++) {
	dist_start += (start[i]-v3[i])*(start[i]-v3[i]);
	dist_end   += (end[i]-v3[i])*(end[i]-v3[i]);
      }
      dist_start = sqrt(dist_start);
      dist_end   = sqrt(dist_end);

      cv::Point3f pt( v3[0], v3[1], v3[2] );
      //std::cout << "path [" << idx << "] start_dist=" << dist_start << " end_dist=" << dist_end << std::endl;
      if ( dist_start<distfromend ) {
	point3d_start.push_back( pt );
      }
      if ( dist_end<distfromend ) {
	point3d_end.push_back( pt );
      }
    }//end of path loop
    
    
    // (2) fit line using opencv function
    std::vector<float> startline; // 6 parameters (vx, vy, vz, x0, y0, z0 )
    cv::fitLine( point3d_start, startline, CV_DIST_L2, 0.0, 0.01, 0.01 );
    std::vector<float> endline; // 6 parameters (vx, vy, vz, x0, y0, z0 )
    cv::fitLine( point3d_end, endline, CV_DIST_L2, 0.0, 0.01, 0.01 );

    // get direction from start to end to help us orient the lines
    std::vector<float> cluster_dir(3); // min -> max
    float cluster_dir_norm = 0.;
    for (int i=0; i<3; i++) {
      cluster_dir[i] = end[i] - start[i];
      cluster_dir_norm += cluster_dir[i]*cluster_dir[i];
    }
    cluster_dir_norm = sqrt(cluster_dir_norm);
    for (int i=0; i<3; i++)
      cluster_dir[i] /= cluster_dir_norm;

    float cosstart = 0;
    float cosend   = 0;
    for (int i=0; i<3; i++) {
      cosstart += cluster_dir[i]*startline[i];
      cosend   += cluster_dir[i]*endline[i];
    }

    // start should be negative, end should be positive
    if ( cosstart>0 ) {
      for (int i=0; i<3; i++)
	startline[i] *= -1.0;
    }
    if ( cosend<0 ) {
      for (int i=0; i<3; i++)
	endline[i] *= -1.0;
    }

    // (3) extend from ends and record clsuters it intersects/(4) mark the cluster image
    int numsteps = distextended/stepsize;

    std::vector< std::set<int> > indices_of_contours_v(3); // index of contours the extensions cross into
    std::vector< std::vector<float> > start_ext;
    std::vector< std::vector<float> > end_ext;    
    for (int isteps=1; isteps<=numsteps; isteps++ ) {

      std::vector<float> stepposmax(3);
      std::vector<float> stepposmin(3);
      for (int i=0; i<3; i++) {
	stepposmax[i] = float(isteps)*stepsize*endline[i]   + end[i];
	stepposmin[i] = float(isteps)*stepsize*startline[i] + start[i];
      }
      
      // x to tick
      stepposmax[0] = stepposmax[0]/cm_per_tick + 3200.0;
      stepposmin[0] = stepposmin[0]/cm_per_tick + 3200.0;
      
      if ( stepposmax[0]>2400 && stepposmax[0]<8448 && stepposmax[1]>-117 && stepposmax[1]<117 && stepposmax[2]>0.3 && stepposmax[2]<1030 ) {
	// project back into the image
	std::vector<int> imgcoordsmax = larcv::UBWireTool::getProjectedImagePixel( stepposmax, meta, img_v.size() );
	imgcoordsmax[0] = meta.row( stepposmax[0] );

	// start point check
	for (int p=0; p<3; p++) {
	  for (int ictr=0; ictr<(int)plane_contours_v[p].size(); ictr++) {
	    const ContourShapeMeta& ctr = plane_contours_v[p][ictr];
	    cv::Point testpt( imgcoordsmax[p+1], imgcoordsmax[0] );
	    double dist = cv::pointPolygonTest( ctr, testpt, false );
	    if ( dist>0 ) {
	      indices_of_contours_v[p].insert(ictr);
	      std::vector<float> pt = stepposmax;
	      pt[0] = (pt[0]-3200.0)*cm_per_tick;
	      end_ext.push_back( pt );
	    }
	  }
	}

	if ( fMakeDebugImage ) {
	  // draw for fun
	  for (int p=0; p<3; p++) {
	    cv::circle( cluster.m_cvimg_debug, cv::Point( imgcoordsmax[p+1], imgcoordsmax[0] ), 1, cv::Scalar(0,255,0), -1 );
	  }
	}
      }
      else {
	if ( m_verbosity>1 )
	  std::cout << " stepmax out of bounds: (" << stepposmax[0] << "," << stepposmax[1] << "," << stepposmax[2] << ")" << std::endl;
      }
      
      if ( stepposmin[0]>2400 && stepposmin[0]<8448 && stepposmin[1]>-117 && stepposmin[1]<117 && stepposmin[2]>0.3 && stepposmin[2]<1030 ) {
	// project back into the image	
	std::vector<int> imgcoordsmin = larcv::UBWireTool::getProjectedImagePixel( stepposmin, meta, img_v.size() );
	imgcoordsmin[0] = meta.row( stepposmin[0] );

	// start point check
	for (int p=0; p<3; p++) {
	  for (int ictr=0; ictr<(int)plane_contours_v[p].size(); ictr++) {
	    const ContourShapeMeta& ctr = plane_contours_v[p][ictr];
	    cv::Point testpt( imgcoordsmin[p+1], imgcoordsmin[0] );
	    double dist = cv::pointPolygonTest( ctr, testpt, false );
	    if ( dist>0 ) {
	      indices_of_contours_v[p].insert(ictr);
	      std::vector<float> pt = stepposmin;
	      pt[0] = (pt[0]-3200.0)*cm_per_tick;
	      start_ext.push_back( pt );	      
	    }
	  }
	}

	if ( fMakeDebugImage ) {
	  // draw for fun
	  for (int p=0; p<3; p++) {
	    cv::circle( cluster.m_cvimg_debug, cv::Point( imgcoordsmin[p+1], imgcoordsmin[0] ), 1, cv::Scalar(0,255,0), -1 );
	  }
	}
      }
      else {
	if ( m_verbosity>1 )
	  std::cout << " stepmin out of bounds: (" << stepposmin[0] << "," << stepposmin[1] << "," << stepposmin[2] << ")" << std::endl;
      }
      
    }//end of step loop

    if ( m_verbosity>0 ) {
      std::cout << "Cluster intersections: " << std::endl;
      for (int p=0; p<3; p++) {
	std::cout << " Plane " << p << ": ";
	for (auto& idx : indices_of_contours_v[p] )
	  std::cout << " " << idx;
	std::cout << std::endl;
      }
    }

    // extend path
    std::vector< std::vector<float> > extendedpath;
    for (int idx=(int)start_ext.size()-1; idx>=0; idx--) {
      extendedpath.push_back( start_ext[idx] );
    }
    for (int idx=0; idx<(int)path3d.size(); idx++)
      extendedpath.push_back( path3d[idx] );
    for (int idx=0; idx<(int)end_ext.size(); idx++)
      extendedpath.push_back( end_ext[idx] );

    std::swap( path3d, extendedpath );

    return indices_of_contours_v;
  }//end of function


  void ContourAStarClusterAlgo::fillInClusterImage( ContourAStarCluster& cluster, const std::vector< std::vector<float> >& path3d,
						    const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v,
						    const std::vector< std::set<int> >& cluster_indices, const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
						    const float maxstepsize, const float tag_qthreshold, const int neighborhood ) {
    // we use the astar path, the clusters it intersects (specified by cluster_indices) to fill in the cluster's CV image
    // we'll use draw contours and fill in points along path in dead region
    int nplanes = (int)cluster_indices.size();
    const larcv::ImageMeta& meta = img_v.front().meta();
    const float cm_per_tick = ::larutil::LArProperties::GetME()->DriftVelocity()*0.5;
    
    // fill in the contours
    for (int p=0; p<nplanes; p++) {
      std::vector< std::vector<cv::Point> > contour_list;      
      for ( auto& idx : cluster_indices[p] )
	contour_list.push_back( plane_contours_v[p][idx] );
      for (int ictr=0; ictr<(int)contour_list.size(); ictr++)
	cv::drawContours( cluster.m_cvimg_v[p], contour_list, ictr, cv::Scalar(255,0,0), CV_FILLED );	
    }

    // fill in the path
    int nnodes = (int)path3d.size();
    for (int inode=0; inode<nnodes-1; inode++) {
      auto const& start = path3d[inode];
      auto const& end   = path3d[inode+1];
      float dir[3];
      float norm = 0;
      for (int i=0; i<3; i++) {
	dir[i] = end[i]-start[i];
	norm += dir[i]*dir[i];
      }
      norm = sqrt(norm);
      for (int i=0; i<3; i++) {
	dir[i] /= norm;
      }
      int nsteps = norm/maxstepsize;
      if ( fabs(maxstepsize*nsteps-norm)>1.0e-2 ) {
	nsteps++;
      }
      float step = norm/float(nsteps);
      for (int istep=1; istep<=nsteps; istep++) {
	std::vector<float> pos(3);
	for (int i=0; i<3; i++)
	  pos[i] = start[i] + istep*step*dir[i];
	std::vector<int> imgcoords = larcv::UBWireTool::getProjectedImagePixel( pos, meta, img_v.size() );
	// change imgcoords to ticks
	imgcoords[0] = pos[0]/cm_per_tick + 3200.0;
	// tick to row
	imgcoords[0] = meta.row( imgcoords[0] );
	// go into image
	int nplanes_w_charge          = 0;
	int nplanes_w_charge_or_badch = 0;
	for (int p=0; p<3; p++) {
	  bool foundq = false;
	  bool foundbad = false;
	  for (int dr=-neighborhood; dr<=neighborhood; dr++) {
	    int row = imgcoords[0]+dr;
	    if ( row<0 || row>=(int)meta.rows() )
	      continue;
	    for (int dc=-neighborhood; dc<=neighborhood; dc++) {
	      int col = imgcoords[p+1]+dc;
	      if ( col<0 || col>=(int)meta.cols() )
		continue;
	      if ( img_v[p].pixel( row, col )>tag_qthreshold ) {
		foundq = true;
		break;
	      }
	      if ( badch_v[p].pixel(row,col)>0 ) {
		foundbad = true;
	      }
	    }// end of col loop
	    if ( foundq )
	      break;
	  }//end of row loop

	  if ( foundq )
	    nplanes_w_charge++;
	  if ( foundq || foundbad)
	    nplanes_w_charge_or_badch++;
	}//end of plane loop

	
	if ( nplanes_w_charge>=2 && nplanes_w_charge_or_badch==nplanes ) {
	  // good point
	  // we fill dead channel regions and parts with charge
	  for (int p=0; p<nplanes; p++) {
	    for (int dr=-neighborhood; dr<=neighborhood; dr++) {
	      int row = imgcoords[0]+dr;
	      if ( row<0 || row>=(int)meta.rows() )
		continue;
	      for (int dc=-neighborhood; dc<=neighborhood; dc++) {
		int col = imgcoords[p+1]+dc;
		if ( col<0 || col>=(int)meta.cols() )
		  continue;
		if ( img_v[p].pixel( row, col )>tag_qthreshold || (badch_v[p].pixel(row,col)>0 && abs(dr)<=1 && abs(dc)<=1) ) {
		  //cluster.m_cvpath_v[p].at<uchar>( cv::Point(col,row) ) = 255;
		  cluster.m_cvimg_v[p].at<uchar>( cv::Point(col,row) ) = 255;		  
		}
	      }//end of col loop
	    }//end of row loop
	  }//en dof plane loop
	}//end of if point is good

	
      }//end of step loop
    }//end of node loop

    // now that we update the images, time to update the contours for both
    // the cluster collection image and the path image

    // cluster image
    cluster.m_current_contours.clear();
    cluster.m_current_contours.resize(3);
    for (int p=0; p<cluster.m_nplanes; p++) {
      std::vector< std::vector<cv::Point> > contour_v;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours( cluster.m_cvimg_v[p], contour_v, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0) );
      for ( auto& ctr : contour_v ) {
	ContourShapeMeta ctrmeta( ctr, img_v[p].meta() );
	cluster.m_current_contours[p].emplace_back( std::move(ctrmeta) );
      }
    }

    // path image contours
    // cluster.m_path_contours.clear();
    // cluster.m_path_contours.resize(3);
    // for (int p=0; p<cluster.m_nplanes; p++) {
    //   std::vector< std::vector<cv::Point> > contour_v;
    //   std::vector<cv::Vec4i> hierarchy;
    //   cv::findContours( cluster.m_cvpath_v[p], contour_v, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0) );
    //   for ( auto& ctr : contour_v ) {
    // 	ContourShapeMeta ctrmeta( ctr, img_v[p].meta() );
    // 	cluster.m_path_contours[p].emplace_back( std::move(ctrmeta) );
    //   }
    // }
    
  }//end of function

  void ContourAStarClusterAlgo::printStageTimes() {
    std::cout << "================================================" << std::endl;
    std::cout << "ContourAStarClusterAlgo::Stage Times" << std::endl;
    std::cout << "Seeding Contour Loop: " << m_stage_times[kContourLoop] << std::endl;
    std::cout << "Seeding Point Poly Test: " << m_stage_times[kPointPolyTest] << std::endl;
    std::cout << "Seeding Create Cluster: " << m_stage_times[kCreateCluster] << std::endl;    
    std::cout << "Seeding Image Prep: " << m_stage_times[kImagePrep] << std::endl;
    std::cout << "Seeding ContourAddition: " << m_stage_times[kAddContours] << std::endl;
    std::cout << "================================================" << std::endl;
  }
  
}
