#include "CACAEndPtFilter.h"

#include <sstream>
#include <assert.h>
#include <ctime>

// larlite
#include "LArUtil/LArProperties.h"

// larcv
#include "CVUtil/CVUtil.h"
#include "UBWireTool/UBWireTool.h"

// larlitecv
#include "TaggerTypes/dwall.h"

namespace larlitecv {

  CACAEndPtFilter::CACAEndPtFilter() {
    fTruthInfoLoaded = false;
    fMakeDebugImage = false;
    m_verbosity = 0;
    m_last_was_duplicate = false;
    m_stage_times.resize( kNumStages, 0 );
    m_debug_set.clear();
  };
    
  
  bool CACAEndPtFilter::isEndPointGood( const larlitecv::BoundarySpacePoint& pt, const larlite::opflash* associated_flash,
					const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v,
					const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,					
					const float max_dtick ) {

    // evaluates end point goodness
    if ( m_verbosity>0 )
      std::cout << __FILE__ << ":" << __LINE__ << " ----------------------------------" << std::endl;
    
    const float cm_per_tick = ::larutil::LArProperties::GetME()->DriftVelocity()*0.5;    

    // make a seed cluster
    clock_t begin_seed = clock();
    if ( fMakeDebugImage )
      m_caca.makeDebugImage();
    larlitecv::ContourAStarCluster cluster = m_caca.makeSeedClustersFrom3DPoint( pt.pos(), img_v, plane_contours_v, -10 );
    clock_t end_seed = clock();
    m_stage_times[kSeedMaking] += float(end_seed-begin_seed)/CLOCKS_PER_SEC;

    // test if a duplicate
    clock_t begin_dup = clock();
    if ( isDuplicateEndPoint( cluster, pt ) ) {
      if ( m_verbosity>0 )
	std::cout << " Duplicate seed cluster." << std::endl;
      m_last_was_duplicate = true;
      m_last_clusters.emplace_back( std::move(cluster) );      
      return false;
    }
    else
      m_last_was_duplicate = false;
    clock_t end_dup = clock();
    m_stage_times[kDuplicateEval] += float(end_dup-begin_dup)/CLOCKS_PER_SEC;

    // extend a non-duplicate cluster
    clock_t begin_ext = clock();
    m_caca.extendSeedCluster( pt.pos(), img_v, badch_v, plane_contours_v, 3, cluster );
    clock_t end_ext = clock();
    m_stage_times[kClusterExtension] += float(end_ext-begin_ext)/CLOCKS_PER_SEC;    

    // check if good path could be made
    if ( cluster.m_path3d.size()==0 ) {
      // if not, register seed cluster, return
      m_last_clusters.emplace_back( std::move(cluster) );      
      return false;
    }


    // fill in most recent past info
    // -----------------------------
    // set the position based on the end closest to the wall
    if ( m_verbosity>0 ) {
      std::cout << "fill in past info." << std::endl;
      std::cout << "front: (" << cluster.m_path3d.front()[0] << "," << cluster.m_path3d.front()[1] << "," << cluster.m_path3d.front()[2] << ")" << std::endl;
      std::cout << "back:  (" << cluster.m_path3d.back()[0] << "," << cluster.m_path3d.back()[1] << "," << cluster.m_path3d.back()[2] << ")" << std::endl;
    }

    std::vector<float> pathdir_front(3,0);
    std::vector<float> pathdir_back(3,0);    

      
    // get the direction of the point 10 cm (or closer out)
    const std::vector<float>& start = cluster.m_path3d.front();
    for ( auto const& pos : cluster.m_path3d ) {
      float dist2start = 0.;      
      for (int i=0; i<3; i++)
	dist2start += ( pos[i]-start[i] )*( pos[i]-start[i] );
      dist2start = sqrt(dist2start);
      if ( dist2start<10.0 ) {
	// update the front dir
	for (int i=0; i<3; i++)
	  pathdir_front[i] = (pos[i]-start[i])/dist2start;
      }
    }
    const std::vector<float>& end = cluster.m_path3d.back();
    for ( int ipos=(int)cluster.m_path3d.size()-1; ipos>=0; ipos-- ) {
      auto const& pos = cluster.m_path3d[ipos];
      float dist2end = 0.;
      for (int i=0; i<3; i++)
	dist2end += ( pos[i]-end[i] )*( pos[i]-end[i] );
      dist2end = sqrt(dist2end);
      if ( dist2end<10.0 ) {
	for (int i=0; i<3; i++)
	  pathdir_back[i] = (pos[i]-end[i])/dist2end;
      }
    }

    if ( m_verbosity>0 ) {
      std::cout << "front dir: (" << pathdir_front[0] << "," << pathdir_front[1] << "," << pathdir_front[2] << ")" << std::endl;
      std::cout << "back dir:  (" << pathdir_back[0] << "," << pathdir_back[1] << "," << pathdir_back[2] << ")" << std::endl;    
    }
    PastClusterInfo_t& info = m_past_info.back();    
    bool usefront = true; // if false use back
    float tick_start = start[0]/cm_per_tick + 3200.0;
    float tick_end   = end[0]/cm_per_tick   + 3200.0;
    if ( m_verbosity>0 ) {
      std::cout << "front tick: " << tick_start << std::endl;
      std::cout << "back tick:  " << tick_end << std::endl;
    }
    
    // use voting system, where we pick the end point that is closest to the edges of the different plane bounds determined by contour collection
    float tick_dist_start = 0;
    float tick_dist_end   = 0;
    // for the start and end point, we add the shortest distance to cluster boundaries for each plane
    // distance for now is just the row distance
    for ( int p=0; p<3; p++) {
      const larcv::ImageMeta& meta = img_v[p].meta();
      if ( cluster.m_plane_rowminmax[p][0]<0 || cluster.m_plane_rowminmax[p][1]<0 )
	continue; // nothing good here
      float cluster_rowmintick = meta.pos_y( cluster.m_plane_rowminmax[p][0] );
      float cluster_rowmaxtick = meta.pos_y( cluster.m_plane_rowminmax[p][1] );
      if ( m_verbosity>0 )
	std::cout << "plane tick bounds [" << cluster_rowmaxtick << "," << cluster_rowmintick << "]" << std::endl;

      float start_dmin = fabs(cluster_rowmintick-tick_start);
      float start_dmax = fabs(cluster_rowmaxtick-tick_start);
      if ( start_dmin<start_dmax )
	tick_dist_start += start_dmin;
      else
	tick_dist_start += start_dmax;

      float end_dmin = fabs(cluster_rowmintick-tick_end);
      float end_dmax = fabs(cluster_rowmaxtick-tick_end);
      if ( end_dmin<end_dmax )
	tick_dist_end += end_dmin;
      else
	tick_dist_end += end_dmax;
    }

    if ( m_verbosity>0 ) {
      std::cout << "front tick total dist: " << tick_dist_start << std::endl;
      std::cout << "back tick total dist:  " << tick_dist_end << std::endl;
    }

    if ( tick_dist_start<tick_dist_end )
      usefront = true;
    else
      usefront = false;
    
    if ( usefront ) {
      for (int i=0; i<3; i++)
	info.dir[i] = pathdir_front[i];
      info.pos = start;
    }
    else {
      // use the end
      for (int i=0; i<3; i++)
	info.dir[i] = pathdir_back[i];
      info.pos = end;
    }
    
    // good cluster with 3D-consistent track found. evaluate goodness.
    // ---------------------------------------------------------------

    m_last_clusters.emplace_back( std::move(cluster) );
    
    if ( pt.type()==larlitecv::kAnode || pt.type()==larlitecv::kCathode ) {
      // perform end test, needs to be in-time with flash and be exiting or entering the right direction
      if ( m_verbosity>0 )
	std::cout << "ticks start=" << tick_start << " end=" << tick_end << std::endl;
      float flash_tick = 3200 + associated_flash->Time()/0.5;
      if ( pt.type()==larlitecv::kCathode )
	flash_tick += (256.0)/cm_per_tick;
      if ( m_verbosity>0 )      
	std::cout << "flash tick=" << flash_tick << std::endl;

      float dir[3] = { 0, 0, 0};
      std::vector<float> flashend;
      float trackend_tick;
      float norm = 0.;      
      if ( fabs(flash_tick-tick_start) < fabs(flash_tick-tick_end)  ) {
	// start end is closest
	flashend = start;
	trackend_tick = tick_start;
	for (int i=0; i<3; i++) {
	  dir[i] = (end[i]-start[i]);
	  norm += dir[i]*dir[i];
	}
      }
      else {
	// end end if closest
	flashend = end;
	trackend_tick = tick_end;
	for (int i=0; i<3; i++) {
	  dir[i] = (start[i]-end[i]);
	  norm += dir[i]*dir[i];
	}
      }
      norm = sqrt(norm);
      if ( norm>0 )
	for (int i=0; i<3; i++) dir[i] /= norm;

      float dtick = fabs(trackend_tick-flash_tick);
      if ( m_verbosity>0 ) {
	std::cout << "dtick=" << dtick << std::endl;
	std::cout << "type=" << pt.type() << std::endl;
	std::cout << "dir[0]=" << dir[0] << std::endl;
      }
      
      if ( dtick < max_dtick ) {
	if ( pt.type()==larlitecv::kAnode && dir[0]>0 )
	  return true;
	if ( pt.type()==larlitecv::kCathode && dir[0]<0 )
	  return true;
      }
      
      return false;
    }
    else if (pt.type()<=larlitecv::kDownstream ) {
      // check it is crossing the correct boundary and goes the right direction
      int target_wall = pt.type();
      float fdwall_target_start = 1.0e5;
      float fdwall_target_end   = 1.0e5;      
      fdwall_target_start = larlitecv::dspecificwall( start, target_wall );
      fdwall_target_end   = larlitecv::dspecificwall( end,   target_wall );      

      float dir[3] = { 0, 0, 0};
      float norm = 0.;
      float dwall_closest = 0.;
      if ( fdwall_target_start < fdwall_target_end ) {
	for (int i=0; i<3; i++) {
	  dir[i]  = end[i] - start[i];
	  norm    += dir[i]*dir[i];
	}
	dwall_closest = fdwall_target_start;
      }
      else {
	for (int i=0; i<3; i++) {
	  dir[i]  = start[i] - end[i];
	  norm    += dir[i]*dir[i];
	}
	dwall_closest = fdwall_target_end;	
      }
	
      norm = sqrt(norm);
      if ( norm>0 ) {
	for (int i=0; i<3; i++)
	  dir[i] /= norm;
      }

      if ( dwall_closest<17.0 ) {
	if ( target_wall==larlitecv::kTop && dir[1]<0 )
	  return true;
	else if ( target_wall==larlitecv::kBottom && dir[1]>0 )
	  return true;
	else if ( target_wall==larlitecv::kUpstream && dir[2]>0 )
	  return true;
	else if ( target_wall==larlitecv::kDownstream && dir[2]<0 )
	  return true;
      }
    }
    else if ( pt.type()==larlitecv::kImageEnd ) {
      return true;
    }
    else {
      std::stringstream ss;
      ss << __FILE__ << ":" << __LINE__ << " unrecognized boundary type = " << pt.type() << std::endl;
      throw std::runtime_error( ss.str() );
    }
    
    return false;
  }

  void CACAEndPtFilter::evaluateEndPoints( const std::vector< const std::vector<larlitecv::BoundarySpacePoint>* >& sp_v, const std::vector< larlite::event_opflash* >& flash_v,
					   const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v,
					   const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
					   const float max_dtick, std::vector< std::vector<int> >& passes_filter ) {
					   
    // Function to evaluate a vector of space points
    // Inputs
    // ------
    // sp_v: a vector of vector of points
    // flash_v: a vector of event_opflash containers
    // img_v: vector of image, each element belongs to different planes
    // badch_v: vector badch-tagged images, each element belongs to different planes
    // plane_contours_v: vector of contours for each plane. made by BMTCV class.
    // dist_from_wall: max dist from wall that good end point can be
    // chi2_threshold: how well it matches flashes (not used for now)
    // max_dtick: distance of ends from paired flash

    // Implicit parameters
    // --------------------
    // fTruthInfoLoaded: if true, then will evaluate MC info
    // m_truthinfo_ptr_v: analysis of truth crossing points from larlitecv/app/MCTruthTools/crossingPointsAnaMethods.h/cxx
    // m_recoinfo_ptr_v:  analysis of reco crossing points from larlitecv/app/MCTruthTools/crossingPointsAnaMethods.h/cxx

    // Outputs
    // -------
    // passes_filter: vector of output, 0=fails, 1=passes, follows structure of sp_v    

    clock_t begin_overall = clock();
    
    // tracking variables
    std::vector<int> good_npasses_caca( larlitecv::kNumEndTypes, 0 );
    std::vector<int> good_nfails_caca(  larlitecv::kNumEndTypes, 0 );
    std::vector<int> good_duplicates_caca(  larlitecv::kNumEndTypes, 0 );    
    std::vector<int> bad_npasses_caca(  larlitecv::kNumEndTypes, 0 );
    std::vector<int> bad_nfails_caca(   larlitecv::kNumEndTypes, 0 );
    std::vector<int> bad_duplicates_caca(  larlitecv::kNumEndTypes, 0 );    
    
    if ( m_verbosity>0 ) {
      std::cout << "==========================================================================" << std::endl;
      std::cout << "CACAENDPTFILTER: evaluate points" << std::endl;
    }
    
    if ( fMakeDebugImage ) {
      clock_t begin = clock();
      m_cvimg_rgbdebug.clear();
      // we create 2 RGB image. We will use this to mark endpoints provided to the filter.
      // (1) reco points which are close to true crossing points
      // (2) reco points which are not close to true crossing points. if no MC, this one not filled.
      const larcv::ImageMeta& meta = img_v.front().meta();
      cv::Mat rgbdebug( meta.rows(), meta.cols(), CV_8UC3 );
      cv::Mat rgbdebug2( meta.rows(), meta.cols(), CV_8UC3 );      
      for (int p=0; p<3; p++) {
	auto const& img = img_v[p];
	cv::Mat cvimg = larcv::as_gray_mat( img, 0.0, 50.0, 0.2 );
	for (size_t r=0; r<meta.rows(); r++) {
	  for (size_t c=0; c<meta.cols(); c++) {
	    rgbdebug.at<cv::Vec3b>( cv::Point(c,r) )[p] = cvimg.at<unsigned char>(cv::Point(c,r) );
	    rgbdebug2.at<cv::Vec3b>( cv::Point(c,r) )[p] = cvimg.at<unsigned char>(cv::Point(c,r) );	    
	  }
	}
      }

      m_cvimg_rgbdebug.emplace_back( std::move(rgbdebug) );
      m_cvimg_rgbdebug.emplace_back( std::move(rgbdebug2) );
      clock_t end = clock();
      m_stage_times[kDebugImages] += float(end-begin)/CLOCKS_PER_SEC;
    }
    
    passes_filter.clear();
    m_past_info.clear();

    bool use_debug_set = false;
    if ( !m_debug_set.empty() ) use_debug_set = true;
    
    int ireco = -1; // overall counter    
    for ( auto const& p_sp_v : sp_v ) {
      std::vector<int> passes_v(p_sp_v->size(),0);

      int isp = -1; // this vector's counter
      for (auto const& sp : *p_sp_v ) {
	ireco++; // counter for all spacepoint indices
	isp++;   // counter for sp index of this vector

	// debug: select event
	if ( use_debug_set ) {
	  if ( m_debug_set.find( ireco )==m_debug_set.end() )
	    continue;
	}
	
	if ( sp.type()==larlitecv::kTop
	     || sp.type()==larlitecv::kBottom
	     || sp.type()==larlitecv::kUpstream
	     || sp.type()==larlitecv::kDownstream	     
	     || sp.type()==larlitecv::kAnode
	     || sp.type()==larlitecv::kCathode
	     || sp.type()==larlitecv::kImageEnd
	     ) {
	  // debug select boundary type
	  //if ( sp.type()==larlitecv::kAnode ) {

	  // clear the cluster vector
	  clearClusters();
	  
	  const larlitecv::BoundaryFlashIndex& flashidx = sp.getFlashIndex();
	  if ( m_verbosity>1 ) {
	    std::cout << "--------------------------------------------------------------" << std::endl;
	    std::cout << "[ipt " << ireco << "] Anode/Cathode space point" << std::endl;
	    std::cout << "  flash index: (" << flashidx.ivec << "," << flashidx.idx << "," << flashidx.popflash << ")" << std::endl;
	  }
	  
	  const larlite::opflash* popflash = NULL;
	  if ( sp.type()==larlitecv::kAnode || sp.type()==larlitecv::kCathode ) {
	    // anode/cathode. get the flash index
	    popflash = &((flash_v.at(flashidx.ivec))->at(flashidx.idx));

	    if ( m_verbosity>1 ) {
	      std::cout << "  flash pointer: " << popflash << std::endl;
	    }
	  }
	  bool passes = isEndPointGood( sp, popflash, img_v, badch_v, plane_contours_v, max_dtick );

	  if ( m_verbosity>0 ) {
	    std::cout << "Result: " << passes << std::endl;
	  }

	  if ( wasLastEndPointDuplicate() ) {
	    std::cout << "Duplicate. Move to next end point." << std::endl;
	    if ( fTruthInfoLoaded ) {
	      if ( (m_recoinfo_ptr_v->at(ireco)).truthmatch==1 )
		good_duplicates_caca[ (int)sp.type() ]++;
	      else
		bad_duplicates_caca[ (int)sp.type() ]++;
	    }
	    good_duplicates_caca[ (int)sp.type() ]++;
	    continue;
	  }

	  // if we got this far, we passed
	  if ( passes )
	    m_past_info.back().passed = 1;
	  
	  const larlitecv::RecoCrossingPointAna_t* recoinfo   = NULL;
	  bool truthmatched = false;

	  // --------------------------------------------------------------------------
	  // Performance Analysis using truth information
	  if ( fTruthInfoLoaded ) {

	    try {
	      recoinfo = &(m_recoinfo_ptr_v->at(ireco));
	    }
	    catch (...) {
	      continue;
	    }

	    if ( m_verbosity>0 ) 
	      std::cout << "Has Truth Match: " << recoinfo->truthmatch << std::endl;	    
	    
	    if ( recoinfo->truthmatch==1 ) {
	      truthmatched = true;
	      m_past_info.back().truthmatched = 1;
	      const larlitecv::TruthCrossingPointAna_t* truthinfo = NULL;
	      try {
		truthinfo = &(m_truthinfo_ptr_v->at(recoinfo->truthmatch_index));
	      }
	      catch (...) {
		continue;
	      }
	      
	      if ( m_verbosity>1 )  {
		std::cout << "Truth crossing position: "
			  << "(" << truthinfo->crossingpt_detsce[0] << "," << truthinfo->crossingpt_detsce[1] << "," << truthinfo->crossingpt_detsce[2] << ")"
			  << " dist=" << recoinfo->truthmatch_dist
			  << std::endl;
		std::cout << "Closest Stored Truth crossing position: "
			  << "(" << recoinfo->truthmatch_detsce_tyz[0] << "," << recoinfo->truthmatch_detsce_tyz[1] << "," << recoinfo->truthmatch_detsce_tyz[2] << ")"
			  << " dist=" << recoinfo->truthmatch_dist
			  << std::endl;		
	      }
	    
	      // good reco point
	      if ( passes ) {
		passes_v[isp] = 1;	  
		good_npasses_caca[(int)sp.type()]++;
	      }
	      else {
		good_nfails_caca[(int)sp.type()]++;
	      }
	    }//end if reco is truth matched
	    else {
	      m_past_info.back().truthmatched = 0;
	      // bad reco point
	      if ( passes ) {
		passes_v[isp] = 1;	  
		bad_npasses_caca[(int)sp.type()]++;
	      }
	      else
		bad_nfails_caca[(int)sp.type()]++;

	      if ( m_verbosity>1) {
		const larlitecv::TruthCrossingPointAna_t* truthinfo = NULL;
		try {
		  truthinfo = &(m_truthinfo_ptr_v->at(recoinfo->truthmatch_index));
		  if ( m_verbosity>0 ) {
		    std::cout << "Closest Truth crossing position: "
			      << "(" << truthinfo->crossingpt_detsce[0] << "," << truthinfo->crossingpt_detsce[1] << "," << truthinfo->crossingpt_detsce[2] << ")"
			      << " dist=" << recoinfo->truthmatch_dist
			      << std::endl;
		    std::cout << "Closest Stored Truth crossing position: "
			      << "(" << recoinfo->truthmatch_detsce_tyz[0] << "," << recoinfo->truthmatch_detsce_tyz[1] << "," << recoinfo->truthmatch_detsce_tyz[2] << ")"
			      << " dist=" << recoinfo->truthmatch_dist
			      << std::endl;
		  }
		}
		catch (...) {
		  if ( m_verbosity>0 ) 
		    std::cout << "No closest truth point on record." << std::endl;
		  continue;
		}
	      }
	      
	    }//end of if non-matched reco point	    
	  }// if truth loaded
	  // end of truth analysis -----------------------------------------------------------------------------------

	  
	  if ( fMakeDebugImage ) {
	    // we draw the cluster and end point
	    clock_t begin = clock();
	    
	    // contours
	    larlitecv::ContourAStarCluster& astar_cluster = getLastCluster();
	    
	    // end point from original boundary point
	    std::vector<int> sp_imgcoords = larcv::UBWireTool::getProjectedImagePixel( sp.pos(), img_v.front().meta(), 3 );

	    // end point from cluster
	    // not yet implemented

	    int img_index = 0;
	    if ( fTruthInfoLoaded ) {
	      if ( truthmatched )
		img_index = 0;
	      else
		img_index = 1;
	    }
	    else {
	      if (passes)
		img_index = 0;
	      else
		img_index = 1;
	    }
	    
	    for ( size_t p=0; p<astar_cluster.m_current_contours.size(); p++) {
	      // first copy into contour container	      
	      std::vector< std::vector<cv::Point> > contour_v;	    	      
	      auto const& contour_shpmeta_v = astar_cluster.m_current_contours[p];
	      // now draw contours
	      for ( auto const& ctr : contour_shpmeta_v ) {
		if ( ctr.size()>0 )
		  contour_v.push_back( ctr );
	      }
	      for (int i=0; i<(int)contour_v.size(); i++)
		cv::drawContours( m_cvimg_rgbdebug[img_index], contour_v, i, cv::Scalar(150,150,150), 1 );
	      // draw end point
	      cv::Scalar ptcolor(0,255,255,255);
	      if ( passes )
		ptcolor = cv::Scalar(255,0,255,255);
	      cv::circle(  m_cvimg_rgbdebug[img_index], cv::Point( sp_imgcoords[p+1], sp_imgcoords[0] ), 3, ptcolor, 1 );
	      // draw end point
	      std::stringstream ptname;	  
	      ptname << "#" << ireco;
	      cv::putText( m_cvimg_rgbdebug[img_index], cv::String(ptname.str()), cv::Point( sp_imgcoords[p+1]+2, sp_imgcoords[0]+2 ), cv::FONT_HERSHEY_SIMPLEX, 0.5, ptcolor );
	      
	    }// end of plane loop
	    clock_t end = clock();
	    m_stage_times[kDebugImages] += float(end-begin)/CLOCKS_PER_SEC;	
	  }//end of if debug image
	  
	}//if correct type

      }//end of space points loop
      
      passes_filter.emplace_back( std::move(passes_v) );
      
    }//end of spacepoints_vv loop
    
    if ( fTruthInfoLoaded && m_verbosity>0 ) {
      std::cout << "CACA Summary" << std::endl;
      for (int i=0; i<larlitecv::kNumEndTypes; i++)
	std::cout << "  Good " << larlitecv::BoundaryEndNames((larlitecv::BoundaryEnd_t)i) << ":   " << good_npasses_caca[i] << "/" << good_npasses_caca[i]+good_nfails_caca[i] << std::endl;
      for (int i=0; i<larlitecv::kNumEndTypes; i++)
	std::cout << "  Bad "  << larlitecv::BoundaryEndNames((larlitecv::BoundaryEnd_t)i) << ":   " << bad_npasses_caca[i] <<  "/" << bad_npasses_caca[i]+bad_nfails_caca[i] << std::endl;
      for (int i=0; i<larlitecv::kNumEndTypes; i++)
	std::cout << "  Duplicates (Good/Bad) "  << larlitecv::BoundaryEndNames((larlitecv::BoundaryEnd_t)i) << ":   "
		  << good_duplicates_caca[i] <<  "/" << bad_duplicates_caca[i] << std::endl;
    }

    clock_t end_overall = clock();
    m_stage_times[kOverall] += float(end_overall-begin_overall)/CLOCKS_PER_SEC;
    
  }

  std::vector< larlitecv::BoundarySpacePoint >  CACAEndPtFilter::regenerateFitleredBoundaryPoints( const std::vector<larcv::Image2D>& img_v ) {

    std::vector< larlitecv::BoundarySpacePoint > filteredsp_v;
    
    for ( auto const& pastinfo : m_past_info ) {
      std::cout << __PRETTY_FUNCTION__ << ": (" << pastinfo.pos[0] << "," << pastinfo.pos[1] << "," << pastinfo.pos[2] << ")" << std::endl;
      larlitecv::BoundarySpacePoint sp( (larlitecv::BoundaryEnd_t)pastinfo.type, pastinfo.pos, pastinfo.dir, img_v.front().meta() );
      sp.setFlashIndex( pastinfo.vecindex, pastinfo.flashindex, pastinfo.popflash );

      if ( fMakeDebugImage ) {
	int img_index = 0;
	if ( pastinfo.truthmatched==0 )
	  img_index = 1;
	cv::Scalar ptcolor(0,255,0,255);
	for (int p=0; p<3; p++)
	  cv::circle(  m_cvimg_rgbdebug[img_index], cv::Point( sp[p].col, sp[p].row ), 3, ptcolor, 1 );
      }
      
      filteredsp_v.emplace_back( std::move(sp) );
    }
    
    return filteredsp_v;
  }

  
  void CACAEndPtFilter::setTruthInformation( const std::vector<larlitecv::TruthCrossingPointAna_t>& truthinfo, const std::vector<larlitecv::RecoCrossingPointAna_t>& recoinfo ) {
    m_truthinfo_ptr_v = &truthinfo;
    m_recoinfo_ptr_v  = &recoinfo;
    fTruthInfoLoaded  = true;
  }

  bool CACAEndPtFilter::isDuplicateEndPoint( const larlitecv::ContourAStarCluster& seedcluster, const larlitecv::BoundarySpacePoint& sp ) {
    // checks the seed cluster against previous clusters.
    // if plane contours a subset of other contour clusters, then a duplicate
    //std::cout << "  duplicate check. number of past clusters=" << m_last_clusters.size() << std::endl;
    //std::cout << "  seedcluster plane indices: " << seedcluster.m_bmtcv_indices.size() << std::endl;
    
    for ( auto const& pastcluster : m_past_info ) {
      // check the plane contour indices
      bool overlaps = true;
      for (size_t p=0; p<seedcluster.m_bmtcv_indices.size(); p++) {
	for ( auto const& contourindex : seedcluster.m_bmtcv_indices[p] )  {

	  // for debug
	  // ---------
	  // std::cout << "seed: p=" << p << " contourindex=" << contourindex << " check in existing cluster: ";
	  // for ( auto const& usedindex : pastcluster.plane_bmtcv_indices[p] )
	  //   std::cout << " " << usedindex;
	  // std::cout << std::endl;
	  
	  if ( pastcluster.plane_bmtcv_indices[p].find(contourindex)==pastcluster.plane_bmtcv_indices[p].end() ) {
	    // index not found within set
	    overlaps = false;
	    break;
	  }
	}
	if ( !overlaps ) {
	  // no need to keep checking
	  break;
	}
      }

      // overlaps, so is a duplicate
      if ( overlaps ) {
	return true;
      }
    }

    // didn't overlap with any, not a duplicate
    // make a past info
    PastClusterInfo_t info(seedcluster);
    info.type       = (int)sp.type();
    info.vecindex   = sp.getFlashIndex().ivec;
    info.flashindex = sp.getFlashIndex().idx;
    info.popflash   = sp.getFlashIndex().popflash;

    m_past_info.emplace_back( std::move(info) );
    return false;
  }

  void CACAEndPtFilter::printStageTimes() {
    std::cout << "==============================================" << std::endl;
    std::cout << "CACAEndPtFilter Timing -----------------------" << std::endl;
    std::cout << "Overall: " << m_stage_times[kOverall] << std::endl;
    std::cout << "Seed making: " << m_stage_times[kSeedMaking] << std::endl;
    std::cout << "Duplicate eval: " << m_stage_times[kDuplicateEval] << std::endl;
    std::cout << "Cluster extension: " << m_stage_times[kClusterExtension] << std::endl;
    std::cout << "Debug Images: " << m_stage_times[kDebugImages] << std::endl;
    m_caca.printStageTimes();
    std::cout << "==============================================" << std::endl;
    
  }

  void CACAEndPtFilter::setDebugSet( const std::vector<int>& ptindex ) {
    m_debug_set.clear();
    for ( auto const& idx : ptindex )
      m_debug_set.insert(idx);
  }

}
