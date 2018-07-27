#include "FlowContourMatch.h"

#include <exception>

#include "larcv/core/DataFormat/ImageMeta.h"

#include "TH2D.h"

namespace larflow {

  FlowMatchData_t::FlowMatchData_t( int srcid, int tarid )
    : src_ctr_id(srcid), tar_ctr_id(tarid)
  {}
  FlowMatchData_t::FlowMatchData_t( const FlowMatchData_t& x ) {
    src_ctr_id = x.src_ctr_id;
    tar_ctr_id = x.tar_ctr_id;
    score      = x.score;
    matchingflow_v = x.matchingflow_v;
  }

  // ==========================================
  // FlowContourMatch Algo
  // ---------------------
  
  FlowContourMatch::FlowContourMatch() {
    m_score_matrix = NULL;
    m_plot_scorematrix = NULL;
    m_src_img2ctrindex = NULL;
    m_tar_img2ctrindex = NULL;    
  }

  FlowContourMatch::~FlowContourMatch() {
    clear();
  }

  void FlowContourMatch::clear( bool clear2d, bool clear3d ) {
    if ( clear2d ) {
      delete [] m_score_matrix;
      delete m_plot_scorematrix;
      m_score_matrix = NULL;
      m_plot_scorematrix = NULL;
      m_flowdata.clear();
      m_src_targets.clear();
      delete m_src_img2ctrindex;
      delete m_tar_img2ctrindex;
      m_src_img2ctrindex = NULL;
      m_tar_img2ctrindex = NULL;
    }
    
    if (clear3d) {
      m_hit2flowdata.clear();
    }
  }

  // ==================================================================================
  // Primary Algo Interface
  // -----------------------
  void FlowContourMatch::match( FlowDirection_t flowdir,
				const larlitecv::ContourCluster& contour_data,
				const larcv::Image2D& src_adc,
				const larcv::Image2D& tar_adc,
				const larcv::Image2D& flow_img,
				const larlite::event_hit& hit_v,
				const float threshold ) {
    
    // produces 3D hits from from one flow image
    int src_planeid = -1;
    int tar_planeid = -1;

    switch ( flowdir ) {
    case kY2U:
      src_planeid = 2;
      tar_planeid = 0;
      break;
    case kY2V:
      src_planeid = 1;
      tar_planeid = 0;
      break;
    default:
      throw std::runtime_error("FlowContourMatch::match: invalid FlowDirection_t option"); // shouldnt be possible
      break;
    }

    // we clear the 2d data, but keep the hit data (and update it)
    clear( true, false );
    
    
    // first we create match data within the image
    _createMatchData( contour_data, flow_img, src_adc, tar_adc );

    // use the match data to score contour-contour matching
    _scoreMatches( contour_data, src_planeid, tar_planeid );

    // use score matrix to define matches
    _greedyMatch();

    // make 3D hits and update hit2flowdata vector
    _make3Dhits( hit_v, src_adc, tar_adc, src_planeid, tar_planeid, threshold, m_hit2flowdata );

  }
  
  
  void FlowContourMatch::_createMatchData( const larlitecv::ContourCluster& contour_data,
					   const larcv::Image2D& flow_img,
					   const larcv::Image2D& src_adc,
					   const larcv::Image2D& tar_adc ) {

    int src_planeid = src_adc.meta().id();
    int tar_planeid = tar_adc.meta().id();
    float threshold = 10;
    
    const larcv::ImageMeta& srcmeta = src_adc.meta();
    const larcv::ImageMeta& tarmeta = tar_adc.meta();
    m_srcimg_meta = &srcmeta;
    m_tarimg_meta = &tarmeta;
    
    // allocate arrays for image pixel to contour index lookup
    m_src_img2ctrindex = new int[m_srcimg_meta->cols()*m_srcimg_meta->rows()];
    m_tar_img2ctrindex = new int[m_tarimg_meta->cols()*m_tarimg_meta->rows()];
    memset( m_src_img2ctrindex, 0, sizeof(int)*m_srcimg_meta->cols()*m_srcimg_meta->rows() );
    memset( m_tar_img2ctrindex, 0, sizeof(int)*m_tarimg_meta->cols()*m_tarimg_meta->rows() );    
    
    for ( int r=0; r<(int)srcmeta.rows(); r++) {
      
      // for each row, we find the available contours on the images.
      // saves us search each time

      std::set< int > tar_ctr_ids;
      std::vector<int> src_cols_in_ctr;
      src_cols_in_ctr.reserve(20);
      std::map<int,int> src_cols2ctrid;

      // std::cout << "------------------------------------------" << std::endl;
      // std::cout << "Find row=" << r << " contours" << std::endl;

      // Find contours on source image in this row
      // std::cout << "source: ";      
      for ( int c=0; c<(int)srcmeta.cols(); c++) {
	if ( src_adc.pixel(r,c)<threshold )
	  continue;

	cv::Point pt( c,r );
	int ictr = 0;
	for ( auto const& ctr : contour_data.m_plane_atomics_v[src_planeid] ) {
	  double result =  cv::pointPolygonTest( ctr, pt, false );
	  if ( result>=0 ) {
	    src_cols_in_ctr.push_back( c );
	    src_cols2ctrid[c] = ictr;	    
	    //std::cout << " " << ictr;
	    m_src_img2ctrindex[ r*m_srcimg_meta->cols() + c ] = ictr;
	    break;
	  }
	  ictr++;
	}
      }
      //std::cout << std::endl;

      // Find Contours on the target image in this row
      //std::cout << "target: ";      
      for ( int c=0; c<(int)tarmeta.cols(); c++) {
	if ( tar_adc.pixel(r,c)<threshold )
	  continue;

	cv::Point pt( c,r );	
	int ictr = 0;	
	for ( auto const& ctr : contour_data.m_plane_atomics_v[tar_planeid] ) {
	  double result =  cv::pointPolygonTest( ctr, pt, false );
	  if ( result>=0 ) {
	    tar_ctr_ids.insert( ictr );
	    m_tar_img2ctrindex[ r*m_tarimg_meta->cols() + c ] = ictr;
	    //std::cout << ictr << " ";
	    break;
	  }
	  ictr++;
	}
      }//end of col loop
      //std::cout << std::endl;      

      // Nothing in this row, move on to the next row
      if ( src_cols_in_ctr.size()==0 || tar_ctr_ids.size()==0 ) {
	//std::cout << "nothing to match" << std::endl;
	continue;
      }

      // now loop over source columns in contours and make matches to target contours
      for ( auto const& source_col : src_cols_in_ctr ) {

	float flow = flow_img.pixel(r,source_col);
	int target_col = source_col+flow;
	cv::Point src_pt( source_col, r );
	cv::Point tar_pt( target_col, r );	
	
	// remember the contour we're in
	int src_ctr_id = src_cols2ctrid[source_col];
	const larlitecv::Contour_t& src_ctr = contour_data.m_plane_atomics_v[src_planeid][src_ctr_id];

	// store the target point for this contour
	auto it_srcctr_targets = m_src_targets.find( src_ctr_id );
	if ( it_srcctr_targets==m_src_targets.end() ) {
	  // create a container
	  m_src_targets.insert( std::pair<int,ContourTargets_t>(src_ctr_id,ContourTargets_t()) );
	  it_srcctr_targets = m_src_targets.find( src_ctr_id );
	}
	TargetPix_t tpix;
	tpix.row = r;
	tpix.col = target_col;
	tpix.srccol = source_col;
	it_srcctr_targets->second.push_back( tpix );
	
	// now, find the distance to the contours on the target row
	for ( auto const& ctrid : tar_ctr_ids ) {
	  float dist = cv::pointPolygonTest( contour_data.m_plane_atomics_v[tar_planeid][ctrid], tar_pt, true );
	  if ( dist>-1.0 )
	    dist = -1.0;
	  
	  dist = fabs(dist);

	  // apply some matching threshold
	  if ( dist>30.0 ) {
	    continue;
	  }

	  // // store the match data
	  SrcTarPair_t idpair = { src_ctr_id, ctrid };
	  auto it_flowdata = m_flowdata.find( idpair );
	  if ( it_flowdata==m_flowdata.end() ) {
	  //   // if the map doesn't have the pair we're looking for, we create the data
	    FlowMatchData_t x( src_ctr_id,  ctrid);
	    m_flowdata.insert( std::pair<SrcTarPair_t,FlowMatchData_t>(idpair,x));
	    it_flowdata = m_flowdata.find(idpair);
	  }

	  FlowMatchData_t& flowdata = it_flowdata->second;
	  FlowMatchData_t::FlowPixel_t flowpix;
	  flowpix.src_wire = source_col;
	  flowpix.tar_wire = target_col;
	  flowpix.row = r;
	  flowpix.pred_miss = std::fabs(dist);
	  flowdata.matchingflow_v.push_back( flowpix );
	  
	}
      }
    }
  }

  void FlowContourMatch::_scoreMatches( const larlitecv::ContourCluster& contour_data, int src_planeid, int tar_planeid ) {
    m_src_ncontours = contour_data.m_plane_atomics_v[src_planeid].size();
    m_tar_ncontours = contour_data.m_plane_atomics_v[tar_planeid].size();
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    std::cout << "scr ncontours: " << m_src_ncontours << std::endl;
    std::cout << "tar ncontours: " << m_tar_ncontours << std::endl;

    if ( m_score_matrix!=NULL )
      delete [] m_score_matrix;
    
    m_score_matrix = new double[m_src_ncontours*m_tar_ncontours]; // should probably its own class
    memset(m_score_matrix, 0, sizeof(double)*m_src_ncontours*m_tar_ncontours );
    
    for ( auto it : m_flowdata ) {
      FlowMatchData_t& flowdata = it.second;
      float score = _scoreMatch( flowdata );
      flowdata.score = score;
      m_score_matrix[ flowdata.src_ctr_id*m_tar_ncontours + flowdata.tar_ctr_id ] = score;
    }

    // normalize it
    for (int is=0; is<m_src_ncontours; is++) {
      float norm_s = 0;
      for (int it=0; it<m_tar_ncontours; it++) {
	norm_s += m_score_matrix[ is*m_tar_ncontours + it ];
      }
      if (norm_s>0 ) {
	for (int it=0; it<m_tar_ncontours; it++) {
	  m_score_matrix[ is*m_tar_ncontours + it ] /= norm_s;
	}
      }
    }
    
  }
  
  float FlowContourMatch::_scoreMatch( const FlowMatchData_t& matchdata ) {
    float score = 0.0;
    int nscores = 0;
    for ( auto const& flow : matchdata.matchingflow_v ) {
      score += 1.0/flow.pred_miss;
      nscores++;
    }
    
    return score;
  }

  void FlowContourMatch::_greedyMatch() {
    // goal is to assign a cluster on the
    // source plane purely to one on the target
    
    for (int is=0; is<m_src_ncontours; is++) {
      float max_s = -1.0;
      int   idx   = 0;
      for (int it=0; it<m_tar_ncontours; it++) {
	float score = m_score_matrix[ is*m_tar_ncontours + it ];
	if ( score>max_s ) {
	  max_s = 0;
	  idx = it;
	}
      }
      if (max_s>0 ) {
	for (int it=0; it<m_tar_ncontours; it++) {
	  if ( it!=idx )
	    m_score_matrix[ is*m_tar_ncontours + it ] = 0;
	  else
	    m_score_matrix[ is*m_tar_ncontours + it ] = 1.0;
	}
      }
    }
    
  }

  void FlowContourMatch::dumpMatchData() {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    for ( auto it : m_flowdata ) {
      std::cout << "[CONTOURS: src(" << it.first[0] << ") -> tar(" << it.first[1] << ")]" << std::endl;
      const FlowMatchData_t& flowdata = it.second;
      std::cout << "  Flow entries " << flowdata.matchingflow_v.size() << std::endl;
      for ( auto const& flow : flowdata.matchingflow_v ) {
	std::cout << "    " << flow.row << ": " << flow.src_wire
		  << " -> " << flow.tar_wire << "  err=" << flow.pred_miss << std::endl;
      }
    }
  }

  TH2D& FlowContourMatch::plotScoreMatrix() {
    if ( m_plot_scorematrix!=NULL ) {
      delete m_plot_scorematrix;
    }
    m_plot_scorematrix = new TH2D( "h2d_flowmatch_scorematrix", ";Source Contour;Target Contour",
				  m_src_ncontours, 0, m_src_ncontours,
				  m_tar_ncontours, 0, m_tar_ncontours );
    for (int is=0; is<m_src_ncontours; is++) {
      for (int it=0; it<m_tar_ncontours; it++) {
	m_plot_scorematrix->SetBinContent( is+1, it+1, m_score_matrix[ is*m_tar_ncontours + it ] );
      }
    }

    m_plot_scorematrix->SetMaximum(1.0);
    m_plot_scorematrix->SetMinimum(0.0);	
    
    return *m_plot_scorematrix;
  }

  void FlowContourMatch::_make3Dhits( const larlite::event_hit& hit_v,
				      const larcv::Image2D& srcimg_adc,
				      const larcv::Image2D& tar_adc,
				      const int src_plane,
				      const int tar_plane,
				      const float threshold,
				      std::vector<HitFlowData_t>& hit2flowdata ) {

    // make3Dhits
    // turn flow predictions and contour matches into 3D hits
    //
    // inputs
    // ------
    // hit_v: vector of hits found on the different planes
    //
    // implicit input from class members
    // ----------------------------------
    // m_score_matrix: tells us which target contours most strongly associated to cluster (from scoreMatches)
    // m_flowdata: data of what clusters associated to which (createMatchData)
    // m_src_img2ctrindex: array mapping (row,col) to contour index for source image (createMatchData)
    // m_tar_img2ctrindex: array mapping (row,col) to contour index for target image (createMatchData)
    //
    // outputs
    // -------
    // returned: vector 3D hits
    //
    // description of method
    // ---------------------
    // 1) loop over hits
    // 2) for each hit, determine if it is within a source contour using m_src_img2ctrindex
    // 3) using flow image, determine target pixel
    // 4) using m_score_matrix, make list of target contour indices that connected with source contour
    // 5) determine match quality
    // 6) depending on match quality, determine matching target column
    // 7) convert source and target columns into wires, row into ticks
    // 8) turn that information, make 3D hit position (X-axis relative to trigger time)
    // 9) ???
    // 9) profit
    
    if ( hit2flowdata.size()!=hit_v.size() ) {
      hit2flowdata.clear();
      hit2flowdata.resize( hit_v.size() );
    }
    
    // for (int hitidx=0; hitidx<(int)hit_v.size(); hitidx++) {
    //   // get hit for this index
    //   const larlite::hit& ahit = hit_v[hitidx];
    //   std::cout << "hit[" << hitidx << "]: " << ahit.StartTick() << std::endl;
    // }
    // std::cin.get();
    
    for (int hitidx=0; hitidx<(int)hit_v.size(); hitidx++) {
      // get hit for this index
      const larlite::hit& ahit = hit_v[hitidx];

      // is this on the source plane? if not, skip
      if ( src_plane!=(int)ahit.WireID().planeID().Plane )
	continue;

      // is it in the image (crop)

      // time limits
      int hit_tstart = 2400+(int)ahit.StartTick();
      int hit_tend   = 2400+(int)ahit.EndTick();
      if ( hit_tend < m_srcimg_meta->min_y() || hit_tstart > m_srcimg_meta->max_y() ) {
	// if not within the tick bounds, skip
	continue;
      }
      if ( hit_tend >= m_srcimg_meta->max_y() )
	hit_tend = m_srcimg_meta->max_y()-1;
      if ( hit_tstart <= m_srcimg_meta->min_y() )
	hit_tstart = m_srcimg_meta->min_y();
      
      // wire bounds
      int wire = (int)ahit.WireID().Wire;
      if ( wire < m_srcimg_meta->min_x() || wire >= m_srcimg_meta->max_x() ) {
	// if not within wire bounds, skip
	continue;
      }

      
      // transfor wire and ticks into col and row
      int wirecol  = m_srcimg_meta->col( wire );
      int rowstart = m_srcimg_meta->row( hit_tstart );
      int rowend   = m_srcimg_meta->row( hit_tend );

      std::cout << "---------------------------------------" << std::endl;
      std::cout << "valid hit: wire=" << wire << " wirecol=" << wirecol 
		<< " tickrange=[" << hit_tstart << "," << hit_tend << "]"
		<< " rowrange=["  << rowstart << "," << rowend << "]"
		<< std::endl;

      // ok we are in the image. check the contours.
      // we loop through contours and check if any of their pixels are inside the hit tick range.
      bool foundcontour = false;
      int  sourcecontourindex = -1;
      for ( auto const& it_ctr : m_src_targets ) {
	int src_ctridx = it_ctr.first; // source image contour index
	const ContourTargets_t& ctrtargets = it_ctr.second; // list of src and target pixels

	// loop over pixels within this source contour, find those that fit within hit
	for ( auto const& pixinfo : ctrtargets ) {
	  //std::cout << "  srcctr[" << src_ctridx << "] (" << pixinfo.row << "," << pixinfo.srccol << ")" << std::endl;
	  
	  if ( wirecol!=pixinfo.srccol ) // source contour pixel does not match, skip
	    continue;
	  if ( rowstart <= pixinfo.row && pixinfo.row <= rowend ) {
	    // found the overlap
	    sourcecontourindex = src_ctridx;
	    foundcontour = true;

	    std::cout << "pixel within hit. source contour=" << src_ctridx
		      << " source(r,c)=(" << pixinfo.row << "," << pixinfo.srccol << ")"
		      << " target(r,c)=(" << pixinfo.row << "," << pixinfo.col << ")"
		      << std::endl;
	    
	    // update the src/target pix from larflow based on
	    // 1) adc value of source pixel
	    // 2) match quality
	    float src_adc    = srcimg_adc.pixel( pixinfo.row, pixinfo.srccol );

	    // match quality
	    // 1-best) target pixel inside (primary?) contour and on charge
	    // 2) target pixel inside contour -- find closest charge inside contour
	    // 3) target pixel outside contour -- find closest charge inside best-score contour
	    
	    // get the data for this hit
	    HitFlowData_t& hitdata = hit2flowdata[hitidx];

	    // so we first calculate the hit quality, using the flow, src image, target image,
	    //  and src-target contour matches (via score matrix)
	    int matchquality = -1; // <<
	    int pastquality  = hitdata.matchquality;
	    int dist2center  = abs( pixinfo.srccol - m_srcimg_meta->cols()/2 );
	    int dist2charge  = -1;
	    std::cout << "  -- past hitquality=" << pastquality << std::endl;
	    
	    // does target point to charge? look within a range of wires
	    int tarcolmin = pixinfo.col-2;
	    int tarcolmax = pixinfo.col+2;
	    if ( tarcolmin<0 )
	      tarcolmin = 0;
	    if ( tarcolmax>=(int)m_tarimg_meta->cols() )
	      tarcolmax = (int)m_tarimg_meta->cols() - 1;

	    // we look for the peak adc value between tarcolmin and tarcolmax
	    float target_adc = 0; // <<
	    int   target_col = 0; // <<
	    bool oncharge = false;
	    for (int tarcol=tarcolmin; tarcol<=tarcolmax; tarcol++) {
	      float tadc = tar_adc.pixel( pixinfo.row, tarcol );
	      if ( target_adc<tadc ) {
		target_adc = tadc;
		target_col = tarcol;
	      }
	    }
	    if ( target_adc>threshold ) {
	      oncharge = true;
	    }

	    // is this pixel in a (matched) contour
	    bool incontour = false;
	    int target_contour = m_tar_img2ctrindex[ int(pixinfo.row*m_tarimg_meta->cols() + target_col) ];
	    if ( target_contour>0 ) {
	      incontour = true;
	    }
	    
	    // quality level 1: oncharge and incontour
	    if ( incontour && oncharge ) {
	      matchquality = 1;
	      dist2charge = 0;
	      std::cout << "  -- hit quality=1 " << " past(" << pastquality << ") " << std::endl;
	    }
	    else if ( incontour && !oncharge && (pastquality<0 || pastquality>=2) ) {
	      // we calculate the matched pixel for this case if in the past we didnt get a better match
	      matchquality = 2;
	      std::cout << "  -- hit quality=2 " << " past(" << pastquality << ") " << std::endl;
		      
	      // we look for the closest pixel inside the contour that has charge, and that is what we match to
	      int possearch_col = target_col+1;
	      while ( possearch_col<(int)m_tarimg_meta->cols() && possearch_col-target_col<30 ) {
		if ( m_tar_img2ctrindex[ int(pixinfo.row*m_tarimg_meta->cols() + possearch_col) ]==target_contour ) {
		  // column in contour
		  float tadc = tar_adc.pixel( pixinfo.row, possearch_col );
		  if ( tadc>threshold ) {
		    break;
		  }
		}
		possearch_col++;
	      }
	      
	      int negsearch_col = target_col-1;
	      while ( negsearch_col>=0 && target_col-negsearch_col<30 ) {	      
		if ( m_tar_img2ctrindex[ int(pixinfo.row*m_tarimg_meta->cols() + negsearch_col) ]==target_contour ) {
		  // column in contour
		  float tadc = tar_adc.pixel( pixinfo.row, negsearch_col );
		  if ( tadc>threshold ) {
		    break;
		  }
		}
		negsearch_col--;
	      }

	      int negdist = abs(negsearch_col-target_col);
	      int posdist = abs(possearch_col-target_col);
	      
	      if (  negdist < posdist ) {
		target_col = negsearch_col;
		dist2charge = negdist;
	      }
	      else {
		target_col  = possearch_col;
		dist2charge = posdist;
	      }
	      target_adc = tar_adc.pixel( pixinfo.row, target_col );
	      
	    }
	    else if ( !incontour && (pastquality<0 || pastquality>=3) ) {
	      // we calculate the matched pixel for this case if in the past we didnt get a better match
	      matchquality = 3;
	      std::cout << "  -- hit quality=3 " << " past(" << pastquality << ") " << std::endl;
	      
	      // check the best matching contour first for charge to match
	      // we find closest charge on row for all match contours.
	      // take the pixel using the best match
	      std::vector<ClosestContourPix_t> matched_contour_list;
	      std::set<int> used_contours;
	      
	      int possearch_col = target_col+1;
	      while ( possearch_col<(int)m_tarimg_meta->cols() && possearch_col-target_col<50 ) {
		float tadc = tar_adc.pixel( pixinfo.row, possearch_col );		
		if ( tadc > threshold )  {
		  int target_contour_idx = m_tar_img2ctrindex[ int(pixinfo.row*m_tarimg_meta->cols() + possearch_col) ];
		  if ( used_contours.find( target_contour_idx )==used_contours.end() ) {
		    // have not search this contour, provide a match candidate
		    ClosestContourPix_t close_ctr_info;
		    close_ctr_info.ctridx = target_contour_idx;
		    close_ctr_info.dist = abs(possearch_col - target_col);
		    close_ctr_info.col	= possearch_col;
		    close_ctr_info.adc  = tadc;		    
		    close_ctr_info.scorematch = m_score_matrix[ int(src_ctridx*m_tar_ncontours + target_contour_idx) ];
		    matched_contour_list.push_back( close_ctr_info );
		    used_contours.insert( target_contour_idx );
		  }
		}
		possearch_col++;
	      }//end of pos loop
	      int negsearch_col = target_col+1;
	      while ( negsearch_col>=0 && target_col-negsearch_col<50 ) {
		float tadc = tar_adc.pixel( pixinfo.row, negsearch_col );
		if (  tadc > threshold )  {
		  int target_contour_idx = m_tar_img2ctrindex[ int(pixinfo.row*m_tarimg_meta->cols() + negsearch_col) ];
		  if ( used_contours.find( target_contour_idx )==used_contours.end() ) {
		    // have not search this contour, provide a match candidate
		    ClosestContourPix_t close_ctr_info;
		    close_ctr_info.ctridx = target_contour_idx;
		    close_ctr_info.dist = abs(negsearch_col - target_col);
		    close_ctr_info.col	= negsearch_col;
		    close_ctr_info.adc  = tadc;		    
		    close_ctr_info.scorematch = m_score_matrix[ src_ctridx*m_tar_ncontours + target_contour_idx ];
		    matched_contour_list.push_back( close_ctr_info );
		    used_contours.insert( target_contour_idx );		    
		  }
		}
		negsearch_col--;
	      }//end of neg loop
	      
	      // ok, now we pick the best one!
	      float best_score = 0.0;
	      float best_adc = 0.0;
	      int best_col;
	      int best_dist;
	      for ( auto& match : matched_contour_list ) {
		if ( best_score < match.scorematch ) {
		  best_col   = match.col;
		  best_score = match.scorematch;
		  best_dist  = match.dist;
		  best_adc   = match.adc;
		}
	      }

	      target_col  = best_col;
	      dist2charge = best_dist;
	      target_adc  = best_adc;
	    }

	    // did we do better?
	    bool update_hitdata = false;
	    if ( matchquality>0 ) {
	      //we found a case we did better or the same

	      if ( matchquality<pastquality || pastquality<0 )  {
		// we did better. replace the hit
		update_hitdata = true;
	      }
	      else {
		if ( matchquality==1 && hitdata.dist2center>dist2center) {
		  // we decide on this by using the src pixel closest to the center of the y-image
		  update_hitdata = true;
		}
		else if ( matchquality==2 && hitdata.dist2charge>dist2charge ) {
		  // we decide on this by using which flow prediction was closest to the eventual charge match
		  update_hitdata = true;
		}
		else if ( matchquality==3 && hitdata.dist2charge>dist2charge ) {
		  // same criterion as 2
		  update_hitdata = true;
		}
	      }
	      
	    }
	    
	    if ( update_hitdata ) {
	      std::cout << "  -- update hit flow data" << std::endl;
	      hitdata.maxamp    = target_adc;
	      hitdata.hitidx    = hitidx;
	      hitdata.srccol    = pixinfo.srccol;
	      hitdata.targetcol = target_col;
	      hitdata.pixrow    = pixinfo.row;
	      hitdata.matchquality = matchquality;
	      hitdata.dist2center = dist2center;
	      hitdata.dist2charge = dist2charge;
	      hitdata.src_ctr_idx = sourcecontourindex;
	      hitdata.tar_ctr_idx = target_contour;
	    }
	    
	  }//if row within hit row range
	}//end of ctrtargets pixel loop

      }//end of loop over list of src-target pairs
      
    }//end of hit index loop


    return;
  }

  std::vector<FlowMatchHit3D> FlowContourMatch::get3Dhits() {
    return get3Dhits( m_hit2flowdata );
  }
  
  std::vector<FlowMatchHit3D> FlowContourMatch::get3Dhits( const std::vector<HitFlowData_t>& hit2flowdata ) {

    // now we have, in principle, the best/modified flow prediction for hits that land on flow predictions
    // we can make 3D hits!
    std::vector<FlowMatchHit3D> output_hit3d_v;
    for (int hitidx=0; hitidx<(int)hit2flowdata.size(); hitidx++) {
      const HitFlowData_t& hitdata = hit2flowdata[ hitidx ];
      if (  hitdata.matchquality<=0 ) {
    	// no good match
    	continue;
      }
      // otherwise make a hit
      FlowMatchHit3D flowhit;
      flowhit.row         = hitdata.pixrow;
      flowhit.srcpixel    = hitdata.srccol;
      flowhit.targetpixel = hitdata.targetcol;
      flowhit.src_ctrid   = hitdata.src_ctr_idx;
      flowhit.tar_ctrid   = hitdata.tar_ctr_idx;
      output_hit3d_v.emplace_back( flowhit );
    }

    return output_hit3d_v;    
  }
  
}
