
namespace larflow {
  
  ContourFlowMatchDict_t createMatchData( const ublarcvapp::ContourClusterAlgo& contour_data,
                                          const larcv::Image2D& flow_img,
                                          const larcv::Image2D& src_adc,
                                          const larcv::Image2D& tar_adc ) {
    
    ContourFlowMatchDict_t matchdict;
    /*

    // we compile the relationships between pixels and the different contour-clusters
    // the goal is to start to see what contours on source and target imager are paired together
    //   through flow predictions
    // things we fill
    // --------------
    // m_src_img2ctrindex: 2D array. position of array corresponds to position of src_adc img.
    //                     value is the index to a contour.
    // m_tar_img2ctrindex: same as above, but for target image
    // m_srcimg_meta: pointer to source image meta
    // m_tarimg_meta: pointer to target image meta
    // m_src_targets: map between contour index and a ContourTargets_t object.
    //                contourtargets object is a container storing TargetPix_t.
    //                TargetPix_t stores info about source -> target pixel pair from the flow predictions
    // m_flowdata: map between src-target contour pair to FlowMatchData_t which contains score
    

    int src_planeid = src_adc.meta().plane();
    int tar_planeid = tar_adc.meta().plane();
    float threshold = 10;
    
    const larcv::ImageMeta& srcmeta = src_adc.meta();
    const larcv::ImageMeta& tarmeta = tar_adc.meta();
    m_srcimg_meta = &srcmeta;
    m_tarimg_meta[kflowdir] = &tarmeta;
    
    // allocate arrays for image pixel to contour index lookup
    m_src_img2ctrindex                       = new int[m_srcimg_meta->cols()*m_srcimg_meta->rows()];
    m_tar_img2ctrindex[kflowdir]             = new int[m_tarimg_meta[kflowdir]->cols()*m_tarimg_meta[kflowdir]->rows()];
    memset( m_src_img2ctrindex, 0,           sizeof(int)*m_srcimg_meta->cols()*m_srcimg_meta->rows() );
    memset( m_tar_img2ctrindex[kflowdir], 0, sizeof(int)*m_tarimg_meta[kflowdir]->cols()*m_tarimg_meta[kflowdir]->rows() );    
    
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
	    m_tar_img2ctrindex[kflowdir][ r*m_tarimg_meta[kflowdir]->cols() + c ] = ictr;
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
	const ublarcvapp::Contour_t& src_ctr = contour_data.m_plane_atomics_v[src_planeid][src_ctr_id];

	// store the target point for this contour
	auto it_srcctr_targets = m_src_targets[kflowdir].find( src_ctr_id );
	if ( it_srcctr_targets==m_src_targets[kflowdir].end() ) {
	  // create a container
	  m_src_targets[kflowdir].insert( std::pair<int,ContourTargets_t>(src_ctr_id,ContourTargets_t()) );
	  it_srcctr_targets = m_src_targets[kflowdir].find( src_ctr_id );
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

	  // apply some matching threshold [WARNING HIDDEN PARAMETER]
	  if ( dist>30.0 ) {
	    continue;
	  }

	  // // store the match data
	  SrcTarPair_t idpair = { src_ctr_id, ctrid };
	  auto it_indexmap = m_flowdata_indexmap[kflowdir].find( idpair );
	  if ( it_indexmap==m_flowdata_indexmap[kflowdir].end() ) {
	  //   // if the map doesn't have the pair we're looking for, we create the data
	    FlowMatchData_t x( src_ctr_id,  ctrid);
            m_flowdata[kflowdir].emplace_back( std::move(x) );
            m_flowdata_indexmap[kflowdir].insert( std::pair<SrcTarPair_t,int>(idpair,m_flowdata[kflowdir].size()-1));
	    //m_flowdata[kflowdir].insert( std::pair<SrcTarPair_t,FlowMatchData_t>(idpair,x));
	    it_indexmap = m_flowdata_indexmap[kflowdir].find(idpair);
	  }
          
	  FlowMatchData_t& flowdata = m_flowdata[kflowdir].at(it_indexmap->second);
	  FlowMatchData_t::FlowPixel_t flowpix;
	  flowpix.src_wire = source_col;
	  flowpix.tar_wire = target_col;
	  flowpix.row = r;
	  flowpix.pred_miss = std::fabs(dist);
	  flowdata.matchingflow_v.push_back( flowpix );
	  
	}
      }
    }
    */

    return matchdict;
  }

}
