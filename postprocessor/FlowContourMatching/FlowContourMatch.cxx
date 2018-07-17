#include "FlowContourMatch.h"

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

  // FlowMatchData_t& FlowMatchData_t::operator=(const FlowMatchData_t &s) {
  //   // copy assignment operator
  //   FlowMatchData_t t(s.src_ctr_id,s.tar_ctr_id);
  //   t.score = s.score;
  //   t.matchingflow_v = s.matchingflow_v;
  //   return t;
  // }
  FlowContourMatch::FlowContourMatch() {
    m_score_matrix = NULL;
    m_plot_scorematrix = NULL;
  }

  FlowContourMatch::~FlowContourMatch() {
    delete [] m_score_matrix;
    delete m_plot_scorematrix;
  }
  
  void FlowContourMatch::createMatchData( const larlitecv::ContourCluster& contour_data, const larcv::Image2D& flow_img, const larcv::Image2D& src_adc, const larcv::Image2D& tar_adc ) {

    int src_planeid = src_adc.meta().id();
    int tar_planeid = tar_adc.meta().id();
    float threshold = 10;
    
    const larcv::ImageMeta& srcmeta = src_adc.meta();
    const larcv::ImageMeta& tarmeta = tar_adc.meta();
    
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
	    //std::cout << ictr << " ";
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

  void FlowContourMatch::scoreMatches( const larlitecv::ContourCluster& contour_data, int src_planeid, int tar_planeid ) {
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
      float score = scoreMatch( flowdata );
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
  
  float FlowContourMatch::scoreMatch( const FlowMatchData_t& matchdata ) {
    float score = 0.0;
    int nscores = 0;
    for ( auto const& flow : matchdata.matchingflow_v ) {
      score += 1.0/flow.pred_miss;
      nscores++;
    }
    
    return score;
  }

  void FlowContourMatch::greedyMatch() {
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

  
}
