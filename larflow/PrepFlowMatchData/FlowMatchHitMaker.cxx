#include "FlowMatchHitMaker.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "larcv/core/PyUtil/PyUtils.h"
#include "larcv/core/Base/larcv_logger.h"
#include "larcv/core/DataFormat/EventSparseImage.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/Geometry.h"
#include "larlite/LArUtil/DetectorProperties.h"
#include "PrepSSNetTriplet.h"
#include <stdexcept>

namespace larflow {
namespace prep {

  /**
   * @brief compile network output into hit candidate information
   *
   * @param[in] pair_probs Pointer to (1,Np) numpy array containing larmatch scores for Np space points
   * @param[in] source_sparseimg Pointer to (Ns,2) numpy array containing indicies (row,col) of Ns non-zero pixels
   * @param[in] target_sparseimg Pointer to (Nt,2) numpy array containing indicies (row,col) of Nt non-zero pixels
   * @param[in] matchpairs Point to (Np,2) numpy array containing index to (source_sparseimg,target_sparseimg) arrays for Np space points
   * @param[in] source_plane Index of source plane
   * @param[in] target_plane Index of target plane
   * @param[in] source_meta  ImageMeta describing parameters for soruce image
   * @param[in] img_v vector of Image2D instances containing ionization information (i.e. the wireplane images)
   * @param[in] ev_chstatus Class containing status of channels, i.e. if good or dead
   *
   */
  int FlowMatchHitMaker::add_match_data( PyObject* pair_probs,
                                         PyObject* source_sparseimg, PyObject* target_sparseimg,
                                         PyObject* matchpairs,
                                         const int source_plane, const int target_plane,
                                         const larcv::ImageMeta& source_meta,
                                         const std::vector<larcv::Image2D>& img_v,
                                         const larcv::EventChStatus& ev_chstatus )
  {

    // enable numpy environment (if not already set)
    std::cout << "setting pyutils ... ";
    import_array1(0);    
    larcv::SetPyUtil();
    std::cout << " done" << std::endl;

    // cast numpy data to C-arrays
    const int dtype = NPY_FLOAT;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);

    // match scores
    npy_intp pair_dims[2];
    float **probs_carray;
    std::cout << "get pair prob dims" << std::endl;
    if ( PyArray_AsCArray( &pair_probs, (void**)&probs_carray, pair_dims, 2, descr )<0 ) {
      larcv::logger::get("PyFlowMatchData::make_larflow_hits").send(larcv::msg::kCRITICAL,__FUNCTION__,__LINE__, "cannot get carray for pair prob matrix");
    }

    // sparse source image
    npy_intp source_dims[2];
    float **source_carray;
    std::cout << "get source dims" << std::endl;    
    if ( PyArray_AsCArray( &source_sparseimg, (void**)&source_carray, source_dims, 2, descr )<0 ) {
      larcv::logger::get("PyFlowMatchData::make_larflow_hits").send(larcv::msg::kCRITICAL,__FUNCTION__,__LINE__, "cannot get carray for source sparse-image matrix");
    }

    // sparse target image
    npy_intp target_dims[2];
    float **target_carray;
    std::cout << "get target dims" << std::endl;        
    if ( PyArray_AsCArray( &target_sparseimg, (void**)&target_carray, target_dims, 2, descr )<0 ) {
      larcv::logger::get("PyFlowMatchData::make_larflow_hits").send(larcv::msg::kCRITICAL,__FUNCTION__,__LINE__, "cannot get carray for target sparse-image matrix");
    }

    // (source,target) indicies for each match
    npy_intp match_dims[2];
    long **matchpairs_carray;
    std::cout << "get index dims" << std::endl;            
    if ( PyArray_AsCArray( &matchpairs, (void**)&matchpairs_carray, match_dims, 2, PyArray_DescrFromType(NPY_LONG) )<0 ) {
      larcv::logger::get("PyFlowMatchData::make_larflow_hits").send(larcv::msg::kCRITICAL,__FUNCTION__,__LINE__, "cannot get carray for match pair matrix");
    }

    std::cout << "match matrix:  (" << match_dims[0]  << "," << match_dims[1]  << ")" << std::endl;    
    std::cout << "prob matrix:   (" << pair_dims[0]   << "," << pair_dims[1]   << ")" << std::endl;    
    std::cout << "source matrix: (" << source_dims[0] << "," << source_dims[1] << ")" << std::endl;
    std::cout << "target matrix: (" << target_dims[0] << "," << target_dims[1] << ")" << std::endl;

    // uboone-specific geo stuff

    int other_plane = -1;
    if ( source_plane==2 ) {
      other_plane = ( target_plane==0 ) ? 1 : 0;
    }
    else if ( source_plane==1 ) {
      other_plane = ( target_plane==0 ) ? 2 : 0;
    }
    else if ( source_plane==0 ) {
      other_plane = ( target_plane==1 ) ? 2 : 1;
    }

    auto const& other_plane_status = ev_chstatus.Status( other_plane );

    // choose threshold
    //float match_prob_threshold = 0.5;
    // if (other_plane==2) 
    //   match_prob_threshold = 0.1;

    bool require_3plane = false;
      
    // loop over each candidate match
    int n_indead = 0;
    int n_2plane = 0;
    int n_3plane = 0;
    int nrepeated = 0;
    for (int ipair=0; ipair<(int)pair_dims[1]; ipair++) {

      // score for this match
      float prob = probs_carray[0][ipair];

      // match threshold
      //if ( prob<_match_score_threshold ) continue;

      // get source and target index (in sparse image)
      int srcidx = matchpairs_carray[ipair][0];
      int taridx = matchpairs_carray[ipair][1];

      // get source and target column, row
      int srccol = (int)source_carray[srcidx][1];
      int srcrow = (int)source_carray[srcidx][0];      
      int tarcol = (int)target_carray[taridx][1];

      // get the other plane wire
      // convert row to tick
      double y(0), z(0);
      //larutil::Geometry::GetME()->IntersectionPoint( srccol, tarcol, (UChar_t)source_plane, (UChar_t)target_plane, y, z );
      float tick = source_meta.pos_y( srcrow );

      if ( y<-117 || y>117 || z<0 || z>1036 ) continue;

      Double_t pos[3] = { 0, y, z };
      float other_wire = larlite::larutil::Geometry::GetME()->WireCoordinate( pos, other_plane );
      float other_adc  = 0;
      try {
        img_v[other_plane].pixel( srcrow, (int)other_wire, __FILE__, __LINE__ );
      }
      catch (std::exception& e ) {
        std::cout << "Error getting other plane pixel, (r,c)=(" << srcrow << "," << other_wire << "). "
                  << "for pos(y,z)=(" << y << ", " << z << "). "
                  << "from (src,tar) intersection=(" << srccol << "," << tarcol << ")"
                  << std::endl;
        std::cout << "error: " << e.what() << std::endl;
        continue;
      }

      std::vector<int> img_coords(3,0);
      img_coords[ source_plane ] = srccol;
      img_coords[ target_plane ] = tarcol;
      img_coords[ other_plane  ] = (int)other_wire;

      // std::cout << "pair[" << ipair << "] (t,y,z)=(" << tick << "," << y << "," << z << ") "
      //           << "imgcoords=[" << img_coords[0] << "," << img_coords[1] << "," << img_coords[2] << "]"
      //           << std::endl;
      //std::cout << "    otherplane=" << other_plane << std::endl;

      // determine if this is a valid point
      // (1) must have charge inall three planes OR
      // (2) charge in two and dead channel in one      
      bool indead = false;
      if ( other_adc<10.0 ) {

        // if event chstatus pointer is not null, we check the ch status
        int chstatus = other_plane_status.Status( other_wire );
        if ( chstatus==4 ) {
          if ( require_3plane ) {
            // good + below threshold so we ignore this match
            continue;
          }
          else {
            // accept anyway
            n_2plane++;
          }
        }
        else {
          // in dead, so accept
          indead = true;
          n_indead++;
        }
      }
      else {
        // all three planes have charge
        n_3plane++;
      }

      // now look for the wire-triple in the match data list
      std::vector<int> triple(6,0);
      triple[ source_plane ] = srccol;
      triple[ target_plane ] = tarcol;
      triple[ other_plane  ] = int(other_wire);
      triple[ 3 ] = (int)tick;
      
      //std::cout << "    triple=" << triple[0] << "," << triple[1] << "," << triple[2] << std::endl;

      auto it = _match_map.find( triple );
      if ( it==_match_map.end() ) {
        // if not in match map, we create a new entry
        match_t m;
        m.set_wire( source_plane, srccol );
        m.set_wire( target_plane, tarcol );
        m.set_wire( other_plane,  int(other_wire) );
        m.tyz = { (float)tick, (float)y, (float)z };
        _matches_v.emplace_back( std::move(m) );
        _match_map[ triple ] = int(_matches_v.size())-1;
        it = _match_map.find( triple );
      }
      else {
        nrepeated++;
      }

      auto& m = _matches_v.at( it->second );
      // set the score given here
      //if ( indead ) prob *= 0.5;
      m.set_score( prob );

    }//end of score loop


    PyArray_Free( pair_probs, (void*)probs_carray );
    PyArray_Free( source_sparseimg, (void*)source_carray );
    PyArray_Free( target_sparseimg, (void*)target_carray );
    PyArray_Free( matchpairs, (void*)matchpairs_carray );    
    
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "process match data. " << std::endl;
    std::cout << "  number of triples with match information: " << _match_map.size() << std::endl;
    std::cout << "  number of: 3-plane=" << n_3plane << " 2-plane=" << n_2plane << " 2-plane+dead=" << n_indead << std::endl;
    std::cout << "  num repeated: " << nrepeated << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    return 0;
  }

  /**
   *
   * \brief uses the match data in _matches_v to make hits
   *
   * makes larlite::larflow3dhit objects based on stored match data
   * the object is a wrapper around a vector<float>.
   *
   * spacepoint proposals below _match_score_threshold will be skipped.
   *
   * larflow3dhit inherits from vector<float>. The values in the vector are as follows:
   * [0-4]:   x,y,z,tpcid,cryoid
   * [5-9]:   5 flow direction scores (deprecated) + 1 max score (deprecated based on 2-flow paradigm. for triplet, [9] is the only score stored 
   * [10-16]: 7 ssnet scores, (bg,track,shower), from larmatch (not 2D sparse ssnet)
   * [17-22]: 6 keypoint label score [nu,track-start,track-end,nu-shower,delta,michel]
   * [23-25]: reserved for plane charge
   * [26-28]: 3D flow direction
   * 
   * @param[in] ev_chstatus Class containing channel status, indicating if good or dead wire
   * @param[out] hit_v vector of space points whose larmatch score was above threshold
   *
   */
  void FlowMatchHitMaker::make_hits( const larcv::EventChStatus& ev_chstatus,
                                     const std::vector<larcv::Image2D>& img_v,
                                     std::vector<larlite::larflow3dhit>& hit_v ) const {

    auto const detp = larutil::DetectorProperties::GetME();
    int ncolumns = 3;
    ncolumns += 7; // flow scores
    ncolumns += 7; // ssnet classes
    ncolumns += 6; // keypoint scores
    ncolumns += 3; // plane charge
    ncolumns += 3; // flow directions    
    // = 29 columns

    //auto const& meta = img_v.front().meta();
    
    int idx = 0;
    unsigned long maxsize = hit_v.size() + _matches_v.size()+10;
    hit_v.reserve(maxsize);
    for ( auto const& m : _matches_v ) {

      // find the highest match score
      const std::vector<float>& scores = m.get_scores();
      float maxscore = 0;
      int maxdir = -1;
      for ( int i=0; i<scores.size(); i++ ) {
        float s = scores[i];
        if ( s>maxscore ) {
          maxscore = s;
          maxdir = i;
        }
      }
      if ( maxscore<_match_score_threshold )
        continue;

      larlite::larflow3dhit hit;
      hit.tick = m.tyz[0];

      auto const& meta = img_v.at( m.img_indices_v.front() ).meta();

      if ( hit.tick<=meta.min_y() || hit.tick>=meta.max_y() )
        continue;
      
      hit.srcwire = m.Y;
      hit.targetwire.resize(6);
      hit.targetwire[0] = m.U;
      hit.targetwire[1] = m.V;
      hit.targetwire[2] = m.Y;
      hit.targetwire[3] = meta.row(hit.tick);
      hit.targetwire[4] = m.tpcid;
      hit.targetwire[5] = m.cryoid;
      hit.idxhit = idx;
      if(m.istruth==0) hit.truthflag = larlite::larflow3dhit::TruthFlag_t::kNoTruthMatch;
      else{ hit.truthflag = larlite::larflow3dhit::TruthFlag_t::kOnTrack;}
      //float x = (m.tyz[0]-3200.0)*cm_per_tick;
      float x = detp->ConvertTicksToX(hit.tick,0,m.tpcid,m.cryoid);
      hit.resize(ncolumns,0);
      hit[0] = x;
      hit[1] = m.tyz[1];
      hit[2] = m.tyz[2];
      hit[3] = m.tpcid;
      hit[4] = m.cryoid;
      
      // store larmatch score
      hit.flowdir = (larlite::larflow3dhit::FlowDirection_t)maxdir;
      hit[9] = maxscore;
      hit.track_score = maxscore;

      try {
	if ( has_ssnet_scores && m.ssnet_scores.size()==larflow::prep::PrepSSNetTriplet::kNumClasses) {
	  for (int c=0; c<larflow::prep::PrepSSNetTriplet::kNumClasses; c++)
	    hit[10 + c] = m.ssnet_scores[c];
	}
      }
      catch ( const std::exception& e ) {
	throw std::runtime_error("ERROR STOREING SSNET SCORES");
      }

      try {
	if ( has_kplabel_scores ) {
	  for (int c=0; c<6; c++)
	    hit[17+c] = m.keypoint_scores[c];
	  
	  if ( hit_v.size()<10 ) {
	    std::cout << " make hit[" << hit_v.size() << "]: kplabels=(";
	    for (int c=0; c<6; c++) {
	      std::cout << hit[17+c];
	      if ( c+1<6 )
		std::cout << ", ";
	      else
		std::cout << ")" << std::endl;
	    }
	  }
	}
      }
      catch ( const std::exception& e ) {
	throw std::runtime_error("ERROR STORING KEYPOINT LABEL SCORES");
      }

      try {
	if ( has_paf ) {
	  for (int i=0; i<3; i++)
	    hit[26+i] = m.paf[i];
	}
      }
      catch ( const std::exception& e ) {
	throw std::runtime_error("ERROR STORING PAF SCORES");
      }
      
      hit_v.emplace_back( std::move(hit) );
    }
    LARCV_NORMAL() << " saved " << hit_v.size() << " hits "
		   << " from "  << _matches_v.size() << " matches" << std::endl;
    
  };

  /**
   *
   * \brief uses the match data in _matches_v to make hits
   *
   * makes larlite::larflow3dhit objects based on stored match data
   * the object is a wrapper around a vector<float>.
   *
   * spacepoint proposals below _match_score_threshold will be skipped.
   *
   * FILLS HIT DATA SCHEMA FOR THE OLD v0 version of larmatch 
   *
   * larflow3dhit inherits from vector<float>. The values in the vector are as follows:
   * [0-2]:   x,y,z
   * [3-9]:   6 flow direction scores + 1 max score (deprecated based on 2-flow paradigm. for triplet, [8] is the only score stored 
   * [10-12]: 3 ssnet scores, (bg,track,shower), from larmatch (not 2D sparse ssnet)
   * [13-15]: 3 keypoint label score [nu,track,shower]
   * [16-18]: 3D flow direction
   * 
   * @param[in] ev_chstatus Class containing channel status, indicating if good or dead wire
   * @param[out] hit_v vector of space points whose larmatch score was above threshold
   *
   */
  void FlowMatchHitMaker::make_hits_v0_scn_network( const larcv::EventChStatus& ev_chstatus,
						    const std::vector<larcv::Image2D>& img_v,
						    std::vector<larlite::larflow3dhit>& hit_v ) const {

    const float cm_per_tick = larutil::LArProperties::GetME()->DriftVelocity()*0.5;
    const int ncolumns = 19;

    auto const& meta = img_v.front().meta();
    
    int idx = 0;
    unsigned long maxsize = hit_v.size() + _matches_v.size()+10;
    hit_v.reserve(maxsize);
    for ( auto const& m : _matches_v ) {

      // find the highest match score
      std::vector<float> scores = m.get_scores();
      float maxscore = 0;
      int maxdir = -1;
      for ( int i=0; i<scores.size(); i++ ) {
        float s = scores[i];
        if ( s>maxscore ) {
          maxscore = s;
          maxdir = i;
        }
      }
      if ( maxscore<_match_score_threshold )
        continue;

      larlite::larflow3dhit hit;
      hit.tick = m.tyz[0];
      hit.srcwire = m.Y;
      hit.targetwire.resize(3);
      hit.targetwire[0] = m.U;
      hit.targetwire[1] = m.V;
      hit.targetwire[2] = m.Y;
      hit.idxhit = idx;
      if(m.istruth==0) hit.truthflag = larlite::larflow3dhit::TruthFlag_t::kNoTruthMatch;
      else{ hit.truthflag = larlite::larflow3dhit::TruthFlag_t::kOnTrack;}
      float x = (m.tyz[0]-3200.0)*cm_per_tick;
      hit.resize(ncolumns,0);
      hit[0] = x;
      hit[1] = m.tyz[1];
      hit[2] = m.tyz[2];

      if ( hit.tick<=meta.min_y() || hit.tick>=meta.max_y() )
        continue;
      
      
      // store scores
      for ( int i=0; i<scores.size(); i++ ) {
        float s = scores[i];
        hit[3+i] = s;
      }
      hit.flowdir = (larlite::larflow3dhit::FlowDirection_t)maxdir;
      hit[9] = maxscore;
      hit.track_score = maxscore;

      if ( has_ssnet_scores && m.ssnet_scores.size()==3) {
        for (int c=0; c<3; c++)
          hit[10 + c] = m.ssnet_scores[c];
      }
      
      if ( has_kplabel_scores ) {
	hit[13] = m.keypoint_scores[0]; // neutrino
	hit[14] = m.keypoint_scores[1]; // track
	hit[15] = m.keypoint_scores[3]; // showers
      }

      if ( has_paf ) {
        for (int i=0; i<3; i++)
          hit[16+i] = m.paf[i];
      }
      
      hit_v.emplace_back( std::move(hit) );
    }
    std::cout << "[FlowMatchHitMaker::make_hits] saved " << hit_v.size() << " hits "
              << " from "  << _matches_v.size() << " matches" << std::endl;
    
  };
  

  /**
   * @brief label container of larflow3dhit using 2D track/shower ssnet output images
   *
   * calculates weighted ssnet score and modifies hit to carry value.
   * the weighted ssnet score for the space point is in `larlite::larflow3dhit::renormed_shower_score`
   *
   * @param[in] ssnet_score_v            SSNet shower score images for each plane
   * @param[inout] larmatch_hit_v        LArMatch hits, modified
   */
  void FlowMatchHitMaker::store_2dssnet_score( larcv::IOManager& iolcv,
					       std::vector<larlite::larflow3dhit>& larmatch_hit_v )
  {
    
    clock_t begin = clock();

    larcv::EventSparseImage* ev_ssnet
      = (larcv::EventSparseImage*)iolcv.get_data( larcv::kProductSparseImage,"sparseuresnetout");
    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_adc->as_vector();
    
    auto const& sparseimg_v = ev_ssnet->SparseImageArray();
    std::cout << "number of sparse images: " << sparseimg_v.size() << std::endl;

    if ( sparseimg_v.size()==0 )
      return;

    // convert into 5-particle ssnet

    struct SSNetData_t {
      int row;
      int col;

      float bg;
      float hip;
      float mip;
      float shr;
      float dlt;
      float mic;
      bool operator<(const SSNetData_t& rhs) const {
	if (row<rhs.row) return true;
	if ( row==rhs.row ) {
	  if ( col<rhs.col ) return true;
	}
	return false;
      };
    };
    
    std::map< std::pair<int,int>, SSNetData_t> data[3];
      
    for ( size_t p=0; p<3; p++ ) {

      auto const& meta = adc_v[p].meta();
      
      if (sparseimg_v.size()>0) {
        auto& spimg = sparseimg_v.at(p);
      
        int nfeatures = spimg.nfeatures();
        int stride = nfeatures+2;
        int npts = spimg.pixellist().size()/stride;
        auto const& spmeta = spimg.meta(0);
        
        for (int ipt=0; ipt<npts; ipt++) {
          int row = spimg.pixellist().at( ipt*stride+0 );
          int col = spimg.pixellist().at( ipt*stride+1 );
          
          int xrow = meta.row( spmeta.pos_y( row ) );
          int xcol = meta.col( spmeta.pos_x( col ) );
          
          // int maxpid = -1;
          // float maxscore = -1;
          // for (int i=0; i<5; i++) {
          //   float score = spimg.pixellist().at( ipt*stride+2+i );
          //   if ( score>maxscore ) {
          //     maxscore = score;
          //     maxpid   = i;
          //   }
          // }
	  
          // float hip = spimg.pixellist().at( ipt*stride+2 );
          // float mip = spimg.pixellist().at( ipt*stride+3 );
          // float shr = spimg.pixellist().at( ipt*stride+4 );
          // float dlt = spimg.pixellist().at( ipt*stride+5 );
          // float mic = spimg.pixellist().at( ipt*stride+6 );
	  
	  SSNetData_t ssnetdata;
	  ssnetdata.row = xrow;
	  ssnetdata.col = xcol;
          ssnetdata.hip = spimg.pixellist().at( ipt*stride+2 );
          ssnetdata.mip = spimg.pixellist().at( ipt*stride+3 );
          ssnetdata.shr = spimg.pixellist().at( ipt*stride+4 );
          ssnetdata.dlt = spimg.pixellist().at( ipt*stride+5 );
          ssnetdata.mic = spimg.pixellist().at( ipt*stride+6 );
	  
	  data[p][ std::pair<int,int>(xrow,xcol) ] = ssnetdata;
	  
        }//end of point loop
      }//end of if five particle ssn data exists
      
    }//end of plane loop
    
        
    for ( auto & hit : larmatch_hit_v ) {

      // 5 particle score, 3 planes. we average ...
      std::vector<float> scores(5,0);
      int nplanes = 0;

      for ( int p=0; p<3; p++) {
	int row = hit.targetwire[3];
	int col = hit.targetwire[p];
	auto it = data[p].find( std::pair<int,int>( row,col ) );
	if ( it!=data[p].end() ) {
	  scores[0] += it->second.hip;
	  scores[1] += it->second.mip;
	  scores[2] += it->second.shr;
	  scores[3] += it->second.dlt;
	  scores[4] += it->second.mic;

	  // softmax
	  // double norm = 0.;
	  // for (int i=0; i<5; i++)
	  //   norm += exp( scores[i] );
	  // for (int i=0; i<5; i++)
	  //   scores[i] = exp( scores[i] )/norm;
	  
	  nplanes++;
	}
      }
      if (nplanes==0)
	continue;

      float renorm = 0.;
      int max_pid = -1;
      float max_val = 0;
      for ( int i=0; i<5; i++) {
	if ( scores[i]>max_val ) {
	  max_val = scores[i];
	  max_pid = i;
	}
	scores[i] /= (float)nplanes;
	renorm += scores[i];
      }
      for ( int i=0; i<5; i++) {
	scores[i] /= renorm;
      }
      
      // stuff into larflow hit
      for (int i=0; i<5; i++)
	hit[3+i] = scores[i];
      hit[8] = max_pid;
      
      hit.renormed_shower_score = scores[2]+scores[3]+scores[4];

    }//end of hit loop
    
    clock_t end = clock();
    double elapsed = double(end-begin)/CLOCKS_PER_SEC;
    
    LARCV_INFO() << " elasped=" << elapsed << " secs" << std::endl;
    
  }
  

  /**
   * compile network output into hit candidate information
   *
   * 
   */
  int FlowMatchHitMaker::add_triplet_match_data( PyObject* triple_probs,
                                                 PyObject* triplet_indices,
                                                 PyObject* imgu_sparseimg,
                                                 PyObject* imgv_sparseimg,
                                                 PyObject* imgy_sparseimg,
                                                 const std::vector< std::vector<float> >& pos_vv,
                                                 const std::vector<larcv::Image2D>& adc_v,
						 const std::vector<int>& img_indices_v,
						 const int tpcid, const int cryoid )
  {

    if ( img_indices_v.size()!=3 ) {
      LARCV_CRITICAL() << "Unexpected number of image indices = " << img_indices_v.size() << ". Expected 3" << std::endl;
    }
    
    // enable numpy environment (if not already set)
    std::cout << "setting pyutils ... ";
    import_array1(0);    
    larcv::SetPyUtil();
    std::cout << " done" << std::endl;

    // check types
    if ( PyArray_TYPE((PyArrayObject*)triple_probs)!=NPY_FLOAT32 ) {
       throw std::runtime_error("[FlowMatchHitMaker] triple_probs array needs to by type np.float32");
    }
    if ( PyArray_TYPE((PyArrayObject*)triplet_indices)!=NPY_LONG ) {
      throw std::runtime_error("[FlowMatchHitMaker] triplet_indices array needs to by type np.long");
    }
    if (PyArray_TYPE((PyArrayObject*)imgu_sparseimg)!=NPY_LONG) {
      throw std::runtime_error("[FlowMatchHitMaker] imgu_sparseimg array needs to by type np.long");
    }
    if (PyArray_TYPE((PyArrayObject*)imgv_sparseimg)!=NPY_LONG) {
      throw std::runtime_error("[FlowMatchHitMaker] imgv_sparseimg array needs to by type np.long");
    }
    if (PyArray_TYPE((PyArrayObject*)imgy_sparseimg)!=NPY_LONG) {
      throw std::runtime_error("[FlowMatchHitMaker] imgy_sparseimg array needs to by type np.long");
    }
      

    // // match scores
    int pair_ndims = PyArray_NDIM( (PyArrayObject*)triplet_indices );
    npy_intp* pair_dims = PyArray_DIMS( (PyArrayObject*)triplet_indices );
    std::cout << "[FlowMatchHitMaker] pair prob dims=(" << pair_dims[0] << ")" << std::endl;

    // triplet indicies for each match
    // npy_intp* index_dims = PyArray_DIMS( (PyArrayObject*)triplet_indices );
    // std::cout << "[FlowMatchHitMaker] index array dims=(" << index_dims[0] << "," << index_dims[1] << ")" << std::endl;
    
    // // sparse images
    // npy_intp imgu_dims[2];
    // std::cout << "[FlowMatchHitMaker] img[u] prob dims=(" << imgu_dims[0] << "," << imgu_dims[1] << ")" << std::endl;

    // npy_intp imgv_dims[2];
    // std::cout << "[FlowMatchHitMaker] img[v] prob dims=(" << imgv_dims[0] << "," << imgv_dims[1] << ")" << std::endl;

    // npy_intp imgy_dims[2];
    // std::cout << "[FlowMatchHitMaker] img[y] prob dims=(" << imgy_dims[0] << "," << imgy_dims[1] << ")" << std::endl;

    bool precalc_pos = ( pos_vv.size()>0 ) ? true : false;
    int nrepeated = 0;
    int nbelowminprob = 0.;
    
    // loop over each candidate triplet
    for (int ipair=0; ipair<(int)pair_dims[0]; ipair++) {

      // score for this match
      float prob = *(float*)PyArray_GETPTR1( (PyArrayObject*)triple_probs, ipair );

      long index[5] = { *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 0 ),
                        *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 1 ),
                        *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 2 ),
			*(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 3 ),
			*(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 4 ) };

      
      std::vector<int> triple(7,0); // (col,col,col,tick,istruth,tpcid,cryoid)
      triple[ 0 ] = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgu_sparseimg, index[0], 1 );
      triple[ 1 ] = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgv_sparseimg, index[1], 1 );
      triple[ 2 ] = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgy_sparseimg, index[2], 1 );
      int row = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgy_sparseimg, index[2], 0 );
      triple[ 3 ] = (int)adc_v[2].meta().pos_y( row );
      triple[ 4 ] = (int)index[3]; //truth
      triple[ 5 ] = (int)tpcid;
      triple[ 6 ] = (int)cryoid;

      if ( triple[0]==0 && triple[1]==0 && triple[2]==0 && triple[3]==2400 ) {	
	std::cout << "zero triple: "
		  << "index=(" << index[0] << "," << index[1] << "," << index[2] << ")"
		  << std::endl;
	std::cin.get();
      }
      
      // match threshold
      if ( prob<_match_score_threshold ) {
        nbelowminprob++;
        //continue;
      }

      std::vector<float> pos(3,0);
      
      if ( precalc_pos ) {
        pos = pos_vv[index[4]];
      }
      else {
        
        // get the other plane wire
        // convert row to tick
        double y(0), z(0);
	//larlite::larutil::Geometry::GetME()->IntersectionPoint( triple[2], triple[0], (UChar_t)2, (UChar_t)0, y, z );

        pos[0] = (triple[3]-3200.0)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
        pos[1] = y;
        pos[2] = z;
        
      }

      auto it = _match_map.find( triple );
      if ( it==_match_map.end() ) {
        // if not in match map, we create a new entry
        match_t m;
	m.cryoid = cryoid;
	m.tpcid  = tpcid;
	m.set_istruth(triple[4]);
        for ( size_t p=0; p<3; p++ )
          m.set_wire( p, triple[p] );
        m.tyz = { (float)triple[3], (float)pos[1], (float)pos[2] };
	m.img_indices_v = img_indices_v;
        _matches_v.emplace_back( std::move(m) );
        _match_map[ triple ] = int(_matches_v.size())-1;
        it = _match_map.find( triple );
      }
      else {
        std::cout << "repeated triple: (" << triple[0] << "," << triple[1] << "," << triple[2] << "," << triple[3] << "," << triple[4] <<")"
		  << " tpcid=" << triple[5]
		  << " cryoid=" << triple[6]
		  << std::endl;
	auto const& orig = it->first;
	std::cout << " orig triplet(" << orig[0] << "," << orig[1] << "," << orig[2] << "," << orig[3] << "," << orig[4] <<") "
		  << " tpcid=" << orig[5]
		  << " cryoid=" << orig[6]
		  << std::endl;
	std::cin.get();
        nrepeated++;
      }

      auto& m = _matches_v.at( it->second );
      // set the score given here
      //if ( indead ) prob *= 0.5;
      m.set_score( prob );

    }//end of score loop
    
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "process match data. " << std::endl;
    std::cout << "  number of triples with match information: " << _match_map.size() << std::endl;
    std::cout << "  num repeated: " << nrepeated << std::endl;
    std::cout << "  num below min prob: " << nbelowminprob << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    return 0;
  }


  /**
   * store ssnet scores in match_t struct
   *
   * add_triplet_match_data needed to have been run first
   *
   */
  int FlowMatchHitMaker::add_triplet_ssnet_scores( PyObject* triplet_indices,
                                                   PyObject* imgu_sparseimg,
                                                   PyObject* imgv_sparseimg,
                                                   PyObject* imgy_sparseimg,
                                                   const larcv::ImageMeta& meta,
                                                   PyObject* ssnet_scores ) {

    has_ssnet_scores = true;

    // check types
    if ( PyArray_TYPE((PyArrayObject*)ssnet_scores)!=NPY_FLOAT32 ) {
       throw std::runtime_error("[FlowMatchHitMaker] triple_probs array needs to by type np.float32");
    }
    if ( PyArray_TYPE((PyArrayObject*)triplet_indices)!=NPY_LONG ) {
      throw std::runtime_error("[FlowMatchHitMaker] triplet_indices array needs to by type np.long");
    }
    if (PyArray_TYPE((PyArrayObject*)imgu_sparseimg)!=NPY_LONG) {
      throw std::runtime_error("[FlowMatchHitMaker] imgu_sparseimg array needs to by type np.long");
    }
    if (PyArray_TYPE((PyArrayObject*)imgv_sparseimg)!=NPY_LONG) {
      throw std::runtime_error("[FlowMatchHitMaker] imgv_sparseimg array needs to by type np.long");
    }
    if (PyArray_TYPE((PyArrayObject*)imgy_sparseimg)!=NPY_LONG) {
      throw std::runtime_error("[FlowMatchHitMaker] imgy_sparseimg array needs to by type np.long");
    }

    
    // match triplets
    int pair_ndims      = PyArray_NDIM( (PyArrayObject*)triplet_indices );
    npy_intp* pair_dims = PyArray_DIMS( (PyArrayObject*)triplet_indices );

    int ss_pair_ndims      = PyArray_NDIM( (PyArrayObject*)ssnet_scores );
    npy_intp* ss_pair_dims = PyArray_DIMS( (PyArrayObject*)ssnet_scores );
    std::cout << "[FlowMatchHitMaker::add_triplet_ssnet_scores] score dims="
              << "(" << ss_pair_dims[0] << "," << ss_pair_dims[1] << ")"
              << std::endl;            

    for (int ipair=0; ipair<(int)ss_pair_dims[0]; ipair++ ) {
      // get the triplet indices
      
      long index[4] = { *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 0 ),
                        *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 1 ),
                        *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 2 ),
                        *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 3 ) };
      
      std::vector<int> triple(5,0); // (col,col,col,tick,istruth)
      triple[ 0 ] = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgu_sparseimg, index[0], 1 );
      triple[ 1 ] = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgv_sparseimg, index[1], 1 );
      triple[ 2 ] = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgy_sparseimg, index[2], 1 );
      int row = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgy_sparseimg, index[2], 0 );
      triple[ 3 ] = (int)meta.pos_y( row );
      triple[ 4 ] = (int)index[3];
    
      // look for the triplet in the map
      auto it = _match_map.find( triple );
      if ( it!=_match_map.end() ) {
        int matchidx = it->second;
        auto& match = _matches_v[matchidx];

        // store the ssnet scores
        match.ssnet_scores.resize(larflow::prep::PrepSSNetTriplet::kNumClasses,0.0);
        //std::cout << "save ssnet scores: ";
	if ( ss_pair_dims[1]==larflow::prep::PrepSSNetTriplet::kNumClasses) {
	  // current 7-class style
	  for (int c=0; c<larflow::prep::PrepSSNetTriplet::kNumClasses; c++) {
	    match.ssnet_scores[c] = (float) *((float*)PyArray_GETPTR2( (PyArrayObject*)ssnet_scores, ipair, c ));
	    //std::cout << match.ssnet_scores[c] << " ";
	  }
	}
	else {
	  // old 3-class style
	  match.ssnet_scores[0] = (float) *((float*)PyArray_GETPTR2( (PyArrayObject*)ssnet_scores, ipair, 0 )); // bg
	  match.ssnet_scores[1] = (float) *((float*)PyArray_GETPTR2( (PyArrayObject*)ssnet_scores, ipair, 1 ));	// shower, assign to electron
	  match.ssnet_scores[3] = (float) *((float*)PyArray_GETPTR2( (PyArrayObject*)ssnet_scores, ipair, 2 ));	// track, assign to muon
	}
	
        //std::cout << std::endl;
      }
      else {
        std::cout << "could not find match for triplet: ("
                  << triple[0] << ","
                  << triple[1] << ","
                  << triple[2] << ","
                  << triple[3] << ","
                  << triple[4] << ")"
                  << std::endl;
        throw std::runtime_error("bad ssnet data");
      }
    }
    
    return 0;
  }

  /**
   * store keypoint label scores in match_t struct
   *
   * add_triplet_match_data needed to have been run first
   *
   */
  int FlowMatchHitMaker::add_triplet_keypoint_scores( PyObject* triplet_indices,
                                                      PyObject* imgu_sparseimg,
                                                      PyObject* imgv_sparseimg,
                                                      PyObject* imgy_sparseimg,
                                                      const larcv::ImageMeta& meta,                                                      
                                                      PyObject* kplabel_scores,
						      const int tpcid, const int cryoid ) {

    has_kplabel_scores = true;

    // check types
    if ( PyArray_TYPE((PyArrayObject*)kplabel_scores)!=NPY_FLOAT32 ) {
       throw std::runtime_error("[FlowMatchHitMaker] kplabel_scores array needs to by type np.float32");
    }
    if ( PyArray_TYPE((PyArrayObject*)triplet_indices)!=NPY_LONG ) {
      throw std::runtime_error("[FlowMatchHitMaker] triplet_indices array needs to by type np.long");
    }
    if (PyArray_TYPE((PyArrayObject*)imgu_sparseimg)!=NPY_LONG) {
      throw std::runtime_error("[FlowMatchHitMaker] imgu_sparseimg array needs to by type np.long");
    }
    if (PyArray_TYPE((PyArrayObject*)imgv_sparseimg)!=NPY_LONG) {
      throw std::runtime_error("[FlowMatchHitMaker] imgv_sparseimg array needs to by type np.long");
    }
    if (PyArray_TYPE((PyArrayObject*)imgy_sparseimg)!=NPY_LONG) {
      throw std::runtime_error("[FlowMatchHitMaker] imgy_sparseimg array needs to by type np.long");
    }    
    
    // match triplets
    int pair_ndims      = PyArray_NDIM( (PyArrayObject*)triplet_indices );
    npy_intp* pair_dims = PyArray_DIMS( (PyArrayObject*)triplet_indices );

    int kp_pair_ndims      = PyArray_NDIM( (PyArrayObject*)kplabel_scores );
    npy_intp* kp_pair_dims = PyArray_DIMS( (PyArrayObject*)kplabel_scores );
    if ( kp_pair_ndims==1)
      std::cout << "[FlowMatchHitMaker::add_triplet_keypoint_scores] score dims=(" << kp_pair_dims[0] << ")" << std::endl;
    else
      std::cout << "[FlowMatchHitMaker::add_triplet_keypoint_scores] score dims=(" << kp_pair_dims[0] << "," << kp_pair_dims[1] << ")" << std::endl;


    for (int ipair=0; ipair<(int)kp_pair_dims[0]; ipair++ ) {
      // get the triplet indices
      
      long index[4] = { *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 0 ),
                        *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 1 ),
                        *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 2 ),
                        *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 3 ) };
      
      std::vector<int> triple(7,0); // (col,col,col,tick,istruth)
      triple[ 0 ] = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgu_sparseimg, index[0], 1 );
      triple[ 1 ] = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgv_sparseimg, index[1], 1 );
      triple[ 2 ] = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgy_sparseimg, index[2], 1 );
      int row = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgy_sparseimg, index[2], 0 );
      triple[ 3 ] = (int)meta.pos_y( row );
      triple[ 4 ] = (int)index[3];
      triple[ 5 ] = (int)tpcid;
      triple[ 6 ] = (int)cryoid;
    
      // look for the triplet in the map
      auto it = _match_map.find( triple );
      if ( it!=_match_map.end() ) {
        int matchidx = it->second;
        auto& match = _matches_v[matchidx];

        // store the kplabel scores]
        if ( kp_pair_ndims>1 ) {
          match.keypoint_scores = std::vector<float>(6,0);
	  if ( kp_pair_dims[1]==6 ) {
	    // current 6-class keypoints
	    for (int iclass=0; iclass<(int)6; iclass++)
	      match.keypoint_scores[iclass] = (float)*((float*)PyArray_GETPTR2( (PyArrayObject*)kplabel_scores, ipair, iclass ));
	  }
	  else if ( kp_pair_dims[1]==3 ) {
	    // old 3-class keypoints
	    match.keypoint_scores[0] = (float)*((float*)PyArray_GETPTR2( (PyArrayObject*)kplabel_scores, ipair, 0 )); // neutrino
	    match.keypoint_scores[1] = (float)*((float*)PyArray_GETPTR2( (PyArrayObject*)kplabel_scores, ipair, 1 )); // track-starts
	    match.keypoint_scores[3] = (float)*((float*)PyArray_GETPTR2( (PyArrayObject*)kplabel_scores, ipair, 2 )); // shower starts
	    //std::cout << (float)*((float*)PyArray_GETPTR2( (PyArrayObject*)kplabel_scores, ipair, 0 )) << std::endl;
	  }
	  else {
	    throw std::runtime_error("bad keypoint label data. wrong number of classes");
	  }
        }
        else {
	  throw std::runtime_error("bad keypoint label data");
          match.keypoint_scores.resize(1,0);
          match.keypoint_scores[0] = (float)*((float*)PyArray_GETPTR1( (PyArrayObject*)kplabel_scores, ipair ) );
        }
      }
      else {
        std::cout << "could not find match for triplet: ("
                  << triple[0] << ","
                  << triple[1] << ","
                  << triple[2] << ","
                  << triple[3] << ","
                  << triple[4] << ")"
		  << " tpcid=" << tpcid
		  << " cryoid=" << cryoid
                  << std::endl;
      }
    }
    
    return 0;
  }

  /**
   * store particle affinity field direction
   *
   * add_triplet_match_data needed to have been run first
   *
   */
  int FlowMatchHitMaker::add_triplet_affinity_field( PyObject* triplet_indices,
                                                     PyObject* imgu_sparseimg,
                                                     PyObject* imgv_sparseimg,
                                                     PyObject* imgy_sparseimg,
                                                     const larcv::ImageMeta& meta,                                                      
                                                     PyObject* paf_pred ) {

    has_paf = true;

    // check types
    if ( PyArray_TYPE((PyArrayObject*)paf_pred)!=NPY_FLOAT32 ) {
       throw std::runtime_error("[FlowMatchHitMaker] paf_pred array needs to by type np.float32");
    }
    if ( PyArray_TYPE((PyArrayObject*)triplet_indices)!=NPY_LONG ) {
      throw std::runtime_error("[FlowMatchHitMaker] triplet_indices array needs to by type np.long");
    }
    if (PyArray_TYPE((PyArrayObject*)imgu_sparseimg)!=NPY_LONG) {
      throw std::runtime_error("[FlowMatchHitMaker] imgu_sparseimg array needs to by type np.long");
    }
    if (PyArray_TYPE((PyArrayObject*)imgv_sparseimg)!=NPY_LONG) {
      throw std::runtime_error("[FlowMatchHitMaker] imgv_sparseimg array needs to by type np.long");
    }
    if (PyArray_TYPE((PyArrayObject*)imgy_sparseimg)!=NPY_LONG) {
      throw std::runtime_error("[FlowMatchHitMaker] imgy_sparseimg array needs to by type np.long");
    }    
    
    // match triplets
    int pair_ndims      = PyArray_NDIM( (PyArrayObject*)triplet_indices );
    npy_intp* pair_dims = PyArray_DIMS( (PyArrayObject*)triplet_indices );

    int paf_pair_ndims      = PyArray_NDIM( (PyArrayObject*)paf_pred );
    npy_intp* paf_pair_dims = PyArray_DIMS( (PyArrayObject*)paf_pred );
    std::cout << "[FlowMatchHitMaker::add_triplet_affinity_field] score dims=(" << paf_pair_dims[0] << ")" << std::endl;    

    for (int ipair=0; ipair<(int)paf_pair_dims[0]; ipair++ ) {
      // get the triplet indices
      
      long index[4] = { *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 0 ),
                        *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 1 ),
                        *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 2 ),
                        *(long*)PyArray_GETPTR2( (PyArrayObject*)triplet_indices, ipair, 3 ) };
      
      std::vector<int> triple(5,0); // (col,col,col,tick,istruth)
      triple[ 0 ] = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgu_sparseimg, index[0], 1 );
      triple[ 1 ] = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgv_sparseimg, index[1], 1 );
      triple[ 2 ] = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgy_sparseimg, index[2], 1 );
      int row = (int)*(long*)PyArray_GETPTR2( (PyArrayObject*)imgy_sparseimg, index[2], 0 );
      triple[ 3 ] = (int)meta.pos_y( row );
      triple[ 4 ] = (int)index[3];
    
      // look for the triplet in the map
      auto it = _match_map.find( triple );
      if ( it!=_match_map.end() ) {
        int matchidx = it->second;
        auto& match = _matches_v[matchidx];

        // store the kplabel scores]
        match.paf.resize(3,0);
        for (int i=0;i<3; i++ ) {
          match.paf[i] = (float)*((float*)PyArray_GETPTR2( (PyArrayObject*)paf_pred, ipair, i ));
        }
      }
      else {
        std::cout << "could not find match for triplet: ("
                  << triple[0] << ","
                  << triple[1] << ","
                  << triple[2] << ","
                  << triple[3] << ","
                  << triple[4] << ")"
                  << std::endl;
      }
    }
    
    return 0;
  }
  
}  
}

