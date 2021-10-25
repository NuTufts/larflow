#include "LArVoxelHitMaker.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/Geometry.h"
#include "larcv/core/DataFormat/EventSparseImage.h"
#include "larcv/core/DataFormat/EventImage2D.h"

namespace larflow {
namespace voxelizer {

  bool LArVoxelHitMaker::_setup_numpy = false;
  
  int LArVoxelHitMaker::add_voxel_labels( PyObject* coord_t,
					  PyObject* larmatch_pred_t,
					  PyObject* ssnet_pred_t,
					  PyObject* kp_pred_t )
  {
    if ( !_setup_numpy ) {
      std::cout << "setting pyutils ... ";
      import_array1(0);    
      std::cout << " done" << std::endl;
    }

    // cast numpy data to C-arrays
    PyArray_Descr *descr_float = PyArray_DescrFromType(NPY_FLOAT);
    PyArray_Descr *descr_long  = PyArray_DescrFromType(NPY_LONG);    

    // voxel coordinate tensor
    npy_intp coord_dims[2];
    long **coord_carray;
    if ( PyArray_AsCArray( &coord_t, (void**)&coord_carray, coord_dims, 2, descr_long )<0 ) {
      LARCV_CRITICAL() << "cannot get carray for COORD array" << std::endl;
    }

    // larmatch score tensor
    npy_intp larmatch_dims[2];
    float **larmatch_carray;
    if ( PyArray_AsCArray( &larmatch_pred_t, (void**)&larmatch_carray, larmatch_dims, 2, descr_float )<0 ) {
      LARCV_CRITICAL() << "cannot get carray for LARMATCH SCORE array" << std::endl;
    }

    // SSNet tensor
    npy_intp ssnet_dims[2];
    float **ssnet_carray;
    if ( PyArray_AsCArray( &ssnet_pred_t, (void**)&ssnet_carray, ssnet_dims, 2, descr_float )<0 ) {
      LARCV_CRITICAL() << "cannot get carray for SSNET array" << std::endl;
    }

    // Keypoint prediction tensor
    npy_intp kp_dims[2];
    float **kp_carray;
    if ( PyArray_AsCArray( &kp_pred_t, (void**)&kp_carray, kp_dims, 2, descr_float )<0 ) {
      LARCV_CRITICAL() << "cannot get carray for KEYPOINT SCORE array" << std::endl;
    }

    size_t nvoxels = coord_dims[0];
    for (size_t ivoxel=0; ivoxel<nvoxels; ivoxel++) {

      std::array<long,3> coord = { coord_carray[ivoxel][0], coord_carray[ivoxel][1], coord_carray[ivoxel][2] };
      voxeldata_t vdata( coord );
      vdata.lm_score = larmatch_carray[ivoxel][1];
      if ( vdata.lm_score < _hit_score_threshold )
	continue;

      for (int i=0; i<7; i++) 
	vdata.ssnet_class_score[i] = ssnet_carray[i][ivoxel];

      for (int i=0; i<6; i++)
	vdata.kp_class_score[i] = kp_carray[i][ivoxel];

      _voxeldata_map[ coord ] = vdata;
    }

    
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
   * [0-2]:   x,y,z
   * [3-8]:   reserved for 2D ssnet: HIP[3], MIP[4], shower[5], michel[6], delta[7], maxclass[8]
   * [9]: larmatch score
   * [10-16]: 3D ssnet other from larvoxelnet
   * [17-22]: 6 keypoint label score [nu,trackstart,trackend,shower,delta,michel]
   * [23-25]: U[23],V[24],Y[25] plane charge 
   * [26-28]: reserved for 3D flow direction
   * 
   *
   */  
  void LArVoxelHitMaker::make_labeled_larflow3dhits( const larflow::prep::PrepMatchTriplets& tripletmaker,
						     const std::vector<larcv::Image2D>& adc_v,
						     larlite::event_larflow3dhit& output_container )
  {

    const float cm_per_tick = larutil::LArProperties::GetME()->DriftVelocity()*0.5;

    int hitidx = -1;
    size_t ntriplet = tripletmaker._triplet_v.size();
    for (size_t itriplet=0; itriplet<ntriplet; itriplet++) {
      auto const& triplet = tripletmaker._triplet_v.at(itriplet);
      auto const& pos     = tripletmaker._pos_v.at(itriplet);

      std::vector<int> coord = _voxelizer.get_voxel_indices( pos );
      std::array<long,3> coordkey = { coord[0], coord[1], coord[2] };

      auto it = _voxeldata_map.find( coordkey );
      if ( it!=_voxeldata_map.end() ) {

	auto const& voxeldata = it->second;

	larlite::larflow3dhit hit;
	hit.resize(29,0);
	hit[0] = pos[0];
	hit[1] = pos[1];
	hit[2] = pos[2];
	int row = triplet[3];
	if ( row<0 || row>=(int)adc_v.front().meta().rows() )
	  continue;
	
	hit.tick = adc_v.front().meta().pos_y( triplet[3] );

	hitidx++;
	
	// hit[3] = voxeldata.ssnet_class_score[1]; // electron
	// hit[4] = voxeldata.ssnet_class_score[2]; // gamma
	// hit[5] = voxeldata.ssnet_class_score[3]; // muon
	// hit[6] = voxeldata.ssnet_class_score[4]; // pion
	// hit[7] = voxeldata.ssnet_class_score[5]; // proton
	// hit[9] = voxeldata.ssnet_class_score[6]; // other
	
	hit[9] = voxeldata.lm_score; // true vs ghost score	
	
	// hit[10] = voxeldata.ssnet_class_score[0]+voxeldata.ssnet_class_score[6]; // bg
	// hit[11] = hit[5]+hit[6]+hit[7]; // track
	// hit[12] = hit[3]+hit[4]; // shower

	hit[10] = voxeldata.ssnet_class_score[0]; //bg
	hit[11] = voxeldata.ssnet_class_score[1];
	hit[12] = voxeldata.ssnet_class_score[2];
	hit[13] = voxeldata.ssnet_class_score[3];
	hit[14] = voxeldata.ssnet_class_score[4];
	hit[15] = voxeldata.ssnet_class_score[5];
	hit[16] = voxeldata.ssnet_class_score[6];			

	hit[17] = voxeldata.kp_class_score[0]; // nu
	hit[18] = voxeldata.kp_class_score[1]; // track-start
	hit[19] = voxeldata.kp_class_score[2]; // track-end
	hit[20] = voxeldata.kp_class_score[3]; // shower
	hit[21] = voxeldata.kp_class_score[4]; // michel
	hit[22] = voxeldata.kp_class_score[5]; // delta
	
	hit[23] = 0.0;
	hit[24] = 0.0;
	hit[25] = 0.0;

	hit[26] = 0.;
	hit[27] = 0.;
	hit[28] = 0.;	

	hit.track_score = voxeldata.lm_score; // true vs ghost score

	//get wires
	hit.targetwire.resize(4,0);
	for (int p=0; p<3; p++) {
	  int sparseidx = triplet[p];
	  hit.targetwire[p] = tripletmaker._sparseimg_vv[p].at(sparseidx).col;
	}
	hit.targetwire[3] = row;
	hit.srcwire = hit.targetwire[2];
	hit.idxhit = hitidx;	
	if( tripletmaker._truth_v.size()==ntriplet && tripletmaker._truth_v[itriplet]==1)
	  hit.truthflag = larlite::larflow3dhit::TruthFlag_t::kOnTrack;
	else
	  hit.truthflag = larlite::larflow3dhit::TruthFlag_t::kNoTruthMatch;	  

	//std::cout << "hit@wires=(" << hit.targetwire[0] << "," << hit.targetwire[1] << "," << hit.targetwire[2] << "," << hit.targetwire[3] << ")" << std::endl;
	for (int p=0; p<3; p++) {
	  auto const& meta = adc_v[p].meta();
	  auto const& img  = adc_v[p];

	  int nfilled = 0;
	  for (int dr=-2; dr<=2; dr++) {
	    int r = row+dr;
	    if ( r<=0 || r>=(int)meta.rows() )
	      continue;

	    for (int dc=-2; dc<=2; dc++) {
	      int c = hit.targetwire[p] + dc;
	      if ( c<=0 || c>=(int)meta.cols() )
		continue;
	      float pix = adc_v[p].pixel(r,c,__FILE__,__LINE__);
	      if ( pix>10.0 ) {
		hit[23+p] += pix;
		nfilled++;
	      }
	    }
	  }//end of radius loop

	  if ( nfilled==0 )
	    hit[23+p] = 0.0;
	  else
	    hit[23+p] /= (float)nfilled;
	  
	}//end of plane loop

	//std::cout << "hit: q=(" << hit[23] << "," << hit[24] << "," << hit[25] << ")" << std::endl;

	output_container.emplace_back( std::move(hit) );
	
      }// end if valid hit
    }//end of triplet loop
    
    return;
  }

  /**
   * @brief label container of larflow3dhit using 2D track/shower ssnet output images
   *
   * calculates weighted ssnet score and modifies hit to carry value.
   * the weighted ssnet score for the space point is in `larlite::larflow3dhit::renormed_shower_score`
   *
   * @param[in] ssnet_score_v            SSNet shower score images for each plane
   * @param[inout] larmatch_hit_v        LArMatch hits, modified
   */
  void LArVoxelHitMaker::store_2dssnet_score( larcv::IOManager& iolcv,
					      larlite::event_larflow3dhit& larmatch_hit_v )
  {
    
    clock_t begin = clock();

    // get sparse image for each plane
    std::vector< const larcv::SparseImage*> spimg_v;

    // try sparseuresnetout
    larcv::EventSparseImage* ev_ssnet
      = (larcv::EventSparseImage*)iolcv.get_data( larcv::kProductSparseImage,"sparseuresnetout");

    if ( ev_ssnet->SparseImageArray().size()==3 ) {
      LARCV_NORMAL() << "use current 'sparseuresnetout' tree" << std::endl;      
      auto& sparseimg_v = ev_ssnet->SparseImageArray();
      for ( auto& spimg : sparseimg_v ) 
	spimg_v.push_back( &spimg );
    }
    else {
      LARCV_NORMAL() << "use old uresnet_plane tree" << std::endl;
      // let's try uresnet_plane%d: old name
      for (int p=0; p<3; p++) {
	char zname[50];	
	sprintf( zname, "uresnet_plane%d", p );
	
	larcv::EventSparseImage* ev_s = (larcv::EventSparseImage*)iolcv.get_data(larcv::kProductSparseImage,zname);
	auto const& s_v = ev_s->SparseImageArray();
	if ( s_v.size()>0 )
	  spimg_v.push_back( &s_v.at(0) );
      }
    }

    if (spimg_v.size()!=3) {
      LARCV_WARNING() << "wrong number of sparse ssnet output images: " << spimg_v.size() << ". Want 3" << std::endl;
      return;
    }


    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_adc->as_vector();
    

    std::cout << "number of sparse images: " << spimg_v.size() << std::endl;


    // convert into 5-particle ssnet

    struct SSNetData_t {
      int row;
      int col;
      
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
      
      if (spimg_v.size()>0) {
        auto const& spimg = *spimg_v.at(p);
      
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
  


}
}
