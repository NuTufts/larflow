#include "LArVoxelHitMaker.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/Geometry.h"

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
    long **kp_carray;
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
	vdata.ssnet_class_score[i] = ssnet_carray[ivoxel][i];

      for (int i=0; i<6; i++)
	vdata.kp_class_score[i] = kp_carray[ivoxel][i];

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
   * [3-7]:   ssnet e-[3], gamma[4], muon[5], pion[6], proton[7]
   * [8]: larmatch score
   * [9]: ssnet other
   * [10-12]: 3 ssnet scores, (bg,track,shower), from larmatch (not 2D sparse ssnet)
   * [13-18]: 6 keypoint label score [nu,trackstart,trackend,shower,delta,michel]
   * [19-21]: U[19],V[20],Y[21] plane charge 
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
	hit.resize(22,0);
	hit[0] = pos[0];
	hit[1] = pos[1];
	hit[2] = pos[2];
	int row = triplet[3];
	if ( row<0 || row>=(int)adc_v.front().meta().rows() )
	  continue;
	
	hit.tick = adc_v.front().meta().pos_y( triplet[3] );

	std::cout << "hit@tick=" << hit.tick << std::endl;

	hitidx++;
	
	hit[3] = voxeldata.ssnet_class_score[1]; // electron
	hit[4] = voxeldata.ssnet_class_score[2]; // gamma
	hit[5] = voxeldata.ssnet_class_score[3]; // muon
	hit[6] = voxeldata.ssnet_class_score[4]; // pion
	hit[7] = voxeldata.ssnet_class_score[5]; // proton
	hit[9] = voxeldata.ssnet_class_score[6]; // other
	
	hit[8] = voxeldata.lm_score; // true vs ghost score
	
	hit[10] = voxeldata.ssnet_class_score[0]+voxeldata.ssnet_class_score[6]; // bg
	hit[11] = hit[5]+hit[6]+hit[7]; // track
	hit[12] = hit[3]+hit[4]; // shower

	hit[13] = voxeldata.kp_class_score[0]; // nu
	hit[14] = voxeldata.kp_class_score[1]; // track-start
	hit[15] = voxeldata.kp_class_score[2]; // track-end
	hit[16] = voxeldata.kp_class_score[3]; // shower
	hit[17] = voxeldata.kp_class_score[4]; // delta
	hit[18] = voxeldata.kp_class_score[5]; // michel
	
	hit[19] = 0.0;
	hit[20] = 0.0;
	hit[21] = 0.0;

	hit.srcwire = triplet[2];
	hit.targetwire = triplet;
	hit.idxhit = hitidx;	
	if( tripletmaker._truth_v.size()==ntriplet && tripletmaker._truth_v[itriplet]==1)
	  hit.truthflag = larlite::larflow3dhit::TruthFlag_t::kOnTrack;
	else
	  hit.truthflag = larlite::larflow3dhit::TruthFlag_t::kNoTruthMatch;	  

	
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
		hit[19+p] += pix;
		nfilled++;
	      }
	    }
	  }//end of radius loop

	  if ( nfilled==0 )
	    hit[19+p] = 0.0;
	  else
	    hit[19+p] /= (float)nfilled;
	  
	}//end of plane loop

	output_container.emplace_back( std::move(hit) );
	
      }// end if valid hit
    }//end of triplet loop

    return;
  }

  


}
}
