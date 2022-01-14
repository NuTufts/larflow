#include "SmallClusterRemoval.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <iostream>

#include "larflow/Reco/cluster_functions.h"

namespace larvoxelprepdata {

  bool SmallClusterRemoval::_setup_numpy = false;
  
  PyObject* SmallClusterRemoval::do_removal( PyObject* coord_array, PyObject* feat_array,
					     int voxelthreshold, float charge_threshold )
  {

    if ( !_setup_numpy ) {
      std::cout << "import array" << std::endl;
      import_array1(0);
      _setup_numpy = true;
    }

    
    if ( PyArray_TYPE((PyArrayObject*)coord_array)!=NPY_LONG ) {
      throw std::runtime_error("[FlowMatchHitMaker] triplet_indices array needs to by type np.long");
    }
    if ( PyArray_TYPE((PyArrayObject*)feat_array)!=NPY_FLOAT32 ) {
      throw std::runtime_error("[FlowMatchHitMaker] triple_probs array needs to by type np.float32");
    }
    
    // coordinate array
    const int dtype = NPY_INT64;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);
    npy_intp pair_dims[2];
    long **coord;
    std::cout << "get pair prob dims" << std::endl;
    if ( PyArray_AsCArray( &coord_array, (void**)&coord, pair_dims, 2, descr )<0 ) {
      std::cout << "Could not load C-array properly. Returning None." << std::endl;
      Py_RETURN_NONE;      
    }

    // transfer voxel coordinates into spacepoints 
    std::vector< std::vector<float> > hit_v;
    hit_v.reserve( pair_dims[0] );
    for (int i=0; i<pair_dims[0]; i++) {
      std::vector< float > hit(3,0);
      for (int v=0; v<3; v++)
	hit[v] = float(coord[i][v])*0.3; // xyz
      std::cout << "[" << i << "] (" << hit[0] << "," << hit[1] << "," << hit[2] << ")" << std::endl;
      hit_v.push_back(hit);
    }

    
    std::vector< larflow::reco::cluster_t > cluster_v;
    float maxdist = 0.35; // a little larger than the voxel pitch
    int minsize = 1;
    int maxkd = 5;
    larflow::reco::cluster_spacepoint_v( hit_v, cluster_v, maxdist, minsize, maxkd );

    std::cout << "number of clusters: " << cluster_v.size() << std::endl;
    std::vector< int > use_v( cluster_v.size(), 0 );
    int tot_passing_voxels = 0;
    for (int c=0; c<(int)cluster_v.size(); c++) {
      std::cout << "  cluster[" << c << "] size=" << cluster_v[c].points_v.size() << std::endl;
      if ( cluster_v[c].points_v.size()>=voxelthreshold ) {

	// sum the charge on each plane
	std::vector<float> planeq(3,0);
	for (int i=0; i<(int)cluster_v[c].hitidx_v.size(); i++) {
	  int hitidx = cluster_v[c].hitidx_v[i];	  	
	  for (int p=0; p<3; p++) {
	    planeq[p] += *((float*)PyArray_GETPTR2( (PyArrayObject*)feat_array,hitidx,p));
	  }
	}
	std::cout << "    charge=(" << planeq[0] << "," << planeq[1] << "," << planeq[2] << ")" << std::endl;
	bool chargeok = false;
	for (int p=0; p<3; p++) {
	  if ( planeq[p]>charge_threshold )
	    chargeok = true;
	}

	if ( chargeok ) {
	  use_v[c] = 1;
	  tot_passing_voxels += (int)cluster_v[c].points_v.size();
	}
      }
    }
    std::cout << "num passing voxels: " << tot_passing_voxels << std::endl;
    
    if ( tot_passing_voxels== 0 ) {
      Py_RETURN_NONE;
    }

    npy_intp pass_dim[] = { pair_dims[0] };
    PyArrayObject* pass_array = (PyArrayObject*)PyArray_SimpleNew( 1, pass_dim, NPY_INT64 );
    PyArray_FILLWBYTE (pass_array, 0);
    
    for (int c=0; c<(int)cluster_v.size(); c++) {
      if ( use_v[c]>0 ) {
	for (int i=0; i<(int)cluster_v[c].hitidx_v.size(); i++) {
	  int hitidx = cluster_v[c].hitidx_v[i];
	  if ( hitidx<0 || hitidx>=pair_dims[0] ) {
	    std::cout << "  out of bounds hit saved: " << hitidx << std::endl;
	  }
	  else {
	    std::cout << "  save idx=" << hitidx << std::endl;
	    *((long*)PyArray_GETPTR1(pass_array,hitidx)) = 1;
	  }
	}
      }
    }

    std::cout << "returning passing array which leaves " << tot_passing_voxels << " total voxels" << std::endl;
    Py_INCREF( pass_array );
    
    return (PyObject*)pass_array;
  }

}
