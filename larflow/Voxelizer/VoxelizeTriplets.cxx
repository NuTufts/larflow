#include "VoxelizeTriplets.h"

#include <iostream>
#include <sstream>

namespace larflow {
namespace voxelizer {

  bool VoxelizeTriplets::_setup_numpy = false;
  
  VoxelizeTriplets::VoxelizeTriplets( std::vector<float> origin,
                                      std::vector<float> dim_len,
                                      float voxel_size ) 
    : larcv::larcv_base("VoxelizeTriplets"),
    _origin(origin),
    _len(dim_len),
    _voxel_size(voxel_size)
  {
    _ndims = (int)origin.size();
    _nvoxels.resize(_ndims,0);

    std::stringstream ss;
    ss << "(";
    for (int v=0; v<_ndims; v++) {
      int nv = (_origin[v] + _len[v])/voxel_size;
      if ( fabs(nv*_voxel_size-dim_len[v])>0.001 )
        nv++;
      _nvoxels[v] = nv;
      ss << nv;
      if ( v+1<_ndims ) ss << ", ";
    }
    ss << ")";

    LARCV_NORMAL() << "Number of voxels defined: " << ss.str() << std::endl;
  }

  int VoxelizeTriplets::get_axis_voxel( int axis, float coord ) const {

    if ( axis<0 || axis>=_ndims )
      throw std::runtime_error("[VoxelizeTriplets::get_axis_voxel] invalid dim");
    
    int vidx = (coord-_origin[axis])/_voxel_size;
    if (vidx<0 || vidx>=_nvoxels[axis] )
      throw std::runtime_error("[VoxelizeTriplets::get_axis_voxel] coordinate given is out of bounds");

    return vidx;
  }
  
  /**
   * takes in precomputed triplet data and outputs voxel data in the form of a python dict
   * contents of the dictionary:
   *  dict["voxcoord"] = (Nv,3) numpy int array; Nv (x,y,z) voxel coordinate
   *  dict["voxlabel"]  = (Nv) numpy int array; 1 if truth voxel, 0 otherwise. 
   *  dict["trip2vidx"] = (Nt) numpy int array; Nt voxel indices referencing the "coord" array
   *  dict["vox2trips_list"] = List of length Nv with (Mi) numpy int arrays. Each array contains triplet index list to combine into the voxel.
   * 
   * inputs:
   * @param[in] triplet_data Instance of triplet data, assumed to be filled already
   * @return Python dictionary as described above. Ownership is transferred to calling namespace.
   *
   */
  PyObject* VoxelizeTriplets::make_voxeldata_dict( const larflow::PrepMatchTriplets& triplet_data )
  {

    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }
    
    // first we need to define the voxels that are filled    
    std::set< std::array<int,3> > voxel_set;
    
    for ( int itriplet=0; itriplet<(int)triplet_data._triplet_v.size(); itriplet++ ) {
      const std::vector<float>& pos = triplet_data._pos_v[itriplet];
      std::array<int,3> coord;
      for (int i=0; i<3; i++)
        coord[i] = get_axis_voxel(i,pos[i]);
      voxel_set.insert( coord );        
    }

    // now we assign voxel to an index
    std::map< std::array<int,3>, int > voxel_list; /// map from voxel coordinate to voxel index
    int idx=0;
    for ( auto& coord : voxel_set ) {
      voxel_list[coord] = idx;
      idx++;
    }
    int nvidx = idx;    

    LARCV_INFO() << "Filling " << nvidx << " voxels from " << triplet_data._triplet_v.size() << " triplets" << std::endl;

    // assign triplets to voxels and vice versa
    std::vector< std::vector<int> > voxelidx_to_tripidxlist( nvidx ); // voxel to triplet vector
    std::vector<int> trip2voxelidx( triplet_data._triplet_v.size(), 0); // triplet to voxel
    
    for ( int itriplet=0; itriplet<(int)triplet_data._triplet_v.size(); itriplet++ ) {
      const std::vector<float>& pos = triplet_data._pos_v[itriplet];
      std::array<int,3> coord;
      for (int i=0; i<3; i++)
        coord[i] = get_axis_voxel(i,pos[i]);
      auto it=voxel_list.find(coord);
      if ( it==voxel_list.end() ) {
        throw std::runtime_error("could not find a voxel we defined!!");
      }
      
      int voxelidx = it->second;
      trip2voxelidx[itriplet] = voxelidx;
      voxelidx_to_tripidxlist[voxelidx].push_back( itriplet );
    }

    // ok now we can make the arrays


    // first the voxel coordinate array
    npy_intp* coord_dims = new npy_intp[2];
    coord_dims[0] = (int)nvidx;
    coord_dims[1] = (int)_ndims;
    PyArrayObject* coord_array = (PyArrayObject*)PyArray_SimpleNew( 2, coord_dims, NPY_LONG );
    for ( auto it=voxel_list.begin(); it!=voxel_list.end(); it++ ) {
      int vidx = it->second;
      const std::array<int,3>& coord = it->first;
      for (int j=0; j<_ndims; j++)
        *((long*)PyArray_GETPTR2( coord_array, (int)vidx, j)) = (long)coord[j];
    }

    // the voxel truth label
    bool has_truth = triplet_data._truth_v.size()==triplet_data._triplet_v.size();
    npy_intp* vlabel_dims = new npy_intp[1];
    vlabel_dims[0] = (int)nvidx;
    PyArrayObject* vlabel_array = (PyArrayObject*)PyArray_SimpleNew( 1, vlabel_dims, NPY_LONG );
    int num_true_voxels = 0;
    for ( auto it=voxel_list.begin(); it!=voxel_list.end(); it++ ) {
      int vidx = it->second;
      int truthlabel = 0.;

      if ( has_truth ) {
        // is there a true pixel?
        std::vector<int>& tripidx_v = voxelidx_to_tripidxlist[vidx];
        for ( auto const& tidx : tripidx_v ) {
          if ( triplet_data._truth_v[tidx]==1 ) {
            truthlabel = 1;
          }
        }
        if ( truthlabel==1 )
          num_true_voxels++;
      }
      *((long*)PyArray_GETPTR1( vlabel_array, vidx )) = truthlabel;
    }

    // the triplet to voxel index array
    npy_intp* trip2vidx_dims = new npy_intp[1];
    trip2vidx_dims[0] = (int)trip2voxelidx.size();
    PyArrayObject* trip2vidx_array = (PyArrayObject*)PyArray_SimpleNew( 1, trip2vidx_dims, NPY_LONG );
    for (int itriplet=0; itriplet<(int)trip2voxelidx.size(); itriplet++ ) {
      *((long*)PyArray_GETPTR1( trip2vidx_array, itriplet )) = (long)trip2voxelidx[itriplet];
    }

    // finally the list of triplet indices for each voxel
    PyObject* tripidx_pylist = PyList_New( nvidx );
    for ( int vidx=0; vidx<nvidx; vidx++ ) {

      // make the array
      npy_intp* tidx_dims = new npy_intp[1];
      std::vector<int>& tripidx_v = voxelidx_to_tripidxlist[vidx];
      tidx_dims[0] = (int)tripidx_v.size();
      PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( 1, tidx_dims, NPY_LONG );
      for ( int i=0; i<tidx_dims[0]; i++ ) {
        *((long*)PyArray_GETPTR1( array, i )) = tripidx_v[i];
      }

      int err = PyList_SetItem( tripidx_pylist, (Py_ssize_t)vidx, (PyObject*)array );
      if (err!=0 ) {
        throw std::runtime_error("Error putting voxel's triplet list to pylist");
      }
      Py_DECREF( array );
    }

    // the dictionary
    PyObject *d = PyDict_New();
    PyObject *key_coord     = Py_BuildValue("s", "voxcoord" );
    PyObject *key_label     = Py_BuildValue("s", "voxlabel" );
    PyObject *key_trip2vidx = Py_BuildValue("s", "trip2vidx" );
    PyObject *key_vox2trips = Py_BuildValue("s", "vox2trips_list" );

    PyDict_SetItem( d, key_coord, (PyObject*)coord_array );
    PyDict_SetItem( d, key_label, (PyObject*)vlabel_array );
    PyDict_SetItem( d, key_trip2vidx, (PyObject*)trip2vidx_array );
    PyDict_SetItem( d, key_vox2trips, (PyObject*)tripidx_pylist );

    Py_DECREF( key_coord );
    Py_DECREF( key_label );
    Py_DECREF( key_trip2vidx );
    Py_DECREF( key_vox2trips );
    Py_DECREF( coord_array );
    Py_DECREF( vlabel_array );
    Py_DECREF( trip2vidx_array );
    Py_DECREF( tripidx_pylist );
    
    return d;
  }


}
}
