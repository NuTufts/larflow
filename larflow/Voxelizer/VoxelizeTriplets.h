#ifndef __VOXELIZE_TRIPLETS_H__
#define __VOXELIZE_TRIPLETS_H__

#include <Python.h>
#include "bytesobject.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <string>
#include <vector>
#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"

namespace larflow {
namespace voxelizer {

  class VoxelizeTriplets : public larcv::larcv_base {

  public:

    VoxelizeTriplets()
      : larcv::larcv_base("VoxelizeTriplets")
      {};
    
    VoxelizeTriplets( std::vector<float> origin,
                      std::vector<float> dim_len,
                      float voxel_size );
    ~VoxelizeTriplets() {};

  protected:

    int _ndims;
    std::vector<float> _origin;
    std::vector<float> _len;
    float _voxel_size;
    std::vector<int>   _nvoxels;

  public:

    int get_axis_voxel( int axis, float coord ) const;
    
    PyObject* make_voxeldata_dict( const larflow::PrepMatchTriplets& triplet_data );

    const std::vector<int>& get_nvoxels() const  { return _nvoxels; };
    const std::vector<float>& get_origin() const { return _origin; };
    const std::vector<float>& get_dim_len() const { return _len; };
    float get_voxel_size() const { return _voxel_size; };


  private:

    static bool _setup_numpy;
    
  };
  
}
}

#endif
