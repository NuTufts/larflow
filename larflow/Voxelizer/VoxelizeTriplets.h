#ifndef __VOXELIZE_TRIPLETS_H__
#define __VOXELIZE_TRIPLETS_H__

#include <Python.h>
#include "bytesobject.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <string>
#include <vector>

#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"

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

    void process_fullchain( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

  protected:

    int _ndims;
    std::vector<float> _origin;
    std::vector<float> _len;
    float _voxel_size;
    std::vector<int>   _nvoxels;

    std::set< std::array<int,3> >      _voxel_set;  ///< set of occupied voxels
    std::map< std::array<int,3>, int > _voxel_list; ///< map from voxel coordinate to voxel index
    std::vector< std::vector<int> >    _voxelidx_to_tripidxlist; // voxel index to vector of triplet indices
    std::vector<int>                   _trip2voxelidx; ///< triplet index to voxel index map
    

    larflow::PrepMatchTriplets _triplet_maker; ///< internal triplet maker, used if making from  images only

  public:

    int get_axis_voxel( int axis, float coord ) const;
    
    
    void make_voxeldata( const larflow::PrepMatchTriplets& triplet_data );
    PyObject* make_voxeldata_dict( const larflow::PrepMatchTriplets& triplet_data );
    PyObject* make_voxeldata_dict();    

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
