#ifndef __LARFLOW_VOXELIZER_TPCVOXELDATA__
#define __LARFLOW_VOXELIZER_TPCVOXELDATA__

#include <vector>
#include <set>
#include <map>

namespace larflow {
namespace voxelizer {

  /**
   * @ingroup Voxelizer
   * @class VoxelData
   * @brief Voxel information for one TPC produced by VoxelizeTriplets
   *
   */
  class TPCVoxelData {

  public:
    
    TPCVoxelData() {};
    virtual ~TPCVoxelData() {};

    // meta data for the label of the TPC
    int _tpcid;
    int _cryoid;
    
    // meta of the tensor we've defined. how to situate it against real detector space.
    int _ndims; ///< number of dimensions of the voxel grid (really only works in 3D)
    std::vector<float> _origin;  ///< origin of the voxel grid in cm
    std::vector<float> _len;     ///< length of the voxel grid in each dimension in cvm
    float _voxel_size;           ///< voxel edge length in cm
    std::vector<int>   _nvoxels; ///< number of voxels in each dimension

    // actual data
    std::set< std::array<long,3> >      _voxel_set;  ///< set of occupied voxels
    std::map< std::array<long,3>, long > _voxel_list; ///< map from voxel coordinate to voxel index

    // map to original triplet data that contains the detector info and labels
    std::vector< std::vector<int> >    _voxelidx_to_tripidxlist; ///< voxel index to vector of triplet indices
    std::vector<int>                   _trip2voxelidx; ///< triplet index to voxel index map   

  };

}
}

#endif
