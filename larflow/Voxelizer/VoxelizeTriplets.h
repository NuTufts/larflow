#ifndef __VOXELIZE_TRIPLETS_H__
#define __VOXELIZE_TRIPLETS_H__

#include <Python.h>
#include "bytesobject.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <string>
#include <vector>

#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/DataFormat/storage_manager.h"

#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"
#include "larflow/KeyPoints/LoaderKeypointData.h"

namespace larflow {
namespace voxelizer {

  /**
   * @ingroup Voxelizer
   * @class VoxelizeTriplets
   * @brief Take spacepoint proposals from larflow::prep::PrepMatchTriplets and convert into voxels
   *
   * This class takes in space point proposals in the form of (U,V,Y) wire combinations from larflow::prep::PrepMatchTriplets.
   * It assigns those space points to a location in a sparse voxel array defined in the constructor.
   * The class also provides tools to get numpy arrays that help to make associations between spacepoints and voxels.
   * Namely, a map from space point to occupied voxel index and a map from occupied voxel to a list of spacepoint indices 
   * that lie within the voxel.
   * This information is intended for use in training sparse convolutional neural networks whose inputs
   * are the voxels.
   *
   * The voxel grid can be defined using the constructor with arguments.
   * If the default constructor is used, a default grid is setup whose values are
   * relevant for the MicroBooNE LArTPC.
   *
   */
  class VoxelizeTriplets {

  public:

    VoxelizeTriplets();

    VoxelizeTriplets( std::vector<float> origin,
                      std::vector<float> dim_len,
                      float voxel_size );
    ~VoxelizeTriplets() {};

    void process_fullchain( larcv::IOManager& iolcv,
                            std::string adc_producer,
                            std::string chstatus_producer,
                            bool has_mc=false );

  protected:

    int _ndims; ///< number of dimensions of the voxel grid (really only works in 3D)
    std::vector<float> _origin;  ///< origin of the voxel grid in cm
    std::vector<float> _len;     ///< length of the voxel grid in each dimension in cvm
    float _voxel_size;           ///< voxel edge length in cm
    std::vector<int>   _nvoxels; ///< number of voxels in each dimension
    void _define_voxels();

    std::set< std::array<int,3> >      _voxel_set;  ///< set of occupied voxels
    std::map< std::array<int,3>, int > _voxel_list; ///< map from voxel coordinate to voxel index
    std::vector< std::vector<int> >    _voxelidx_to_tripidxlist; ///< voxel index to vector of triplet indices
    std::vector<int>                   _trip2voxelidx; ///< triplet index to voxel index map   

  public:

    larflow::prep::PrepMatchTriplets _triplet_maker; ///< internal triplet maker, used if making from  images only    

    int get_axis_voxel( int axis, float coord ) const;
    
    
    void make_voxeldata( const larflow::prep::PrepMatchTriplets& triplet_data );
    PyObject* make_voxeldata_dict( const larflow::prep::PrepMatchTriplets& triplet_data );
    PyObject* make_voxeldata_dict();

    int make_ssnet_voxel_labels( const larflow::keypoints::LoaderKeypointData& data,
				 PyArrayObject*& ssnet_array,
				 PyArrayObject*& ssnet_weight );
				       
    PyObject* get_full_voxel_labelset_dict( const larflow::keypoints::LoaderKeypointData& data );
    int make_kplabel_arrays( const larflow::keypoints::LoaderKeypointData& data,
			     PyArrayObject* match_array,
			     PyArrayObject*& kplabel_label,
			     PyArrayObject*& kplabel_weight );
    

    /** @brief get the number of total voxels */   
    const std::vector<int>& get_nvoxels() const  { return _nvoxels; };

    /** @brief get the origin of the voxel grid */
    const std::vector<float>& get_origin() const { return _origin; };

    /** @brief get the lengths of each dimension of the voxel grid */
    const std::vector<float>& get_dim_len() const { return _len; };

    /** @brief get the voxel edge length */
    float get_voxel_size() const { return _voxel_size; };

    void set_voxel_size_cm( float width_cm );

    std::vector<int> get_voxel_indices( const std::vector<float>& xyz ) const;

    
  protected:
    
    
  private:

    static bool _setup_numpy;
    
  };
  
}
}

#endif
