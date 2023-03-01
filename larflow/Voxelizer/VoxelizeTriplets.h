#ifndef __VOXELIZE_TRIPLETS_H__
#define __VOXELIZE_TRIPLETS_H__

#include <Python.h>
#include "bytesobject.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <string>
#include <vector>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/SparseTensor3D.h"
#include "larcv/core/DataFormat/Particle.h"
#include "larlite/DataFormat/storage_manager.h"
#include "ublarcvapp/MCTools/SimChannelVoxelizer.h"

#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"
#include "larflow/PrepFlowMatchData/MatchTriplets.h"
#include "larflow/KeyPoints/PrepKeypointData.h"
#include "larflow/KeyPoints/LoaderKeypointData.h"

#include "TPCVoxelData.h"

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
  class VoxelizeTriplets : public larcv::larcv_base {

  public:

    VoxelizeTriplets();

    // VoxelizeTriplets( std::vector<float> origin,
    //                   std::vector<float> dim_len,
    //                   float voxel_size );
    
    ~VoxelizeTriplets() {};

    void process_fullchain( larcv::IOManager& iolcv,
                            std::string adc_producer,
                            std::string chstatus_producer,
                            bool has_mc=false );

  protected:

    int _ndims; ///< number of dimensions of the voxel grid (really only works in 3D)
    std::vector<float> _len;     ///< length of the voxel grid in each dimension in cvm
    void _define_voxels();

    ublarcvapp::mctools::SimChannelVoxelizer _simchan_voxelizer; ///< we use this class to define the voxels in each TPCS

    // std::set< std::array<int,3> >      _voxel_set;  ///< set of occupied voxels
    // std::map< std::array<int,3>, int > _voxel_list; ///< map from voxel coordinate to voxel index
    // std::vector< std::vector<int> >    _voxelidx_to_tripidxlist; ///< voxel index to vector of triplet indices
    // std::vector<int>                   _trip2voxelidx; ///< triplet index to voxel index map

  public:

    std::vector< TPCVoxelData > _voxel_data_v;

  public:

    larflow::prep::PrepMatchTriplets _triplet_maker; ///< internal triplet maker, used if making from  images only    

    int get_axis_voxel( int axis, float coord ) const;
    
    PyObject* get_full_voxel_labelset_dict( const larflow::keypoints::LoaderKeypointData& data );
    
    PyObject* make_full_voxel_labelset_dict( const larflow::voxelizer::TPCVoxelData& voxdata,
					     const larflow::prep::MatchTriplets& tripletdata,
					     const larflow::prep::SSNetLabelData& ssnetdata,
					     const larflow::keypoints::KeypointData& kpdata );

    std::vector< larcv::SparseTensor3D >
    make_mlreco_semantic_label_sparse3d( const larflow::voxelizer::TPCVoxelData& voxdata,
					 const larflow::prep::MatchTriplets& triplet_data,
					 const larflow::prep::SSNetLabelData& ssnetdata );

    std::vector< larcv::SparseTensor3D >
    make_mlreco_cosmicorigin_label_sparse3d( const larflow::voxelizer::TPCVoxelData& voxdata,
					     const larflow::prep::MatchTriplets& triplet_data );
    
    std::vector< larcv::SparseTensor3D >
    make_mlreco_cluster_label_sparse3d( const larflow::voxelizer::TPCVoxelData& voxdata,
					const larflow::prep::MatchTriplets& tripletdata,
					const larflow::keypoints::KeypointData& kpdata,					
					std::vector<larcv::Particle>& particle_v,
					std::vector<larcv::Particle>& rejected_v );

    // TPCVoxelData functions
    larflow::voxelizer::TPCVoxelData make_voxeldata( const larflow::prep::MatchTriplets& triplet_data );

    int fill_tpcvoxeldata_semantic_labels( const larflow::prep::SSNetLabelData& ssnetdata,
					   const larflow::prep::MatchTriplets& triplet_data,
					   larflow::voxelizer::TPCVoxelData& voxeldata );
    
    int fill_tpcvoxeldata_planecharge( const larflow::prep::MatchTriplets& triplet_data,
				       larflow::voxelizer::TPCVoxelData& voxdata );

    int fill_tpcvoxeldata_cosmicorigin( const larflow::prep::MatchTriplets& triplet_data,
					larflow::voxelizer::TPCVoxelData& voxdata );

    int fill_tpcvoxeldata_instance_labels( const larflow::prep::MatchTriplets& tripletdata,
					   larflow::voxelizer::TPCVoxelData& voxdata );

    // Make dictionary of numpy arrays with voxel data
    
    PyObject* make_voxeldata_dict( const larflow::voxelizer::TPCVoxelData& voxdata,
				   const larflow::prep::MatchTriplets& triplet_data );
    PyObject* make_voxeldata_dict();

    int make_ssnet_voxel_label_nparray( const larflow::prep::SSNetLabelData& ssnetdata,
					const larflow::voxelizer::TPCVoxelData& voxeldata,
					PyArrayObject*& ssnet_array,
					PyArrayObject*& ssnet_weight );
    
    PyObject* make_ssnet_dict_labels( const larflow::voxelizer::TPCVoxelData& voxeldata,
				      const larflow::prep::MatchTriplets& data );
				       
    int make_kplabel_arrays( const larflow::keypoints::KeypointData& kpdata,
			     const larflow::voxelizer::TPCVoxelData& voxdata,
			     PyArrayObject* match_array,
			     PyArrayObject*& kplabel_label,
			     PyArrayObject*& kplabel_weight,
			     float sigma=10.0 );
    
    // PyObject* make_kplabel_dict_fromprep( const larflow::keypoints::PrepKeypointData& data,
    // 					  PyObject* voxel_match_array );

    PyObject* make_instance_dict_labels( const larflow::voxelizer::TPCVoxelData& voxdata,
					 const larflow::prep::MatchTriplets& tripletdata );
    
    PyObject* make_origin_dict_labels( const larflow::voxelizer::TPCVoxelData& voxdata,
				       const larflow::prep::MatchTriplets& data );

    // /** @brief get the number of total voxels */   
    // const std::vector<int>& get_nvoxels() const  { return _nvoxels; };

    // /** @brief get the origin of the voxel grid */
    // const std::vector<float>& get_origin() const { return _origin; };

    /** @brief get the lengths of each dimension of the voxel grid */
    const std::vector<float>& get_dim_len() const { return _len; };

    // /** @brief get the voxel edge length */
    // float get_voxel_size() const { return _voxel_size; };

    void set_voxel_size_cm( float width_cm );

    std::vector<long> get_voxel_indices( const std::vector<float>& xyz );

    larcv::Voxel3DMeta make_meta( const TPCVoxelData& voxdata );

    void nu_only( bool only_nu ) { _nu_only=only_nu; };
    
  protected:

    bool _nu_only;
    
  private:

    static bool _setup_numpy;
    
  };
  
}
}

#endif
