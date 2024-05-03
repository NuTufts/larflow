#ifndef __LARFLOW_OPTICALMODEL_FLASHMATCHDATAPREP_H__
#define __LARFLOW_OPTICALMODEL_FLASHMATCHDATAPREP_H__

#include <Python.h>
#include "bytesobject.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

/**
 * @ingroup larflow_opticalmodel
 * @class OpModelMCDataPrep
 *
 * @brief Prep Simulation-derived data. 
 * 
 * This class is an extension of the ublarcvapp::mctools::FlashMatcherV2 class.
 *
 * We absorb its algorithms to build opreco-mctrack/mcshower matches.
 * But we extend it to tag matches where the charge voxel data from
 * larflow::voxelizer::Voxelizer seems to be missing voxels.
 * 
 */

#include "larlite/DataFormat/storage_manager.h"
#include "ublarcvapp/MCTools/FlashMatcherV2.h"
#include "larflow/Voxelizer/VoxelizeTriplets.h"


namespace larflow {
namespace opticalmodel {

  class OpModelMCDataPrep : public ::ublarcvapp::mctools::FlashMatcherV2 {
    
  public:

    OpModelMCDataPrep()
      : ublarcvapp::mctools::FlashMatcherV2()
    {};
    virtual ~OpModelMCDataPrep() {};

    void process( larlite::storage_manager& mgr,
		  const larflow::voxelizer::VoxelizeTriplets& voxelizer );
    
    void tagBadFlashMatches( const larflow::voxelizer::VoxelizeTriplets& voxelizer,
			     larlite::storage_manager& ioll );

    void filterBadMCDataMatches();

    void getChargeVoxelsForFlash( const ublarcvapp::mctools::RecoFlash_t& recoflash,
				  const larflow::voxelizer::VoxelizeTriplets& voxelizer,
				  std::vector< std::vector<int>   >& voxel_indices,
				  std::vector< std::vector<float> >& voxel_features );
    
    std::vector<int> flash_isgood;
    std::vector<float> flash_track_frac_intpc_w_charge;
    std::vector<float> flash_track_frac_intpc;
    
    void printMatches() const;

    std::vector<float> get_recoflash_pe( const ublarcvapp::mctools::RecoFlash_t& recoflash,
					 larlite::storage_manager& ioll );
    
    PyObject* make_opmodel_data_dict( const ublarcvapp::mctools::RecoFlash_t& recoflash,
				      const larflow::voxelizer::VoxelizeTriplets& voxelizer,
				      larlite::storage_manager& ioll );

    virtual void clear();
    
  private:
    
    static bool _setup_numpy;
    
  };
  
}
}
  
#endif
