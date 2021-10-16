#ifndef __LARVOXEL_HIT_MAKER_H__
#define __LARVOXEL_HIT_MAKER_H__


#include <Python.h>
#include "bytesobject.h"
#include <vector>
#include <map>
#include <array>
#include "larlite/DataFormat/larflow3dhit.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/ImageMeta.h"
#include "larcv/core/DataFormat/Image2D.h"

#include "VoxelizeTriplets.h"


namespace larflow {
namespace voxelizer {

  /**
   * @ingroup Voxelizer
   * @class LArVoxelHitMaker
   * @brief Processes output of LArVoxel network and assigns labels to triplets
   *
   * @author Taritree Wongjirad (taritree.wongjirad@tuts.edu)
   * @date $Data 2020/07/22 17:00$
   *
   *
   * Revision history
   * 2020/07/22: Added doxygen documentation. 
   * 
   *
   */  
  class LArVoxelHitMaker : public larcv::larcv_base
  {
  public:
    
    LArVoxelHitMaker()
      : larcv::larcv_base("LArVoxelHitMaker"),
	_hit_score_threshold(0.5)
    {};
    virtual ~LArVoxelHitMaker() {};

    /**
     * @struct voxellabel_t
     * @brief  internal struct storing network output for single voxel
     *
     */
    struct voxeldata_t {            
      voxeldata_t()
      {};

      voxeldata_t( std::array<long,3> co )
	: coord(co)
      {};
      float lm_score;
      std::array<long,3> coord; ///< indices of voxel in tensor
      std::array<float,7> ssnet_class_score;
      std::array<float,6> kp_class_score;
    };

    float _hit_score_threshold;    
    std::map< std::array<long,3>, voxeldata_t > _voxeldata_map;
    VoxelizeTriplets _voxelizer;
    /**
     * \brief reset state and clear member containers
     */
    void clear() {};

    int add_voxel_labels( PyObject* coord_t,
			  PyObject* larmatch_pred_t,
			  PyObject* ssnet_pred_t,
			  PyObject* kp_pred_t );

    void make_labeled_larflow3dhits( const larflow::prep::PrepMatchTriplets& tripletmaker,
				     const std::vector<larcv::Image2D>& adc_v,				     
				     larlite::event_larflow3dhit& output_container );
    void store_2dssnet_score( larcv::IOManager& iolcv,
			      larlite::event_larflow3dhit& larmatch_hit_v );
 
  private:

    static bool _setup_numpy;
    
  };

}
}

#endif
