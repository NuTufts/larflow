#ifndef __LARFLOW_SPATIALEMBED_SPARTIALEMBED3DNETPRODUCTS_H__
#define __LARFLOW_SPATIALEMBED_SPARTIALEMBED3DNETPRODUCTS_H__

#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include <string>

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larflow/Voxelizer/VoxelizeTriplets.h"

namespace larflow {
namespace spatialembed {

  /** 
   * @class SpatialEmbed3DNetProducts
   * @ingroup SpatialEmbed
   * @brief Integrate spatialembed3d network outputs into larmatch products
   *
   */  
  class SpatialEmbed3DNetProducts : public larcv::larcv_base {

  public:

    SpatialEmbed3DNetProducts()
      : larcv::larcv_base("SpatialEmbed3DNetProducts")
      {};
    virtual ~SpatialEmbed3DNetProducts() {};

    std::vector<int> assignVoxelsToCluster( PyObject* voxel_coord_array,
					    PyObject* seed_score_array,
					    PyObject* embedding_array,
					    const std::vector<int>& tensor_dim_lens );

  protected:

    
  private:

    static bool _setup_numpy;
    
  };
  
}
}

#endif
