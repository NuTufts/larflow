#ifndef __LARFLOW_SPATIALEMBED_SPARTIALEMBED3DNETPRODUCTS_H__
#define __LARFLOW_SPATIALEMBED_SPARTIALEMBED3DNETPRODUCTS_H__

#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include <string>

#include "TTree.h"

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
      : larcv::larcv_base("SpatialEmbed3DNetProducts"),
      _tree_simple(nullptr),
      _in_voxelidx_v(nullptr),
      _in_cluster_id(nullptr)
      {};
    virtual ~SpatialEmbed3DNetProducts() {};

    std::vector< std::vector<int> >
      assignVoxelsToCluster( PyObject* voxel_coord_array,
                             PyObject* seed_score_array,
                             PyObject* embedding_array,
                             const std::vector<int>& tensor_dim_lens );

    void bindSimpleOutputVariables( TTree* atree );
    void fillVoxelClusterID( PyObject* voxel_coord_ndarray, PyObject* cluster_idx_ndarray );
    void setTreeBranches( TTree& input_tree );
    PyObject* getEntryDataAsNDarray( int entry );
                                   
  protected:

    TTree* _tree_simple;
    std::vector< std::vector<int> > _voxelidx_v;
    std::vector< int >              _cluster_id;

    std::vector< std::vector<int> >* _in_voxelidx_v;
    std::vector< int >*              _in_cluster_id;
    
  private:

    static bool _setup_numpy;
    
  };
  
}
}

#endif
