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
      _in_cluster_id(nullptr),
      _in_embed_pos_v(nullptr),
      _in_seed_v(nullptr)      
      {};
    virtual ~SpatialEmbed3DNetProducts() {};

    std::vector< std::vector<int> >
      assignVoxelsToCluster( PyObject* voxel_coord_array,
                             PyObject* seed_score_array,
                             PyObject* embedding_array,
                             const std::vector<int>& tensor_dim_lens );

    void bindSimpleOutputVariables( TTree* atree );
    void fillVoxelClusterID( PyObject* voxel_coord_ndarray,
                             PyObject* cluster_idx_ndarray,
                             PyObject* embed_pos_ndarray=nullptr,
                             PyObject* seed_ndarray=nullptr );
    void setTreeBranches( TTree& input_tree );
    PyObject* getEntryDataAsNDarray( int entry );
    PyObject* getEntryEmbedPosAsNDarray( int entry );
    PyObject* getEntrySeedScoreAsNDarray( int entry );        
                                   
  protected:

    TTree* _tree_simple;
    std::vector< std::vector<int> >   _voxelidx_v;  ///< outer vector over voxels, inner vector over tensor dims
    std::vector< std::vector<int> >   _cluster_id;  ///< outer vector over classes, inner vector over voxels
    std::vector< std::vector<float> > _embed_pos_v; ///< outer vector over voxels, inner vector is 3-vector
    std::vector< std::vector<float> > _seed_v;      ///< outer vector over voxels, innver vector over classes

    std::vector< std::vector<int> >*   _in_voxelidx_v;
    std::vector< std::vector<int> >*   _in_cluster_id;
    std::vector< std::vector<float> >* _in_embed_pos_v;
    std::vector< std::vector<float> >* _in_seed_v;
    
  private:

    static bool _setup_numpy;
    
  };
  
}
}

#endif
