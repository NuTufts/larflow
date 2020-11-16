#include "SpatialEmbed3DNetProducts.h"

namespace larflow {
namespace spatialembed {

  /**
   * @brief assign voxels to clusters using spatialembednet output
   *
   */
  std::vector<std::vector<int>>
  SpatialEmbed3DNetProducts::assignVoxelsToCluster( PyObject* voxel_coord_ndarray,
						    PyObject* seed_score_ndarray,
						    PyObject* embedding_ndarray,
						    const std::vector<int>& tensor_dim_lens )
  {

    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    // coordinate array
    PyArray_Descr *descr_long = PyArray_DescrFromType(NPY_LONG);
    npy_intp coord_dims[2];
    long **coord_carray;
    if ( PyArray_AsCArray( &voxel_coord_ndarray, (void**)&coord_carray, coord_dims, 2, descr_long )<0 ) {
      LARCV_CRITICAL() << "Cannot get carray for voxel coord tensor" << std::endl;
      throw std::runtime_error("Cannot get carray for voxel coord tensor");
    }

    // determine the number of batches by find the maximum batch index
    // also make a list of indices for each batch
    std::map<int,std::vector<int> > batch_indices;
    int nbatches = 0;
    for (int i=0; i<coord_dims[0]; i++) {

      int ibatch = coord_carray[i][3];

      auto it=batch_indices.find( ibatch );

      if ( it==batch_indices.end() ) {
	batch_indices[ibatch] = std::vector<int>();
	batch_indices[ibatch].reserve( coord_dims[0] );
      }

      batch_indices[ibatch].push_back( i );
      
      if ( coord_carray[i][3]>nbatches )
	nbatches = coord_carray[i][3];
    }
    nbatches++;

    
    // make output array
    std::vector< std::vector<int> > batch_output_v(nbatches);

    // loop over batches
    for (int ibatch=0; ibatch<nbatches; ibatch++) {
      // array indices for this batch
      auto const& b_indices = batch_indices[ibatch];

      // output array
      std::vector<int>& cluster_id = batch_output_v[ibatch];
      cluster_id.resize(b_indices.size(),-1);

      

    }

    
  }
  
  
}
}
