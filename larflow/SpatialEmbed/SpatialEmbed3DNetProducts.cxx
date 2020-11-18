#include "SpatialEmbed3DNetProducts.h"
#include <sstream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "TFile.h"

namespace larflow {
namespace spatialembed {

  bool SpatialEmbed3DNetProducts::_setup_numpy = false;
  
  /**
   * @brief assign voxels to clusters using spatialembednet output
   *
   */
  std::vector<std::vector<int>>
  SpatialEmbed3DNetProducts::assignVoxelsToCluster( PyObject* voxel_coord_ndarray,
						    PyObject* seed_score_ndarray,
						    PyObject* embedding_ndarray,
						    const std::vector<int>& tensor_dim_lens)
  {

    if ( !_setup_numpy ) {
      import_array1( std::vector<std::vector<int> >() );
      _setup_numpy = true;
    }

    // get c-array view of data
    
    // coordinate array
    PyArray_Descr *descr_long  = PyArray_DescrFromType(NPY_LONG);
    PyArray_Descr *descr_float = PyArray_DescrFromType(NPY_FLOAT);
    
    npy_intp coord_dims[2];
    long **coord_carray;
    if ( PyArray_AsCArray( &voxel_coord_ndarray, (void**)&coord_carray, coord_dims, 2, descr_long )<0 ) {
      LARCV_CRITICAL() << "Cannot get carray for voxel coord tensor" << std::endl;
      throw std::runtime_error("Cannot get carray for voxel coord tensor");
    }

    npy_intp seed_dims[2];
    float **seed_carray;
    if ( PyArray_AsCArray( &seed_score_ndarray, (void**)&seed_carray, seed_dims, 2, descr_float )<0 ) {
      LARCV_CRITICAL() << "Cannot get carray for seed score tensor" << std::endl;
      throw std::runtime_error("Cannot get carray for seed score tensor");
    }

    npy_intp embed_dims[2];
    float **embed_carray;
    if ( PyArray_AsCArray( &embedding_ndarray, (void**)&embed_carray, embed_dims, 2, descr_float )<0 ) {
      LARCV_CRITICAL() << "Cannot get carray for embedding tensor" << std::endl;
      throw std::runtime_error("Cannot get carray for embedding tensor");
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

      // form clusters:
      // (1) identify highest seed voxel
      // (2) use it to define the centroid and bandwith
      // (3) accept voxels using score

      
      

    }

    
  }

  /**
   * @brief save voxel coord and cluster id to the root tree
   *
   */
  void SpatialEmbed3DNetProducts::fillVoxelClusterID( PyObject* voxel_coord_ndarray,
                                                      PyObject* cluster_idx_ndarray,
                                                      PyObject* embed_pos_ndarray )
  {

    if ( !_setup_numpy ) {
      import_array();
      _setup_numpy = true;
    }

    if ( !_tree_simple ) {
      std::stringstream ss;
      ss << "Output tree not yet set. Make sure to call 'bindSimpleOutputVariables' first." << std::endl;
      LARCV_CRITICAL() << ss.str();
      throw std::runtime_error(ss.str());
    }
    
    // get c-array view of data
    
    // coordinate array
    PyArray_Descr *descr_long  = PyArray_DescrFromType(NPY_LONG);
        
    npy_intp coord_dims[2];
    long **coord_carray;
    if ( PyArray_AsCArray( &voxel_coord_ndarray, (void**)&coord_carray, coord_dims, 2, descr_long )<0 ) {
      LARCV_CRITICAL() << "Cannot get carray for voxel coord tensor" << std::endl;
      throw std::runtime_error("Cannot get carray for voxel coord tensor");
    }

    npy_intp cluster_dims[1];
    long *cluster_carray;
    if ( PyArray_AsCArray( &cluster_idx_ndarray, (void*)&cluster_carray, cluster_dims, 1, descr_long )<0 ) {
      LARCV_CRITICAL() << "Cannot get carray for voxel cluster tensor" << std::endl;
      throw std::runtime_error("Cannot get carray for voxel cluster tensor");
    }

    PyArray_Descr *descr_float  = PyArray_DescrFromType(NPY_FLOAT);    
    npy_intp embed_dims[2];
    float **embed_carray = nullptr;
    if ( embed_pos_ndarray && PyArray_AsCArray( &embed_pos_ndarray, (void**)&embed_carray, embed_dims, 2, descr_float )<0 ) {
      LARCV_CRITICAL() << "Cannot get carray for embed position tensor" << std::endl;
      throw std::runtime_error("Cannot get carray for embed position tensor");      
    }

    // determine the number of batches by find the maximum batch index
    // also make a list of indices for each batch
    std::map<int,std::vector<int> > batch_indices;    
    for (int i=0; i<coord_dims[0]; i++) {

      int ibatch = coord_carray[i][3];

      auto it=batch_indices.find( ibatch );

      if ( it==batch_indices.end() ) {
	batch_indices[ibatch] = std::vector<int>();
	batch_indices[ibatch].reserve( coord_dims[0] );
      }

      batch_indices[ibatch].push_back( i );
    }
    int nbatches = batch_indices.size();

    // loop over batches
    int ibatch = 0;
    for (auto it=batch_indices.begin(); it!=batch_indices.end(); it++) {
      
      // array indices for this batch
      auto const& b_indices = it->second;

      // clear arrays
      _voxelidx_v.clear();
      _voxelidx_v.reserve( b_indices.size() );
      _cluster_id.resize( b_indices.size(), 0 );
      _embed_pos_v.clear();
      _embed_pos_v.reserve( b_indices.size() );      

      for ( auto const& i : b_indices ) {
        std::vector<int> coord_i = { (int)coord_carray[i][0], (int)coord_carray[i][1], (int)coord_carray[i][2] };
        _voxelidx_v.push_back( coord_i );
        _cluster_id[i] = cluster_carray[i];
        if ( embed_pos_ndarray ) {
          std::vector<float> embed_i = { (float)embed_carray[i][0], (float)embed_carray[i][1], (float)embed_carray[i][2] };
          _embed_pos_v.push_back( embed_i );
        }
      }

      _tree_simple->Fill();
      ibatch++;
    }

    LARCV_INFO() << "filled " << ibatch << " entries." << std::endl;
  }

  void SpatialEmbed3DNetProducts::bindSimpleOutputVariables( TTree* atree )
  {
    _tree_simple = atree;
    atree->Branch("voxelindex",&_voxelidx_v);
    atree->Branch("cluster_id",&_cluster_id);
    atree->Branch("embed_pos",&_embed_pos_v);        
  }

  void SpatialEmbed3DNetProducts::setTreeBranches( TTree& input_tree ) {
    _tree_simple = &input_tree;
    _tree_simple->SetBranchAddress( "voxelindex", &_in_voxelidx_v );
    _tree_simple->SetBranchAddress( "cluster_id", &_in_cluster_id );
    _tree_simple->SetBranchAddress( "embed_pos", &_in_embed_pos_v );
  }

  /**
   * @brief return a numpy array for the entry
   *
   */
  PyObject* SpatialEmbed3DNetProducts::getEntryDataAsNDarray( int entry )
  {

    if ( !SpatialEmbed3DNetProducts::_setup_numpy ) {
      import_array1(0);
      SpatialEmbed3DNetProducts::_setup_numpy = true;
    }

    // DECLARE TENSORS and dict keys
    unsigned long bytes = _tree_simple->GetEntry(entry);
    if ( bytes==0 ) {
      Py_INCREF(Py_None);
      return Py_None;
    }

    size_t nvoxels = _in_voxelidx_v->size();
    
    // coord tensor
    npy_intp coord_t_dim[] = { (long int)nvoxels, 4 };
    PyArrayObject* coord_t = (PyArrayObject*)PyArray_SimpleNew( 2, coord_t_dim, NPY_LONG );
    
    // FILL TENSORS
    for (size_t i=0; i<nvoxels; i++ ) {
      for (size_t j=0; j<3; j++)
        *((long*)PyArray_GETPTR2(coord_t,i,j)) = (*_in_voxelidx_v)[i][j];
      *((long*)PyArray_GETPTR2(coord_t,i,3)) = (*_in_cluster_id)[i];
    }
          
    return (PyObject*)coord_t;
    
  }

  /**
   * @brief return a numpy array for the entry
   *
   */
  PyObject* SpatialEmbed3DNetProducts::getEntryEmbedPosAsNDarray( int entry )
  {

    if ( !SpatialEmbed3DNetProducts::_setup_numpy ) {
      import_array1(0);
      SpatialEmbed3DNetProducts::_setup_numpy = true;
    }

    // DECLARE TENSORS and dict keys
    unsigned long bytes = _tree_simple->GetEntry(entry);
    if ( bytes==0 ) {
      Py_INCREF(Py_None);
      return Py_None;
    }

    size_t nvoxels = _in_embed_pos_v->size();
    
    // embed tensor
    npy_intp embed_t_dim[] = { (long int)nvoxels, 4 };
    PyArrayObject* embed_t = (PyArrayObject*)PyArray_SimpleNew( 2, embed_t_dim, NPY_FLOAT );
    
    // FILL TENSORS
    for (size_t i=0; i<nvoxels; i++ ) {
      for (size_t j=0; j<3; j++)
        *((float*)PyArray_GETPTR2(embed_t,i,j)) = (*_in_embed_pos_v)[i][j];
      *((long*)PyArray_GETPTR2(embed_t,i,3)) = (*_in_cluster_id)[i];
    }
          
    return (PyObject*)embed_t;
    
  }
  
  
}
}
