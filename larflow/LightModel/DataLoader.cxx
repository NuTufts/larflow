#include "DataLoader.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <iostream>

#include "TRandom3.h"

namespace larflow {
namespace lightmodel {

  bool DataLoader::_setup_numpy = false;
  
  /**
   * @brief constructor given list of input files
   *
   * @param[in] input_v List of paths to input ROOT files containing ground truth data
   *
   */
  DataLoader::DataLoader( std::vector<std::string>& input_v )
    : tclusterflash(nullptr)
  {
    input_files.clear();
    input_files = input_v;
    load_tree();
  }

  DataLoader::~DataLoader()
  {
    if ( tclusterflash ) delete tclusterflash;
  }

  /**
   * @brief load TTree class data members and define TBranch variables
   *
   */
  void DataLoader::load_tree() {
    std::cout << "[DataLoader::load_tree()]" << std::endl;

    tclusterflash  = new TChain("preppedTree");

    for (auto const& infile : input_files ) {
      std::cout << "add " << infile << " to chains" << std::endl;
      tclusterflash->Add(infile.c_str());
    }

    voxel_row = 0;
    voxel_col = 0;
    voxel_depth = 0;
    voxel_charge = 0;
    flash_vector = 0;
    
    tclusterflash->SetBranchAddress(  "voxel_row",           &voxel_row );
    tclusterflash->SetBranchAddress(  "voxel_col",           &voxel_col );
    tclusterflash->SetBranchAddress(  "voxel_depth",           &voxel_depth );
    tclusterflash->SetBranchAddress(  "voxel_charge",           &voxel_charge );
    tclusterflash->SetBranchAddress(  "flash_vector",           &flash_vector );

    std::cout << "Initialized tree" << std::endl;
    
  }

  /**
   * @brief load event data for the different trees
   *
   * @param[in] entry number
   * @return number of bytes loaded from tree data. returns 0 if end of file or error.
   */
  unsigned long DataLoader::load_entry( int entry )
  {
    unsigned long bytes = tclusterflash->GetEntry(entry);

    if ( !bytes ) {
      throw std::runtime_error("out of file-bounds");
    }
    
    std::cout << "Got entry " << entry << std::endl;

    // after grabbing entry, check size of vectors
    size = voxel_row->size();

    std::cout << "Voxel vector size: " << size << std::endl;
    
    return bytes;

  }

  // load entry and return some data
  // for use in making batches
  DataLoader::ClusterFlashPair_t DataLoader::getTreeEntry(int entry) {
    
    unsigned long bytes = tclusterflash->GetEntry(entry);
    
    if ( !bytes ) {
      throw std::runtime_error("out of file-bounds");
    }

    size = voxel_row->size();
    _current_entry = entry;
    
    DataLoader::ClusterFlashPair_t data;    
    data.voxel_row_v = *voxel_row;
    data.voxel_col_v = *voxel_col;
    data.voxel_depth_v = *voxel_depth;
    data.voxel_charge_v = *voxel_charge;
    data.flash_vector_v = *flash_vector;
    
    return data;

  }

  /**
   * @brief get total entries
   *
   * @return number of entries in the tclusterflash ROOT tree (chain)
   */
  unsigned long DataLoader::GetEntries()
  {
    return tclusterflash->GetEntries();
  }

  // uses info from one entry of tree
  PyObject* DataLoader::make_arrays() {
    
    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }
    
    // CLUSTER INFO ARRAYS: COORD & FEATURE TENSOR
    PyArrayObject* voxel_coord_array = nullptr; // row, col, depth 
    PyArrayObject* voxel_feature_array = nullptr; // charge values
    make_clusterinfo_arrays( voxel_coord_array, voxel_feature_array, size );
    PyObject *coord_array_key = Py_BuildValue("s","coord_array");
    PyObject *charge_array_key = Py_BuildValue("s","charge_array");
    
    // FLASH INFO ARRAY
    PyArrayObject* flashinfo_label = nullptr;
    make_flashinfo_arrays( flashinfo_label );
    PyObject *flash_label_key = Py_BuildValue("s","flash_info");

    std::cout << "Made arrays n all that" << std::endl;

    PyObject *d = PyDict_New();
    PyDict_SetItem(d, flash_label_key, (PyObject*)flashinfo_label);
    PyDict_SetItem(d, coord_array_key, (PyObject*)voxel_coord_array);
    PyDict_SetItem(d, charge_array_key, (PyObject*)voxel_feature_array);

    Py_DECREF(flashinfo_label);
    Py_DECREF(voxel_coord_array);
    Py_DECREF(voxel_feature_array);
    
    //    return make_flashinfo_arrays( flashinfo_label );
    return d;
    
  }

  // Make arrays given a batch
  PyObject* DataLoader::make_arrays( const std::vector< ClusterFlashPair_t >& data_v ) const
  {

    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    size_t nbatches = data_v.size(); // number of elements (clusterflash pairs) in the batch
    size_t nvoxels_tot = 0; // total number of voxels (this is our new "N")
    
    std::cout << "Converting data for " << nbatches << " clusters" << std::endl;

    for (size_t ibatch=0; ibatch < nbatches; ibatch++) { // for each element in the batch
      
      auto const& clusterdata = data_v[ibatch]; // grab 1 cluster
      size_t nvoxels = clusterdata.voxel_row_v.size(); // N for that 1 cluster
      nvoxels_tot += nvoxels; // total added up N (will be dimension for coord + feature array for batch)
      
    }

    std::cout << "Converting data for " << nvoxels_tot << " total voxels into numpy arrays" << std::endl;

    // COORD TENSOR: CONTAINS voxel row, col, depth, batch
    npy_intp coord_t_dim[] = { (long int)nvoxels_tot, 4 };
    PyArrayObject* coord_t = (PyArrayObject*)PyArray_SimpleNew( 2, coord_t_dim, NPY_LONG );
    PyObject *coord_t_key = Py_BuildValue("s", "coord_t");

    
    // FEATURE TENSOR: voxel charge
    npy_intp feat_t_dim[] = { (long int)nvoxels_tot };
    PyArrayObject* feat_t = (PyArrayObject*)PyArray_SimpleNew( 1, feat_t_dim, NPY_FLOAT );
    PyObject *feat_t_key = Py_BuildValue("s", "feat_t");

    // FLASHINFO TENSOR: 32 flash values for the PMTs
    npy_intp flash_t_dim[] = { 32*(int)nbatches };
    PyArrayObject* flash_t = (PyArrayObject*)PyArray_SimpleNew( 1, flash_t_dim, NPY_FLOAT );
    PyObject *flash_t_key = Py_BuildValue("s", "flash_t");

    // Loop through and fill arrays
    size_t nvoxels_filled = 0;
    
    for ( size_t ibatch=0; ibatch<nbatches; ibatch++ ) { // for each cluster

      auto const& clusterdata = data_v[ibatch];
      size_t nvoxels = clusterdata.voxel_row_v.size(); // N
      std::cout << "The number of voxels N for this cluster is: " << nvoxels << std::endl;

      // fill coord tensor
      for (size_t i = 0; i < nvoxels; i++ ) {

	long row = clusterdata.voxel_row_v.at(i);
	long col = clusterdata.voxel_col_v.at(i);
	long depth = clusterdata.voxel_depth_v.at(i);
	long batch = ibatch;
	*((long*)PyArray_GETPTR2(coord_t,nvoxels_filled+i,0)) = (long)row;
	*((long*)PyArray_GETPTR2(coord_t,nvoxels_filled+i,1)) = (long)col;
	*((long*)PyArray_GETPTR2(coord_t,nvoxels_filled+i,2)) = (long)depth;
	*((long*)PyArray_GETPTR2(coord_t,nvoxels_filled+i,3)) = (long)batch;

      }

      // fill feature tensor
      for (size_t i = 0; i < nvoxels; i++ ) {

	float label = clusterdata.voxel_charge_v.at( i );
	*((float*)PyArray_GETPTR1(feat_t,nvoxels_filled+i)) = (float)label;
	
      }

      // fill flash tensor
      for (size_t i = 0; i < 32; i++ ) {
	
        float label = clusterdata.flash_vector_v.at( i );
        *((float*)PyArray_GETPTR1(flash_t,32*ibatch+i)) = (float)label;
	
      }
      
      nvoxels_filled += nvoxels;
      
    }

    PyArray_ENABLEFLAGS(coord_t,      NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS(feat_t,       NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS(flash_t,       NPY_ARRAY_OWNDATA);

    // Create and fill dictionary
    PyObject *d = PyDict_New();
    PyDict_SetItem(d, coord_t_key,      (PyObject*)coord_t);
    PyDict_SetItem(d, feat_t_key,       (PyObject*)feat_t);
    PyDict_SetItem(d, flash_t_key,       (PyObject*)flash_t);

    Py_DECREF(coord_t_key);
    Py_DECREF(feat_t_key);
    Py_DECREF(flash_t_key);
    
    return d;
    
  }
  
  PyObject* DataLoader::getTrainingDataBatch(int batch_size) {
    
    std::vector< DataLoader::ClusterFlashPair_t > data_batch;
    data_batch.reserve(batch_size);

    _num_entries = tclusterflash->GetEntries();
    _rand = new TRandom3(0);
    
    int ntries = 0;

    // for an arb number of tries
    while ( ntries<batch_size*10 && data_batch.size()<batch_size ) {

      std::cout << "batch_size: " << batch_size << std::endl;
      std::cout << "data_batch.size(): " << data_batch.size() << std::endl;

      // try shuffle, dumb
      try {
	//	std::cout << "_num_entries: " <<  _num_entries << std::endl;
	_current_entry = _rand->Integer(_num_entries);
	auto data = getTreeEntry(_current_entry);
	if (data.voxel_row_v.size() > 0) 
	  data_batch.emplace_back( std::move(data) );
	} catch (...) {
	std::cout << "CATCH!" << std::endl;
	_current_entry = 0;
	}
      
	ntries++;
    }

    if ( data_batch.size()==batch_size ) {
      std::cout << "data_batch.size()==batch_size, making arrays now..." << std::endl;
      return make_arrays( data_batch );
    }
    
    Py_INCREF(Py_None);
    return Py_None;
    
  }
  
  int DataLoader::make_clusterinfo_arrays( PyArrayObject*& voxel_coord_array,
					   PyArrayObject*& voxel_feature_array,
					   int N) {


    // make coordinate array
    int coord_nd = 2;
    npy_intp coord_dims[] = { N, 4 };
    voxel_coord_array = (PyArrayObject*)PyArray_SimpleNew( coord_nd, coord_dims, NPY_LONG );
    
    // loop through each coord value

    for (int j = 0; j < coord_dims[0]; j++) {
      
      //      for (int i = 0; i < coord_dims[1]; i++ ) {
      long row = voxel_row->at( j );
      long col = voxel_col->at( j );
      long depth = voxel_depth->at( j );
      long batch = 0;
      *((long*)PyArray_GETPTR2(voxel_coord_array,j,0)) = (long)row;
      *((long*)PyArray_GETPTR2(voxel_coord_array,j,1)) = (long)col;
      *((long*)PyArray_GETPTR2(voxel_coord_array,j,2)) = (long)depth;
      *((long*)PyArray_GETPTR2(voxel_coord_array,j,3)) = (long)batch;
      //      }
      
    }

    // make feature (charge) array
    int feature_nd = 1;
    npy_intp feature_dims[] = { N };
    voxel_feature_array = (PyArrayObject*)PyArray_SimpleNew( feature_nd, feature_dims, NPY_FLOAT );

    // loop through each charge values
    for (int i = 0; i < feature_dims[0]; i++ ) {

      float label = voxel_charge->at( i );

      *((float*)PyArray_GETPTR1(voxel_feature_array,i)) = (float)label;
      
    }
    
    return 0;
  }
  
  
  int DataLoader::make_flashinfo_arrays( PyArrayObject*& flashinfo_label ) {

    // make flash array
    int flashinfo_nd = 1;
    npy_intp flashinfo_dims[] = { 32 };
    flashinfo_label = (PyArrayObject*)PyArray_SimpleNew( flashinfo_nd, flashinfo_dims, NPY_FLOAT );

    // loop through the 32 PMT values
    for (int i = 0; i < flashinfo_dims[0]; i++ ) {

      float label = flash_vector->at( i );

      *((float*)PyArray_GETPTR1(flashinfo_label,i)) = (float)label;
      
    }

    std::cout << "End of make array fn" << std::endl;
    
    return 0;
  }

}
}
