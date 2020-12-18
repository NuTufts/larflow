#include "DataLoader.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <iostream>

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
    //    tkeypoint = new TChain("keypointlabels");
    //    tssnet    = new TChain("ssnetlabels");

    for (auto const& infile : input_files ) {
      std::cout << "add " << infile << " to chains" << std::endl;
      tclusterflash->Add(infile.c_str());
      //      tkeypoint->Add(infile.c_str());
      //      tssnet->Add(infile.c_str());
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
   * @return number of bytes loaded from the tkeypoint tree data. returns 0 if end of file or error.
   */
  unsigned long DataLoader::load_entry( int entry )
  {
    unsigned long bytes = tclusterflash->GetEntry(entry);
    
    std::cout << "Got entry " << entry << std::endl;
    
    
    return bytes;


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

  /**
   * @brief return a ground truth data, return a subsample of all truth matches
   *
   * returns a python dictionary. The dictionary contents are:
   * \verbatim embed:rst:leading-asterisk
   *  * "matchtriplet":numpy array with sparse image indices for each place, representing pixels a candidate space point project into
   *  * "match_weight":weight of "matchtriplet" examples
   *  * "positive_indices":indices of entries in "matchtriplet" array that correspond to good/true spacepoints
   *  * "ssnet_label":class label for space point
   *  * "ssnet_top_weight":weight based on topology (i.e. on boundary, near nu-vertex)
   *  * "ssnet_class_weight":weight based on class frequency
   *  * "kplabel":keypoint score numpy array
   *  * "kplabel_weight":weight for keypoint label
   *  * "kpshift":shift in 3D from space point position to nearest keypoint
   * \endverbatim
   *
   * @param[in]  num_max_samples maximum number of space points for which we return ground truth data
   * @param[out] nfilled The number of space points, for which we actually return data
   * @param[in]  withtruth withtruth If true, return info on whether space point is true (i.e. good)
   * @return Python dictionary object with various numpy arrays
   *                        
   */

  /*
    
    PyObject* DataLoader::sample_data( const int& num_max_samples,
    int& nfilled,
    bool withtruth ) {
  */
  PyObject* DataLoader::make_arrays() {
    
    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }
    
    // CLUSTER INFO ARRAYS
    PyArrayObject* voxel_coord_array = nullptr; // row, col, depth 
    PyArrayObject* voxel_feature_array = nullptr; // charge values
    make_clusterinfo_arrays( voxel_coord_array, voxel_feature_array );
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

  
  int DataLoader::make_clusterinfo_arrays( PyArrayObject*& voxel_coord_array,
					   PyArrayObject*& voxel_feature_array ) {


    // make coordinate array
    int coord_nd = 2;
    npy_intp coord_dims[] = { 10, 4 };
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
    /*
    for (int i = 0; i < coord_dims[1]; i++ ) {
      long label = voxel_col->at( i );
      *((long*)PyArray_GETPTR2(voxel_coord_array,1,i)) = (long)label;
    }
    
    for (int i = 0; i < coord_dims[1]; i++ ) {
      long label = voxel_depth->at( i );
      *((long*)PyArray_GETPTR2(voxel_coord_array,2,i)) = (long)label;
    }
    
    for (int i = 0; i < coord_dims[1]; i++ ) {
      long label = 0;
      *((long*)PyArray_GETPTR2(voxel_coord_array,3,i)) = (long)label;
    }
    */
    
    // make feature (charge) array
    int feature_nd = 1;
    npy_intp feature_dims[] = { 10 };
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
