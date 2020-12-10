#include "DataLoader.h"

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
  PyArrayObject* DataLoader::make_arrays() {
    
    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }
  
    /*
    // CLUSTER INFO ARRAYS
    PyArrayObject* index_array = nullptr;
    PyArrayObject* feature_array = nullptr;
    make_clusterinfo_arrays( index_array, feature_array );
    */
    
    // FLASH INFO ARRAY
    PyArrayObject* flashinfo_label = nullptr;

    std::cout << "Made arrays n all that" << std::endl;
    
    return make_flashinfo_arrays( flashinfo_label );
    
  }

  /*
  int DataLoader::make_clusterinfo_arrays( PyArrayObject*& index_array,
					   PyArrayObject*& feature_array ) {
    
    // make feature (charge) array
    int feature_nd = 1;
    npy_intp feature_dims[] = { 32 };
    flashinfo_label = (PyArrayObject*)PyArray_SimpleNew( flashinfo_nd, flashinfo_dims, NPY_FLOAT );

    // loop through the 32 PMT values
    for (int i = 0; i < flashinfo_dims[0]; i++ ) {

      float label = flash_v->at( i );

      *((float*)PyArray_GETPTR1(flashinfo_label,i)) = (float)label;
      
    }
    
    return 0;
  }
  */
  
  PyArrayObject* DataLoader::make_flashinfo_arrays( PyArrayObject*& flashinfo_label ) {


    
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
    
    return flashinfo_label;
  }

}
}
