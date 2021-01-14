#ifndef __LOADER_LIGHTMODEL_DATA_H__
#define __LOADER_LIGHTMODEL_DATA_H__

/**
 * class used to help load light model ground truth data for training
 *
 */

#include <Python.h>
#include "bytesobject.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <string>
#include <vector>

#include "TRandom3.h"
#include "TChain.h"

namespace larflow {
namespace lightmodel {

  /**
   * @ingroup Lightmodel
   * @class DataLoader
   * @brief load light model ground truth data for training
   *
   * The purpose of this class is to be able to read the tree (from the input files)
   * and produec numpy arrays for convolutional neural network training.
   *
   * 
   */
  class DataLoader {
    
  public:
    
  DataLoader()
    : tclusterflash(nullptr){};     // constructor w/ initialization list
    
    DataLoader( std::vector<std::string>& input_v );
    
    virtual ~DataLoader();
    
    std::vector<std::string> input_files; ///< list of input ROOT files to load

    /** @brief add an individual ROOT file to be loaded */
    void add_input_file( std::string input ) { input_files.push_back(input); };

    TChain* tclusterflash;   ///< TTree containing cluster and flash info (TChain can be thought of as a TTree loading data over several input files)

    std::vector<int>*         voxel_row;    ///< pointer to voxel row loaded from tclusterflash ROOT tree
    std::vector<int>*         voxel_col;    ///< pointer to voxel col loaded from tclusterflash ROOT tree
    std::vector<int>*         voxel_depth;     ///< pointer to voxel depth loaded from tclusterflash ROOT tree
    std::vector<float>*       voxel_charge; ///< pointer to voxel charge loaded from tclusterflash ROOT tree
    std::vector<float>*       flash_vector; ///< pointer to flash info loaded from tclusterflash ROOT tree

    // struct to contain info for one cluster+flash pair
    struct ClusterFlashPair_t {
      std::vector<int>         voxel_row_v;    ///< vector of voxel row values for one cluster
      std::vector<int>         voxel_col_v;    ///< vector of voxel col values for one cluster
      std::vector<int>         voxel_depth_v;     ///< vector of voxel depth values for one cluster
      std::vector<float>       voxel_charge_v; ///< vector of voxel charge values for one cluster

      std::vector<float>       flash_vector_v; ///< vector of flash info associated to the cluster
    };

    // list containing all cluster+flash pairs
    //typedef std::vector<ClusterFlashPair_t> PairList_t;
    
    void load_tree();
    unsigned long load_entry( int entry );
    ClusterFlashPair_t getTreeEntry(int entry); // return data for 1 entry in tree
    unsigned long GetEntries();    

    PyObject* getTrainingDataBatch(int batch_size);

    PyObject* make_arrays(); // makes for just one entry
    PyObject* make_arrays( const std::vector< ClusterFlashPair_t >& data_v ) const;

  private:

    long size; // size of a given entry
    
  protected:

    unsigned long _current_entry;
    unsigned long _num_entries;
    TRandom3* _rand;
    
    int make_clusterinfo_arrays( PyArrayObject*& voxel_coord_array,
				 PyArrayObject*& voxel_feature_array,
				 int N );
    
    int make_flashinfo_arrays( PyArrayObject*& flashinfo_label );

    
    
    static bool _setup_numpy; ///< if true setup numpy by calling import_numpy(0)
    
  };
  
}
}

#endif
