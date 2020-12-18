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
    
    /*
      DataLoader()
      : ttriplet(nullptr),
      tkeypoint(nullptr),
      tssnet(nullptr),
      _exclude_neg_examples(true)
      {};
    */
    
    
  DataLoader()
    : tclusterflash(nullptr){};     // constructor w/ initialization list
    
    DataLoader( std::vector<std::string>& input_v );
    
    virtual ~DataLoader();
    
    std::vector<std::string> input_files; ///< list of input ROOT files to load

    /** @brief add an individual ROOT file to be loaded */
    void add_input_file( std::string input ) { input_files.push_back(input); };

    TChain* tclusterflash;   ///< TTree containing cluster and flash info (TChain can be thought of as a TTree loading data over several input files)
    
     //    TChain* tkeypoint;  ///< TTree containing keypoint information (TChain can be thought of as a TTree loading data over several input files)
    //TChain* tssnet;     ///< TTree containing sssnet information (TChain can be though of as a TTree loading data over several input files)

    /*
    // @@@ Replace these below w/ each branch in the tree? (all vectors)
    std::vector< std::vector<float> >*       kplabel_v[3];    ///< pointer to keypoint labels loaded from the tkeypoint ROOT tree
    std::vector< std::vector<float> >*       kpshift_v;       ///< pointer to vector to closet true keypoint loaded from the tkeypoint ROOT tree
    std::vector< int   >*                    ssnet_label_v;   ///< pointer to ssnet label for each space point loaded from the tssnet ROOT tree
    std::vector< float >*                    ssnet_weight_v;  ///< pointer to class-weights for each space point loaded from the tssnet ROOT tree
    */

    std::vector<int>*         voxel_row;    ///< pointer to voxel row loaded from tclusterflash ROOT tree
    std::vector<int>*         voxel_col;    ///< pointer to voxel col loaded from tclusterflash ROOT tree
    std::vector<int>*         voxel_depth;     ///< pointer to voxel depth loaded from tclusterflash ROOT tree
    std::vector<float>*       voxel_charge; ///< pointer to voxel charge loaded from tclusterflash ROOT tree
    std::vector<float>*       flash_vector; ///< pointer to flash info loaded from tclusterflash ROOT tree

    /** @brief set flag that, if True, only loads true (non-ghost) spacepoints for training ssnet and keypoint labels (default is false)*/
    //    void exclude_false_triplets( bool exclude ) { _exclude_neg_examples = exclude; };
    void load_tree();
    unsigned long load_entry( int entry );
    unsigned long GetEntries();    

    /*
    PyObject* sample_data( const int& num_max_samples,
                           int& nfilled,
                           bool withtruth );
    */

    PyObject* make_arrays();

  private:

    long size; // size of a given entry
    
  protected:
        
    int make_clusterinfo_arrays( PyArrayObject*& voxel_coord_array,
				 PyArrayObject*& voxel_feature_array,
				 int N );
    
    /*    
    int make_flashinfo_arrays( const int& num_max_samples,
			   int& nfilled,
                             bool withtruth,
                             std::vector<int>& pos_match_index,
                             PyArrayObject* match_array,
                             PyArrayObject*& kplabel_label,
                             PyArrayObject*& kplabel_weight );
    */
    /*
    int make_kpshift_arrays( const int& num_max_samples,
                             int& nfilled,
                             bool withtruth,
                             PyArrayObject* match_array,
                             PyArrayObject*& kpshift_label );
    */
    int make_flashinfo_arrays( PyArrayObject*& flashinfo_label );
    
    
    static bool _setup_numpy; ///< if true setup numpy by calling import_numpy(0)
    //    bool _exclude_neg_examples;  ///< if flag set to true, only true (i.e. non-ghost) spacepoints are loaded for training ssnet and keypoint labels
    
  };
  
}
}

#endif
