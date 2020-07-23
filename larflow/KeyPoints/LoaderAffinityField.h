#ifndef __LOADER_AFFINITY_FIELD_DATA_H__
#define __LOADER_AFFINITY_FIELD_DATA_H__

#include <Python.h>
#include "bytesobject.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <string>
#include <vector>

#include "TChain.h"
#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"

namespace larflow {
namespace keypoints {

  /**
   * @ingroup Keypoints
   * @class LoaderAffinityField
   * @brief used to help load affinity field ground truth data for training
   *
   * This class will load data from the `AffinityFieldTree` produced by
   * larflow::keypoints::PrepAffinityField.
   *
   * This data comes in the form of a 3D direction for each
   * proposed spacepoint in an image. This is the target output of the network.
   * 
   * Functions are provided to get these training labels in the form 
   * of numpy arrays in order to interface with DL frameworks. 
   * (We usually develop with pytorch in mind.)
   *
   * Usage: (to do)
   *
   */
  class LoaderAffinityField {

  public:

    LoaderAffinityField()
      : tpaf(nullptr),
      _exclude_neg_examples(true)
    {};
    
    LoaderAffinityField( std::vector<std::string>& input_v );
    virtual ~LoaderAffinityField();


    std::vector<std::string> input_files; ///< list of input files to load

    /** 
     * @brief add input ROOT file to be loaded 
     *
     * @param[in] input path to ROOT file containing `AffinityFieldTree`
     */    
    void add_input_file( std::string input ) { input_files.push_back(input); };

    TChain* tpaf; ///< pointer to TChain for `AffinityFieldTree`
    std::vector< std::vector<float> >*  _label_v; ///< direction labels for each space point in the image

    /** @brief set flag that, if true, will only return labels for true (i.e. non-ghost) points */
    void exclude_false_triplets( bool exclude ) { _exclude_neg_examples = exclude; };
    void load_tree();
    unsigned long load_entry( int entry );
    unsigned long GetEntries();
    PyObject* get_match_data( PyObject* triplet_matches_pyobj, bool exclude_neg_examples );
    
  protected:

    int make_paf_arrays( const int nfilled,
                         const std::vector<int>& pos_match_index,
                         const bool exclude_neg_examples,
                         PyArrayObject* match_array,
                         PyArrayObject*& paf_label,
                         PyArrayObject*& paf_weight );
        
    static bool _setup_numpy; ///< flag indicating if import_numpy() has been called
    bool _exclude_neg_examples; ///< flag that, if true, causes get_match_data() to return only true (i.e. non-ghost) points
    
  };
  
}
}

#endif
