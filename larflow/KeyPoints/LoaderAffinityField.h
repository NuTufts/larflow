#ifndef __LOADER_AFFINITY_FIELD_DATA_H__
#define __LOADER_AFFINITY_FIELD_DATA_H__

/**
 * class used to help load affinity field ground truth data for training
 *
 */

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

  class LoaderAffinityField {

  public:

    LoaderAffinityField()
      : tpaf(nullptr),
      _setup_numpy(false),
      _exclude_neg_examples(true)
    {};
    
    LoaderAffinityField( std::vector<std::string>& input_v );
    virtual ~LoaderAffinityField();


    std::vector<std::string> input_files;
    void add_input_file( std::string input ) { input_files.push_back(input); };

    TChain* tpaf;
    std::vector< std::vector<float> >*  _label_v;

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
        
    bool _setup_numpy;
    bool _exclude_neg_examples;
    
  };
  
}
}

#endif
