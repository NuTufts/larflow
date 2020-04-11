#ifndef __LOADER_KEYPOINT_DATA_H__
#define __LOADER_KEYPOINT_DATA_H__

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

  class LoaderKeypointData {

  public:

    LoaderKeypointData()
      : ttriplet(nullptr),
      tkeypoint(nullptr),
      tssnet(nullptr),
      _setup_numpy(false)
    {};
    
    LoaderKeypointData( std::vector<std::string>& input_v );
    virtual ~LoaderKeypointData();


    std::vector<std::string> input_files;
    void add_input_file( std::string input ) { input_files.push_back(input); };

    TChain* ttriplet;
    TChain* tkeypoint;
    TChain* tssnet;

    std::vector<larflow::PrepMatchTriplets>* triplet_v;
    std::vector< std::vector<float> >*       kplabel_v;
    std::vector< std::vector<float> >*       kpshift_v;
    std::vector< int   >*                    ssnet_label_v;    
    std::vector< float >*                    ssnet_weight_v;

    void load_tree();
    unsigned long load_entry( int entry );
    unsigned long GetEntries();    
    PyObject* sample_data( const int& num_max_samples, int& nfilled, bool withtruth );
    
  protected:
    
    int make_ssnet_arrays( const int& num_max_samples,
                           int& nfilled,
                           bool withtruth,
                           std::vector<int>& pos_match_index,
                           PyArrayObject* match_array,
                           PyArrayObject*& ssnet_label,
                           PyArrayObject*& ssnet_top_weight,
                           PyArrayObject*& ssnet_class_weight );

    int make_kplabel_arrays( const int& num_max_samples,
                             int& nfilled,
                             bool withtruth,
                             std::vector<int>& pos_match_index,
                             PyArrayObject* match_array,
                             PyArrayObject*& kplabel_label,
                             PyArrayObject*& kplabel_weight );

    int make_kpshift_arrays( const int& num_max_samples,
                             int& nfilled,
                             bool withtruth,
                             PyArrayObject* match_array,
                             PyArrayObject*& kpshift_label );
    
    bool _setup_numpy;
    
  };
  
}
}

#endif
