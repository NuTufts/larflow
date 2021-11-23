#ifndef __LOADER_KEYPOINT_DATA_H__
#define __LOADER_KEYPOINT_DATA_H__

/**
 * class used to help load keypoint ground truth data for training
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

  /**
   * @ingroup Keypoints
   * @class LoaderKeypointData
   * @brief load keypoint ground truth data for training
   *
   * The class, larflow::keypoints::PrepKeypointData, stores keypoint information
   * into a ROOT tree (which is stored in a ROOT file).
   *
   * The purpose of this class is to be able to read the tree (from the input files)
   * and produec numpy arrays for convolutional neural network training.
   *
   * Since the keypoint information does not make sense without 
   * the space point information, this loader also provides that too.
   *
   * example usage: (to do)
   *
   */
  class LoaderKeypointData : public larcv::larcv_base {

  public:

    LoaderKeypointData()
      : larcv::larcv_base("LoaderKeypointData"),
	ttriplet(nullptr),
	tkeypoint(nullptr),
	tssnet(nullptr),
	_run(0),
	_subrun(0),
	_event(0),
	_exclude_neg_examples(true)
    {};
    
    LoaderKeypointData( std::vector<std::string>& input_v );
    virtual ~LoaderKeypointData();


    std::vector<std::string> input_files; ///< list of input ROOT files to load

    /** @brief add an individual ROOT file to be loaded */
    void add_input_file( std::string input ) { input_files.push_back(input); };

    TChain* ttriplet;   ///< TTree containing space point information (TChain can be thought of as a TTree loading data over several input files)
    TChain* tkeypoint;  ///< TTree containing keypoint information (TChain can be thought of as a TTree loading data over several input files)
    TChain* tssnet;     ///< TTree containing sssnet information (TChain can be though of as a TTree loading data over several input files)

    // Branch variables
    int _run;    ///< run number ID
    int _subrun; ///< subrun number ID
    int _event;  ///< event number ID
    std::vector<larflow::prep::PrepMatchTriplets>* triplet_v; ///< pointer to triplet data loaded from ttriplet ROOT tree
    std::vector< std::vector<float> >*       kplabel_v[6];    ///< pointer to keypoint labels loaded from the tkeypoint ROOT tree
    std::vector< std::vector<float> >*       kppos_v[6];      ///< pointer to keypoint labels loaded from the tkeypoint ROOT tree    
    std::vector< std::vector<float> >*       kpshift_v;       ///< pointer to vector to closet true keypoint loaded from the tkeypoint ROOT tree
    std::vector< int   >*                    ssnet_label_v;   ///< pointer to ssnet label for each space point loaded from the tssnet ROOT tree
    std::vector< float >*                    ssnet_weight_v;  ///< pointer to class-weights for each space point loaded from the tssnet ROOT tree

    /** @brief set flag that, if True, only loads true (non-ghost) spacepoints for training ssnet and keypoint labels (default is false)*/
    void exclude_false_triplets( bool exclude ) { _exclude_neg_examples = exclude; };
    void load_tree();
    unsigned long load_entry( int entry );
    unsigned long GetEntries();    
    PyObject* sample_data( const int& num_max_samples,
                           int& nfilled,
                           bool withtruth );

    int run()    { if (_run)    return _run;    else return -1; };
    int subrun() { if (_subrun) return _subrun; else return -1; };
    int event()  { if (_event)  return _event;  else return -1; };    

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

    
    static bool _setup_numpy; ///< if true setup numpy by calling import_numpy(0)
    bool _exclude_neg_examples;  ///< if flag set to true, only true (i.e. non-ghost) spacepoints are loaded for training ssnet and keypoint labels
    
  };
  
}
}

#endif
