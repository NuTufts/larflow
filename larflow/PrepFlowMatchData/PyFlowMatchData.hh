#ifndef __LARFLOW_PYFLOWMATCHDATA_H__
#define __LARFLOW_PYFLOWMATCHDATA_H__

struct _object;
typedef _object PyObject;

#include <Python.h>
#include "bytesobject.h"
#include "PrepFlowMatchData.hh"

namespace larflow {

  // get a randomly sampled set of flow pixel pairs, returned in a numpy array
  // used for training
  PyObject* sample_pair_array( const int& nsamples, const FlowMatchMap& matchdata, int& nfilled, bool with_truth=false );

  // get a set of flow pixel pairs for a squential set of source pixels
  // used for deploy
  PyObject* get_chunk_pair_array( const int& start_source_pixel_index,
                                  const int& max_num_pairs,
                                  const FlowMatchMap& matchdata,
                                  int& last_source_pixel_index,
                                  int& num_pairs_filled,
                                  bool with_truth=false );

  // internal function that makes array, filling it with source pixels specified by indices
  PyObject* _make_pair_array( const FlowMatchMap& matchdata,
                              const std::vector<size_t>& idx_v,
                              const int start_idx,
                              const int max_num_pairs,
                              int& nsource_pixels_covered,
                              int& num_pairs_filled,
                              bool withtruth );
  
}

#endif
