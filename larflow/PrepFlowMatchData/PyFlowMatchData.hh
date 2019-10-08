#ifndef __LARFLOW_PYFLOWMATCHDATA_H__
#define __LARFLOW_PYFLOWMATCHDATA_H__

struct _object;
typedef _object PyObject;

#include <Python.h>
#include "bytesobject.h"
#include "PrepFlowMatchData.hh"

namespace larflow {

  PyObject* sample_pair_array( const int& nsamples, const FlowMatchMap& matchdata, int& nfilled, bool with_truth=false );
  
}

#endif
