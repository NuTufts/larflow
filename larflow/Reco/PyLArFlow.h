#ifndef __PY_LARFLOW_H__
#define __PY_LARFLOW_H__

/*
 * python bindings to support conversion of larflow information, mostly for plotting in plotly
 *
 */

#include <Python.h>
#include <vector>
#include "DataFormat/larflowcluster.h"

namespace larflow {
namespace reco {

  class PyLArFlow {
  public:
    PyLArFlow();
    virtual ~PyLArFlow() {};

    static PyObject* as_ndarray_larflowcluster_wcharge( const larlite::larflowcluster& lfcluster );
    static PyObject* as_ndarray_larflowcluster_wssnet(  const larlite::larflowcluster& lfcluster );
    static PyObject* as_ndarray_larflowcluster_wprob(   const larlite::larflowcluster& lfcluster );  
    static PyObject* as_ndarray_larflowcluster_wdeadch( const larlite::larflowcluster& lfcluster );    

    static PyLArFlow _g_instance;
    static bool      _g_once;
  protected:
    static int import_ndarray();
  };

}
}

#endif
