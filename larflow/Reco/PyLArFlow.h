#ifndef __PY_LARFLOW_H__
#define __PY_LARFLOW_H__

#include <Python.h>
#include <vector>
#include "DataFormat/larflowcluster.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class PyLArFlow
   * @brief singleton providing python bindings to support conversion of larflow information, mostly for plotting in plotly
   * 
   */  
  class PyLArFlow {
  public:
    PyLArFlow();
    virtual ~PyLArFlow() {};
    
    static PyObject* as_ndarray_larflowcluster_wcharge( const larlite::larflowcluster& lfcluster );
    static PyObject* as_ndarray_larflowcluster_wssnet(  const larlite::larflowcluster& lfcluster );
    static PyObject* as_ndarray_larflowcluster_wprob(   const larlite::larflowcluster& lfcluster );  
    static PyObject* as_ndarray_larflowcluster_wdeadch( const larlite::larflowcluster& lfcluster );    

    static PyLArFlow _g_instance; ///< pointer to single instance
    static bool      _g_once;     ///< flag indicating that we've called import_numpy to setup numpy environment
  protected:
    static int import_ndarray();  ///< call import_numpy to setup numpy environment
  };

}
}

#endif
