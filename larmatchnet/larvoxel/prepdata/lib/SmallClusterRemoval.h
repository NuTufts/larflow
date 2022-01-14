#ifndef __SMALL_CLUSTER_REMOVAL__
#define __SMALL_CLUSTER_REMOVAL__

#include <Python.h>
#include "bytesobject.h"

namespace larvoxelprepdata {

  class SmallClusterRemoval {

  public:

    SmallClusterRemoval() {};
    virtual ~SmallClusterRemoval() {};

    PyObject* do_removal( PyObject* coord_array, PyObject* feat_array,
			  int voxelthreshold, float charge_threshold );

  private:

    static bool _setup_numpy;
    
  };

}


#endif
