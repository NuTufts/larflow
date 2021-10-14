#ifndef __LARFLOW_PREP_INSTANCE_DATA_LOADER_H__
#define __LARFLOW_PREP_INSTANCE_DATA_LOADER_H__

#include <Python.h>
#include "bytesobject.h"

#include "larcv/core/Base/larcv_base.h"
#include "PrepMatchTriplets.h"

namespace larflow {
namespace prep {

  class InstanceDataLoader : public larcv::larcv_base {
  public:

    InstanceDataLoader()
      : larcv::larcv_base("InstanceDataLoader")
      {};
    virtual ~InstanceDataLoader() {};

    PyObject* getDataDict( const larflow::prep::PrepMatchTriplets& tripletdata );

    
  };

}
}

#endif
