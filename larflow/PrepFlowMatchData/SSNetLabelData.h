#ifndef __LARFLOW_PREP_SSNETLABELDATA_H__
#define __LARFLOW_PREP_SSNETLABELDATA_H__

#include <Python.h>
#include "bytesobject.h"

#include <vector>

namespace larflow {
namespace prep {

  class SSNetLabelData {
  public:

    SSNetLabelData();
    virtual ~SSNetLabelData() {};

    int tpcid;
    int cryoid;
    std::vector< int >                _ssnet_num_v;       ///< number of each class    
    std::vector< int >                _ssnet_label_v;     ///< label: see larflow/larflow/PrepFloatMatchData/PrepSSNetTriplet.h
    std::vector< float >              _ssnet_weight_v;    ///< triplet weights based on topology
    //std::vector< std::vector<int> >   _pid_label_v;     ///< particle ID label
    //std::vector< float >              _pid_weight_v;    ///< class balancing weight
    std::vector< std::vector<float> > _boundary_weight_v; ///< upweights at boundary, vertex

    PyObject* make_ssnet_arrays();

  private:

    static bool _setup_numpy;
    
  };
  
}
}

#endif
