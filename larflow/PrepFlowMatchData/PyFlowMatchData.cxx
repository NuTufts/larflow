#include "PyFlowMatchData.hh"
#include <numpy/ndarrayobject.h>

#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

#include "larcv/core/PyUtil/PyUtils.h"

namespace larflow {

  PyObject* sample_pair_array( const int& nsamples, const FlowMatchMap& matchdata, int& nfilled, bool withtruth ) {

    //larcv::SetPyUtil();
    import_array1(0);

    size_t nsource_indices = matchdata.nsourceIndices();
    
    std::vector<size_t> idx_v( nsource_indices );
    for ( size_t i=0; i<nsource_indices; i++ ) idx_v[i] = i;
    unsigned seed =  std::chrono::system_clock::now().time_since_epoch().count();
    shuffle (idx_v.begin(), idx_v.end(), std::default_random_engine(seed));
    
    npy_intp* dims = new npy_intp[2];
    dims[0] = nsamples;
    dims[1] = (withtruth) ? 3 : 2;
      
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( 2, dims, NPY_INT );
    nfilled = 0;
    for ( size_t isrc=0; isrc<matchdata.nsourceIndices(); isrc++ ) {
      size_t srcidx = idx_v[isrc];
      const std::vector<int>& target_v = matchdata.getTargetIndices(srcidx);
      const std::vector<int>* truth_v  = nullptr;
      if ( withtruth ) truth_v = &matchdata.getTruthVector(srcidx);
      //std::cout << "  srcidx=" << srcidx << ": number of target indices=" << target_v.size() << " nfilled=" << nfilled << std::endl;
      for ( size_t itar=0; itar<target_v.size(); itar++ ) {
        int taridx = target_v[itar];
        *((int*)PyArray_GETPTR2( array, nfilled, 0)) = srcidx;
        *((int*)PyArray_GETPTR2( array, nfilled, 1)) = taridx;
        if ( withtruth ) {
          *((int*)PyArray_GETPTR2( array, nfilled, 2)) = (*truth_v)[itar];
        }
        nfilled++;
        if (nfilled==nsamples)
          break;
      }
      if ( nfilled==nsamples )
        break;
    }
    return (PyObject*)array;
  }

}
