#include "PyLArFlow.h"

#include "bytesobject.h"
#ifdef USE_PYTHON3
#include "numpy/arrayobject.h"
#include <cassert>
#else
#include <numpy/ndarrayobject.h>
#endif
#include "larcv/core/PyUtil/PyUtils.h"

namespace larflow {
namespace reco {
  
  PyLArFlow::PyLArFlow() {}
  PyLArFlow PyLArFlow::_g_instance;
  bool      PyLArFlow::_g_once = false;

  int PyLArFlow::import_ndarray() {
    if ( _g_once )
      return 0;
      
    std::cout << "[" << __FUNCTION__ << "] load numpy C-api." << std::endl;
    _g_once = true;      
    import_array1(0);
  }
  
  PyObject* PyLArFlow::as_ndarray_larflowcluster_wcharge( const larlite::larflowcluster& lfcluster ) {
    import_ndarray();

    npy_intp dim_data[2];
    dim_data[0] = lfcluster.size();
    dim_data[1] = 4;

    //std::cout << "[" << __FUNCTION__ << "]" << std::endl;

    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( 2, dim_data, NPY_FLOAT );

    for ( size_t ihit=0; ihit<lfcluster.size(); ihit++ ) {
      // position and charge
      float totq = 0.;
      int npix = 0;
      for ( size_t i=0; i<3; i++ ) {
        *((float*)PyArray_GETPTR2( array, ihit, i )) = lfcluster[ihit][i];
        if ( lfcluster[ihit][3+i]>0 ) {
          totq += lfcluster[ihit][3+i];
          npix++;
        }
      }
      if ( npix>0 ) totq /= (float)npix;
      *((float*)PyArray_GETPTR2( array, ihit, 3 )) = totq;
    }//end of hit loop

    return (PyObject*)array;
  }

  PyObject* PyLArFlow::as_ndarray_larflowcluster_wssnet( const larlite::larflowcluster& lfcluster ) {
    //larcv::SetPyUtil();
    import_ndarray();

    npy_intp dim_data[2];
    dim_data[0] = lfcluster.size();
    dim_data[1] = 4;

    //std::cout << "[" << __FUNCTION__ << "] allocating array" << std::endl;        

    PyArrayObject* array = nullptr;
    try {
      array = (PyArrayObject*)PyArray_SimpleNew( 2, dim_data, NPY_FLOAT );
    }
    catch ( std::exception& e ) {
      std::cout << "error: " << e.what() << std::endl;
      throw std::runtime_error( e.what() );
    }

    //std::cout << "[" << __FUNCTION__ << "] loop over hits, n=" << lfcluster.size() << std::endl;
    for ( size_t ihit=0; ihit<lfcluster.size(); ihit++ ) {
      // position and ssnet
      //std::cout << "[" << __FUNCTION__ << "] hit data idx=" << ihit << std::endl;
      for ( size_t i=0; i<3; i++ ) {
        *((float*)PyArray_GETPTR2( array, ihit, i )) = lfcluster[ihit][i];
      }
      *((float*)PyArray_GETPTR2( array, ihit, 3 )) = lfcluster[ihit].shower_score;
    }//end of hit loop

    return (PyObject*)array;
  }

  PyObject* PyLArFlow::as_ndarray_larflowcluster_wprob( const larlite::larflowcluster& lfcluster ) {
    import_ndarray();

    npy_intp dim_data[2];
    dim_data[0] = lfcluster.size();
    dim_data[1] = 4;

    PyArrayObject* array = nullptr;
    try {
      array = (PyArrayObject*)PyArray_SimpleNew( 2, dim_data, NPY_FLOAT );
    }
    catch ( std::exception& e ) {
      std::cout << "error: " << e.what() << std::endl;
      throw std::runtime_error( e.what() );
    }

    for ( size_t ihit=0; ihit<lfcluster.size(); ihit++ ) {
      // position and ssnet
      for ( size_t i=0; i<3; i++ ) {
        *((float*)PyArray_GETPTR2( array, ihit, i )) = lfcluster[ihit][i];
      }
      *((float*)PyArray_GETPTR2( array, ihit, 3 )) = lfcluster[ihit][6];
    }//end of hit loop

    return (PyObject*)array;
  }

  PyObject* PyLArFlow::as_ndarray_larflowcluster_wdeadch( const larlite::larflowcluster& lfcluster ) {
    import_ndarray();
    
    npy_intp dim_data[2];
    dim_data[0] = lfcluster.size();
    dim_data[1] = 4;

    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( 2, dim_data, NPY_FLOAT );

    for ( size_t ihit=0; ihit<lfcluster.size(); ihit++ ) {
      // position and charge
      float totq = 0.;
      int npix = 0;
      for ( size_t i=0; i<3; i++ ) {
        *((float*)PyArray_GETPTR2( array, ihit, i )) = lfcluster[ihit][i];
        if ( lfcluster[ihit][3+i]>0 ) {
          totq += lfcluster[ihit][3+i];
          npix++;
        }
      }
      if ( npix>=3 ) 
        *((float*)PyArray_GETPTR2( array, ihit, 3 )) = 10.0;
      else
        *((float*)PyArray_GETPTR2( array, ihit, 3 )) =  5.0;
      
    }//end of hit loop

    return (PyObject*)array;
  }

}
}
