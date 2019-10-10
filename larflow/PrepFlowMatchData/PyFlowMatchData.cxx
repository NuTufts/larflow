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

    int num_source_pixels = 0;
    return _make_pair_array( matchdata, idx_v, 0, nsamples, num_source_pixels, nfilled, withtruth );
  }

  PyObject* get_chunk_pair_array( const int& start_source_pixel_index,
                                  const int& max_num_pairs,
                                  const FlowMatchMap& matchdata,
                                  int& last_source_pixel_index,
                                  int& num_pairs_filled,
                                  bool with_truth ) {
    // how many source pixels are in the matchmap
    size_t nsource_indices = matchdata.nsourceIndices();
    
    if ( nsource_indices <= start_source_pixel_index ) {
      throw std::runtime_error("[PyFlowMatchData.cxx:get_chunk_pair_array] starting source pixel index bigger than number of source pixels");
    }

    size_t num_indices = nsource_indices-start_source_pixel_index;
    std::vector<size_t> idx_v( num_indices );
    for ( size_t i=0; i<num_indices; i++ ) {
      idx_v[i] = start_source_pixel_index+i;
    }
    num_pairs_filled = 0;
    return _make_pair_array( matchdata, idx_v, 0, max_num_pairs, last_source_pixel_index, num_pairs_filled, with_truth );
  }

  PyObject* _make_pair_array( const FlowMatchMap& matchdata,
                              const std::vector<size_t>& idx_v,
                              const int start_idx,
                              const int max_num_pairs,
                              int& nsource_pixels_covered,
                              int& num_pairs_filled,
                              bool withtruth ) {
    import_array1(0);
    
    npy_intp* dims = new npy_intp[2];
    dims[0] = max_num_pairs;

    // if we want truth, we include additional value with 1=correct match, 0=false    
    dims[1] = (withtruth) ? 3 : 2;

    // output array
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( 2, dims, NPY_INT );

    // number of pairs we've stored
    num_pairs_filled = 0;
    // number of source pixels we've used
    nsource_pixels_covered  = 0;
    for ( size_t isrc=start_idx; isrc<idx_v.size(); isrc++ ) {
      size_t srcidx = idx_v[isrc];
      const std::vector<int>& target_v = matchdata.getTargetIndices(srcidx);
      const std::vector<int>* truth_v  = nullptr;
      if ( withtruth ) truth_v = &matchdata.getTruthVector(srcidx);

      // number of pairs for this source pixel
      size_t numpairs = target_v.size();

      if ( num_pairs_filled+numpairs>max_num_pairs ) {
        // if filling these pairs leads to an incomplete sample,
        // we stop
        break;
      }
      
      //std::cout << "  srcidx=" << srcidx << ": number of target indices=" << target_v.size() << " nfilled=" << nfilled << std::endl;
      for ( size_t itar=0; itar<target_v.size(); itar++ ) {
        int taridx = target_v[itar];
        *((int*)PyArray_GETPTR2( array, num_pairs_filled, 0)) = srcidx;
        *((int*)PyArray_GETPTR2( array, num_pairs_filled, 1)) = taridx;
        if ( withtruth ) {
          *((int*)PyArray_GETPTR2( array, num_pairs_filled, 2)) = (*truth_v)[itar];
        }
        num_pairs_filled++;
        if (num_pairs_filled==max_num_pairs)
          break;
      }

      nsource_pixels_covered++;
      
      if ( num_pairs_filled==max_num_pairs )
        break;
    }//end of indices loop

    // zero rest of array
    if ( num_pairs_filled<max_num_pairs ) {
      for ( size_t i=num_pairs_filled; i<max_num_pairs; i++ ) {
        for (int j=0; j<dims[1]; j++) {
          *((int*)PyArray_GETPTR2( array, i, j)) = 0;
        }
      }
    }

    // return the array
    return (PyObject*)array;
    
  }
}
