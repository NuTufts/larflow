#include "PyFlowMatchData.hh"
#include <numpy/ndarrayobject.h>

#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

#include "larcv/core/PyUtil/PyUtils.h"
#include "larcv/core/Base/larcv_logger.h"
#include "core/LArUtil/LArProperties.h"
#include "core/LArUtil/Geometry.h"

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

  /**
   * return (N,2 or 3) numpy array where each entry is pair of source and target indices of sparse-matrices
   *
   * This function is expected to be called by the above functions.
   *
   * matchdata[in]   Match data class made using larflow::PrepFlowMatchData.
   * idx_v[in]       Index of pair to grab.
   * start_idx[in]   Index of idx_v to start at.
   * max_num_pairs[in] Maximum number of pairs to return.
   * nsource_pixels_covered[out] the number of source pixels we've used in this array.
   * num_pairs_filled[out]       the number of pairs we've returned
   * withtruth[in]   if True, a third value is given for each pair, indiciating if pair is a true/good pair
   **/
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
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( 2, dims, NPY_LONG );

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
        *((long*)PyArray_GETPTR2( array, num_pairs_filled, 0)) = (long)srcidx;
        *((long*)PyArray_GETPTR2( array, num_pairs_filled, 1)) = (long)taridx;
        if ( withtruth ) {
          *((long*)PyArray_GETPTR2( array, num_pairs_filled, 2)) = (*truth_v)[itar];
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
          *((long*)PyArray_GETPTR2( array, i, j)) = 0;
        }
      }
    }

    // return the array
    return (PyObject*)array;
    
  }


  /**
   *
   */
  void make_larflow_hits( PyObject* pair_probs,
                          PyObject* source_sparseimg, PyObject* target_sparseimg,
                          PyObject* matchpairs,
                          const int source_plane,
                          const int target_plane,
                          const larcv::ImageMeta& source_meta,
                          const std::vector<larcv::Image2D>& img_v,
                          larlite::event_larflow3dhit& hit_v,
                          const larcv::EventChStatus* ev_chstatus ) {
    larcv::SetPyUtil();
    
    const int dtype = NPY_FLOAT;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);

    npy_intp pair_dims[2];
    float **probs_carray;    
    if ( PyArray_AsCArray( &pair_probs, (void**)&probs_carray, pair_dims, 2, descr )<0 ) {
      larcv::logger::get("PyFlowMatchData::make_larflow_hits").send(larcv::msg::kCRITICAL,__FUNCTION__,__LINE__, "cannot get carray for pair prob matrix");
    }

    npy_intp source_dims[2];
    float **source_carray;    
    if ( PyArray_AsCArray( &source_sparseimg, (void**)&source_carray, source_dims, 2, descr )<0 ) {
      larcv::logger::get("PyFlowMatchData::make_larflow_hits").send(larcv::msg::kCRITICAL,__FUNCTION__,__LINE__, "cannot get carray for source sparse-image matrix");
    }

    npy_intp target_dims[2];
    float **target_carray;    
    if ( PyArray_AsCArray( &target_sparseimg, (void**)&target_carray, target_dims, 2, descr )<0 ) {
      larcv::logger::get("PyFlowMatchData::make_larflow_hits").send(larcv::msg::kCRITICAL,__FUNCTION__,__LINE__, "cannot get carray for target sparse-image matrix");
    }

    npy_intp match_dims[2];
    long **matchpairs_carray;
    if ( PyArray_AsCArray( &matchpairs, (void**)&matchpairs_carray, match_dims, 2, PyArray_DescrFromType(NPY_LONG) )<0 ) {
      larcv::logger::get("PyFlowMatchData::make_larflow_hits").send(larcv::msg::kCRITICAL,__FUNCTION__,__LINE__, "cannot get carray for match pair matrix");
    }

    // std::cout << "match matrix:  (" << match_dims[0]  << "," << match_dims[1]  << ")" << std::endl;    
    // std::cout << "prob matrix:   (" << pair_dims[0]   << "," << pair_dims[1]   << ")" << std::endl;    
    // std::cout << "source matrix: (" << source_dims[0] << "," << source_dims[1] << ")" << std::endl;
    // std::cout << "target matrix: (" << target_dims[0] << "," << target_dims[1] << ")" << std::endl;

    const float cm_per_tick = larutil::LArProperties::GetME()->DriftVelocity()*0.5;
    int other_plane = -1;
    if ( source_plane==2 ) {
      other_plane = ( target_plane==0 ) ? 1 : 0;
    }
    else if ( source_plane==1 ) {
      other_plane = ( target_plane==0 ) ? 2 : 0;
    }
    else if ( source_plane==0 ) {
      other_plane = ( target_plane==1 ) ? 2 : 1;
    }

    for (int ipair=0; ipair<(int)pair_dims[1]; ipair++) {

      float prob = probs_carray[0][ipair];

      if ( prob<0.5 ) continue;
      
      int srcidx = matchpairs_carray[ipair][0];
      int taridx = matchpairs_carray[ipair][1];
      int srccol = (int)source_carray[srcidx][1];
      int srcrow = (int)source_carray[srcidx][0];      
      int tarcol = (int)target_carray[taridx][1];
      //int tarrow = (int)target_carray[taridx][0];      

      float tick = source_meta.pos_y( srcrow );
      float x = (tick-3200.0)*cm_per_tick;
      double y, z;
      larutil::Geometry::GetME()->IntersectionPoint( srccol, tarcol, (UChar_t)source_plane, (UChar_t)target_plane, y, z );

      Double_t pos[3] = { 0, y, z };
      float other_wire = larutil::Geometry::GetME()->WireCoordinate( pos, other_plane );
      float other_adc  = img_v[other_plane].pixel( srcrow, (int)other_wire );

      bool indead = false;
      if ( other_adc<10.0 ) {
        if ( ev_chstatus==0 )
          continue;

        // if event chstatus pointer is not null, we check the ch status
        int chstatus = ev_chstatus->Status( other_plane ).Status( other_wire );
        if ( chstatus==4 ) {
          // good, so we ignore this match
          continue;
        }
        indead = true;
      }
      
      larlite::larflow3dhit lfhit;
      lfhit.resize(3,0);
      lfhit.srcwire = int(srccol);
      if ( target_plane==0 ) {
        lfhit.flowdir = larlite::larflow3dhit::kY2U;
        lfhit.targetwire[0] = tarcol;
        lfhit.targetwire[1] = (int)other_wire;
      }
      else {
        lfhit.flowdir = larlite::larflow3dhit::kY2V;
        lfhit.targetwire[0] = (int)other_wire;
        lfhit.targetwire[1] = tarcol;        
      }
      lfhit.tick = tick;
      lfhit.targetwire.resize(2,0);
      lfhit[0] = x;
      lfhit[1] = y;
      lfhit[2] = z;

      // if we are in the dead region, we put the score between 0-0.5
      lfhit.track_score = (!indead) ? prob : 0.5*prob;      

      hit_v.emplace_back( std::move(lfhit) );
    }

  }

  void make_larflow_hits_with_deadchs( PyObject* pair_probs,
                                       PyObject* source_sparseimg, PyObject* target_sparseimg,
                                       PyObject* matchpairs,
                                       const int source_plane,
                                       const int target_plane,
                                       const larcv::ImageMeta& source_meta,
                                       const std::vector<larcv::Image2D>& img_v,
                                       const larcv::EventChStatus& ev_chstatus,
                                       larlite::event_larflow3dhit& hit_v ) {
    
    make_larflow_hits( pair_probs,
                       source_sparseimg, target_sparseimg,
                       matchpairs,
                       source_plane,
                       target_plane,
                       source_meta,
                       img_v, hit_v, &ev_chstatus );
    
  }
  
}
