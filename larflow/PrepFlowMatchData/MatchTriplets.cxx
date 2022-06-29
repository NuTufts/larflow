#include "MatchTriplets.h"

#include <sstream>
#include <chrono>       // std::chrono::system_clock
#include <random>       // std::default_random_engine

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>


namespace larflow {
namespace prep {

  bool MatchTriplets::_setup_numpy = false;

  /**
   * @brief clear the containers we fill when we run process
   *
   */
  void MatchTriplets::clear()
  {

    _imgmeta_v.clear();
    _sparseimg_vv.clear();
    
    _triplet_v.clear();
    _trip_cryo_tpc_v.clear();
    _truth_v.clear();
    _truth_2plane_v.clear();
    _weight_v.clear();
    _flowdir_v.clear();
    _triarea_v.clear();
    _pos_v.clear();
    _instance_id_v.clear();
    _ancestor_id_v.clear();
    _pdg_v.clear();

    _match_minspan_v.clear();
    _match_maxspan_v.clear();    
    _match_cyclespan_v.clear();
    
  }
  
  /**
   * @brief return a numpy array containing the sparse image information
   *
   * @param[in] plane Plane index for sparse image requested.
   * @return numpy array with shape (N,3) containing info from sparse matrix. each row contains (row,col,pixel value).
   */
  PyObject* MatchTriplets::make_sparse_image( int plane ) {
    
    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    npy_intp* dims = new npy_intp[2];
    dims[0] = (int)_sparseimg_vv[plane].size();
    dims[1] = 3;

    // output array
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( 2, dims, NPY_FLOAT );

    for ( size_t idx=0; idx<_sparseimg_vv[plane].size(); idx++ ) {
      *((float*)PyArray_GETPTR2( array, (int)idx, 0)) = (float)_sparseimg_vv[plane][idx].row;
      *((float*)PyArray_GETPTR2( array, (int)idx, 1)) = (float)_sparseimg_vv[plane][idx].col;
      *((float*)PyArray_GETPTR2( array, (int)idx, 2)) = (float)_sparseimg_vv[plane][idx].val;      
    }

    //Py_DECREF( array ); // breaks
      
    return (PyObject*)array;
  }

  /**
   * @brief return a dictionary of numpy arrays containing triplet information
   *
   * @param[in] withtruth arrays containing truth information are returned (if truth data available)
   * @return dictionary with numpy arrays
   */  
  PyObject* MatchTriplets::get_all_triplet_data( const bool withtruth )
  {
    std::vector<int> idx_v(_triplet_v.size());
    for (int i=0; i<(int)_triplet_v.size(); i++)
      idx_v[i] = i;
    int nsamples = 0;
    return make_triplet_array( _triplet_v.size(), idx_v, 0, withtruth, nsamples );
  }

  /**
   * @brief utility function to get imgcoord of triplet
   *
   * @param[in] idx_triplet Index of triplet to return info for.
   * @return a vector<int> containing (col,col,col,row)
   *
   */
  std::vector<int> MatchTriplets::get_triplet_imgcoord_rowcol( int idx_triplet )
  {
    if ( idx_triplet<0 || idx_triplet>=(int)_triplet_v.size() ) {
      std::stringstream msg;
      msg << "[PrepMatchTriplets::get_triplet_imgcoord_rowcol.L" << __LINE__ << "] "
          << "triplet index requested (" << idx_triplet << ") is out of bounds. "
          << "values should be between [0," << _triplet_v.size() << ")."
          << std::endl;
      throw std::runtime_error(msg.str());
    }

    auto const& triplet = _triplet_v[idx_triplet];
    std::vector<int> imgcoord = {0,0,0,triplet[3]};
    for (int p=0; p<3; p++ ) {
      imgcoord[p] = _sparseimg_vv[p][triplet[p]].col;
    }
    return imgcoord;
  }
  
  /**
   * @brief Make ndarray containing charge data from the wire planes for all spacepoint proposals
   * 
   * the dictionary contains the following:
   * @verbatim embed:rst:leading-asterisk 
   *  * `imgcoord_t`: (N,4) numpy array containing (col,col,col,row) in 2D dimension.
   *  * `instance_t`: (N,1) instance labels. ID is the geant4 track id.
   *  * `segment_t`:  (N,1) particle class labels. labels follow values in larcv/core/DataFormat/DataFormatTypes.h.
   *  * `ancestor_t`: (N,1) ancestor labels. ID is the geant4 ancestor id (not yet implemented).
   * @endverbatim
   *
   * @return dictionary with numpy arrays
   */
  PyObject* MatchTriplets::make_spacepoint_charge_array()
  {

    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }


    if ( _triplet_v.size()!=_pos_v.size() ) {
      std::stringstream ss;
      ss << "[MatchTriplets::make_spacepoint_charge_array] "
         << "pos=" << _pos_v.size() << ") "
         << " do not match triplet_v size = " << _triplet_v.size()
         << std::endl;
      throw std::runtime_error( ss.str() );
    }   

    long int npts =  _triplet_v.size();
    
    // space point
    npy_intp spacepoint_t_dim[] = { npts, 6 }; // (x,y,z,Q_u,Q_v,_Q_y)
    PyArrayObject* spacepoint_t = (PyArrayObject*)PyArray_SimpleNew( 2, spacepoint_t_dim, NPY_FLOAT );
    PyObject *spacepoint_t_key = Py_BuildValue("s", "spacepoint_t");

    int ifilled = 0;
    for (size_t itriplet=0; itriplet<npts; itriplet++) {
      auto const& triplet = _triplet_v[itriplet];

      // fill (x,y,z)
      for (int i=0; i<3; i++)
	*((float*)PyArray_GETPTR2(spacepoint_t,itriplet,i)) = _pos_v[itriplet][i];
      for (int p=0; p<3; p++ ) {
	*((float*)PyArray_GETPTR2(spacepoint_t,itriplet,3+p)) = _sparseimg_vv[p][triplet[p]].val;
      }
      ifilled++;
    }

    // Create and fill dictionary
    PyObject *d = PyDict_New();
    PyDict_SetItem(d, spacepoint_t_key, (PyObject*)spacepoint_t);
    Py_DECREF( spacepoint_t );
    Py_DECREF( spacepoint_t_key );
    
    // if we have it, provide truth label for triplets
    int ntruepts = 0;
    for (auto const& truth : _truth_v ) {
      if ( truth ) ntruepts++;
    }
    std::cout << "[MatchTriplets::make_spacepoint_charge_array] number of true points: " << ntruepts << " of " << _truth_v.size() << std::endl;
    
    if (ntruepts>0) {
      
      if ( _truth_v.size() != _triplet_v.size() )  {
	throw std::runtime_error("has truth, but truth vector different size than triplet vector");
      }

      npy_intp truth_t_dim[] = { (long int)npts };
      PyArrayObject* truth_t = (PyArrayObject*)PyArray_SimpleNew( 1, truth_t_dim, NPY_LONG );
      PyObject *truth_t_key = Py_BuildValue("s", "truetriplet_t");

      for (size_t itriplet=0; itriplet<npts; itriplet++) {
        *((long*)PyArray_GETPTR1( truth_t, itriplet )) = (long)_truth_v[itriplet];
      }
         
      PyDict_SetItem(d, truth_t_key, (PyObject*)truth_t);     
      Py_DECREF( truth_t );    
      Py_DECREF( truth_t_key );

      // Particle type ID
      npy_intp segment_t_dim[] = { (long int)npts };
      PyArrayObject* segment_t = (PyArrayObject*)PyArray_SimpleNew( 1, segment_t_dim, NPY_LONG );
      PyObject *segment_t_key = Py_BuildValue("s", "segment_t");
      
      for (size_t itriplet=0; itriplet<npts; itriplet++) {
        *((long*)PyArray_GETPTR1( segment_t, itriplet )) = (long)_pdg_v[itriplet];
      }
      
      PyDict_SetItem(d, segment_t_key, (PyObject*)segment_t);     
      Py_DECREF( segment_t );    
      Py_DECREF( segment_t_key );
      
      
    }

    return d;
  }


  /**
   * @brief return a numpy array with indices to the sparse matrix ADC array
   *
   * @param[in] max_num_samples Maximum number of samples to return. Dim[0] of returned array.
   * @param[in] idx_v List of triplet_v indices to use
   * @param[in] start_idx First index in idx_v to copy
   * @param[in] withtruth If true, additional element added to Dim[1]
                          which contains if triplet is true match (1) or fals match (0).
   * @param[out] nsamples Returns the number of indices we copied.
   * @return A numpy array, with type NPY_LONG and dimensions (max_num_samples, 5)
   *         columns: [u-index,v-index,y-index,truthlabel,triplet-index]
   *
   */
  PyObject* MatchTriplets::make_triplet_array( const int max_num_samples,
					       const std::vector<int>& idx_v,
					       const int start_idx,
					       const bool withtruth,
					       int& nsamples ) const
  {

    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    int nd = 2;
    int ndims2 = 5; // (index0, index1, index2, truth, tripindex)
    npy_intp dims[] = { max_num_samples, ndims2 };

    // output array
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_LONG );

    // number of pairs we've stored
    nsamples = 0;
    
    int end_idx = start_idx + max_num_samples;
    end_idx = ( end_idx>(int)idx_v.size() )   ?  idx_v.size() : end_idx; // cap to number of indices

    // std::cout << "[MatchTriplets::make_triplet_array] withtruth=" << withtruth << " "
    //           << "make numpy array with indices from triplets[" << start_idx << ":" << end_idx << "]"
    //           << std::endl;
    
    for ( int idx=start_idx; idx<end_idx; idx++ ) {
      int tripidx = idx_v[idx];
      for (size_t p=0; p<3; p++ )
        *((long*)PyArray_GETPTR2( array, nsamples, p)) = (long)_triplet_v[tripidx][p];
      if ( withtruth ) {
        *((long*)PyArray_GETPTR2( array, nsamples, 3)) = (long)_truth_v[tripidx];
        *((long*)PyArray_GETPTR2( array, nsamples, 4)) = (long)tripidx;
      }
      else {
        *((long*)PyArray_GETPTR2( array, nsamples, 3)) = 0;        
        *((long*)PyArray_GETPTR2( array, nsamples, 4)) = (long)tripidx;
      }
      nsamples++;
      if (nsamples==max_num_samples)
        break;
    }//end of indices loop

    //std::cout << "[MatchTriplets::make_triplet_array] nsamples=" << nsamples << std::endl;

    // zero rest of array
    if ( nsamples<max_num_samples ) {
      for ( size_t i=nsamples; i<max_num_samples; i++ ) {
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
   * @brief randomly select a set of triplet matches
   *
   */
  PyObject* MatchTriplets::sample_triplet_matches( const int& nsamples,
                                                       int& nfilled,
                                                       bool withtruth ) const
  {

    std::vector<int> idx_v( _triplet_v.size() );
    for ( size_t i=0; i<_triplet_v.size(); i++ ) idx_v[i] = (int)i;
    if ( _kshuffle_indices_when_sampling ) {
      unsigned seed =  std::chrono::system_clock::now().time_since_epoch().count();
      shuffle (idx_v.begin(), idx_v.end(), std::default_random_engine(seed));
    }
    
    return make_triplet_array( nsamples, idx_v, 0, withtruth, nfilled );

  }
     
  /**
   *
   * @brief get sequential set of triplet indices
   *
   */
  PyObject* MatchTriplets::get_chunk_triplet_matches( const int& start_index,
                                                          const int& max_num_pairs,
                                                          int& last_index,
                                                          int& num_pairs_filled,
                                                          bool with_truth ) {
    
    last_index = start_index + max_num_pairs;
    last_index = ( last_index>(int)_triplet_v.size() ) ? (int)_triplet_v.size() : last_index;
    num_pairs_filled = last_index-start_index;
    std::vector<int> idx_v( num_pairs_filled, 0 );    
    
    for ( int i=start_index; i<last_index; i++ ) {
      idx_v[i-start_index] = (int)i;
    }

    return make_triplet_array( max_num_pairs, idx_v, 0, with_truth, num_pairs_filled );

  }
  
  /**
   * @brief return a numpy array with data from _match_span_v variable
   *
   * @return numpy array, 1-dim, dtype=float
   *
   */
  PyObject* MatchTriplets::get_matchspan_array() const
  {

    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    int nd = 2;
    int ndims = (int)_triplet_v.size(); // (index0, index1, index2, truth, tripindex)
    npy_intp dims[] = { ndims, 3 };

    // output array
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_FLOAT );

    for ( int idx=0; idx<(int)_match_maxspan_v.size(); idx++ ) {
      *((float*)PyArray_GETPTR2( array, idx, 0 )) = (float)_match_maxspan_v[idx];
      *((float*)PyArray_GETPTR2( array, idx, 1 )) = (float)_match_minspan_v[idx];
      *((float*)PyArray_GETPTR2( array, idx, 2 )) = (float)_match_cyclespan_v[idx];
    }//end of indices loop

    // return the array
    return (PyObject*)array;
    
  }
  
}
}
