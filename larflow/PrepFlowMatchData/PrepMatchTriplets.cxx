#include "PrepMatchTriplets.h"
#include "FlowTriples.h"

#include <numpy/ndarrayobject.h>

#include <sstream>
#include <ctime>
#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

namespace larflow {


  void PrepMatchTriplets::process( const std::vector<larcv::Image2D>& adc_v,
                                   const std::vector<larcv::Image2D>& badch_v,
                                   const float adc_threshold ) {

    std::clock_t start = std::clock();
    
    // first we make a common sparse image
    _sparseimg_vv = larflow::FlowTriples::make_initial_sparse_image( adc_v, adc_threshold );
      

    // then we make the flow triples for each of the six flows
    std::vector< larflow::FlowTriples > triplet_v( larflow::kNumFlows );
    for (int sourceplane=0; sourceplane<(int)adc_v.size(); sourceplane++ ) {
      for (int targetplane=0; targetplane<(int)adc_v.size(); targetplane++ ) {

        if ( sourceplane==targetplane ) continue;

        int flowindex = (int)larflow::LArFlowConstants::getFlowDirection( sourceplane, targetplane );
        // save pixel position, rather than index: we will reindex after adding dead channels
        triplet_v[ flowindex ] = FlowTriples( sourceplane, targetplane,
                                              adc_v, badch_v,
                                              _sparseimg_vv, 10.0, false );        
      }
    }

    // collect the unique dead channel additions to each plane
    std::set< std::pair<int,int> > deadpixels_to_add[ adc_v.size() ];

    // add the unique dead channel additions
    std::cout << "sparse pixel totals before deadch additions: "
              << "(" << _sparseimg_vv[0].size() << "," << _sparseimg_vv[1].size() << "," << _sparseimg_vv[2].size() << ")"
              << std::endl;
    
    for ( auto& triplet : triplet_v ) {

      int otherplane = triplet.get_other_plane_index();
      
      std::vector<FlowTriples::PixData_t>& pix_v = triplet.getDeadChToAdd()[ otherplane ];
      
      for ( auto& pix : pix_v ) {
        auto it = deadpixels_to_add[ otherplane ].find( std::pair<int,int>( pix.row, pix.col ) );
        if ( it==deadpixels_to_add[ otherplane ].end() ) {
          // unique, add to sparse image
          pix.val = 1.0;
          _sparseimg_vv[otherplane].push_back( pix );
        }
      }
    }

    std::cout << "sparse pixel totals before deadch additions: "
              << "(" << _sparseimg_vv[0].size() << "," << _sparseimg_vv[1].size() << "," << _sparseimg_vv[2].size() << ")"
              << std::endl;
    
    // sort all pixels
    for ( auto& pix_v : _sparseimg_vv ) {
      std::sort( pix_v.begin(), pix_v.end() );
    }

    // condense and reindex matches
    std::set< std::vector<int> > triplet_set;
    _triplet_v.clear();
    _triplet_v.reserve( 500000  );
    
    for ( auto& triplet_data : triplet_v ) {    
      int srcplane = triplet_data.get_source_plane_index();
      int tarplane = triplet_data.get_target_plane_index();
      int othplane = triplet_data.get_other_plane_index();
      for ( auto& trip : triplet_data.getTriples() ) {

        std::vector<FlowTriples::PixData_t> pix_v(3);
        pix_v[ srcplane ] = FlowTriples::PixData_t( trip[3], trip[0], 0.0 );
        pix_v[ tarplane ] = FlowTriples::PixData_t( trip[3], trip[1], 0.0 );
        pix_v[ othplane ] = FlowTriples::PixData_t( trip[3], trip[2], 0.0 );

        auto it_src = std::lower_bound( _sparseimg_vv[srcplane].begin(), _sparseimg_vv[srcplane].end(), pix_v[ srcplane ] );
        auto it_tar = std::lower_bound( _sparseimg_vv[tarplane].begin(), _sparseimg_vv[tarplane].end(), pix_v[ tarplane ] );
        auto it_oth = std::lower_bound( _sparseimg_vv[othplane].begin(), _sparseimg_vv[othplane].end(), pix_v[ othplane ] );

        if ( it_src==_sparseimg_vv[srcplane].end()
             || it_tar==_sparseimg_vv[tarplane].end()
             || it_oth==_sparseimg_vv[othplane].end() ) {
          std::stringstream ss;
          ss << "Did not find one of sparse image pixels for col triplet=(" << trip[0] << "," << trip[1] << "," << trip[2] << ")";
          ss << " found-index=("
             << it_src-_sparseimg_vv[srcplane].begin() << ","
             << it_tar - _sparseimg_vv[tarplane].begin() << ","
             << it_oth - _sparseimg_vv[othplane].begin() << ")"
             << std::endl;
          throw std::runtime_error( ss.str() );
        }
        if ( (*it_src).row!=trip[3]
             || (*it_tar).row!=trip[3]
             || (*it_oth).row!=trip[3]
             || (*it_src).col!=trip[0]
             || (*it_tar).col!=trip[1]
             || (*it_oth).col!=trip[2] ) {             
          
          std::stringstream ss;
          ss << "found the wrong pixel. searching for triplet=(" << trip[0] << "," << trip[1] << "," << trip[2] << "," << trip[3] << ") and got"
             << " src(c,r)=(" << (*it_src).col << "," << (*it_src).row << ")"
             << " tar(c,r)=(" << (*it_tar).col << "," << (*it_tar).row << ")"
             << " oth(c,r)=(" << (*it_oth).col << "," << (*it_oth).row << ")"
             << std::endl;
          throw std::runtime_error( ss.str() );
        }
             

        std::vector<int> imgcoord_v(4);
        imgcoord_v[ srcplane ] = trip[0];
        imgcoord_v[ tarplane ] = trip[1];
        imgcoord_v[ othplane ] = trip[2];
        imgcoord_v[ 3 ]        = trip[3];
        auto it_trip = triplet_set.find( imgcoord_v );
        if ( it_trip==triplet_set.end() ) {
          triplet_set.insert( imgcoord_v );

          std::vector<int> imgindex_v(4);
          imgindex_v[ srcplane ] = it_src - _sparseimg_vv[srcplane].begin();
          imgindex_v[ tarplane ] = it_tar - _sparseimg_vv[tarplane].begin();
          imgindex_v[ othplane ] = it_oth - _sparseimg_vv[othplane].begin();
          imgindex_v[ 3 ]        = trip[3];

          _triplet_v.push_back( imgindex_v );
        }
        
      }

    }

    std::clock_t end = std::clock();
    std::cout << "[PrepMatchTriplets] made total of " << _triplet_v.size()
              << " unique index triplets. time elapsed=" << float(end-start)/float(CLOCKS_PER_SEC)
              << std::endl;

  }//end of process method

  std::vector<TH2D> PrepMatchTriplets::plot_sparse_images( const std::vector<larcv::Image2D>& adc_v,
                                                           std::string hist_stem_name )
  {
    std::vector<TH2D> out_v;
    for ( int p=0; p<(int)adc_v.size(); p++ ) {
      std::stringstream ss;
      ss << "htriples_plane" << p << "_" << hist_stem_name;
      auto const& meta = adc_v[p].meta();
      TH2D hist( ss.str().c_str(), "",
                 meta.cols(), meta.min_x(), meta.max_x(),
                 meta.rows(), meta.min_y(), meta.max_y() );

      for ( auto const& pix : _sparseimg_vv[p] ) {
        hist.SetBinContent( pix.col+1, pix.row+1, pix.val );
      }
      
      out_v.emplace_back(std::move(hist));
    }
    return out_v;
  }

  /**
   * use larflow truth images to assign good versus bad match for triples
   *
   */
  void PrepMatchTriplets::make_truth_vector( const std::vector<larcv::Image2D>& larflow_v )
  {

    _truth_v.resize( _triplet_v.size(), 0 );
    _truth_2plane_v.resize( _triplet_v.size() );

    const int true_match_span = 5;
    const int min_required_connections = 1;

    int ntriplet_truth = 0;
    std::vector< int > ndoublet_truth( (int)larflow::kNumFlows, 0 );
    
    for ( size_t itrip=0; itrip<_triplet_v.size(); itrip++ ) {
      // for each triplet, we look for truth flows that connect the planes
      auto const& triplet = _triplet_v[itrip];
      _truth_2plane_v[itrip].resize( (int)larflow::kNumFlows, 0 );
      
      std::vector< const FlowTriples::PixData_t* > pix_v( _sparseimg_vv.size() );
      for (size_t p=0; p<_sparseimg_vv.size(); p++)
        pix_v[p] = &_sparseimg_vv[p][ triplet[p] ];

      int ngood_connections = 0;
      for ( int idir=0; idir<larflow::kNumFlows; idir++ ) {
        int srcplane, tarplane;
        larflow::LArFlowConstants::getFlowPlanes( (FlowDir_t)idir, srcplane, tarplane );
        float pixflow = larflow_v[idir].pixel( pix_v[srcplane]->row, pix_v[srcplane]->col );
        int target_col = pix_v[srcplane]->col + (int)pixflow;
        if ( abs(target_col-pix_v[tarplane]->col)<true_match_span ) {
          ngood_connections++;
          _truth_2plane_v[itrip][idir] = 1;
          ndoublet_truth[idir]++;
        }
      }//end of loop over flow directions
      
      if ( ngood_connections>=min_required_connections ) {
        _truth_v[itrip] = 1;
        ntriplet_truth++;
      }
    }//end of trips loop

    std::cout << "[PrepMatchTriplets::make_truth_vector] " << std::endl;
    std::cout << "  number of triplets: " << _triplet_v.size() << std::endl;
    std::cout << "  number of sparse pixels: [ "
              << _sparseimg_vv[0].size() << ", "
              << _sparseimg_vv[1].size() << ", "
              << _sparseimg_vv[2].size() << " ]"
              << std::endl;
    std::cout << "  number of true-match triplets: " << ntriplet_truth << std::endl;
    std::cout << "  doublet truth: [";
    for (auto& n : ndoublet_truth ) std::cout << " " << n << ",";
    std::cout << " ]" << std::endl;
    
  }

  /**
   * plot truth image for debug
   */
  std::vector<TH2D> PrepMatchTriplets::plot_truth_images( const std::vector<larcv::Image2D>& adc_v,
                                                          std::string hist_stem_name )
  {
    std::vector<TH2D> out_v;

    for ( int p=0; p<(int)adc_v.size(); p++ ) {
      std::stringstream ss;
      ss << "htriples_truth_plane" << p << "_" << hist_stem_name;
      auto const& meta = adc_v[p].meta();
      TH2D hist( ss.str().c_str(), "",
                 meta.cols(), meta.min_x(), meta.max_x(),
                 meta.rows(), meta.min_y(), meta.max_y() );

      out_v.emplace_back( std::move(hist) );
    }
    
    for (int i=0; i<(int)_triplet_v.size(); i++ ) {
      auto& trip  = _triplet_v[i];
      auto& truth = _truth_v[i];
      std::vector< const FlowTriples::PixData_t* > pix_v( trip.size(), 0 );
      for (int p=0; p<(int)adc_v.size(); p++ ) {
        pix_v[p] = &_sparseimg_vv[p][ trip[p] ];
        out_v[p].SetBinContent( pix_v[p]->col+1, pix_v[p]->row+1, 1 + 10*truth );
      }
    }
    return out_v;
  }

  /**
   * return a numpy array containing the sparse image information
   *
   */
  PyObject* PrepMatchTriplets::make_sparse_image( int plane ) {
    
    import_array1(0);

    npy_intp* dims = new npy_intp[2];
    dims[0] = (int)_sparseimg_vv[plane].size();

    // if we want truth, we include additional value with 1=correct match, 0=false    
    dims[1] = 3;

    // output array
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( 2, dims, NPY_FLOAT );

    for ( size_t idx=0; idx<_sparseimg_vv[plane].size(); idx++ ) {
      *((float*)PyArray_GETPTR2( array, (int)idx, 0)) = (float)_sparseimg_vv[plane][idx].col;
      *((float*)PyArray_GETPTR2( array, (int)idx, 1)) = (float)_sparseimg_vv[plane][idx].row;
      *((float*)PyArray_GETPTR2( array, (int)idx, 2)) = (float)_sparseimg_vv[plane][idx].val;      
    }
    
    return (PyObject*)array;
  }

  /**
   * return a numpy array with indices to the sparse matrix object.
   *
   * use a vector with index of match pair to choose matches.   
   *
   */
  PyObject* PrepMatchTriplets::make_2plane_match_array( larflow::FlowDir_t kdir,
                                                        const int max_num_samples,
                                                        const std::vector<int>& idx_v,
                                                        const int start_idx,
                                                        const bool withtruth,
                                                        int& nsamples )
  {
    
    import_array1(0);

    npy_intp* dims = new npy_intp[2];
    dims[0] = max_num_samples;

    // if we want truth, we include additional value with 1=correct match, 0=false    
    dims[1] = (withtruth) ? 3 : 2;

    // output array
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( 2, dims, NPY_LONG );

    int srcplane,tarplane;
    larflow::LArFlowConstants::getFlowPlanes( kdir, srcplane, tarplane );

    // number of pairs we've stored
    nsamples = 0;
    
    int end_idx = start_idx + max_num_samples;
    end_idx = ( end_idx>(int)idx_v.size() )   ?  idx_v.size() : end_idx; // cap to number of indices
      
    for ( int idx=start_idx; idx<end_idx; idx++ ) {
      int tripidx = idx_v[idx];
      *((long*)PyArray_GETPTR2( array, nsamples, 0)) = (long)_triplet_v[tripidx][srcplane];
      *((long*)PyArray_GETPTR2( array, nsamples, 1)) = (long)_triplet_v[tripidx][tarplane];
      if ( withtruth ) {
        *((long*)PyArray_GETPTR2( array, nsamples, 2)) = (long)_truth_2plane_v[tripidx][(int)kdir];
      }
      nsamples++;
      if (nsamples==max_num_samples)
        break;

    }//end of indices loop

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
   * randomly select a set of 2 plane indices
   *
   */
  PyObject* PrepMatchTriplets::sample_2plane_matches( larflow::FlowDir_t kdir,
                                                      const int& nsamples,
                                                      int& nfilled,
                                                      bool withtruth ) {

    std::vector<int> idx_v( _triplet_v.size() );
    for ( size_t i=0; i<_triplet_v.size(); i++ ) idx_v[i] = (int)i;
    unsigned seed =  std::chrono::system_clock::now().time_since_epoch().count();
    shuffle (idx_v.begin(), idx_v.end(), std::default_random_engine(seed));

    return make_2plane_match_array( kdir, nsamples, idx_v, 0, withtruth, nfilled );

  }

  /**
   *
   * randomly select a set of 2 plane indices
   *
   */
  PyObject* PrepMatchTriplets::get_chunk_2plane_matches( larflow::FlowDir_t kdir,
                                                         const int& start_index,
                                                         const int& max_num_pairs,
                                                         int& last_index,
                                                         int& num_pairs_filled,
                                                         bool with_truth ) {
    
    std::vector<int> idx_v( max_num_pairs, 0 );
    last_index = start_index + max_num_pairs;
    last_index = ( last_index>(int)_triplet_v.size() ) ? (int)_triplet_v.size() : last_index;
    
    for ( int i=start_index; i<last_index; i++ ) {
      idx_v[i] = (int)i;
    }

    return make_2plane_match_array( kdir, max_num_pairs, idx_v, 0, with_truth, num_pairs_filled );

  }
  
  
}
