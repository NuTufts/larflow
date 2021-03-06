#include "PrepMatchTriplets.h"
#include "FlowTriples.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <sstream>
#include <ctime>
#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

#include "larlite/core/LArUtil/LArProperties.h"
#include "larlite/core/LArUtil/Geometry.h"
#include "larcv/core/PyUtil/PyUtils.h"
#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "ublarcvapp/UBWireTool/UBWireTool.h"
#include "ublarcvapp/UBImageMod/EmptyChannelAlgo.h"

#include "TRandom3.h"

namespace larflow {
namespace prep {

  bool PrepMatchTriplets::_setup_numpy = false;
  
  /**
   * clear the containers we fill when we run process
   *
   */
  void PrepMatchTriplets::clear()
  {

    _imgmeta_v.clear();
    _sparseimg_vv.clear();
    _triplet_v.clear();
    _truth_v.clear();
    _truth_2plane_v.clear();
    _weight_v.clear();
    _flowdir_v.clear();
    _triarea_v.clear();
    _pos_v.clear();
    
  }
  
  /**
   * @brief convenience function that gets the data needed from an larcv::IOManager instance and runs the process method.
   *
   *
   * @param[in] iolcv   larcv::IOManager instance containing needed data objects
   * @param[in] wire_producer     Name of tree containing event ADC images
   * @param[in] chstatus_producer Name of tree containing badch status
   * @param[in] adc_threshold     Threshold value for pixels we will consider (typically 10.0)
   * @param[in] check_wire_intersection Check that triplet produces a good wire intersection inside the TPC.
   *                                    Also saves the 3D position for each triplet. (Makes slower; For debug.)
   */
  void PrepMatchTriplets::process( larcv::IOManager& iolcv,
                                   std::string wire_producer,
                                   std::string chstatus_producer,
                                   const float adc_threshold,
                                   const bool check_wire_intersection )
  {
    
    // create bad channel maker algo
    ublarcvapp::EmptyChannelAlgo badchmaker;

    // get images
    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,wire_producer);
    
    // get chstatus
    larcv::EventChStatus* ev_badch
      = (larcv::EventChStatus*)iolcv.get_data(larcv::kProductChStatus,chstatus_producer);
    
    std::vector<larcv::Image2D> badch_v = badchmaker.makeBadChImage( 4, 3, 2400, 6*1008, 3456, 6, 1, *ev_badch );
    
    process( ev_adc->Image2DArray(), badch_v, adc_threshold, check_wire_intersection );
    
  } 
  
  /**
   * @brief make the possible hit triplets from all the wire plane images
   *
   * this function is expected to populate the following data members:
   *  _sparseimg_vv
   *  _imgmeta_v
   *  _triplet_v
   *  _flowdir_v
   *
   * @param[in] adc_v   Vector of wire plane images.
   * @param[in] badch_v Vector of bad channel images.
   * @param[in] adc_threshold Threshold value for pixels we will consider.
   * @param[in] check_wire_intersection Check that triplet produces a good wire intersection inside the TPC.
   *                                    Also saves the 3D position for each triplet. (Makes slower; For debug.)
   */
  void PrepMatchTriplets::process( const std::vector<larcv::Image2D>& adc_v,
                                   const std::vector<larcv::Image2D>& badch_v,
                                   const float adc_threshold,
                                   const bool check_wire_intersection )
  {

    std::clock_t start = std::clock();
    
    // first we make a common sparse image
    _sparseimg_vv = larflow::prep::FlowTriples::make_initial_sparse_image( adc_v, adc_threshold );
    _imgmeta_v.clear();
    for ( auto const& img : adc_v )
      _imgmeta_v.push_back( img.meta() );
      

    // then we make the flow triples for each of the six flows
    // we order the flows based on the quality of the planes, with the Y plane being better
    
    FlowDir_t flow_order[] = { kY2V, kY2U, kV2Y, kU2Y, kU2V, kV2U };
    
    std::vector< larflow::prep::FlowTriples > triplet_v( larflow::kNumFlows );
    for (int flowindex=0; flowindex<(int)larflow::kNumFlows; flowindex++) {

      // if ( flowindex!=kV2Y )
      //   continue;
      
      int sourceplane, targetplane;
      larflow::LArFlowConstants::getFlowPlanes( (larflow::FlowDir_t)flowindex, sourceplane, targetplane );

      // save pixel position, rather than index: we will reindex after adding dead channels
      triplet_v[flowindex]  = FlowTriples( sourceplane, targetplane,
                                           adc_v, badch_v,
                                           _sparseimg_vv, 10.0, false );
    }

    // collect the unique dead channel additions to each plane
    std::set< std::pair<int,int> > deadpixels_to_add[ adc_v.size() ];

    // add the unique dead channel additions
    std::cout << "sparse pixel totals before deadch additions: "
              << "(" << _sparseimg_vv[0].size() << "," << _sparseimg_vv[1].size() << "," << _sparseimg_vv[2].size() << ")"
              << std::endl;

    for (int flowindex=0; flowindex<(int)larflow::kNumFlows; flowindex++) {
      
      // if ( flowindex!=kV2Y )
      //   continue;

      auto& triplet = triplet_v[flowindex];
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
    
    std::cout << "sparse pixel totals after deadch additions: "
              << "(" << _sparseimg_vv[0].size() << "," << _sparseimg_vv[1].size() << "," << _sparseimg_vv[2].size() << ")"
              << std::endl;
    
    // sort all pixels
    for ( auto& pix_v : _sparseimg_vv ) {
      std::sort( pix_v.begin(), pix_v.end() );
    }
    
    // condense and reindex matches
    std::set< std::vector<int> > triplet_set;
    _triplet_v.clear();
    _flowdir_v.clear();
    _triarea_v.clear();
    _pos_v.clear();
    _triplet_v.reserve( 200000 );
    _flowdir_v.reserve( 200000 );
    _triarea_v.reserve( 200000 );
    _pos_v.reserve( 200000 );

    int   n_not_crosses = 0;
    float n_bad_triarea = 0;

    for (int iflow=0; iflow<(int)larflow::kNumFlows; iflow++) {
      
      larflow::FlowDir_t flowdir=flow_order[iflow];

      // for debug
      // if ( flowdir!=kY2V )
      //   continue;
      
      auto& triplet_data = triplet_v[ (int)flowdir ];
      
      int srcplane = triplet_data.get_source_plane_index();
      int tarplane = triplet_data.get_target_plane_index();
      int othplane = triplet_data.get_other_plane_index();

      std::cout << "[PrepMatchTriplets] combine matches from flow "
                << larflow::LArFlowConstants::getFlowName(flowdir)
                << " planes={" << srcplane << " -> " << tarplane << ", " << othplane << "}"
                << std::endl;            
      
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

        // check triplet 3d consistency, using ublarcvapp tool
        int crosses = 1;
        double tri_area = 0.;
        std::vector<float> intersection = {0,0,0};
        
        if ( check_wire_intersection ) {

          UInt_t src_ch = larutil::Geometry::GetME()->PlaneWireToChannel( (UInt_t)srcplane, (UInt_t)trip[0]);
          UInt_t tar_ch = larutil::Geometry::GetME()->PlaneWireToChannel( (UInt_t)tarplane, (UInt_t)trip[1] );
          Double_t y,z;
          bool crosses  = larutil::Geometry::GetME()->ChannelsIntersect( src_ch, tar_ch, y, z );

          if ( !crosses ) {
            n_not_crosses++;
            continue;
          }          
          intersection[0] = (adc_v[0].meta().pos_y( imgcoord_v[3] )-3200.0)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
          intersection[1] = y;
          intersection[2] = z;
          
        }
        
        
        auto it_trip = triplet_set.find( imgcoord_v );
        if ( it_trip==triplet_set.end() ) {
          triplet_set.insert( imgcoord_v );

          std::vector<int> imgindex_v(4);
          imgindex_v[ srcplane ] = it_src - _sparseimg_vv[srcplane].begin();
          imgindex_v[ tarplane ] = it_tar - _sparseimg_vv[tarplane].begin();
          imgindex_v[ othplane ] = it_oth - _sparseimg_vv[othplane].begin();
          imgindex_v[ 3 ]        = trip[3];

          _triplet_v.push_back( imgindex_v );
          _flowdir_v.push_back( flowdir );
          if ( check_wire_intersection) {
            _triarea_v.push_back( tri_area );
            _pos_v.push_back( intersection );
          }
        }
        
      }

    }

    std::clock_t end = std::clock();
    std::cout << "[PrepMatchTriplets] made total of " << _triplet_v.size()
              << " unique index triplets. time elapsed=" << float(end-start)/float(CLOCKS_PER_SEC)
              << std::endl;
    std::cout << "[PrepMatchTriplets] number removed for not intersecting: " << n_not_crosses << std::endl;

  }//end of process method

  /**
   * @brief plot the sparse image pixels in a th2d
   *
   * @param[in] hist_stem_name Stem of name given to generated histograms.
   * @return    vector of TH2D that visualize the sparse images.
   *
   */
  std::vector<TH2D> PrepMatchTriplets::plot_sparse_images( std::string hist_stem_name )
  {
    std::vector<TH2D> out_v;
    for ( int p=0; p<(int)_imgmeta_v.size(); p++ ) {
      std::stringstream ss;
      ss << "htriples_plane" << p << "_" << hist_stem_name;
      auto const& meta = _imgmeta_v[p];
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
   * @brief use larflow truth images to assign good versus bad match for triplets
   *
   * this method populates the values for:
   * _truth_v: 1 or 0 if triplet is a correct match
   * _truth_2plane_v: value for all 6 flows, 1 or 0 if correct flow
   *
   * There should be a value of either (1) for correct triplet
   *  and (2) for false triplet
   *
   * @param[in] larflow_v Vector of Image2D which contain true flow information between planes.
   *
   */
  void PrepMatchTriplets::make_truth_vector( const std::vector<larcv::Image2D>& larflow_v )
  {

    _truth_v.clear();
    _truth_2plane_v.clear();
    _truth_v.resize( _triplet_v.size(), 0 );
    _truth_2plane_v.resize( _triplet_v.size() );

    const int true_match_span = 3;
    const int min_required_connections = 1;

    int ntriplet_truth = 0;
    std::vector< int > ndoublet_truth( (int)larflow::kNumFlows, 0 );
    
    for ( size_t itrip=0; itrip<_triplet_v.size(); itrip++ ) {
      // for each triplet, we look for truth flows that connect the planes
      auto const& triplet = _triplet_v[itrip];
      larflow::FlowDir_t flow_dir_origin = _flowdir_v[itrip];
      _truth_2plane_v[itrip].resize( (int)larflow::kNumFlows, 0 );
      
      // for debug
      //if ( flow_dir_origin!=kY2V ) continue;

      int srcplane, tarplane;
      larflow::LArFlowConstants::getFlowPlanes( flow_dir_origin, srcplane, tarplane );
      int othplane = larflow::LArFlowConstants::getOtherPlane( srcplane, tarplane );
      
      std::vector< const FlowTriples::PixData_t* > pix_v( _sparseimg_vv.size() );
      pix_v[srcplane] = &_sparseimg_vv[srcplane][ triplet[srcplane] ];
      pix_v[tarplane] = &_sparseimg_vv[tarplane][ triplet[tarplane] ];
      pix_v[othplane] = &_sparseimg_vv[othplane][ triplet[othplane] ]; 

      int ngood_connections = 0;
      float pixflow  = larflow_v[(int)flow_dir_origin].pixel( pix_v[srcplane]->row, pix_v[srcplane]->col );
      int target_col = pix_v[srcplane]->col + (int)pixflow;
      if ( fabs(pixflow)!=0 && pixflow>-3999 && abs(target_col-pix_v[tarplane]->col)<true_match_span ) {
        ngood_connections++;
        _truth_2plane_v[itrip][(int)flow_dir_origin] = 1;
        ndoublet_truth[(int)flow_dir_origin]++;
      }
      
      if ( ngood_connections>=min_required_connections ) {
        _truth_v[itrip] = 1;
        ntriplet_truth++;
      }
      else {
        _truth_v[itrip] = 0;
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
   * @brief plot truth image for debug
   *
   * @param[in] hist_stem_name Stem of name given to histograms made.
   * @return Vector of TH2D that plots the information.
   *
   */
  std::vector<TH2D> PrepMatchTriplets::plot_truth_images( std::string hist_stem_name )
  {
    std::vector<TH2D> out_v;

    for ( int p=0; p<(int)_imgmeta_v.size(); p++ ) {
      std::stringstream ss;
      ss << "htriples_truth_plane" << p << "_" << hist_stem_name;
      auto const& meta = _imgmeta_v[p];
      TH2D hist( ss.str().c_str(), "",
                 meta.cols(), meta.min_x(), meta.max_x(),
                 meta.rows(), meta.min_y(), meta.max_y() );

      out_v.emplace_back( std::move(hist) );
    }
    
    for (int i=0; i<(int)_triplet_v.size(); i++ ) {
      auto& trip  = _triplet_v[i];
      auto& truth = _truth_v[i];
      std::vector< const FlowTriples::PixData_t* > pix_v( trip.size(), 0 );
      for (int p=0; p<(int)_imgmeta_v.size(); p++ ) {
        pix_v[p] = &_sparseimg_vv[p][ trip[p] ];
        int col = pix_v[p]->col+1;
        int row = pix_v[p]->row+1;
        if ( out_v[p].GetBinContent( col+1, row+1 )<10 )
          out_v[p].SetBinContent( col+1, row+1, 1 + 10*truth );
      }
    }
    return out_v;
  }

  /**
   * @brief return a numpy array containing the sparse image information
   *
   * @param[in] plane Plane index for sparse image requested.
   * @return numpy array with shape (N,3) containing info from sparse matrix. each row contains (row,col,pixel value).
   */
  PyObject* PrepMatchTriplets::make_sparse_image( int plane ) {
    
    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    npy_intp* dims = new npy_intp[2];
    dims[0] = (int)_sparseimg_vv[plane].size();

    // if we want truth, we include additional value with 1=correct match, 0=false    
    dims[1] = 3;

    // output array
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( 2, dims, NPY_FLOAT );

    for ( size_t idx=0; idx<_sparseimg_vv[plane].size(); idx++ ) {
      *((float*)PyArray_GETPTR2( array, (int)idx, 0)) = (float)_sparseimg_vv[plane][idx].row;
      *((float*)PyArray_GETPTR2( array, (int)idx, 1)) = (float)_sparseimg_vv[plane][idx].col;
      *((float*)PyArray_GETPTR2( array, (int)idx, 2)) = (float)_sparseimg_vv[plane][idx].val;      
    }

    
    return (PyObject*)array;
  }

  /**
   * @brief return a numpy array with indices to the sparse matrix object.
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

    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    int nd = 2;
    int ndims2 = 3;
    npy_intp dims[] = { max_num_samples, ndims2 };

    // output array
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_LONG );

    int srcplane,tarplane;
    larflow::LArFlowConstants::getFlowPlanes( kdir, srcplane, tarplane );

    // number of pairs we've stored
    nsamples = 0;
    
    int end_idx = start_idx + max_num_samples;
    end_idx = ( end_idx>(int)idx_v.size() )   ?  idx_v.size() : end_idx; // cap to number of indices

    std::cout << "[PrepMatchTriplets::make_2plane_match_array] src=" << srcplane << " tar=" << tarplane << " withtruth=" << withtruth << " "
              << "make numpy array with indices from triplets[" << start_idx << ":" << end_idx << "]" << std::endl;
    
    for ( int idx=start_idx; idx<end_idx; idx++ ) {
      int tripidx = idx_v[idx];
      *((long*)PyArray_GETPTR2( array, nsamples, 0)) = (long)_triplet_v[tripidx][srcplane];
      *((long*)PyArray_GETPTR2( array, nsamples, 1)) = (long)_triplet_v[tripidx][tarplane];
      if ( withtruth ) {
        *((long*)PyArray_GETPTR2( array, nsamples, 2)) = (long)_truth_2plane_v[tripidx][(int)kdir];
      }
      else {
        *((long*)PyArray_GETPTR2( array, nsamples, 2)) = 0;
      }
      nsamples++;
      if (nsamples==max_num_samples)
        break;

    }//end of indices loop

    std::cout << "[PrepMatchTriplets::make_2plane_match_array] nsamples=" << nsamples << std::endl;

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
   * @brief randomly select a set of 2 plane indices
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
   * @brief get sequential set of matches
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
      idx_v[i-start_index] = (int)i;
    }

    return make_2plane_match_array( kdir, max_num_pairs, idx_v, 0, with_truth, num_pairs_filled );

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
  PyObject* PrepMatchTriplets::make_triplet_array( const int max_num_samples,
                                                   const std::vector<int>& idx_v,
                                                   const int start_idx,
                                                   const bool withtruth,
                                                   int& nsamples )
  {

    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    int nd = 2;
    int ndims2 = 5;
    npy_intp dims[] = { max_num_samples, ndims2 };

    // output array
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_LONG );

    // number of pairs we've stored
    nsamples = 0;
    
    int end_idx = start_idx + max_num_samples;
    end_idx = ( end_idx>(int)idx_v.size() )   ?  idx_v.size() : end_idx; // cap to number of indices

    std::cout << "[PrepMatchTriplets::make_triplet_array] withtruth=" << withtruth << " "
              << "make numpy array with indices from triplets[" << start_idx << ":" << end_idx << "]"
              << std::endl;
    
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

    std::cout << "[PrepMatchTriplets::make_triplet_array] nsamples=" << nsamples << std::endl;

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
  PyObject* PrepMatchTriplets::sample_triplet_matches( const int& nsamples,
                                                       int& nfilled,
                                                       bool withtruth ) {

    std::vector<int> idx_v( _triplet_v.size() );
    for ( size_t i=0; i<_triplet_v.size(); i++ ) idx_v[i] = (int)i;
    unsigned seed =  std::chrono::system_clock::now().time_since_epoch().count();
    shuffle (idx_v.begin(), idx_v.end(), std::default_random_engine(seed));
    
    return make_triplet_array( nsamples, idx_v, 0, withtruth, nfilled );

  }
  
  /**
   *
   * @brief get sequential set of triplet indices
   *
   */
  PyObject* PrepMatchTriplets::get_chunk_triplet_matches( const int& start_index,
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
   *
   * @brief select sample biased towards triplets that score poorly in past iteration of network.
   * 
   * For hard-example training.
   *
   */
  PyObject* PrepMatchTriplets::sample_hard_example_matches( const int& nsamples,
                                                            const int& nhard_samples,
                                                            PyObject* triplet_scores,                                                            
                                                            int& nfilled,
                                                            bool withtruth ) {

    // if number of triplets less than the requested triplet sample, just pass the triplets back
    if (nsamples<=(int)_triplet_v.size()) {
      return sample_triplet_matches( nsamples, nfilled, withtruth );
    }
    
    std::vector<int> idx_v( _triplet_v.size(), 0 );
    for ( size_t i=0; i<_triplet_v.size(); i++ ) idx_v[i] = (int)i;
    unsigned seed =  std::chrono::system_clock::now().time_since_epoch().count();
    shuffle (idx_v.begin(), idx_v.end(), std::default_random_engine(seed));


    TRandom3 sampler( seed );
    // now we sample for bad events
    int nbad = 0;
    int nsaved =  0;
    std::set<int> saved_idx_set;
    std::vector<int> saved_idx_v( _triplet_v.size(), 0 );
    for (size_t i=0; i<_triplet_v.size(); i++) {
      int idx = idx_v[i];
      float past_score = *((float*)PyArray_GETPTR1( (PyArrayObject*)triplet_scores, (int)idx ));
      float weight = fabs(past_score-(float)_truth_v[idx]);
      if (nbad<nhard_samples) {
        // first try to get a set number of bad examples
        if ( sampler.Uniform()>weight ) {
          saved_idx_v[nsaved] = idx;        
          saved_idx_set.insert(idx);
          nsaved++;
          if ( weight>0.5 )
            nbad++;
        }
      }
      else {
        saved_idx_v[nsaved] = idx;
        saved_idx_set.insert(idx);
        nsaved++;
      }
    }

    if ( nsaved<nsamples ) {
      // go back and grab triplets we haven't used
      for (size_t i=0; i<_triplet_v.size(); i++) {
        int idx = idx_v[i];        
        if ( saved_idx_set.find(idx)!=saved_idx_set.end() ) {
          continue;
        }
        saved_idx_v[nsaved] = idx;
        saved_idx_set.insert(idx);
        nsaved++;
        if ( nsaved==nsamples )
          break;
      }
    }
    saved_idx_v.resize(nsaved);

    return make_triplet_array( nsamples, saved_idx_v, 0, withtruth, nfilled );

  }
  
 
}  
}
