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
    _match_span_v.clear();
    _match_span_v.resize( _triplet_v.size(), 0 );    

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
      int match_span = abs(target_col-pix_v[tarplane]->col);
      if ( match_span>10 )
	match_span = 10;

      if ( fabs(pixflow)!=0 && pixflow>-3999 && match_span<true_match_span ) {
        ngood_connections++;
        _truth_2plane_v[itrip][(int)flow_dir_origin] = 1;
        ndoublet_truth[(int)flow_dir_origin]++;
      }
      _match_span_v[itrip] = match_span;
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
   * @brief use instance ID image to label space points
   *
   * @param[in] instance_img_v Vector of Image2D which contain track id.
   *
   */
  void PrepMatchTriplets::make_instanceid_vector( const std::vector<larcv::Image2D>& instance_img_v )
  {

    _instance_id_v.resize( _triplet_v.size(), 0);

    int nids = 0;
    
    for ( size_t itrip=0; itrip<_triplet_v.size(); itrip++ ) {

      if ( _truth_v[itrip]==0 ) {
        _instance_id_v[itrip] = 0;
        continue;
      }
      
      // for each triplet, we look for truth flows that connect the planes
      auto const& triplet = _triplet_v[itrip];
      std::vector<int> imgcoord = {0,0,0,triplet[3]};
      for (int p=0; p<3; p++ ) {
        imgcoord[p] = _sparseimg_vv[p][triplet[p]].col;
      }

      std::map< int, int > id_votes;

      for (int p=0; p<3; p++ ) {
        long plane_id = instance_img_v[p].pixel( imgcoord[3], imgcoord[p], __FILE__, __LINE__ );
        
        if ( plane_id>0 ) {
          // see if its in the daughter to mother map
          auto it_showermap = _shower_daughter2mother.find( plane_id );
          if ( it_showermap != _shower_daughter2mother.end() ) {
            plane_id = (long)it_showermap->second;
          }
        }
        
        if ( id_votes.find(plane_id)==id_votes.end() )
          id_votes[plane_id] = 0;
        id_votes[plane_id] += 1;
      }

      int maxid = 0;
      int nvotes = 0;
      for (auto it=id_votes.begin(); it!=id_votes.end(); it++) {
        if ( it->first>0 && it->second>nvotes) {
          nvotes = it->second;
          maxid = it->first;
        }
      }

      if ( maxid>0 )
        nids++;

      _instance_id_v[itrip] = maxid;
      
    }//end of trips loop
    
    std::cout << "[PrepMatchTriplets::make_instanceid_vector] " << std::endl;
    std::cout << "  number labeled: " << nids << " of " << _triplet_v.size() << std::endl;
              
    
  }

  /**
   * @brief use ancestor ID image to label space points
   *
   * @param[in] ancestor_img_v Vector of Image2D which contain track id.
   *
   */
  void PrepMatchTriplets::make_ancestorid_vector( const std::vector<larcv::Image2D>& ancestor_img_v )
  {

    _ancestor_id_v.resize( _triplet_v.size(), 0);

    int nids = 0;
    int ntrue_triplets =0;
    
    for ( size_t itrip=0; itrip<_triplet_v.size(); itrip++ ) {

      if ( _truth_v[itrip]==0 ) {
        _ancestor_id_v[itrip] = 0;
        continue;
      }
        
      ntrue_triplets++;
      
      // for each triplet, we look for truth flows that connect the planes
      auto const& triplet = _triplet_v[itrip];
      std::vector<int> imgcoord = {0,0,0,triplet[3]};
      for (int p=0; p<3; p++ ) {
        imgcoord[p] = _sparseimg_vv[p][triplet[p]].col;
      }

      std::map< int, int > id_votes;

      for (int p=0; p<3; p++ ) {
        int plane_id = ancestor_img_v[p].pixel( imgcoord[3], imgcoord[p], __FILE__, __LINE__ );
        if ( id_votes.find(plane_id)==id_votes.end() )
          id_votes[plane_id] = 0;
        id_votes[plane_id] += 1;
      }

      int maxid = 0;
      int nvotes = 0;
      for (auto it=id_votes.begin(); it!=id_votes.end(); it++) {
        if ( it->first>0 && it->second>nvotes) {
          nvotes = it->second;
          maxid = it->first;
        }
      }

      if ( maxid>0 )
        nids++;

      _ancestor_id_v[itrip] = maxid;
      
    }//end of trips loop
    
    std::cout << "[PrepMatchTriplets::make_ancestorid_vector] " << std::endl;
    std::cout << "  number labeled: " << nids
	      << " of " << _triplet_v.size() << " total "
	      << " and " << ntrue_triplets << "true" << std::endl;
    
  }

  /**
   * @brief use segment image to label particle class
   *
   * The id numbers contained in segment image correspond to enum values of larcv::ROIType_t.
   * ROIType_t is found in larcv/core/DataFormat/DataFormatTyps.h
   * 
   * @param[in] segment_img_v Vector of Image2D which containing larcv particle IDs
   *
   */
  void PrepMatchTriplets::make_segmentid_vector( const std::vector<larcv::Image2D>& segment_img_v,
                                                 const std::vector<larcv::Image2D>& adc_v )
  {

    _pdg_v.resize( _triplet_v.size(), 0);
    _origin_v.resize( _triplet_v.size(), 0 );

    int nids = 0;
    
    for ( size_t itrip=0; itrip<_triplet_v.size(); itrip++ ) {

      if ( _truth_v[itrip]==0 ) {
        _pdg_v[itrip] = 0;
        _origin_v[itrip] = 0;
        continue;
      }
      
      // for each triplet, we look for truth flows that connect the planes
      auto const& triplet = _triplet_v[itrip];
      std::vector<int> imgcoord = {0,0,0,triplet[3]};
      for (int p=0; p<3; p++ ) {
        imgcoord[p] = _sparseimg_vv[p][triplet[p]].col;
      }

      std::map< int, float > id_votes;
      int nsegvotes = 0;

      for (int p=0; p<3; p++ ) {
        int plane_id = segment_img_v[p].pixel( imgcoord[3], imgcoord[p], __FILE__, __LINE__ );
        float pixval = adc_v[p].pixel( imgcoord[3], imgcoord[p], __FILE__, __LINE__ );
        if ( id_votes.find(plane_id)==id_votes.end() )
          id_votes[plane_id] = 0;
        //id_votes[plane_id] += pixval;
        id_votes[plane_id] += 1;
	if ( plane_id>=(int)larcv::kROIEminus )
	  nsegvotes++;
      }

      // also check the instance image for an id
      if ( _instance_id_v.size()>itrip ) {
        int iid = _instance_id_v[itrip];        
        auto it_class = _instance2class_map.find( iid );
        if ( it_class!=_instance2class_map.end() ) {
          int pid = it_class->second;
          if ( id_votes.find(pid)==id_votes.end() )
            id_votes[pid] = 0;
          id_votes[pid] += 10; // basically overrides segment image
        }
      }
      
      int maxid = 0;
      float nvotes = 0;
      for (auto it=id_votes.begin(); it!=id_votes.end(); it++) {
        if ( it->first>0 && it->second>0 && (it->second>nvotes) ) {
          nvotes = it->second;
          maxid = it->first;
        }
      }
      
      if ( maxid>0 )
        nids++;

      _pdg_v[itrip] = maxid;
      
      if ( nsegvotes>2 ) // need 3 planes with segment data (no dead wires on the segment images)
        _origin_v[itrip] = 1;
      else
        _origin_v[itrip] = 0;
      
    }//end of trips loop
    
    std::cout << "[PrepMatchTriplets::make_segmentid_vector] " << std::endl;
    std::cout << "  number labeled: " << nids << " of " << _triplet_v.size() << std::endl;
    
  }

  /**
   * @brief use mcreco data to label spacepoint origin
   *
   * Set flag as:
   *  origin=0 : unknown
   *  origin=1 : neutrino
   *  origin=2 : cosmic
   * 
   * @param[in] segment_img_v Vector of Image2D which containing larcv particle IDs
   *
   */
  void PrepMatchTriplets::make_origin_vector_frommcreco( larlite::storage_manager& ioll )    
  {

    _origin_v.resize( _triplet_v.size(), 0 );
    
    if ( _ancestor_id_v.size() != _triplet_v.size() ) {
      throw std::runtime_error("Ancestor ID vector not filled yet. Run make_ancestorid_vector first.");
    }

    // need instance to ancestor map
    std::map< int, int > trackid_to_ancestor;
    std::map< int, int > ancestor_to_origin;
    larlite::event_mctrack* ev_mctrack
      = (larlite::event_mctrack*)ioll.get_data( larlite::data::kMCTrack, "mcreco");
    for ( auto const& mctrack : *ev_mctrack ) {
      trackid_to_ancestor[(int)mctrack.TrackID()] = (int)mctrack.AncestorTrackID();
      if ( mctrack.TrackID()==mctrack.AncestorTrackID() ) {
	ancestor_to_origin[(int)mctrack.TrackID()] = mctrack.Origin();
      }
    }
    
    int nids = 0;
    
    for ( size_t itrip=0; itrip<_triplet_v.size(); itrip++ ) {

      if ( _truth_v[itrip]==0 ) {
        _origin_v[itrip] = 0;
        continue;
      }

      int trackid = _instance_id_v[itrip];
      auto it_t2a = trackid_to_ancestor.find( trackid );
      if ( it_t2a!=trackid_to_ancestor.end() ) {
	// found ancestor label
	int aid = it_t2a->second;
	auto it_a2o = ancestor_to_origin.find(aid);
	if ( it_a2o!=ancestor_to_origin.end() ) {
	  _origin_v[itrip] = it_a2o->second;
	  nids++;
	}
      }
      
    }//end of trips loop
    
    std::cout << "[PrepMatchTriplets::make_origin_vector_frommcreco] " << std::endl;
    std::cout << "  number labeled: " << nids << " of " << _triplet_v.size() << std::endl;
    
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
   * use a vector with indices of match pair to choose matches.   
   *
   * @param[in]  kdir Two-plane  flow direction to return data for
   * @param[in]  max_num_samples maximum number of flow samples to return
   * @param[in]  idx_v           list of indices. can be larger than the requested number of samples.
   * @param[in]  start_idx       index of idx_v to start samples
   * @param[in]  withtruth       return array with truth label
   * @param[out] nsamples        number of flow samples returned
   * @return numpy array with indices
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
   * @param[in]  kdir      Two-plane flow direction to return data for
   * @param[in]  nsamples  maximum number of samples requested
   * @param[in]  nfilled   number of samples returned
   * @param[in]  withtruth array should return truth label
   * @return numpy array with indices
   * 
   */
  PyObject* PrepMatchTriplets::sample_2plane_matches( larflow::FlowDir_t kdir,
                                                      const int& nsamples,
                                                      int& nfilled,
                                                      bool withtruth ) {

    std::vector<int> idx_v( _triplet_v.size() );
    for ( size_t i=0; i<_triplet_v.size(); i++ ) idx_v[i] = (int)i;
    if ( _kshuffle_indices_when_sampling ) {
      unsigned seed =  std::chrono::system_clock::now().time_since_epoch().count();
      shuffle (idx_v.begin(), idx_v.end(), std::default_random_engine(seed));
    }

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

    // std::cout << "[PrepMatchTriplets::make_triplet_array] withtruth=" << withtruth << " "
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

    //std::cout << "[PrepMatchTriplets::make_triplet_array] nsamples=" << nsamples << std::endl;

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
    if ( _kshuffle_indices_when_sampling ) {
      shuffle (idx_v.begin(), idx_v.end(), std::default_random_engine(seed));
    }
    
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

  /**
   * @brief make all the initial truth labels
   *
   * runs all the truth label making functions. 
   *
   * @param[in] iolcv larcv IO manager containing event data
   */
  void PrepMatchTriplets::process_truth_labels( larcv::IOManager& iolcv,
                                                larlite::storage_manager& ioll,
                                                std::string wire_producer )
  {
    
    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,wire_producer);
    larcv::EventImage2D* ev_larflow =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"larflow");
    larcv::EventImage2D* ev_instance =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"instance");
    larcv::EventImage2D* ev_ancestor =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"ancestor");
    larcv::EventImage2D* ev_segment =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"segment");


    larlite::event_mctrack* ev_mctrack
      = (larlite::event_mctrack*)ioll.get_data(larlite::data::kMCTrack, "mcreco" );
    larlite::event_mcshower* ev_mcshower
      = (larlite::event_mcshower*)ioll.get_data(larlite::data::kMCShower, "mcreco" );
    fill_daughter2mother_map( *ev_mcshower );
    fill_class_map( *ev_mctrack, *ev_mcshower );
    
    make_truth_vector( ev_larflow->as_vector() );
    make_instanceid_vector( ev_instance->as_vector() );
    make_ancestorid_vector( ev_ancestor->as_vector() );
    make_segmentid_vector( ev_segment->as_vector(), ev_adc->as_vector() );
    make_origin_vector_frommcreco( ioll );
  }

  /**
   * @brief Make ndarray using only true triplets
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
  PyObject* PrepMatchTriplets::make_triplet_ndarray( bool true_pt_only )
  {

    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }


    if ( _triplet_v.size()!=_truth_v.size()
         || _triplet_v.size()!=_instance_id_v.size()
         || _triplet_v.size()!=_pos_v.size()
         || _triplet_v.size()!=_pdg_v.size() ) {
      std::stringstream ss;
      ss << "[PrepMatchTriplets::make_triplet_ndarray] "
         << "truth vectors (truth=" << _truth_v.size() << ", "
         << "instance_id=" << _instance_id_v.size() << ", "
         << "pdg=" << _pdg_v.size() << ", "
         << "pos=" << _pos_v.size() << ") "
         << " do not match triplet_v size = " << _triplet_v.size()
         << std::endl;
      throw std::runtime_error( ss.str() );
    }
    
    int ntruepts = 0;
    for (auto const& truth : _truth_v ) {
      if ( truth ) ntruepts++;
    }

    std::cout << "[PrepMatchTriplets::make_truthonly_triplet_ndarray] number of true points: " << ntruepts << std::endl;

    long int npts = (true_pt_only) ? ntruepts : _triplet_v.size();
    
    // space point
    npy_intp spacepoint_t_dim[] = { npts, 3 };
    PyArrayObject* spacepoint_t = (PyArrayObject*)PyArray_SimpleNew( 2, spacepoint_t_dim, NPY_FLOAT );
    PyObject *spacepoint_t_key = Py_BuildValue("s", "spacepoint_t");

    int ifilled = 0;
    std::vector<int> idx_v(npts);
    for (size_t i=0; i<_truth_v.size(); i++) {
      if ( (true_pt_only && _truth_v[i]==1) || !true_pt_only ) {
        for (int j=0; j<3; j++) {
          *((float*)PyArray_GETPTR2(spacepoint_t,ifilled,j)) = _pos_v[i][j];
        }
        idx_v[ifilled] = i;
        ifilled++;
      }
      if (ifilled>=npts)
        break;
    }
    if ( ifilled!=npts ) {
      throw std::runtime_error("mismatch in num expected filled points");
    }
    
    // image coords
    npy_intp imgcoord_t_dim[] = { (long int)npts, 4 };
    PyArrayObject* imgcoord_t = (PyArrayObject*)PyArray_SimpleNew( 2, imgcoord_t_dim, NPY_LONG );
    PyObject *imgcoord_t_key = Py_BuildValue("s", "imgcoord_t");

    // instance label
    npy_intp instance_t_dim[] = { (long int)npts };
    PyArrayObject* instance_t = (PyArrayObject*)PyArray_SimpleNew( 1, instance_t_dim, NPY_LONG );
    PyObject *instance_t_key = Py_BuildValue("s", "instance_t");

    // particle class label
    npy_intp segment_t_dim[] = { (long int)npts };
    PyArrayObject* segment_t = (PyArrayObject*)PyArray_SimpleNew( 1, segment_t_dim, NPY_LONG );
    PyObject *segment_t_key = Py_BuildValue("s", "segment_t");

    // origin label
    npy_intp origin_t_dim[] = { (long int)npts };
    PyArrayObject* origin_t = (PyArrayObject*)PyArray_SimpleNew( 1, origin_t_dim, NPY_LONG );
    PyObject *origin_t_key = Py_BuildValue("s", "origin_t");
    
    // truth label for triplet
    npy_intp truth_t_dim[] = { (long int)npts };
    PyArrayObject* truth_t = (PyArrayObject*)PyArray_SimpleNew( 1, truth_t_dim, NPY_LONG );
    PyObject *truth_t_key = Py_BuildValue("s", "truetriplet_t");

    // truth span for triplet
    npy_intp span_t_dim[] = { (long int)npts };
    PyArrayObject* span_t = (PyArrayObject*)PyArray_SimpleNew( 1, span_t_dim, NPY_LONG );
    PyObject *span_t_key = Py_BuildValue("s", "truespan_t");
    
    ifilled = 0;
    for (auto& idx : idx_v ) {
      auto const& triplet = _triplet_v[idx];
      *((long*)PyArray_GETPTR2(imgcoord_t,ifilled,3)) = triplet[3];
      for (int p=0; p<3; p++ ) {
        *((long*)PyArray_GETPTR2(imgcoord_t,ifilled,p)) = _sparseimg_vv[p][triplet[p]].col;
      }
      *((long*)PyArray_GETPTR1(instance_t,ifilled)) = _instance_id_v[idx];
      *((long*)PyArray_GETPTR1(segment_t,ifilled))  = _pdg_v[idx];
      *((long*)PyArray_GETPTR1(truth_t,ifilled))    = (long)_truth_v[idx];
      *((long*)PyArray_GETPTR1(origin_t,ifilled))   = (long)_origin_v[idx];
      *((long*)PyArray_GETPTR1(span_t,ifilled))     = (long)_match_span_v[idx];      
      ifilled++;
    }

    // Create and cfill dictionary
    PyObject *d = PyDict_New();
    PyDict_SetItem(d, spacepoint_t_key, (PyObject*)spacepoint_t);
    PyDict_SetItem(d, imgcoord_t_key,   (PyObject*)imgcoord_t);
    PyDict_SetItem(d, instance_t_key,   (PyObject*)instance_t);
    PyDict_SetItem(d, segment_t_key,    (PyObject*)segment_t);
    PyDict_SetItem(d, origin_t_key,     (PyObject*)origin_t);
    PyDict_SetItem(d, truth_t_key,      (PyObject*)truth_t);
    PyDict_SetItem(d, span_t_key,       (PyObject*)span_t);         

    Py_DECREF( spacepoint_t );
    Py_DECREF( imgcoord_t );
    Py_DECREF( instance_t );
    Py_DECREF( segment_t );
    Py_DECREF( truth_t );    
    Py_DECREF( spacepoint_t_key );
    Py_DECREF( imgcoord_t_key );
    Py_DECREF( instance_t_key );
    Py_DECREF( segment_t_key );
    Py_DECREF( origin_t_key );
    Py_DECREF( truth_t_key );
    Py_DECREF( span_t_key );

    return d;
  }

  /**
   * @brief utility function to get imgcoord of triplet
   *
   * @param[in] idx_triplet Index of triplet to return info for.
   * @return a vector<int> containing (col,col,col,row)
   *
   */
  std::vector<int> PrepMatchTriplets::get_triplet_imgcoord_rowcol( int idx_triplet )
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
   * @brief stores map between daughter geant4 track ids to mother ids
   *
   * Used in conjuction with the 2D instance image labels.
   *
   */
  void PrepMatchTriplets::fill_daughter2mother_map( const std::vector<larlite::mcshower>& shower_v )
  {

    _shower_daughter2mother.clear();
    
    for (auto const& shower : shower_v ) {
      long showerid = shower.TrackID();
      if ( showerid<0 ) showerid *= -1;
      const std::vector<unsigned int>& dlist = shower.DaughterTrackID();      
      for (auto const& daughterid : dlist ) {
        _shower_daughter2mother[(unsigned long)daughterid]= (unsigned long)showerid;
      }
    }
    
  }

  /**
   * @brief stores map between instance id and particle class
   *
   * Used in conjuction with the SSNet class labels
   *
   */
  void PrepMatchTriplets::fill_class_map( const std::vector<larlite::mctrack>&  track_v,
                                          const std::vector<larlite::mcshower>& shower_v )
  {

    _instance2class_map.clear();
    
    for (auto const& shower : shower_v ) {
      long showerid = shower.TrackID();
      if ( showerid<0 ) showerid *= -1;
      int  pid = shower.PdgCode();
      _instance2class_map[(unsigned long)showerid] = pid;
      for ( auto const& daughterid : shower.DaughterTrackID() ) {
        long id = (daughterid<0) ? daughterid*-1 : daughterid;
        _instance2class_map[(unsigned long)id] = pid;
      }
    }
    
    for (auto const& track : track_v ) {
      long trackid = track.TrackID();
      if ( trackid<0 ) trackid *= -1;
      int  pid = track.PdgCode();
      _instance2class_map[(unsigned long)trackid] = pid;
    }

    for (auto it=_instance2class_map.begin(); it!=_instance2class_map.end(); it++ ) {
      int larcv_class = 0;
      switch ( it->second ) {
      case 11:
      case -11:
        larcv_class = (int)larcv::kROIEminus;
        break;
      case 13:
      case -13:
        larcv_class = (int)larcv::kROIMuminus;
        break;
      case 211:
      case -211:
        larcv_class = (int)larcv::kROIPiminus;
        break;
      case 2212:
      case 2112:
        larcv_class = (int)larcv::kROIProton;
        break;
      case 22:
        larcv_class = (int)larcv::kROIGamma;
        break;
      case 111:
        larcv_class = (int)larcv::kROIPizero;
        break;
      case 130:
      case 310:
      case 311:
      case 312:
        larcv_class = (int)larcv::kROIKminus;
        break;
      default:
        larcv_class = (int)larcv::kROIUnknown;
        break;
      }
      it->second = larcv_class;
    }//end of iterator loop
    
  }

  PyObject* PrepMatchTriplets::get_all_triplet_data( const bool withtruth )
  {
    std::vector<int> idx_v(_triplet_v.size());
    for (int i=0; i<(int)_triplet_v.size(); i++)
      idx_v[i] = i;
    int nsamples = 0;
    return make_triplet_array( _triplet_v.size(), idx_v, 0, withtruth, nsamples );
  }

  /**
   * @brief Make ndarray using only true triplets
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
  PyObject* PrepMatchTriplets::make_spacepoint_charge_array()
  {

    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }


    if ( _triplet_v.size()!=_pos_v.size() ) {
      std::stringstream ss;
      ss << "[PrepMatchTriplets::make_spacepoint_charge_array] "
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
    std::cout << "[PrepMatchTriplets::make_spacepoint_charge_array] number of true points: " << ntruepts << std::endl;
    
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
      
    }

    return d;
  }
