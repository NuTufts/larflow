#include "OpModelMCDataPrep.h"

#include "ublarcvapp/MCTools/MCPos2ImageUtils.h"

#include "larlite/DataFormat/opflash.h"

namespace larflow {
namespace opticalmodel {

  bool OpModelMCDataPrep::_setup_numpy = false;

  void OpModelMCDataPrep::process( larlite::storage_manager& mgr,
				   const larflow::voxelizer::VoxelizeTriplets& voxelizer )
  {
    ublarcvapp::mctools::FlashMatcherV2::process(mgr);
    tagBadFlashMatches( voxelizer, mgr );
    filterBadMCDataMatches();
  }
  
  void OpModelMCDataPrep::tagBadFlashMatches( const larflow::voxelizer::VoxelizeTriplets& voxelizer,
					      larlite::storage_manager& ioll )
  {

    if ( _verbose_level>=1 )
      std::cout << "[OpModelMCDataPrep::tagBadFlashMatches]" << std::endl;
    
    // get mctrack list
    larlite::event_mctrack* ev_mctrack
    = (larlite::event_mctrack*)ioll.get_data(larlite::data::kMCTrack,"mcreco");
    
    // loop over matches produced by FlashMatcherV2
    // check if it passes through voxels with charge
    flash_isgood.clear();
    flash_track_frac_intpc.clear();
    flash_track_frac_intpc_w_charge.clear();
    
    flash_isgood.resize( recoflash_v.size(), 1 );
    flash_track_frac_intpc.resize( recoflash_v.size(), 0.0 );        
    flash_track_frac_intpc_w_charge.resize( recoflash_v.size(), 0.0 );

    const float stepsize = 0.5;

    // get voxelizer origin
    std::vector<float> voxel_origin   = voxelizer.get_origin();
    std::vector<float> voxel_axis_len = voxelizer.get_dim_len();
    std::vector<float> voxel_max(3,0);
    for (int i=0; i<3; i++)
      voxel_max[i] = voxel_origin[i] + voxel_axis_len[i];

    std::vector<float> tpc_min = { 0.0, -117.0, 0.0 };
    std::vector<float> tpc_max = { 256.0, +117.0, 1036.0 };
    
    for ( int iflash=0; iflash<recoflash_v.size(); iflash++ ) {

      flash_isgood[iflash] = 1; // default is to pass unless proven otherwise
      flash_track_frac_intpc[iflash] = 0.0;
      flash_track_frac_intpc_w_charge[iflash] = 0.0;
      
      
      auto& recoflash = recoflash_v.at(iflash);
      if ( _verbose_level>=2 ) {
	std::cout << "== [FLASH " << iflash << " ] ========================" << std::endl;
	printRecoMatchInfo( recoflash, iflash );
      }

      int ancestorid = recoflash.ancestorid;

      std::set< int > index_visited;
      int nvoxel_w_charge = 0;
      size_t npts_tested = 0;
      size_t npts_in_voxel = 0;
      size_t npts_out_voxel = 0;      
      size_t npts_in_tpc = 0;
      size_t npts_in_tpc_wcharge = 0;

      std::vector<int> trackid_list = recoflash.trackid_list();

      
      
      for (int itrackid=0; itrackid<(int)trackid_list.size(); itrackid++) {

	int trackid = trackid_list.at(itrackid);


	// get list of reco points
	// we get info about this track already parsed by the pgraph class
	auto pnode = mcpg.findTrackID(trackid); // returns pointer to MCPixelPGraph::Node_t struct

	if ( _verbose_level>=2 )
	  std::cout << "[node] " << mcpg.strNodeInfo( *pnode ) << std::endl;
	
	if ( pnode->type==0 && pnode->pid!=2112 ) {
	  
	  // is a track
	  if ( _verbose_level>=2 )
	    std::cout << "  [ flash " << iflash << ", trackid " << trackid << " ]" << std::endl;		  

	  // used the stored vector index to get larlite::mctrack object
	  auto const& mctrackinfo = ev_mctrack->at( pnode->vidx );
	  
	  // convert track trajectory to list of points
	  bool apply_t0_shift = true;
	  bool apply_sce = true;
	  std::vector< std::vector<float> > reco_traj
	    = ublarcvapp::mctools::MCPos2ImageUtils::Get()->getRecoSpacepoints( mctrackinfo,
										apply_t0_shift,
										apply_sce );

	  // follow along trajectory
	  for (int istep=0; istep<(int)reco_traj.size()-1; istep++) {

	    /// each point is (x,y,z,tick)
	    auto& pt = reco_traj.at(istep);
	    auto& ptnext = reco_traj.at(istep+1);

	    auto& mcstep      = mctrackinfo.at(istep);
	    auto& mcstep_next = mctrackinfo.at(istep+1);
	    std::vector<float> mcpt   = { (float)mcstep.X(), (float)mcstep.Y(), (float)mcstep.Z() };
	    std::vector<float> mcnext = {
	      (float)mcstep_next.X(),
	      (float)mcstep_next.Y(),
	      (float)mcstep_next.Z() };

	    
	    
	    float len = 0.0;
	    std::vector<float> stepdir(3,0);
	    for (int i=0; i<3; i++) {
	      stepdir[i] = ptnext[i]-pt[i];
	      len += stepdir[i]*stepdir[i];
	    }
	    int nsubsteps = len/stepsize+1;
	    float substepsize = len/nsubsteps;

	    int saw_charge = 0;
	    
	    for (int isub=0; isub<nsubsteps; isub++) {
	      float f = float(isub)/float(nsubsteps);
	      std::vector<float> testpt(3,0);
	      std::vector<float> truept(3,0);	      
	      for (int i=0; i<3; i++) {
		testpt[i] = pt[i]*(1-f) + f*ptnext[i];
		truept[i] = mcpt[i]*(1-f) + f*mcnext[i];
	      }
	      float tick = pt[3]*(1-f) + f*ptnext[3];

	      npts_tested += 1;

	      bool intpc = true;
	      for (int i=0; i<3; i++) {
		if ( truept[i]<=tpc_min[i] || truept[i]>=tpc_max[i] )
		  intpc = false;
	      }
	      if ( intpc )
		npts_in_tpc += 1;

	      // out of voxel boundary
	      bool invoxel_bounds = true;
	      for (int i=0; i<3; i++) {
		if ( testpt[i]<=voxel_origin[i] || testpt[i]>=voxel_max[i] ) {
		  invoxel_bounds = false;
		}
	      }

	      if (invoxel_bounds) {
		npts_in_voxel++;
	      }
	      else {
		npts_out_voxel++;
	      }
	      
	      // // out of the image bounds
	      // if (tick<=2400 || tick>=2400+1008*6 )
	      //continue;

	      if ( !invoxel_bounds )
		continue;
	      
	      // get the voxel our test point is in
	      std::vector<int> voxel_indices = voxelizer.get_voxel_indices( testpt );
	      int voxelindex = voxelizer.get_voxel_index( voxel_indices );
	      if ( voxelindex<0 )
		continue;

	      saw_charge = 1;
	      
	      if ( intpc )
		npts_in_tpc_wcharge += 1;
	      
	      auto it_visited = index_visited.find( voxelindex );
	      if ( it_visited==index_visited.end() ) {
		index_visited.insert( voxelindex );

		std::vector<float> voxcharge_v = voxelizer.get_voxel_charge( voxelindex );
		float chargesum = 0.0;
		for ( auto& q : voxcharge_v )
		  chargesum += q;
		
		if ( chargesum>0 )
		  nvoxel_w_charge++;
	      }//end of if newly visited voxel
	      
	    }//end of substep loop

	    if ( _verbose_level>=2 ) {
	      std::cout << "  [" << istep << "] "
			<< "truept=("
			<< mcpt[0] << " cm,"
			<< mcpt[1] << " cm,"
			<< mcpt[2] << " cm,"
			<< mcstep.T()*1.0e-3 << " usec) "
			<< "recopt=(" << pt[0] << " cm," << pt[1] << " cm," << pt[2] << " cm, tick=" << pt[3] << ") "
			<< " charge_voxel=" << saw_charge
			<< std::endl;
	    }
	    
	  }//end of track step loop
	  
	}//end of if track type  node
      }//end of trackid loop inside recoflash

      float frac_in_voxel = float(npts_in_voxel)/float(npts_tested);
      float frac_out_voxel = 0.;
      float frac_intpc = 0.;
      float frac_intpc_wcharge = 0.;

      if ( _verbose_level>=2 )
	std::cout << "npts tested: " << npts_tested << std::endl;
      
      if ( npts_tested>0 ) {

	frac_out_voxel = float(npts_out_voxel)/float(npts_tested);
	frac_intpc = float(npts_in_tpc)/float(npts_tested);

	if ( npts_in_tpc>0 )
	  frac_intpc_wcharge = float(npts_in_tpc_wcharge)/float(npts_in_tpc);

	if ( _verbose_level>=2 ) {
	  std::cout << "npts in voxel: " << npts_in_voxel << " frac=" << frac_in_voxel << std::endl;
	  std::cout << "npts outside voxel: " << npts_out_voxel << " frac=" << frac_out_voxel << std::endl;
	  std::cout << "npts in tpc: " << npts_in_tpc << " frac=" << frac_intpc << std::endl;
	  std::cout << "npts in tpc w/ charge: " << npts_in_tpc_wcharge << " frac=" << frac_intpc_wcharge << std::endl;
	}
      }
      
      if ( frac_intpc_wcharge < 0.8 ) {
	flash_isgood[iflash] = 0;
      }
      
      flash_track_frac_intpc[iflash] = frac_intpc;
      flash_track_frac_intpc_w_charge[iflash] = frac_intpc_wcharge;
      
      // std::cout << "[OpModelMCDataPrep] [flash: " << iflash << "]"
      // 		<< " track fraction w charge: " << flash_track_frac_w_charge[iflash]
      // 		<< std::endl;
      if ( _verbose_level>=2 )
	std::cout << "------------------------------------" << std::endl;
      
    }//end of recoflash loop
    
    
  }//end of tag flash for complete track charge

  void OpModelMCDataPrep::filterBadMCDataMatches()
  {
    std::vector< ublarcvapp::mctools::RecoFlash_t > passes_v;
    std::vector< ublarcvapp::mctools::RecoFlash_t > reject_v;

    std::vector<float> tmp_intpc_w_charge = flash_track_frac_intpc_w_charge;
    std::vector<float> tmp_intpc = flash_track_frac_intpc;
    flash_track_frac_intpc_w_charge.clear();
    flash_track_frac_intpc.clear();

    for ( int iflash=0; iflash<recoflash_v.size(); iflash++ ) {

      auto& recoflash = recoflash_v.at(iflash);

      if ( flash_isgood[iflash]==1 ) {
	passes_v.emplace_back( std::move(recoflash) );
	flash_track_frac_intpc_w_charge.push_back( tmp_intpc_w_charge[iflash] );
	flash_track_frac_intpc.push_back( tmp_intpc[iflash] );	
      }
      else {
	reject_v.emplace_back( std::move(recoflash) );
	//flash_track_frac_intpc_w_charge.push_back( tmp_intpc_w_charge[iflash] );	
	//flaash_track_frac_intpc.push_back( tmp_intpc[iflash] );
      }
      
    }

    for ( auto& reject : filtered_v ) {
      reject_v.emplace_back( std::move(reject) );
      //flash_track_frac_intpc.push_back( tmp_intpc[iflash] );
    }

    std::swap( recoflash_v, passes_v );
    std::swap( filtered_v, reject_v );
    
    passes_v.clear();
    reject_v.clear();
    
  }

  void OpModelMCDataPrep::getChargeVoxelsForFlash( const ublarcvapp::mctools::RecoFlash_t& recoflash,
						   const larflow::voxelizer::VoxelizeTriplets& voxelizer,
						   std::vector< std::vector<int>   >& voxel_indices,
						   std::vector< std::vector<float> >& voxel_features )
  {
    
    // collect voxel indices and charge features
    int ancestorid = recoflash.ancestorid;
    
    std::vector<int> trackid_list = recoflash.trackid_list();
    const std::set<int>& trackid_set = recoflash.trackid_v;

    voxel_indices.clear();
    voxel_features.clear(); /// charge sum for voxels

    auto const& voxel_map = voxelizer.get_voxel_map();
    
    for ( auto const& it_voxel_map : voxel_map ) {

      auto const& indices_array = it_voxel_map.first;
      auto const& vindex = it_voxel_map.second;
      
      //auto const& tripidxlist = voxelizer._voxelidx_to_tripidxlist.at(vindex);
      auto const& tripidxlist = voxelizer.get_triplet_idx_list( vindex );
      
      bool accept = false;
      for ( auto const& tripidx : tripidxlist ) {
	int tid = voxelizer._triplet_maker._instance_id_v.at(tripidx);
	int aid = voxelizer._triplet_maker._ancestor_id_v.at(tripidx);
	auto it_tid = trackid_set.find( tid );
	auto it_aid = trackid_set.find( aid );
	if ( it_tid!=trackid_set.end() || it_aid!=trackid_set.end() || aid==ancestorid || tid==ancestorid ) {
	  accept = true;
	  break;
	}
      }
      
      if (!accept)
	continue;
      
      // store the indices and charge
      std::vector<int> indices_v  = { indices_array[0], indices_array[1], indices_array[2] };      
      std::vector<float> charge_v = voxelizer.get_voxel_charge( vindex );
      
      voxel_indices.push_back( indices_v );
      voxel_features.push_back( charge_v );
      
    }
    
  }
  
  void OpModelMCDataPrep::printMatches() const
  {

    std::cout << "[OpModelMCDataPrep::printMatches] ====================" << std::endl;
    int iflash = 0;
    for ( auto const& flash : recoflash_v ) {
      std::cout << ublarcvapp::mctools::FlashMatcherV2::strRecoMatchInfo( flash, iflash )
		<< " frac_intpc=" << flash_track_frac_intpc[iflash]
		<< " intpc_w_charge=" << flash_track_frac_intpc_w_charge[iflash]
		<< std::endl;
      iflash++;
    }
    std::cout << "======================================================" << std::endl;
    
  }

  std::vector<float> OpModelMCDataPrep::get_recoflash_pe( const ublarcvapp::mctools::RecoFlash_t& recoflash,
							  larlite::storage_manager& ioll )
  {

    std::string producer = "none";
    std::vector<float> flashpe_v(32,0.0);

    if ( recoflash.producerid>=0 ) {

      //int group = 0;
      if ( recoflash.producerid==0 ) {
	producer = "simpleFlashBeam";
	//group = 0;
      }
      else if (recoflash.producerid==1) {
	producer = "simpleFlashCosmic";
	//group = 2;
      }
      
      larlite::event_opflash* ev_opflash_v =
	(larlite::event_opflash*)ioll.get_data( larlite::data::kOpFlash, producer );

      auto const& opflash = ev_opflash_v->at( recoflash.index );

      // this is probably mc version and data version specific -- careful
      for (int igroup=0; igroup<4; igroup++) {
	float group_sum = 0.;	
	for (int ipmt=0; ipmt<32; ipmt++) {
	  flashpe_v[ipmt] = 0.;
	  int iopdet = 100*igroup + ipmt;
	  if ( iopdet<opflash.nOpDets() ) {
	    float pe = opflash.PE( iopdet );
	    group_sum += pe;
	    flashpe_v[ipmt] = pe;
	  }
	}
	if ( group_sum>0.0 ) {
	  break;
	}
      }//end of group loop
    }//end of if producer

    return flashpe_v;
    
  }
  
  PyObject* OpModelMCDataPrep::make_opmodel_data_dict( const ublarcvapp::mctools::RecoFlash_t& recoflash,
						       const larflow::voxelizer::VoxelizeTriplets& voxelizer,
						       larlite::storage_manager& ioll )
  {
    
    // ok now we can make the arrays
    if ( !_setup_numpy ) {
      std::cout << "[OpModelMCDataPrep::" << __FUNCTION__ << ".L" << __LINE__ << "] setup numpy" << std::endl;
      import_array1(0);
      _setup_numpy = true;
    }       

    std::string producer = "none";

    std::vector<float> flashpe_v = get_recoflash_pe( recoflash, ioll );
    std::vector< std::vector<int> > voxel_coord_v;
    std::vector< std::vector<float> > voxel_plane_charge_v;
    
    getChargeVoxelsForFlash( recoflash, voxelizer, voxel_coord_v, voxel_plane_charge_v );

    // voxel coordinate array
    int ndims = 3;
    size_t nvoxels = voxel_coord_v.size();
    
    npy_intp* coord_dims = new npy_intp[2];
    coord_dims[0] = (int)nvoxels;
    coord_dims[1] = ndims;
    PyArrayObject* coord_array = (PyArrayObject*)PyArray_SimpleNew( 2, coord_dims, NPY_LONG );
    for (size_t ii=0; ii<nvoxels; ii++) {
      for (int j=0; j<ndims; j++) {
        *((long*)PyArray_GETPTR2( coord_array, (int)ii, j)) = (long)voxel_coord_v[ii][j];
      }
    }

    // voxel charge array
    npy_intp* feat_dims = new npy_intp[2];
    feat_dims[0] = (int)nvoxels;
    feat_dims[1] = ndims;
    PyArrayObject* feat_array = (PyArrayObject*)PyArray_SimpleNew( 2, feat_dims, NPY_FLOAT );
    for (size_t ii=0; ii<nvoxels; ii++) {
      for (int j=0; j<ndims; j++) {
        *((float*)PyArray_GETPTR2( feat_array, (int)ii, j)) = (float)voxel_plane_charge_v[ii][j];
      }
    }

    // voxel PE array
    npy_intp* pe_dims = new npy_intp[2];
    pe_dims[0] = 1;
    pe_dims[1] = (int)flashpe_v.size();
    PyArrayObject* pe_array = (PyArrayObject*)PyArray_SimpleNew( 2, pe_dims, NPY_FLOAT );
    for (int j=0; j<(int)flashpe_v.size(); j++) {
      *((float*)PyArray_GETPTR2( pe_array, 0, j)) = (float)flashpe_v[j];
    }
    
    // the dictionary
    PyObject *d = PyDict_New();
    PyObject *key_coord     = Py_BuildValue("s", "voxcoord" );
    PyObject *key_feat      = Py_BuildValue("s", "voxcharge" );
    PyObject *key_flashpe   = Py_BuildValue("s", "flashpe" );

    // add to dictionary
    PyDict_SetItem( d, key_coord,   (PyObject*)coord_array );
    PyDict_SetItem( d, key_feat,    (PyObject*)feat_array );    
    PyDict_SetItem( d, key_flashpe, (PyObject*)pe_array );

    Py_DECREF( key_coord );
    Py_DECREF( key_feat );
    Py_DECREF( key_flashpe );

    return d;
    
  }

  
}
}
