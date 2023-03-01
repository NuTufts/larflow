#include "VoxelizeTriplets.h"

#include <iostream>
#include <sstream>
#include <math.h> 

#include "larlite/LArUtil/LArProperties.h"
#include "larflow/PrepFlowMatchData/PrepSSNetTriplet.h"

namespace larflow {
namespace voxelizer {

  bool VoxelizeTriplets::_setup_numpy = false;

  /** @brief construct where default dimensions are used to define voxel grid
   *
   * Origin is set to:
   * \verbatim embed:rst:leading-asterisks
   *  * x: (-801 ticks)*(0.5 usec/tick)*(drift velocity cm/usec) cm
   *  * y: -120.0 cm
   *  * z: 0.0 cm
   * \endverbatim
   *
   * The drift velocity is retrieved from larlite::LArProperties.
   *
   * The length is set to:
   * \verbatim embed:rst:leading-asterisks
   *  * x: (1010 pixel rows)*(6 ticks/pixel)*(0.5 usec/tick)*(drift velocity cm/usec) cm
   *  * y: 240.0 cm
   *  * z: 1037.0 cm
   * \endverbatim
   *
   * The voxel size is set to 0.3 cm. This is the wire pitch in MicroBoone.
   * 
   */
  VoxelizeTriplets::VoxelizeTriplets()
    : larcv::larcv_base("VoxelizeTriplets"),
      _nu_only(false)
  {
    set_voxel_size_cm(0.3); // will call function to define voxels in each tpc
  }

  // I WANT TO BE ABLE TO DEFINE THE ORIGIN TOO, BUT NOT CLEAR HOW TO DO THAT RIGHT NOW
  // UNDER THE NEED TO HANDLE MORE THAN 1 TPC.
  // REQUIRES DEFINING AND ORIGIN OFFSET FROM THE DEFAULT SOMEHOW FOR EACH TPC
  // /**
  //  * @brief constructor where voxel grid can be specified
  //  *
  //  * @param[in] origin  Position in 3D where origin of the voxel grid is located (in cm).
  //  * @param[in] dim_len Total length of each dimension (should be 3) in cm.
  //  * @param[in] voxel_size Length of height, width, and depth of an individual voxel in cm.
  //  *
  //  */  
  // VoxelizeTriplets::VoxelizeTriplets( std::vector<float> origin,
  //                                     std::vector<float> dim_len,
  //                                     float voxel_size ) 
  //   : larcv::larcv_base("VoxelizeTriplets"),
  //     _origin(origin),
  //   _len(dim_len),
  //   _voxel_size(voxel_size)
  // {
  //   _define_voxels();
  // }

  /**
   * @brief using the origin, grid length, and voxel size define the voxel grid
   *
   */
  void VoxelizeTriplets::_define_voxels()
  {
    _ndims = 3;

    // we use simchannelvoxelizer to define voxels. handles the multiple tpcs.
    _simchan_voxelizer.defineTPCVoxels( _len );

    auto const& geo  = *(larlite::larutil::Geometry::GetME());
    for (int icryo=0; icryo<(int)geo.Ncryostats(); icryo++) {
      for (int itpc=0; itpc<(int)geo.NTPCs(icryo); itpc++) {
	auto const& tpcinfo = _simchan_voxelizer.getTPCInfo( icryo, itpc );

	std::cout << "[VoxelizeTriplets::" << __FUNCTION__ << ".L" << __LINE__ << "] " << std::endl;
	std::cout << " CRYO[" << icryo << "] TPC[" << itpc << "] " << std::endl;
	std::cout << " NVOXELS: (";
	for (int v=0; v<_ndims; v++) {
	  std::cout << tpcinfo._num_voxels_v[v];
	  if ( v!=2 )
	    std::cout << ",";
	}
	std::cout << ")" << std::endl;

	std::cout << " VOXEL LENGTH: (";
	for (int v=0; v<_ndims; v++) {
	  std::cout << tpcinfo._voxel_dim_v[v];
	  if ( v!=2 )
	    std::cout << ",";
	}
	std::cout << ")" << std::endl;
	
	std::cout << " BOUNDS: ";
	for (int v=0; v<_ndims; v++) {
	  std::cout << "[" << tpcinfo._origin_cm_v[v] << "," << tpcinfo._origin_cm_v[v] + tpcinfo._voxel_dim_v[v]*tpcinfo._num_voxels_v[v] << "] ";
	}
	std::cout << std::endl;
	
      }
    }    
  }

  /**
   * @brief set/reset voxel definition
   *
   */
  void VoxelizeTriplets::set_voxel_size_cm( float width_cm )
  {
    const float driftv = larutil::LArProperties::GetME()->DriftVelocity();
    float dtick = std::floor( width_cm/larutil::DetectorProperties::GetME()->GetXTicksCoefficient() );
    // LARCV_DEBUG() << "setting voxel edges to be " << width_cm << " cm --> dtick=" << dtick << std::endl;
    dtick = 6; // over-ride
    _len.resize(3,0);
    _len[0] = dtick;
    _len[1] = width_cm;
    _len[2] = width_cm;
    _define_voxels();
  }

  // /**
  //  * @brief get the voxel bin along one of the dimensions
  //  *
  //  * @param[in] axis Dimension we want
  //  * @param[in] coord Coordinate in the dimension we want
  //  * @return voxel bin index along the given dimension
  //  */
  // int VoxelizeTriplets::get_axis_voxel( int axis, float coord ) const {

  //   if ( axis<0 || axis>=_ndims ) {
  //     std::stringstream ss;
  //     ss << "[VoxelizeTriplets::" << __FUNCTION__ << ".L" << __LINE__ << "] invalid dim given: " << axis << " (_ndims=" << _ndims << ")" << std::endl;
  //     throw std::runtime_error(ss.str());
  //   }
    
  //   int vidx = (coord-_origin[axis])/_voxel_size;
  //   if (vidx<0 || vidx>=_nvoxels[axis] ) {
  //     std::stringstream ss;
  //     ss << "[VoxelizeTriplets::" << __FUNCTION__ << ".L" << __LINE__ << "]";
  //     ss << " dim[" << axis << "] coordinate[" << coord << "] "
  //        << "given is out of bounds [" << _origin[axis] << "," << _origin[axis]+_len[axis] << "]"
  //        << " vidx=" << vidx << " bounds[0," << _nvoxels[axis] << ")"
  //        << std::endl;
  //     throw std::runtime_error(ss.str());
  //   }

  //   return vidx;
  // }
  
  /**
   *
   * @brief takes in precomputed triplet data and saves filled voxels and maps between voxels and triplets
   *
   * populates the following data members:
   * \verbatim embed:rst:leading-astericks
   *  * _voxel_set
   *  * _voxel_list
   *  * _voxelidx_to_tripidxlist
   *  * _trip2voxelidx
   * \endverbatim
   *
   * @param[in] triplet_data Instance of triplet data, assumed to be filled already
   *
   */
  larflow::voxelizer::TPCVoxelData
  VoxelizeTriplets::make_voxeldata( const larflow::prep::MatchTriplets& triplet_data )
  {

    // configure voxel based on tpc and cryo ID
    // (do later -- not needed for MicroBooNE for now)

    auto& tpcinfo = _simchan_voxelizer.getTPCInfo( triplet_data._cryoid, triplet_data._tpcid );
    
    larflow::voxelizer::TPCVoxelData voxdata;
    voxdata._ndims  = _ndims;
    voxdata._origin = tpcinfo._origin_cm_v;
    voxdata._len    = tpcinfo._voxel_dim_v;
    voxdata._voxel_size = 0.3;
    voxdata._nvoxels = tpcinfo._num_voxels_v;
    voxdata._tpcid  = triplet_data._tpcid;
    voxdata._cryoid = triplet_data._cryoid;    
            
    // first we need to define the voxels that are filled
    voxdata._voxel_set.clear();

    unsigned long n_uncontained = 0;

    for ( int itriplet=0; itriplet<(int)triplet_data._triplet_v.size(); itriplet++ ) {
      const std::vector<float>& pos = triplet_data._pos_v[itriplet]; // this is (x,y,z)
      const float tick = triplet_data._wirecoord_v[itriplet][3];
      std::vector<long> vox_index;
      bool contained = _simchan_voxelizer.getVoxelIndexWithTPCInfo( tick, pos[1], pos[2], tpcinfo, vox_index );
      if ( contained ) {
	std::array<long,3> coord;
	for (int i=0; i<3; i++)
	  coord[i] = vox_index[i];
	voxdata._voxel_set.insert( coord );
      }
      else {
	std::cout << "uncontained? pos=(" << pos[0] << "," << pos[1] << "," << pos[2] << ") tick=" << tick
		  << " index=(" << vox_index[0] << "," << vox_index[1] << "," << vox_index[2] << ")"
		  << std::endl;
	n_uncontained++;
      }
    }

    // now we assign voxel to an index
    voxdata._voxel_list.clear();
    long idx=0;
    for ( auto& coord : voxdata._voxel_set ) {
      voxdata._voxel_list[coord] = idx;
      idx++;
    }
    long nvidx = idx;
    
    LARCV_INFO() << "CRYO[" << voxdata._cryoid << "] TPC[" << voxdata._tpcid << "]" << std::endl;
    LARCV_INFO() << "  Filling " << nvidx << " voxels from " << triplet_data._triplet_v.size() << " triplets" << std::endl;
    LARCV_INFO() << "  Fillfrac=" << float(nvidx)/((float)voxdata._nvoxels[0]*(float)voxdata._nvoxels[1]*(float)voxdata._nvoxels[2])*100.0 << "%"
		 << std::endl;
    
    // assign triplets to voxels and vice versa
    voxdata._voxelidx_to_tripidxlist.clear();
    voxdata._voxelidx_to_tripidxlist.resize(nvidx);
    voxdata._trip2voxelidx.clear();
    voxdata._trip2voxelidx.resize( triplet_data._triplet_v.size(), 0 );
    
    for ( int itriplet=0; itriplet<(int)triplet_data._triplet_v.size(); itriplet++ ) {
      const std::vector<float>& pos = triplet_data._pos_v[itriplet]; // this is (x,y,z)
      const float tick = triplet_data._wirecoord_v[itriplet][3];
      std::vector<long> vox_index;
      bool contained = _simchan_voxelizer.getVoxelIndexWithTPCInfo( tick, pos[1], pos[2], tpcinfo, vox_index );
      if ( !contained )
	continue;
      std::array<long,3> coord;
      for (int i=0; i<3; i++)
        coord[i] = vox_index[i];
      auto it=voxdata._voxel_list.find(coord);
      if ( it==voxdata._voxel_list.end() ) {
        throw std::runtime_error("could not find a voxel we defined!!");
      }
      
      int voxelidx = it->second;
      voxdata._trip2voxelidx[itriplet] = voxelidx;
      voxdata._voxelidx_to_tripidxlist[voxelidx].push_back( itriplet );
    }

    int number_multipt_voxel = 0;
    int max_pt_in_voxel = 0;    
    for ( auto const& voxel_tripletlist : voxdata._voxelidx_to_tripidxlist ) {
      if ( voxel_tripletlist.size()>1 )
	number_multipt_voxel++;
      if ( voxel_tripletlist.size()>max_pt_in_voxel )
	max_pt_in_voxel = (int)voxel_tripletlist.size();
    }

    LARCV_INFO() << "Number of filled voxels: " <<voxdata._voxelidx_to_tripidxlist.size() << std::endl;
    LARCV_INFO() << "  number of uncontained triplets: " << n_uncontained << std::endl;
    LARCV_INFO() << "  number of multipt voxel: " << number_multipt_voxel << std::endl;
    LARCV_INFO() << "  max pts in a voxel: " << max_pt_in_voxel << std::endl;
    
    // Fill a TPCVoxelData object for This MatchTriplet instance
    return voxdata;
  }

  /**
   * @brief fill TPCVoxelData ssnet voxel labels
   *
   * fill TPCVoxelData::_voxel_ssnetid.
   *  
   */
  int VoxelizeTriplets::fill_tpcvoxeldata_semantic_labels( const larflow::prep::SSNetLabelData& ssnetdata,
							   const larflow::prep::MatchTriplets& triplet_data,
							   larflow::voxelizer::TPCVoxelData& voxdata )
  {

    // clear label map
    voxdata._voxel_realedep.clear();
    voxdata._voxel_ssnetid.clear();
    
    // Loop over the voxels
    int nvidx = (int)voxdata._voxel_set.size();

    int iidx = 0;
    
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {

      int vidx = it->second;

      // voxel coordinate
      const std::array<long,3>& coord = it->first;
      const std::vector<int>& tripidx_v = voxdata._voxelidx_to_tripidxlist[vidx]; // index of triplet
	    
      // the class label: mlreco uses the 5-particle labels for larcv
      // find class label with most (non background) triplets
      std::vector<int> nclass( larflow::prep::PrepSSNetTriplet::kNumClasses, 0 );      
      for ( auto const& tripidx : tripidx_v ) {
	int triplet_label = ssnetdata._ssnet_label_v.at(tripidx);
	if ( triplet_label>=0 && triplet_label<larflow::prep::PrepSSNetTriplet::kNumClasses )
	  nclass[triplet_label]++;
      }
      int max_class = -1;
      int max_class_n = 0;
      for (int iclass=1; iclass<larflow::prep::PrepSSNetTriplet::kNumClasses; iclass++) {
	if ( nclass[iclass]>max_class_n ) {
	  max_class = iclass;
	  max_class_n = nclass[iclass];
	}
      }
      
      // now that we have decided the class, set the voxetset label
      if (max_class>0 && max_class_n>0 ) {
	int mlreco_label = 0;
	switch (max_class) {
	case 0: // BG class
	  mlreco_label = 5;
	  break;
	case 1: // electron
	  mlreco_label = 1;
	  break;
	case 2: // gamma
	  mlreco_label = 0;
	  break;
	case 3: // muon
	  mlreco_label = 2;
	  break;
	case 4: // pion
	  mlreco_label = 3;
	  break;
	case 5: // proton
	  mlreco_label = 4;
	  break;
	default: // other (kaons usually): dump it into the pion class
	  mlreco_label = 3;
	  break;
	}

	voxdata._voxel_ssnetid[vidx] = mlreco_label;
	if (mlreco_label!=5 )
	  voxdata._voxel_realedep[vidx] = 1; // real
	else
	  voxdata._voxel_realedep[vidx] = 0; // ghost
      }
      else {
	voxdata._voxel_ssnetid[vidx]  = 5; // no label, so set to background
	voxdata._voxel_realedep[vidx] = 0; // ghost
      }

      iidx++;
    }//end of loop over voxels
    
    return 0;
  }

  /**
   * @brief fill charge from three planes for voxels
   *
   */
  int VoxelizeTriplets::fill_tpcvoxeldata_planecharge( const larflow::prep::MatchTriplets& triplet_data,
						       larflow::voxelizer::TPCVoxelData& voxdata )
  {

    voxdata._voxel_planecharge.clear();
    
    // Loop over the voxels
    int nvidx = (int)voxdata._voxel_set.size();

    int iidx = 0 ;
    
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {

      int vidx = it->second;

      // voxel coordinate
      const std::array<long,3>& coord = it->first;
      const std::vector<int>& tripidx_v = voxdata._voxelidx_to_tripidxlist[vidx]; // index of triplet
	    
      // ave plane charge of spacepoints associated to this voxel
      std::vector<float> pixsum_v(3,0.0);
      std::vector<int> npix_v(3,0);
      for ( auto const& tripidx : tripidx_v ) {
	auto const& tripindices = triplet_data._triplet_v[tripidx];
	for (int p=0; p<3; p++) {
	  const larflow::prep::FlowTriples::PixData_t& pixdata = triplet_data._sparseimg_vv.at(p).at(tripindices[p]);
	  pixsum_v[p] += pixdata.val;
	  npix_v[p]++;
	}
      }

      std::vector<float> charge_v(3,0);
      
      // fill charge voxelset(s)
      for (int p=0; p<3; p++) {
	if (npix_v[p]>0) {
	  charge_v[p] = pixsum_v[p]/float(npix_v[p]);
	}
	else {
	  charge_v[p] = 0.0;
	}
      }
      
      voxdata._voxel_planecharge[vidx] = charge_v;
      
    }//end of voxel list
    
    return 0;
  }

  
  /**
   * @brief fill not-neutrino origin label for tpcvoxeldata
   *
   */
  int VoxelizeTriplets::fill_tpcvoxeldata_cosmicorigin( const larflow::prep::MatchTriplets& triplet_data,
							larflow::voxelizer::TPCVoxelData& voxdata )
  {

    voxdata._voxel_origin.clear();
    
    // Loop over the voxels
    int nvidx = (int)voxdata._voxel_set.size();

    int iidx = 0 ;
    
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {

      int vidx = it->second;

      // voxel coordinate
      const std::array<long,3>& coord = it->first;
      const std::vector<int>& tripidx_v = voxdata._voxelidx_to_tripidxlist[vidx]; // index of triplet
	          
      // cosmic origin label: 0=neutrino, 1=cosmic
      // not cosmic if at least one nu origin triplet inside this voxel
      bool isnu = false;
      for ( auto const& tripidx : tripidx_v ) {
	int origin_label = triplet_data._origin_v.at(tripidx);
	if ( origin_label==1 )
	  isnu = true;
      }

      if (isnu)
	voxdata._voxel_origin[vidx] = 0;
      else
	voxdata._voxel_origin[vidx] = 1;
      
      iidx++;
    }//end of loop over voxels
    
    return 0;
  }

  /**
   * @brief produces particle cluster labels for voxels, refines particle list
   *
   */
  int VoxelizeTriplets::fill_tpcvoxeldata_instance_labels( const larflow::prep::MatchTriplets& tripletdata,
							   larflow::voxelizer::TPCVoxelData& voxdata )
  {

    voxdata._voxel_instanceid.clear();
    voxdata._id2trackid.clear();
    
    int nvidx = (int)voxdata._voxel_set.size();

    // compile unique IDs and their frequency
    std::map<long,long> instance2id; //key: geant4 track id, value: new instance code
    std::map<long,long> idcounts;    //key: new instance code, value: number of voxels
    int nids = 0;
    for ( auto const& instanceid : tripletdata._instance_id_v ) {
      if ( instanceid==0 ) // ID[0] is reserved for no TrackID assigned to triplet
	continue;
      
      int id = 0;
      if ( instance2id.find( instanceid )==instance2id.end() ) {
	// New instanceID/trackID found, assign new sequential ID
	id = nids;
	instance2id[instanceid] = id; 
	idcounts[id] = 0;
	voxdata._id2trackid[id] = instanceid;
	voxdata._trackid2id[instanceid] = id;
	nids++;
	//LARCV_DEBUG() << "trackid[" << instanceid << "] -> sequentialid[" << id << "]" << std::endl;
      }
      idcounts[id]++;
    }
    
    std::vector<int> vox_nclass( nids+1, 0 ); // count voxels with an assigned id label
    std::vector<int> nvotes_id( nids+1, 0 ); // vector to vote for instance label for voxel
    int nmissing = 0;
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {
      int vidx = it->second; // voxel index
      const std::array<long,3>& coord = it->first; // voxel coordinates      

      // find class label with most (non background) triplets
      const std::vector<int>& tripidx_v = voxdata._voxelidx_to_tripidxlist[vidx]; // index of triplet

      // clear the values
      memset( nvotes_id.data(), 0, sizeof(int)*nvotes_id.size() );

      for ( auto const& tripidx : tripidx_v ) {
	int instance_label = tripletdata._instance_id_v.at(tripidx);

	auto it = instance2id.find( instance_label );
	if ( it!=instance2id.end() ) {
	  // found the instance, get the id label
	  int idlabel = it->second;
	  nvotes_id[idlabel]++;
	}
      }
      
      int max_id = -1;
      int max_id_n = 0;
      for ( int i=0; i<(int)nvotes_id.size(); i++ ) {
	if ( nvotes_id[i]>max_id_n ) {
	  max_id_n = nvotes_id[i];
	  max_id = i;
	}
      }
      
      if (max_id>=0 ) {
	// determined a winning ID for this voxel
	voxdata._voxel_instanceid[vidx] = max_id;
	vox_nclass[max_id]++;
      }
      else {
	voxdata._voxel_instanceid[vidx] = nids;
	vox_nclass[nids]++;
      }
      
    }//loop over voxels
    
    return 0;
    
  }  
  
  /**
   * @brief takes in precomputed triplet data and outputs voxel data in the form of a python dict
   * 
   * Contents of the dictionary:
   * \verbatim embed:rst:leading-asterisks
   *  * dict["voxcoord"] = (Nv,3) numpy int array; Nv (x,y,z) voxel coordinate
   *  * dict["voxlabel"]  = (Nv) numpy int array; 1 if truth voxel, 0 otherwise. 
   *  * dict["trip2vidx"] = (Nt) numpy int array; Nt voxel indices referencing the "coord" array
   *  * dict["vox2trips_list"] = List of length Nv with (Mi) numpy int arrays. Each array contains triplet index list to combine into the voxel.
   * \endverbatim
   * 
   * Uses member containers filled in make_voxeldata().
   *
   * @param[in] triplet_data larlite::prep::PrepMatchTriplets class containing spacepoint proposals
   * @return Python dictionary as described above. Ownership is transferred to calling namespace.
   *
   */
  PyObject* VoxelizeTriplets::make_voxeldata_dict( const larflow::voxelizer::TPCVoxelData& voxdata,
						   const larflow::prep::MatchTriplets& triplet_data )
  {
    
    // ok now we can make the arrays
    if ( !_setup_numpy ) {
      LARCV_INFO() << " setup numpy" << std::endl;
      import_array1(0);
      _setup_numpy = true;
    }       

    int nvidx = (int)voxdata._voxel_set.size();
    
    // first the voxel coordinate array
    npy_intp* coord_dims = new npy_intp[2];
    coord_dims[0] = (int)nvidx;
    coord_dims[1] = (int)voxdata._ndims;
    PyArrayObject* coord_array = (PyArrayObject*)PyArray_SimpleNew( 2, coord_dims, NPY_LONG );
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {
      int vidx = it->second;
      const std::array<long,3>& coord = it->first;
      for (int j=0; j<_ndims; j++)
        *((long*)PyArray_GETPTR2( coord_array, (int)vidx, j)) = (long)coord[j];
    }
    //std::cout << "  made coord array" << std::endl;

    // voxel feature array: charge on planes, taking mean
    npy_intp* feat_dims = new npy_intp[2];
    feat_dims[0] = (int)nvidx;
    feat_dims[1] = (int)3;
    PyArrayObject* feat_array = (PyArrayObject*)PyArray_SimpleNew( 2, feat_dims, NPY_FLOAT );
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {
      int vidx = it->second; // voxel index

      // ave plane charge of spacepoints associated to this voxel
      const std::vector<int>& tripidx_v = voxdata._voxelidx_to_tripidxlist[vidx]; // index of triplet
      std::vector<float> pixsum_v(3,0.0);
      std::vector<int> npix_v(3,0);
      for ( auto const& tripidx : tripidx_v ) {
	auto const& tripindices = triplet_data._triplet_v[tripidx];
	for (int p=0; p<3; p++) {
	  const larflow::prep::FlowTriples::PixData_t& pixdata = triplet_data._sparseimg_vv.at(p).at(tripindices[p]);
	  pixsum_v[p] += pixdata.val;
	  npix_v[p]++;
	}
      }
      for (int p=0; p<3; p++) {
	if (npix_v[p]>0) {
	  *((float*)PyArray_GETPTR2( feat_array, (int)vidx, p)) = pixsum_v[p]/float(npix_v[p]);
	}
	else {
	  *((float*)PyArray_GETPTR2( feat_array, (int)vidx, p)) = 0.0;
	}
      }
    }

    // the voxel truth label
    bool has_truth = triplet_data._truth_v.size()==triplet_data._triplet_v.size();
    npy_intp* vlabel_dims = new npy_intp[1];
    vlabel_dims[0] = (int)nvidx;
    PyArrayObject* vlabel_array = (PyArrayObject*)PyArray_SimpleNew( 1, vlabel_dims, NPY_LONG );
    int num_true_voxels = 0;
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {
      int vidx = it->second;
      int truthlabel = 0.;

      if ( has_truth ) {
        // is there a true pixel?
        const std::vector<int>& tripidx_v = voxdata._voxelidx_to_tripidxlist[vidx];
        for ( auto const& tidx : tripidx_v ) {
          if ( triplet_data._truth_v[tidx]==1 ) {
            truthlabel = 1;
          }
        }
        if ( truthlabel==1 )
          num_true_voxels++;
      }
      *((long*)PyArray_GETPTR1( vlabel_array, vidx )) = truthlabel;
    }
    //std::cout << "  made truth array, ntrue=" << num_true_voxels << std::endl;

    // weights
    npy_intp* lmweight_dims = new npy_intp[1];
    lmweight_dims[0] = (int)nvidx;
    PyArrayObject* lmweight_array = (PyArrayObject*)PyArray_SimpleNew( 1, lmweight_dims, NPY_FLOAT );
    float frac_pos = (float)num_true_voxels/(float)nvidx;
    float frac_neg = 1.0-frac_pos;
    float w_neg = 1.0/frac_neg;
    float w_pos = 1.0/frac_pos;
    float w_norm = (float)num_true_voxels*w_pos + (float)(nvidx-num_true_voxels)*w_neg;
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {
      int vidx = it->second;
      long truthlabel = *((long*)PyArray_GETPTR1( vlabel_array, vidx ));

      if ( truthlabel )
	*((float*)PyArray_GETPTR1( lmweight_array, vidx )) = w_pos/w_norm;
      else
	*((float*)PyArray_GETPTR1( lmweight_array, vidx )) = w_neg/w_norm;
    }
    //std::cout << "  made weight array: f_pos=" << w_pos/w_norm << " f_neg=" << w_neg/w_norm << std::endl;
    
    

    // the triplet to voxel index array
    npy_intp* trip2vidx_dims = new npy_intp[1];
    trip2vidx_dims[0] = (int)voxdata._trip2voxelidx.size();
    PyArrayObject* trip2vidx_array = (PyArrayObject*)PyArray_SimpleNew( 1, trip2vidx_dims, NPY_LONG );
    for (int itriplet=0; itriplet<(int)voxdata._trip2voxelidx.size(); itriplet++ ) {
      *((long*)PyArray_GETPTR1( trip2vidx_array, itriplet )) = (long)voxdata._trip2voxelidx[itriplet];
    }
    //std::cout << "  made triplet-to-voxelindex array" << std::endl;    
    
    // finally the list of triplet indices for each voxel
    PyObject* tripidx_pylist = PyList_New( nvidx );
    for ( int vidx=0; vidx<nvidx; vidx++ ) {

      // make the array
      npy_intp* tidx_dims = new npy_intp[1];
      const std::vector<int>& tripidx_v = voxdata._voxelidx_to_tripidxlist[vidx];
      tidx_dims[0] = (int)tripidx_v.size();
      PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( 1, tidx_dims, NPY_LONG );
      for ( int i=0; i<tidx_dims[0]; i++ ) {
        *((long*)PyArray_GETPTR1( array, i )) = tripidx_v[i];
      }

      int err = PyList_SetItem( tripidx_pylist, (Py_ssize_t)vidx, (PyObject*)array );
      if (err!=0 ) {
        throw std::runtime_error("Error putting voxel's triplet list to pylist");
      }
      //Py_DECREF( array );
    }
    //std::cout << "  made voxel-index to triplet-list list" << std::endl;        

    // the dictionary
    PyObject *d = PyDict_New();
    PyObject *key_coord     = Py_BuildValue("s", "voxcoord" );
    PyObject *key_feat      = Py_BuildValue("s", "voxfeat" );    
    PyObject *key_label     = Py_BuildValue("s", "voxlabel" );
    PyObject *key_trip2vidx = Py_BuildValue("s", "trip2vidx" );
    PyObject *key_vox2trips = Py_BuildValue("s", "vox2trips_list" );
    PyObject *key_lmweight  = Py_BuildValue("s", "voxlmweight" );    

    PyDict_SetItem( d, key_coord, (PyObject*)coord_array );
    PyDict_SetItem( d, key_feat, (PyObject*)feat_array );    
    PyDict_SetItem( d, key_label, (PyObject*)vlabel_array );
    PyDict_SetItem( d, key_trip2vidx, (PyObject*)trip2vidx_array );
    PyDict_SetItem( d, key_vox2trips, (PyObject*)tripidx_pylist );
    PyDict_SetItem( d, key_lmweight, (PyObject*)lmweight_array );

    //std::cout << "  dereference" << std::endl;
    Py_DECREF( key_coord );
    Py_DECREF( key_feat );    
    Py_DECREF( key_label );
    Py_DECREF( key_trip2vidx );
    Py_DECREF( key_vox2trips );
    Py_DECREF( key_lmweight );

    Py_DECREF( coord_array );
    Py_DECREF( feat_array );    
    Py_DECREF( vlabel_array );
    Py_DECREF( trip2vidx_array );
    Py_DECREF( tripidx_pylist );
    Py_DECREF( lmweight_array );
    
    return d;
  }

  /**
   * @brief calls make_voxeldata_dict with internal triplet maker data
   * 
   * @return Python dictionary as described above. Ownership is transferred to calling namespace.
   *
   */
  PyObject* VoxelizeTriplets::make_voxeldata_dict()
  {

    // NEEDS FIX
    
    PyObject* pylist = PyList_New(0); // create an empty list
    // for ( auto& matchtriplet : _triplet_maker._match_triplet_v ) {
    //   PyObject* voxel_dict = make_voxeldata_dict( matchtriplet );
    //   int status = PyList_Append( pylist, voxel_dict );
    //   if ( status!=0 )
    // 	LARCV_ERROR() << "Could not append voxel dictionary for TPC triplet data" << std::endl;
    //   // need to dereference the dict here?
    //   Py_DECREF( voxel_dict );
    // }
    
    return pylist;
  }
  
  /**
   * @brief process data from image to make triplet and voxel data
   *
   * This method uses an internal instance of larflow::prep::PrepMatchTriplets, _triplet_maker.
   * The internal instance is used to extract spacepoint proposals from the wire plane images
   * and then pass that info to make_voxeldata().
   *
   * @param[in]  iolcv LArCV IOManager containing event data
   * @param[in]  adc_producer Root tree name containing wire images for event
   * @param[in]  chstatus_producer Root tree name containing channel status info for event
   * @param[in]  has_mc The IOManager is expected to contain truth information from Monte Carlo simulations
   *
   */
  void VoxelizeTriplets::process_fullchain( larcv::IOManager& iolcv,
                                            std::string adc_producer,
                                            std::string chstatus_producer,
                                            bool has_mc )
  {

    _triplet_maker.clear();

    const float adc_threshold = 10.;
    const bool calc_triplet_pos3d = true;
    _triplet_maker.process( iolcv, adc_producer, chstatus_producer, adc_threshold, calc_triplet_pos3d );

    if ( has_mc ) {
      larcv::EventImage2D* ev_larflow =    
        (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"larflow");    
      larcv::EventImage2D* ev_instance =    
        (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"instance");    
      _triplet_maker.make_truth_vector( ev_larflow->Image2DArray(), ev_instance->Image2DArray() );
    }

    for ( auto& matchtriplet : _triplet_maker._match_triplet_v )
      make_voxeldata( matchtriplet );

    // persistancy:

    // save larflow3dhits for visualization
    // save class in ttree data for training
    
  }

  /**
   * @brief get voxel indices from non-t0 corrected space point coordinate
   *
   */
  std::vector<long> VoxelizeTriplets::get_voxel_indices( const std::vector<float>& xyz )
  {

    TVector3 worldLoc( xyz[0], xyz[1], xyz[2]);
    
    std::vector<int> ctid = larlite::larutil::Geometry::GetME()->GetContainingCryoAndTPCIDs( worldLoc );
    int cryoid = ctid[0];
    int tpcid  = ctid[1];
    auto& tpcinfo = _simchan_voxelizer.getTPCInfo( cryoid, tpcid );
    float tick = larutil::DetectorProperties::GetME()->ConvertXToTicks( xyz[0], 0, tpcid, cryoid );
    std::vector<long> vox_index(3,0);
    bool contained = _simchan_voxelizer.getVoxelIndexWithTPCInfo( tick, xyz[1], xyz[2], tpcinfo, vox_index );

    if ( !contained ) {

      std::vector<float> origin = tpcinfo._origin_cm_v;
      
      std::stringstream ss;
      ss << "[VoxelizeTriplets::get_voxel_indices.L" << __LINE__ << "] "
	 << " Space point provided (" << xyz[0] << "," << xyz[1] << "," << xyz[2] << ") is out of bounds" << std::endl;
      for (int i=0; i<3; i++) {
	float maxbound = origin[i] + tpcinfo._num_voxels_v[i]*tpcinfo._voxel_dim_v[i];
	ss << " for axis=" << i << " bounds=[" << origin[i] << "," << maxbound << ")"
	   << std::endl;
      }
      throw std::runtime_error( ss.str() );
    }
    return vox_index;
  }

  /**
   * @brief make TPC voxel labels for ssnet
   *
   * collate the voxels and ssnet label data and output a numpy array.
   *  
   */
  int VoxelizeTriplets::make_ssnet_voxel_label_nparray( const larflow::prep::SSNetLabelData& ssnetdata,
							const larflow::voxelizer::TPCVoxelData& voxeldata,
							PyArrayObject*& ssnet_array,
							PyArrayObject*& ssnet_weight )
  {

    // start the numpy environment or something
    if ( !_setup_numpy ) {
      LARCV_INFO() << "setup numpy" << std::endl;
      import_array1(0);
      _setup_numpy = true;
    }       

    int nvidx = (int)voxeldata._voxel_set.size();
    
    // voxel ssnet label array: charge on planes, taking mean
    npy_intp* ssnet_dims = new npy_intp[1];
    ssnet_dims[0] = (int)nvidx;
    ssnet_array = (PyArrayObject*)PyArray_SimpleNew( 1, ssnet_dims, NPY_LONG );

    std::vector<int> vox_nclass( larflow::prep::PrepSSNetTriplet::kNumClasses, 0 );
    for ( auto it=voxeldata._voxel_list.begin(); it!=voxeldata._voxel_list.end(); it++ ) {
      int vidx = it->second; // voxel index

      // find class label with most (non background) triplets
      const std::vector<int>& tripidx_v = voxeldata._voxelidx_to_tripidxlist[vidx]; // index of triplet
      std::vector<int> nclass( larflow::prep::PrepSSNetTriplet::kNumClasses, 0 );      
      for ( auto const& tripidx : tripidx_v ) {
	int triplet_label = ssnetdata._ssnet_label_v.at(tripidx);
	if ( triplet_label>=0 && triplet_label<larflow::prep::PrepSSNetTriplet::kNumClasses )
	  nclass[triplet_label]++;
      }
      int max_class = -1;
      int max_class_n = 0;
      for (int iclass=1; iclass<larflow::prep::PrepSSNetTriplet::kNumClasses; iclass++) {
	if ( nclass[iclass]>max_class_n ) {
	  max_class = iclass;
	  max_class_n = nclass[iclass];
	}
      }
      
      if (max_class>0 && max_class_n>0 ) {
	*((long*)PyArray_GETPTR1( ssnet_array, (int)vidx)) = max_class;
	vox_nclass[max_class]++;
      }
      else {
	*((long*)PyArray_GETPTR1( ssnet_array, (int)vidx)) = 0;
	vox_nclass[0]++;
      }
      
    }

    // weights
    npy_intp* ssnet_weight_dims = new npy_intp[1];
    ssnet_weight_dims[0] = (int)nvidx;
    ssnet_weight = (PyArrayObject*)PyArray_SimpleNew( 1, ssnet_weight_dims, NPY_FLOAT );
    std::vector<float> class_weight( larflow::prep::PrepSSNetTriplet::kNumClasses, 0 );
    float w_norm = larflow::prep::PrepSSNetTriplet::kNumClasses*nvidx;
    for (int iclass=0; iclass<larflow::prep::PrepSSNetTriplet::kNumClasses; iclass++) {
      float w_class = ( vox_nclass[iclass]>0 ) ? (float)nvidx/(float)vox_nclass[iclass]/w_norm : 0.;
      class_weight[iclass] = w_class;
    }

    for (int ivdx=0; ivdx<nvidx; ivdx++) {
      long iclass = *((long*)PyArray_GETPTR2( ssnet_array, (int)ivdx, 0));
      *((float*)PyArray_GETPTR1( ssnet_weight, (int)ivdx)) = class_weight[iclass];
    }
    
    
    //std::cout << "  made ssnet truth array" << std::endl;
    return 0;
  }
  
  /**
   * @brief make voxel labels for ssnet
   */
  PyObject* VoxelizeTriplets::make_ssnet_dict_labels( const larflow::voxelizer::TPCVoxelData& voxdata,
						      const larflow::prep::MatchTriplets& data )
  {

    // ok now we can make the arrays
    if ( !_setup_numpy ) {
      LARCV_INFO() << " setup numpy" << std::endl;
      import_array1(0);
      _setup_numpy = true;
    }       

    int nvidx = (int)voxdata._voxel_set.size();
    
    // voxel ssnet label array: charge on planes, taking mean
    npy_intp* ssnet_dims = new npy_intp[1];
    ssnet_dims[0] = (int)nvidx;
    PyArrayObject* ssnet_array = (PyArrayObject*)PyArray_SimpleNew( 1, ssnet_dims, NPY_LONG );

    std::vector<int> vox_nclass( larflow::prep::PrepSSNetTriplet::kNumClasses, 0 );
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {
      int vidx = it->second; // voxel index

      // find class label with most (non background) triplets
      const std::vector<int>& tripidx_v = voxdata._voxelidx_to_tripidxlist[vidx]; // index of triplet
      std::vector<int> nclass( larflow::prep::PrepSSNetTriplet::kNumClasses, 0 );      
      for ( auto const& tripidx : tripidx_v ) {
	int triplet_label = data._pdg_v.at(tripidx);

	int final_label = 0;
	switch (triplet_label) {
	case 3:
	  final_label = 1;
	  break;
	case 4:
	  final_label = 2;
	  break;
	case 5:
	  final_label = 2;
	  break;
	case 6:
	  final_label = 3;
	  break;
	case 7:
	case 8:
	  final_label = 4;
	  break;
	case 9:
	  final_label = 5;
	  break;
	};

	
	if ( final_label>=0 && final_label<larflow::prep::PrepSSNetTriplet::kNumClasses )
	  nclass[final_label]++;
      }
      int max_class = -1;
      int max_class_n = 0;
      for (int iclass=1; iclass<larflow::prep::PrepSSNetTriplet::kNumClasses; iclass++) {
	if ( nclass[iclass]>max_class_n ) {
	  max_class = iclass;
	  max_class_n = nclass[iclass];
	}
      }

      if (max_class>0 && max_class_n>0 ) {
	    
	*((long*)PyArray_GETPTR1( ssnet_array, (int)vidx)) = max_class;
	vox_nclass[max_class]++;
      }
      else {
	*((long*)PyArray_GETPTR1( ssnet_array, (int)vidx)) = 0;
	vox_nclass[0]++;
      }
      
    }

    // weights
    npy_intp* ssnet_weight_dims = new npy_intp[1];
    ssnet_weight_dims[0] = (int)nvidx;
    PyArrayObject* ssnet_weight = (PyArrayObject*)PyArray_SimpleNew( 1, ssnet_weight_dims, NPY_FLOAT );
    std::vector<float> class_weight( larflow::prep::PrepSSNetTriplet::kNumClasses, 0 );
    float w_norm = larflow::prep::PrepSSNetTriplet::kNumClasses*nvidx;
    for (int iclass=0; iclass<larflow::prep::PrepSSNetTriplet::kNumClasses; iclass++) {
      float w_class = ( vox_nclass[iclass]>0 ) ? (float)nvidx/(float)vox_nclass[iclass]/w_norm : 0.;
      class_weight[iclass] = w_class;
    }

    for (int ivdx=0; ivdx<nvidx; ivdx++) {
      long iclass = *((long*)PyArray_GETPTR2( ssnet_array, (int)ivdx, 0));
      *((float*)PyArray_GETPTR1( ssnet_weight, (int)ivdx)) = class_weight[iclass];
    }
    
    PyObject *d = PyDict_New();
    PyObject* key_label  = Py_BuildValue("s","voxssnet");
    PyObject* key_weight = Py_BuildValue("s","voxssnetweight");
    PyDict_SetItem( d, key_label,  (PyObject*)ssnet_array );
    PyDict_SetItem( d, key_weight, (PyObject*)ssnet_weight );
    
    Py_DECREF( key_label );
    Py_DECREF( key_weight );
    Py_DECREF( ssnet_array );
    Py_DECREF( ssnet_weight );
    
    //std::cout << "  made ssnet truth array" << std::endl;
    return d;
  }
  
  /**
   * @brief get full label set for voxels
   * @param data Class holding the triplet data and truth labels loaded inside
   * @return dictionary with numpy arrays
   */
  PyObject* VoxelizeTriplets::get_full_voxel_labelset_dict( const larflow::keypoints::LoaderKeypointData& data ) 
  {

    size_t num_tpcs = data.triplet_v->size(); // number of sets of triplet data. each entry is for a TPC in the detector.

    PyObject* tpclist = PyList_New(0);

    for ( size_t idata=0; idata<num_tpcs; idata++ ) {
      const larflow::prep::MatchTriplets& tripletdata = data.triplet_v->at(idata); /// proposed spacepoints and labels
      const larflow::keypoints::KeypointData& kpdata  = data.kpdata_v->at(idata);  /// keypoints along with info to calc score label
      const larflow::prep::SSNetLabelData& ssnetdata  = data.ssnet_v->at(idata);   /// ssnet labels for the spacepoints

      larflow::voxelizer::TPCVoxelData voxdata = make_voxeldata(tripletdata); /// make maps from voxels to spacepoints
      
      // get larmatch voxels and truth labels (ghost vs true label)
      PyObject* larmatch_dict = VoxelizeTriplets::make_voxeldata_dict( voxdata, tripletdata );
    
      PyObject *ssnet_label_key  = Py_BuildValue("s", "ssnet_labels" );
      PyObject *ssnet_weight_key = Py_BuildValue("s", "ssnet_weights" );    
      PyArrayObject* ssnet_array  = nullptr;
      PyArrayObject* ssnet_weight = nullptr;
      make_ssnet_voxel_label_nparray( ssnetdata, voxdata, ssnet_array, ssnet_weight );
      PyDict_SetItem(larmatch_dict, ssnet_label_key,  (PyObject*)ssnet_array);
      PyDict_SetItem(larmatch_dict, ssnet_weight_key, (PyObject*)ssnet_weight);    

      PyArrayObject* larmatch_labels = (PyArrayObject*)PyDict_GetItemString( larmatch_dict, "voxlabel" );
      PyArrayObject* kplabel  = nullptr;
      PyArrayObject* kpweight = nullptr;
      make_kplabel_arrays( kpdata, voxdata, larmatch_labels, kplabel, kpweight );
      PyObject *kp_label_key = Py_BuildValue("s", "kplabel" );
      PyDict_SetItem(larmatch_dict, kp_label_key, (PyObject*)kplabel );
      PyObject *kp_weight_key = Py_BuildValue("s", "kpweight" );    
      PyDict_SetItem(larmatch_dict, kp_weight_key, (PyObject*)kpweight );

      // instance labels
      PyObject* dict_instance_labels = make_instance_dict_labels( voxdata, tripletdata );
      int mergeok = PyDict_Update( larmatch_dict, dict_instance_labels );
      if ( mergeok!=0 ) {
	throw std::runtime_error( "voxelizetriplet::get_full_voxel_labelset_dict: merge with instance label dict failed");
      }
      
      // origin labels
      PyObject* dict_origin_labels = make_origin_dict_labels( voxdata, tripletdata );
      mergeok = PyDict_Update( larmatch_dict, dict_origin_labels );
      if ( mergeok!=0 ) {
	throw std::runtime_error( "voxelizetriplet::get_full_voxel_labelset_dict: merge with origin label dict failed");
      }

      Py_DECREF(ssnet_label_key);
      Py_DECREF(ssnet_weight_key);
      Py_DECREF(kp_label_key);
      Py_DECREF(kp_weight_key);
      
      Py_DECREF(kplabel);
      Py_DECREF(kpweight);
      Py_DECREF(ssnet_array);
      Py_DECREF(ssnet_weight);
      Py_DECREF(dict_origin_labels);
      Py_DECREF(dict_instance_labels);

      PyList_Append( tpclist, larmatch_dict );

    }


    return tpclist;
  }

  /**
   * @brief get full label set for voxels
   * @param data Class holding the triplet data and truth labels loaded inside
   * @return dictionary with numpy arrays
   */
  PyObject* VoxelizeTriplets::make_full_voxel_labelset_dict( const larflow::voxelizer::TPCVoxelData& voxdata,
							     const larflow::prep::MatchTriplets& tripletdata,
							     const larflow::prep::SSNetLabelData& ssnetdata,
							     const larflow::keypoints::KeypointData& kpdata )
  {

    // const larflow::prep::MatchTriplets& tripletdata = data.triplet_v->at(idata); /// proposed spacepoints and labels
    // const larflow::keypoints::KeypointData& kpdata  = data.kpdata_v->at(idata);  /// keypoints along with info to calc score label
    // const larflow::prep::SSNetLabelData& ssnetdata  = data.ssnet_v->at(idata);   /// ssnet labels for the spacepoints
      
    // get larmatch voxels and truth labels (ghost vs true label)
    PyObject* larmatch_dict = VoxelizeTriplets::make_voxeldata_dict( voxdata, tripletdata );
    
    PyObject *ssnet_label_key  = Py_BuildValue("s", "ssnet_labels" );
    PyObject *ssnet_weight_key = Py_BuildValue("s", "ssnet_weights" );    
    PyArrayObject* ssnet_array  = nullptr;
    PyArrayObject* ssnet_weight = nullptr;
    make_ssnet_voxel_label_nparray( ssnetdata, voxdata, ssnet_array, ssnet_weight );
    PyDict_SetItem(larmatch_dict, ssnet_label_key,  (PyObject*)ssnet_array);
    PyDict_SetItem(larmatch_dict, ssnet_weight_key, (PyObject*)ssnet_weight);    

    PyArrayObject* larmatch_labels = (PyArrayObject*)PyDict_GetItemString( larmatch_dict, "voxlabel" );
    PyArrayObject* kplabel  = nullptr;
    PyArrayObject* kpweight = nullptr;
    make_kplabel_arrays( kpdata, voxdata, larmatch_labels, kplabel, kpweight );
    PyObject *kp_label_key = Py_BuildValue("s", "kplabel" );
    PyDict_SetItem(larmatch_dict, kp_label_key, (PyObject*)kplabel );
    PyObject *kp_weight_key = Py_BuildValue("s", "kpweight" );    
    PyDict_SetItem(larmatch_dict, kp_weight_key, (PyObject*)kpweight );

    // instance labels
    PyObject* dict_instance_labels = make_instance_dict_labels( voxdata, tripletdata );
    int mergeok = PyDict_Update( larmatch_dict, dict_instance_labels );
    if ( mergeok!=0 ) {
      throw std::runtime_error( "voxelizetriplet::get_full_voxel_labelset_dict: merge with instance label dict failed");
    }
      
    // origin labels
    PyObject* dict_origin_labels = make_origin_dict_labels( voxdata, tripletdata );
    mergeok = PyDict_Update( larmatch_dict, dict_origin_labels );
    if ( mergeok!=0 ) {
      throw std::runtime_error( "voxelizetriplet::get_full_voxel_labelset_dict: merge with origin label dict failed");
    }

    Py_DECREF(ssnet_label_key);
    Py_DECREF(ssnet_weight_key);
    Py_DECREF(kp_label_key);
    Py_DECREF(kp_weight_key);
    
    Py_DECREF(kplabel);
    Py_DECREF(kpweight);
    Py_DECREF(ssnet_array);
    Py_DECREF(ssnet_weight);
    Py_DECREF(dict_origin_labels);
    Py_DECREF(dict_instance_labels);
    
    return larmatch_dict;
  }
  
  /**
   * @brief make keypoint ground truth numpy arrays
   *
   * @param[in]  num_max_samples Max number of samples to return
   * @param[out] nfilled number of samples actually returned
   * @param[in]  withtruth if true, return flag indicating if true/good space point
   * @param[out] pos_match_index vector index in return samples for space points which are true/good
   * @param[in]  match_array numpy array containing indices to sparse image for each spacepoint
   * @param[out] kplabel_array numpy array containing ssnet class labels for each spacepoint
   * @param[out] kplabel_weight numpy array containing weight for each spacepoint
   * @return always returns 0  
   */
  int VoxelizeTriplets::make_kplabel_arrays( const larflow::keypoints::KeypointData& kpdata,
					     const larflow::voxelizer::TPCVoxelData& voxdata,
					     PyArrayObject* larmatch_label_array,
					     PyArrayObject*& kplabel_array,
					     PyArrayObject*& kplabel_weight,
					     float sigma )
  {
    
    // ok now we can make the arrays
    if ( !_setup_numpy ) {
      LARCV_INFO() << " setup numpy" << std::endl;
      import_array1(0);
      _setup_numpy = true;
    }       

    int nvidx = (int)voxdata._voxel_set.size();
    int nclasses = 6; //number of keypoint classes    
    
    // voxel ssnet label array: charge on planes, taking mean
    npy_intp* kplabel_dims = new npy_intp[2];
    kplabel_dims[0] = (int)nclasses;
    kplabel_dims[1] = (int)nvidx;    
    kplabel_array = (PyArrayObject*)PyArray_SimpleNew( 2, kplabel_dims, NPY_FLOAT );

    auto& tpcinfo = _simchan_voxelizer.getTPCInfo( voxdata._cryoid, voxdata._tpcid );
    
    /// ------- ///
    
    //float sigma = 10.0; // cm
    float sigma2 = sigma*sigma; // cm^2

    std::vector<int> npos(nclasses,0);
    std::vector<int> nneg(nclasses,0);
    
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {
      int vidx = it->second; // voxel index, index in the array we are filling
      const std::array<long,3>& arr_index = it->first; // index in the dense 3D array

      std::vector<float> vox_center(3,0);
      for (int i=0; i<3; i++)
	vox_center[i] = ((float)arr_index[i]+0.5)*tpcinfo._voxel_dim_v[i] + tpcinfo._origin_cm_v[i]; // position of voxel center
      
      //std::cout << "vox-center: (" << vox_center[0] << "," << vox_center[1] << "," << vox_center[2] << ")" << std::endl;
      
      long larmatch_truth_label = 1.0;
      if ( larmatch_label_array!=NULL )
	larmatch_truth_label = *((long*)PyArray_GETPTR1(larmatch_label_array,vidx));
	  
      // for each class, calculate distance to closest true keypoint
      // use smallest label
      for (int c=0; c<6; c++) {

	if ( larmatch_truth_label==0 ) {
	  *((float*)PyArray_GETPTR2(kplabel_array,c,vidx)) = 0.0;
	  nneg[c]++;
	  continue;
	}
	
	const std::vector<larflow::keypoints::KPdata>& kpd_v = kpdata._kpd_v;
	int nkp = kpd_v.size();
	float min_dist_kp = 1.0e9;
	int   max_kp_idx = -1;
	for (int ikp=0; ikp<nkp; ikp++) {
	  const std::vector<float>& pos = kpd_v.at(ikp).keypt;
	  float dist = 0;
	  for (int i=0; i<3; i++)
	    dist += ( pos[i]-vox_center[i] )*( pos[i]-vox_center[i] );
	  if ( dist<min_dist_kp ) {
	    min_dist_kp = dist;
	    max_kp_idx = ikp;
	  }
	  // if ( c==0 ) {
	  //   std::cout << "  true kp[" << ikp << "]: (" << pos[0] << "," << pos[1] << "," << pos[2] << ") dist=" << sqrt(dist) << std::endl;
	  // }	 	  
	}

	// label for pixel
	if ( max_kp_idx>=0 ) {
	  float labelscore = exp(-min_dist_kp/sigma2);
	  if (labelscore>0.05 ) {
	    *((float*)PyArray_GETPTR2(kplabel_array,c,vidx)) = labelscore;
	    npos[c]++;	    
	  }
	  else {
	    *((float*)PyArray_GETPTR2(kplabel_array,c,vidx)) = 0.0;
	    nneg[c]++;	    
	  }
	}
	else {
	  *((float*)PyArray_GETPTR2(kplabel_array,c,vidx)) = 0.0;
	  nneg[c]++;
	}
      }
    }//end of voxel list
	
    // weights to balance positive and negative examples
    int kpweight_nd = 2;
    npy_intp kpweight_dims[] = { nclasses, (long)nvidx };
    kplabel_weight = (PyArrayObject*)PyArray_SimpleNew( kpweight_nd, kpweight_dims, NPY_FLOAT );

    for (int c=0; c<nclasses; c++ ) {
      float w_pos = (npos[c]) ? float(npos[c]+nneg[c])/float(npos[c]) : 0.0;
      float w_neg = (nneg[c]) ? float(npos[c]+nneg[c])/float(nneg[c]) : 0.0;
      float w_norm = w_pos*npos[c] + w_neg*nneg[c];

      //std::cout << "Keypoint class[" << c << "] WEIGHT: W(POS)=" << w_pos/w_norm << " W(NEG)=" << w_neg/w_norm << std::endl;
    
      for (int i=0; i<kpweight_dims[1]; i++ ) {

	float labelscore = *((float*)PyArray_GETPTR2(kplabel_array,c,i));
	if ( labelscore>0.05 ) {
	  if ( w_pos>0.0 )
	    *((float*)PyArray_GETPTR2(kplabel_weight,c,i)) = w_pos/w_norm;
	  else
	    *((float*)PyArray_GETPTR2(kplabel_weight,c,i)) = 0.0;
	}
	else {
	  if ( w_neg>0.0 )
	    *((float*)PyArray_GETPTR2(kplabel_weight,c,i)) = w_neg/w_norm;
	  else
	    *((float*)PyArray_GETPTR2(kplabel_weight,c,i)) = 0.0;
	}
      }//end of class loop
    }

    return 0;
  }

  /**
   * @brief make keypoint ground truth numpy arrays
   *
   * @param[in]  num_max_samples Max number of samples to return
   * @param[out] nfilled number of samples actually returned
   * @param[in]  withtruth if true, return flag indicating if true/good space point
   * @param[out] pos_match_index vector index in return samples for space points which are true/good
   * @param[in]  match_array numpy array containing indices to sparse image for each spacepoint
   * @return always returns 0  
   */
  // PyObject* VoxelizeTriplets::make_kplabel_dict_fromprep( const larflow::keypoints::PrepKeypointData& data,
  // 							  PyObject* plarmatch_label_array )
  // {

  //   // ok now we can make the arrays
  //   if ( !_setup_numpy ) {
  //     LARCV_INFO() << " setup numpy" << std::endl;
  //     import_array1(0);
  //     _setup_numpy = true;
  //   }

  //   PyArrayObject* larmatch_label_array =  (PyArrayObject*)plarmatch_label_array;

  //   int nvidx = (int)_voxel_set.size();
  //   int nclasses = 6; //number of keypoint classes    
    
  //   // voxel ssnet label array: charge on planes, taking mean
  //   npy_intp* kplabel_dims = new npy_intp[2];
  //   kplabel_dims[0] = (int)nclasses;
  //   kplabel_dims[1] = (int)nvidx;    
  //   PyArrayObject* kplabel_array = (PyArrayObject*)PyArray_SimpleNew( 2, kplabel_dims, NPY_FLOAT );

  //   /// ------- ///
    
  //   float sigma = 10.0; // cm
  //   float sigma2 = sigma*sigma; // cm^2

  //   std::vector<int> npos(nclasses,0);
  //   std::vector<int> nneg(nclasses,0);

  //   for ( auto it=_voxel_list.begin(); it!=_voxel_list.end(); it++ ) {
  //     int vidx = it->second; // voxel index, index in the array we are filling
  //     const std::array<long,3>& arr_index = it->first; // index in the dense 3D array

  //     std::vector<float> vox_center(3,0);
  //     for (int i=0; i<3; i++)
  // 	vox_center[i] = ((float)arr_index[i]+0.5)*get_voxel_size() + get_origin()[i]; // position of voxel center

  //     //std::cout << "vox-center: (" << vox_center[0] << "," << vox_center[1] << "," << vox_center[2] << ")" << std::endl;

  //     long larmatch_truth_label = 1.0;
  //     if ( larmatch_label_array!=NULL )
  // 	larmatch_truth_label = *((long*)PyArray_GETPTR1(larmatch_label_array,vidx));
	  
  //     // for each class, calculate distance to closest true keypoint
  //     // use smallest label
  //     for (int c=0; c<6; c++) {

  // 	if ( larmatch_truth_label==0 ) {
  // 	  *((float*)PyArray_GETPTR2(kplabel_array,c,vidx)) = 0.0;
  // 	  nneg[c]++;
  // 	  continue;
  // 	}
	
  // 	const std::vector< std::vector<float> >& pos_v = (kpdata._kppos_v[c]);
  // 	int nkp = pos_v.size();
  // 	float min_dist_kp = 1.0e9;
  // 	int   max_kp_idx = -1;
  // 	for (int ikp=0; ikp<nkp; ikp++) {
  // 	  const std::vector<float>& pos = pos_v[ikp];
  // 	  float dist = 0;
  // 	  for (int i=0; i<3; i++)
  // 	    dist += ( pos[i]-vox_center[i] )*( pos[i]-vox_center[i] );
  // 	  if ( dist<min_dist_kp ) {
  // 	    min_dist_kp = dist;
  // 	    max_kp_idx = ikp;
  // 	  }
  // 	  // if ( c==0 ) {
  // 	  //   std::cout << "  true kp[" << ikp << "]: (" << pos[0] << "," << pos[1] << "," << pos[2] << ") dist=" << sqrt(dist) << std::endl;
  // 	  // }	 	  
  // 	}

  // 	// label for pixel
  // 	if ( max_kp_idx>=0 ) {
  // 	  float labelscore = exp(-min_dist_kp/sigma2);
  // 	  if (labelscore>0.05 ) {
  // 	    *((float*)PyArray_GETPTR2(kplabel_array,c,vidx)) = labelscore;
  // 	    npos[c]++;	    
  // 	  }
  // 	  else {
  // 	    *((float*)PyArray_GETPTR2(kplabel_array,c,vidx)) = 0.0;
  // 	    nneg[c]++;	    
  // 	  }
  // 	}
  // 	else {
  // 	  *((float*)PyArray_GETPTR2(kplabel_array,c,vidx)) = 0.0;
  // 	  nneg[c]++;
  // 	}
  //     }
  //   }//end of voxel list
	
  //   // weights to balance positive and negative examples
  //   int kpweight_nd = 2;
  //   npy_intp kpweight_dims[] = { nclasses, (long)nvidx };
  //   PyArrayObject* kplabel_weight = (PyArrayObject*)PyArray_SimpleNew( kpweight_nd, kpweight_dims, NPY_FLOAT );

  //   for (int c=0; c<nclasses; c++ ) {
  //     float w_pos = (npos[c]) ? float(npos[c]+nneg[c])/float(npos[c]) : 0.0;
  //     float w_neg = (nneg[c]) ? float(npos[c]+nneg[c])/float(nneg[c]) : 0.0;
  //     float w_norm = w_pos*npos[c] + w_neg*nneg[c];

  //     //std::cout << "Keypoint class[" << c << "] WEIGHT: W(POS)=" << w_pos/w_norm << " W(NEG)=" << w_neg/w_norm << std::endl;
    
  //     for (int i=0; i<kpweight_dims[1]; i++ ) {

  // 	float labelscore = *((float*)PyArray_GETPTR2(kplabel_array,c,i));
  // 	if ( labelscore>0.05 ) {
  // 	  if ( w_pos>0.0 )
  // 	    *((float*)PyArray_GETPTR2(kplabel_weight,c,i)) = w_pos/w_norm;
  // 	  else
  // 	    *((float*)PyArray_GETPTR2(kplabel_weight,c,i)) = 0.0;
  // 	}
  // 	else {
  // 	  if ( w_neg>0.0 )
  // 	    *((float*)PyArray_GETPTR2(kplabel_weight,c,i)) = w_neg/w_norm;
  // 	  else
  // 	    *((float*)PyArray_GETPTR2(kplabel_weight,c,i)) = 0.0;
  // 	}
  //     }//end of class loop
  //   }

  //   PyObject *d = PyDict_New();
  //   PyObject* key_kplabel  = Py_BuildValue("s","voxkplabel");
  //   PyObject* key_kpweight = Py_BuildValue("s","voxkpweight");
  //   PyDict_SetItem( d, key_kplabel,  (PyObject*)kplabel_array );
  //   PyDict_SetItem( d, key_kpweight, (PyObject*)kplabel_weight );
    
  //   Py_DECREF( key_kplabel );
  //   Py_DECREF( key_kpweight );
  //   Py_DECREF( kplabel_array );
  //   Py_DECREF( kplabel_weight );

  //   return d;
  // }

  /**
   * @brief make voxel labels for instance tags
   */
  PyObject* VoxelizeTriplets::make_instance_dict_labels( const larflow::voxelizer::TPCVoxelData& voxdata,
							 const larflow::prep::MatchTriplets& tripletdata )
  {

    // ok now we can make the arrays
    if ( !_setup_numpy ) {
      LARCV_INFO() << " setup numpy" << std::endl;
      import_array1(0);
      _setup_numpy = true;
    }       
    
    int nvidx = (int)voxdata._voxel_set.size();

    // compile unique IDs
    std::map<int,int> instance2id;
    std::map<int,int> idcounts;
    int nids = 0;
    for ( auto const& instanceid : tripletdata._instance_id_v ) {
      if ( instanceid==0 )
	continue;
      
      int id = 0;
      if ( instance2id.find( instanceid )==instance2id.end() ) {
	id = nids+1;
	instance2id[instanceid] = nids+1; // we start at 1
	idcounts[id] = 0;
	nids++;
      }
      idcounts[id]++;
    }
    
    // voxel ssnet label array: charge on planes, taking mean
    npy_intp* dims = new npy_intp[1];
    dims[0] = (int)nvidx;
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( 1, dims, NPY_LONG );

    std::vector<int> vox_nclass( nids+1, 0 ); // count voxels with an assigned id label
    std::vector<int> nvotes_id( nids+1, 0 ); // vector to vote for instance label for voxel
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {
      int vidx = it->second; // voxel index

      // find class label with most (non background) triplets
      const std::vector<int>& tripidx_v = voxdata._voxelidx_to_tripidxlist[vidx]; // index of triplet

      // clear the values
      memset( nvotes_id.data(), 0, sizeof(int)*nvotes_id.size() );

      for ( auto const& tripidx : tripidx_v ) {
	int instance_label = tripletdata._instance_id_v.at(tripidx);

	auto it = instance2id.find( instance_label );
	if ( it!=instance2id.end() ) {
	  // found the instance, get the id label
	  int idlabel = it->second;
	  nvotes_id[idlabel]++;
	}
      }
      
      int max_id = -1;
      int max_id_n = 0;
      for ( int i=0; i<(int)nvotes_id.size(); i++ ) {
	if ( nvotes_id[i]>max_id_n ) {
	  max_id_n = nvotes_id[i];
	  max_id = i;
	}
      }
      
      if (max_id>0 ) {
	*((long*)PyArray_GETPTR1( array, (int)vidx)) = max_id;
	vox_nclass[max_id]++;
      }
      else {
	*((long*)PyArray_GETPTR1( array, (int)vidx)) = 0;
	vox_nclass[0]++;
      }
      
    }

    // weights
    // npy_intp* weight_dims = new npy_intp[1];
    // weight_dims[0] = (int)nvidx;
    // PyArrayObject* weight = (PyArrayObject*)PyArray_SimpleNew( 1, weight_dims, NPY_FLOAT );
    // std::vector<float> class_weight( larflow::prep::PrepSSNetTriplet::kNumClasses, 0 );
    // float w_norm = larflow::prep::PrepSSNetTriplet::kNumClasses*nvidx;
    // for (int iclass=0; iclass<larflow::prep::PrepSSNetTriplet::kNumClasses; iclass++) {
    //   float w_class = ( vox_nclass[iclass]>0 ) ? (float)nvidx/(float)vox_nclass[iclass]/w_norm : 0.;
    //   class_weight[iclass] = w_class;
    // }

    // for (int ivdx=0; ivdx<nvidx; ivdx++) {
    //   long iclass = *((long*)PyArray_GETPTR2( array, (int)ivdx, 0));
    //   *((float*)PyArray_GETPTR1( weight, (int)ivdx)) = class_weight[iclass];
    // }

    // save instance map
    PyObject* idmap = PyDict_New();
    for ( auto it=instance2id.begin(); it!=instance2id.end(); it++ ) {
      PyObject* key_instance = Py_BuildValue("i",it->first);
      PyObject* key_id       = Py_BuildValue("i",it->second);
      PyDict_SetItem( idmap, key_instance, key_id );
      Py_DECREF( key_instance );
      Py_DECREF( key_id );
    }
    
    PyObject *d = PyDict_New();
    PyObject* key_label  = Py_BuildValue("s","voxinstance");
    PyObject* key_map    = Py_BuildValue("s","voxinstance2id");
    
    //PyObject* key_weight = Py_BuildValue("s","voxinstanceweight");
    PyDict_SetItem( d, key_label,  (PyObject*)array );
    //PyDict_SetItem( d, key_weight, (PyObject*)weight );
    PyDict_SetItem( d, key_map, idmap );
    
    Py_DECREF( key_label );
    //Py_DECREF( key_weight );
    Py_DECREF( array );
    //Py_DECREF( weight );
    Py_DECREF( key_map );
    Py_DECREF( idmap );    
    
    //std::cout << "  made ssnet truth array" << std::endl;
    return d;
  }

  /**
   * @brief make voxel labels for origin tag
   */
  PyObject* VoxelizeTriplets::make_origin_dict_labels( const larflow::voxelizer::TPCVoxelData& voxdata,
						       const larflow::prep::MatchTriplets& data )
  {

    // ok now we can make the arrays
    if ( !_setup_numpy ) {
      LARCV_INFO() << " setup numpy" << std::endl;
      import_array1(0);
      _setup_numpy = true;
    }       

    int nvidx = (int)voxdata._voxel_set.size();
    
    // voxel ssnet label array: charge on planes, taking mean
    npy_intp* dims = new npy_intp[1];
    dims[0] = (int)nvidx;
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( 1, dims, NPY_LONG );

    std::vector<int> vox_nclass( 2, 0 ); // count voxels with an assigned origin label
    std::vector<int> nvotes( 2, 0 ); // vector to vote for origin label for voxel
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {
      int vidx = it->second; // voxel index

      // find class label with most (non background) triplets
      const std::vector<int>& tripidx_v = voxdata._voxelidx_to_tripidxlist[vidx]; // index of triplet

      // clear the values
      memset( nvotes.data(), 0, sizeof(int)*nvotes.size() );

      for ( auto const& tripidx : tripidx_v ) {
	int origin_label = data._origin_v.at(tripidx);
	if ( origin_label==1 )
	  nvotes[1]++;
	else
	  nvotes[0]++;
      }
      
      if ( nvotes[1]>0 ) {
	*((long*)PyArray_GETPTR1( array, (int)vidx)) = 1;
	vox_nclass[1]++;
      }
      else {
	*((long*)PyArray_GETPTR1( array, (int)vidx)) = 0;
	vox_nclass[0]++;
      }
      
    }

    // weights
    npy_intp* weight_dims = new npy_intp[1];
    weight_dims[0] = (int)nvidx;
    PyArrayObject* weight = (PyArrayObject*)PyArray_SimpleNew( 1, weight_dims, NPY_FLOAT );
    std::vector<float> class_weight( 2, 0 );
    float w_norm = 2*nvidx;
    for (int iclass=0; iclass<2; iclass++) {
      float w_class = ( vox_nclass[iclass]>0 ) ? (float)nvidx/(float)vox_nclass[iclass]/w_norm : 0.;
      class_weight[iclass] = w_class;
    }
    
    for (int ivdx=0; ivdx<nvidx; ivdx++) {
      long iclass = *((long*)PyArray_GETPTR2( array, (int)ivdx, 0));
      *((float*)PyArray_GETPTR1( weight, (int)ivdx)) = class_weight[iclass];
    }
    
    PyObject *d = PyDict_New();
    PyObject* key_label  = Py_BuildValue("s","voxorigin");
    PyObject* key_weight = Py_BuildValue("s","voxoriginweight");
    PyDict_SetItem( d, key_label,  (PyObject*)array );
    PyDict_SetItem( d, key_weight, (PyObject*)weight );
    
    Py_DECREF( key_label );
    Py_DECREF( key_weight );
    Py_DECREF( array );
    Py_DECREF( weight );

    //std::cout << "  made ssnet truth array" << std::endl;
    return d;
  }

  /**
   * @brief produces the semantic labels for mlreco
   *
   */
  std::vector< larcv::SparseTensor3D >
  VoxelizeTriplets::make_mlreco_semantic_label_sparse3d( const larflow::voxelizer::TPCVoxelData& voxdata,
							 const larflow::prep::MatchTriplets& triplet_data,
							 const larflow::prep::SSNetLabelData& ssnetdata )
  {

    LARCV_NORMAL() << "Start conversion" << std::endl;

    if ( voxdata._voxel_planecharge.size()==0 )
      LARCV_ERROR() << "Plane charge labels not filled" << std::endl;

    if ( voxdata._voxel_ssnetid.size()==0 )
      LARCV_ERROR() << "SSNet ID labels not filled" << std::endl;

    if ( voxdata._voxel_realedep.size()==0 )
      LARCV_ERROR() << "Real Edep labels not filled" << std::endl;

    if ( _nu_only && voxdata._voxel_origin.size()==0 ) {
      LARCV_ERROR() << "NU-ONLY mode is TRUE, but there are no origin labels. Run: fill_tpcvoxeldata_cosmicorigin() prior to this function." << std::endl;
      throw std::runtime_error("NU-ONLY mode is TRUE, but there are no origin labels.");
    }

    larcv::Voxel3DMeta meta = make_meta( voxdata );
    LARCV_NORMAL() << "Meta: " << meta.dump() << std::endl;    

    // Convert our data into larcv::VoxelSet
    larcv::VoxelSet charge[3]; // charge value from each plane
    larcv::VoxelSet label;

    // Loop over the voxels
    int nvidx = (int)voxdata._voxel_set.size();

    int iidx = 0 ;
    
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {

      // if ( iidx>0 && iidx%10000==0 )
      // 	LARCV_DEBUG() << "  processing voxel " << iidx << " of " << nvidx << std::endl;
      
      int vidx = it->second;
      auto const it_origin = voxdata._voxel_origin.find(vidx);
      if ( it_origin==voxdata._voxel_origin.end() ) {
	LARCV_ERROR() << "No origin label for voxel index=" << vidx << std::endl;
      }

      // skip non-neutrino voxels
      if ( _nu_only && it_origin->second==1 )
	continue;

      // voxel coordinate
      const std::array<long,3>& coord = it->first;
      const std::vector<int>& tripidx_v = voxdata._voxelidx_to_tripidxlist[vidx]; // index of triplet

      auto it_charge = voxdata._voxel_planecharge.find(vidx);
      if ( it_charge==voxdata._voxel_planecharge.end() ) {
	throw std::runtime_error("could not find plane charge for voxel index [vidx]");
      }
      auto const& charge_v = it_charge->second;
      
      // fill charge voxelset(s)
      for (int p=0; p<3; p++)
	charge[p].push_back( meta.index(coord[0], coord[1], coord[2]), charge_v[p] );
      
      // the class label: mlreco uses the 5-particle labels for larcv
      auto it_ssnet = voxdata._voxel_ssnetid.find(vidx);
      if ( it_ssnet!=voxdata._voxel_ssnetid.end() ) 
	label.push_back( meta.index(coord[0], coord[1], coord[2]), (float)it_ssnet->second );
      
      iidx++;
    }//end of loop over voxels

    // sort these voxelsets
    for (int p=0; p<3; p++)
      charge[p].sort();
    label.sort();
      
    std::vector< larcv::SparseTensor3D > out_v;
    for (int p=0; p<3; p++) {
      larcv::SparseTensor3D x( std::move(charge[p]), meta );
      out_v.emplace_back( std::move( x ) );
    }
    larcv::SparseTensor3D xlabel( std::move(label), meta );    
    out_v.emplace_back( std::move(xlabel) );

    return out_v;
  }
  
  /**
   * @brief produces the cosmic origin label
   *
   */
  std::vector< larcv::SparseTensor3D >
  VoxelizeTriplets::make_mlreco_cosmicorigin_label_sparse3d( const larflow::voxelizer::TPCVoxelData& voxdata,
							     const larflow::prep::MatchTriplets& triplet_data )
  {

    LARCV_NORMAL() << "Start conversion" << std::endl;    

    if ( voxdata._voxel_origin.size()==0 )
      throw std::runtime_error("No cosmic origin labels set");

    larcv::Voxel3DMeta meta = make_meta( voxdata );
    LARCV_DEBUG() << "Meta: " << meta.dump() << std::endl;    

    // Convert our data into larcv::VoxelSet
    larcv::VoxelSet label;

    // Loop over the voxels
    int nvidx = (int)voxdata._voxel_set.size();

    int iidx = 0 ;
    
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {

      int vidx = it->second;

      // skip non-neutrino voxels
      auto const it_origin = voxdata._voxel_origin.find(vidx);
      if ( it_origin==voxdata._voxel_origin.end() ) {
	LARCV_ERROR() <<  "could not find voxel by index in TPCVoxelData" << std::endl;
      }
	
      if ( _nu_only && it_origin->second==1 )
	continue;      

      // voxel coordinate
      const std::array<long,3>& coord = it->first;
      const std::vector<int>& tripidx_v = voxdata._voxelidx_to_tripidxlist[vidx]; // index of triplet

      // cosmic origin label: 0=neutrino, 1=cosmic
      label.push_back( meta.index(coord[0], coord[1], coord[2]), it_origin->second );
      
      iidx++;
    }//end of loop over voxels
    
    // sort these voxelsets
    label.sort();
    
    larcv::SparseTensor3D xlabel( std::move(label), meta );    
    std::vector< larcv::SparseTensor3D > out_v;
    out_v.emplace_back( std::move(xlabel) );
    
    return out_v;
  }
  
  /**
   * @brief produces particle cluster labels for voxels, refines particle list
   *
   */
  std::vector< larcv::SparseTensor3D >
  VoxelizeTriplets::make_mlreco_cluster_label_sparse3d( const larflow::voxelizer::TPCVoxelData& voxdata,
							const larflow::prep::MatchTriplets& tripletdata,
							const larflow::keypoints::KeypointData& kpdata,							
							std::vector<larcv::Particle>& particle_v,
							std::vector<larcv::Particle>& rejected_v )
  {
    
    LARCV_DEBUG() << "start" << std::endl;
    larcv::Voxel3DMeta meta = make_meta( voxdata );
    LARCV_DEBUG() << "Meta: " << meta.dump() << std::endl;
    
    if ( voxdata._voxel_instanceid.size()==0 ) {
      throw std::runtime_error("Instance ID labels for voxels have not been filled");
    }

    if ( _nu_only && voxdata._voxel_origin.size()==0 ) {
      LARCV_ERROR() << "NU-ONLY mode is TRUE, but there are no origin labels. Run: fill_tpcvoxeldata_cosmicorigin() prior to this function." << std::endl;
      throw std::runtime_error("NU-ONLY mode is TRUE, but there are no origin labels.");
    }
    
    
    std::vector<larcv::Particle> saved_v;
    std::vector< larcv::SparseTensor3D > out_v;    
    larcv::VoxelSet instance_labels;

    int nvidx = (int)voxdata._voxel_set.size();
    std::vector<int> vox_nclass( voxdata._id2trackid.size()+1, 0 );

    // if we cut down the labels, we need to remap again.
    std::map<long,long> origid2seqid; // original id in voxdata._voxel_origin to new sequential ID

    long nseqid = 0;
    for ( auto it=voxdata._voxel_list.begin(); it!=voxdata._voxel_list.end(); it++ ) {
      int vidx = it->second; // voxel index

      auto const it_origin = voxdata._voxel_origin.find(vidx);
      if ( it_origin==voxdata._voxel_origin.end() ) {
	LARCV_ERROR() << "No origin label for voxel index=" << vidx << std::endl;
      }      
      
      if ( _nu_only && it_origin->second!=0 )
	continue;
      
      const std::array<long,3>& coord = it->first; // voxel coordinates      

      auto it_iid = voxdata._voxel_instanceid.find(vidx);
      long instanceid = it_iid->second;

      auto it_seqid = origid2seqid.find( instanceid );
      if ( it_seqid==origid2seqid.end() ) {
	// not found in map
	origid2seqid[instanceid] = nseqid;
	nseqid++;
	it_seqid = origid2seqid.find( instanceid );
      }
      
      instance_labels.push_back( meta.index(coord[0], coord[1], coord[2]), (float)it_seqid->second );
      vox_nclass[it_seqid->second]++;
    }//loop over voxels
    
    
    // filter the particle list to only include ids in the voxel
    LARCV_DEBUG() << "pre-filterd particles=" << particle_v.size() << std::endl;
    int num_updated = 0;
    for ( auto& particle : particle_v ) {
      
      int trackid = (int)particle.track_id();
      auto it=voxdata._trackid2id.find( trackid );

      long labelid = it->second;
      
      // translate trackid to instanceid
      bool has_voxels = it!=voxdata._trackid2id.end();

      if ( has_voxels ) {
	// translate instanceid to sequential id used for output label
	auto it_seqid = origid2seqid.find( labelid );
	if ( it_seqid==origid2seqid.end() ) {
	  has_voxels = false;
	}
	else {
	  labelid = it_seqid->second;
	}
      }
      
      if ( !has_voxels ) {
	// not found in both original label nor the final filter: reject
	rejected_v.emplace_back( std::move(particle) );
	continue;
      }
      
      // now we want to adjust the info for the particle.
      // we need to see if we have a refined keypoint made
      int found_kpdata = 0;
      for ( auto& kp : kpdata._kpd_v ) {
	if ( kp.trackid==trackid ) {
	  // found the matching kpdata object for this particle
	  found_kpdata++;
	  
	  if ( kp.kptype==kTrackEnd ) {
	    // modify the end point of the particle
	    float t = particle.last_step().T();
	    particle.last_step( kp.keypt[0], kp.keypt[1], kp.keypt[2], t );
	  }
	  else {
	    // else modify the start point of the particle for all types
	    float t = particle.first_step().T();
	    // std::cout << "trackid=" << trackid
	    // 	      << " pos=(" << kp.keypt[0] << ", " << kp.keypt[1] << "," << kp.keypt[2] << ")"
	    // 	      << " t=" << t 
	    // 	      << std::endl;
	    particle.first_step( kp.keypt[0], kp.keypt[1], kp.keypt[2], t );
	  }
	}
      }//loop over all vectors
      if ( found_kpdata>0 ) {
	num_updated++;
	LARCV_DEBUG() << "KPdata updated TRACKID[" << trackid << "] nmatch=" << found_kpdata
		      << " labelid=" << labelid
		      << " shape=" << particle.shape()
		      << " pdg=" << particle.pdg_code()
		      << " start=(" << particle.first_step().X() << "," << particle.first_step().Y() << "," << particle.first_step().Z() << ")"
		      << " t=" << particle.first_step().T()
		      << " num_voxels=" << vox_nclass[labelid]
		      << std::endl;
      }
      else {
	LARCV_DEBUG() << "No KP match for TRACKID[" << trackid << "]"
		      << " shape=" << particle.shape()
		      << " pdg=" << particle.pdg_code()
		      << " start=(" << particle.first_step().X() << "," << particle.first_step().Y() << "," << particle.first_step().Z() << ")"
		      << " t=" << particle.first_step().T()
		      << std::endl;
      }
      
      // change particle track id to id label in tensor
      particle.track_id( (unsigned int)labelid );
      
      // update the number of voxels
      particle.num_voxels( vox_nclass[labelid] );

      // store this particle in the save list
      saved_v.emplace_back( std::move(particle) );
    }
    std::swap( particle_v, saved_v );
    LARCV_DEBUG() << "particles updated with KPdata info: " << num_updated << std::endl;
    LARCV_DEBUG() << "post-filtered particles=" << particle_v.size() << std::endl;
    
    instance_labels.sort();
    
    larcv::SparseTensor3D xlabel( std::move(instance_labels), meta );    
    out_v.emplace_back( std::move(xlabel) );
    
    return out_v;
    
  }  

  /** 
   * @brief make larcv::Voxel3DMeta from the 3d tensor parameters in TPCVoxelData
   *
   */
  larcv::Voxel3DMeta VoxelizeTriplets::make_meta( const larflow::voxelizer::TPCVoxelData& voxdata )
  {
    // voxel 0-dimension is in ticks. convert to x-coordinate cm using drift distance, relative to trigger
    float xmin = larutil::DetectorProperties::GetME()->ConvertTicksToX( voxdata._origin[0], 0, voxdata._tpcid, voxdata._cryoid );
    float xmax = voxdata._origin[0] + voxdata._len[0]*voxdata._nvoxels[0]; // in ticks
    xmax = larutil::DetectorProperties::GetME()->ConvertTicksToX(xmax, 0, voxdata._tpcid, voxdata._cryoid ); // in x cm
    float ymax = voxdata._origin[1] + voxdata._len[1]*voxdata._nvoxels[1];
    float zmax = voxdata._origin[2] + voxdata._len[2]*voxdata._nvoxels[2];
    larcv::Voxel3DMeta meta;    
    meta.set(xmin,voxdata._origin[1],voxdata._origin[2],
	     xmax,ymax,zmax,
	     voxdata._nvoxels[0], voxdata._nvoxels[1], voxdata._nvoxels[2],
	     larcv::kUnitCM);
    return meta;
  }
  
}
}
