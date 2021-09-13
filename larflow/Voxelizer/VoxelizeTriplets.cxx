#include "VoxelizeTriplets.h"

#include <iostream>
#include <sstream>

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
  {

    _origin.clear();
    _len.clear();
    
    const float driftv = larutil::LArProperties::GetME()->DriftVelocity();
    
    _origin.resize(3,0);
    _origin[0] = (2399-3200)*0.5*driftv;    
    _origin[1] = -120.0;
    _origin[2] = 0;
 
    _len.resize(3,0);
    _len[0] = 1010*6*0.5*driftv;    
    _len[1] = 2.0*120.0;
    _len[2] = 1037.0;

    _voxel_size = 0.3;
    
    _define_voxels();
  }

  /**
   * @brief constructor where voxel grid can be specified
   *
   * @param[in] origin  Position in 3D where origin of the voxel grid is located (in cm).
   * @param[in] dim_len Total length of each dimension (should be 3) in cm.
   * @param[in] voxel_size Length of height, width, and depth of an individual voxel in cm.
   *
   */  
  VoxelizeTriplets::VoxelizeTriplets( std::vector<float> origin,
                                      std::vector<float> dim_len,
                                      float voxel_size ) 
    : _origin(origin),
    _len(dim_len),
    _voxel_size(voxel_size)
  {
    _define_voxels();
  }


  /**
   * @brief using the origin, grid length, and voxel size define the voxel grid
   *
   */
  void VoxelizeTriplets::_define_voxels()
  {
    _ndims = (int)_origin.size();
    _nvoxels.resize(_ndims,0);

    std::stringstream ss;
    ss << "(";
    for (int v=0; v<_ndims; v++) {
      int nv = (_len[v])/_voxel_size;
      if ( fabs(nv*_voxel_size-_len[v])>0.001 )
        nv++;
      _nvoxels[v] = nv;
      ss << nv;
      if ( v+1<_ndims ) ss << ", ";
    }
    ss << ")";

    std::cout << "[VoxelizeTriplets::" << __FUNCTION__ << ".L" << __LINE__ << "] "
              << "Number of voxels defined: " << ss.str() << ", ndims=" << _ndims
              << std::endl;
    std::cout << "[VoxelizeTriplets::" << __FUNCTION__ << ".L" << __LINE__ << "] "    
              << "BOUNDS: [" << _origin[0] << "," << _origin[0]+_len[0] << "] "
              << "[" << _origin[1] << "," << _origin[1]+_len[1] << "] "
              << "[" << _origin[2] << "," << _origin[2]+_len[2] << "] "
              << std::endl;    
  }

  /**
   * @brief set/reset voxel definition
   *
   */
  void VoxelizeTriplets::set_voxel_size_cm( float width_cm )
  {
    _voxel_size = width_cm;
    _define_voxels();
  }

  /**
   * @brief get the voxel bin along one of the dimensions
   *
   * @param[in] axis Dimension we want
   * @param[in] coord Coordinate in the dimension we want
   * @return voxel bin index along the given dimension
   */
  int VoxelizeTriplets::get_axis_voxel( int axis, float coord ) const {

    if ( axis<0 || axis>=_ndims ) {
      std::stringstream ss;
      ss << "[VoxelizeTriplets::" << __FUNCTION__ << ".L" << __LINE__ << "] invalid dim given: " << axis << " (_ndims=" << _ndims << ")" << std::endl;
      throw std::runtime_error(ss.str());
    }
    
    int vidx = (coord-_origin[axis])/_voxel_size;
    if (vidx<0 || vidx>=_nvoxels[axis] ) {
      std::stringstream ss;
      ss << "[VoxelizeTriplets::" << __FUNCTION__ << ".L" << __LINE__ << "]";
      ss << " dim[" << axis << "] coordinate[" << coord << "] "
         << "given is out of bounds [" << _origin[axis] << "," << _origin[axis]+_len[axis] << "]"
         << " vidx=" << vidx << " bounds[0," << _nvoxels[axis] << ")"
         << std::endl;
      throw std::runtime_error(ss.str());
    }

    return vidx;
  }
  
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
  void VoxelizeTriplets::make_voxeldata( const larflow::prep::PrepMatchTriplets& triplet_data )
  {
    
    // first we need to define the voxels that are filled    
    _voxel_set.clear();
    
    for ( int itriplet=0; itriplet<(int)triplet_data._triplet_v.size(); itriplet++ ) {
      const std::vector<float>& pos = triplet_data._pos_v[itriplet];
      std::array<int,3> coord;
      for (int i=0; i<3; i++)
        coord[i] = get_axis_voxel(i,pos[i]);
      _voxel_set.insert( coord );        
    }

    // now we assign voxel to an index
    _voxel_list.clear();
    int idx=0;
    for ( auto& coord : _voxel_set ) {
      _voxel_list[coord] = idx;
      idx++;
    }
    int nvidx = idx;    

    std::cout << "[VoxelizeTriplets::" << __FUNCTION__ << ".L" << __LINE__ << "] "    
              << "Filling " << nvidx << " voxels from " << triplet_data._triplet_v.size() << " triplets"
              << " fillfrac=" << float(nvidx)/((float)_nvoxels[0]*(float)_nvoxels[1]*(float)_nvoxels[2])*100.0 << "%"
              << std::endl;

    // assign triplets to voxels and vice versa
    _voxelidx_to_tripidxlist.clear();
    _voxelidx_to_tripidxlist.resize(nvidx);
    _trip2voxelidx.clear();
    _trip2voxelidx.resize( triplet_data._triplet_v.size(), 0 );
    
    for ( int itriplet=0; itriplet<(int)triplet_data._triplet_v.size(); itriplet++ ) {
      const std::vector<float>& pos = triplet_data._pos_v[itriplet];
      std::array<int,3> coord;
      for (int i=0; i<3; i++)
        coord[i] = get_axis_voxel(i,pos[i]);
      auto it=_voxel_list.find(coord);
      if ( it==_voxel_list.end() ) {
        throw std::runtime_error("could not find a voxel we defined!!");
      }
      
      int voxelidx = it->second;
      _trip2voxelidx[itriplet] = voxelidx;
      _voxelidx_to_tripidxlist[voxelidx].push_back( itriplet );
    }
    
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
  PyObject* VoxelizeTriplets::make_voxeldata_dict( const larflow::prep::PrepMatchTriplets& triplet_data )
  {
    
    // ok now we can make the arrays
    if ( !_setup_numpy ) {
      std::cout << "[VoxelizeTriplets::" << __FUNCTION__ << ".L" << __LINE__ << "] setup numpy" << std::endl;
      import_array1(0);
      _setup_numpy = true;
    }       

    int nvidx = (int)_voxel_set.size();
    
    // first the voxel coordinate array
    npy_intp* coord_dims = new npy_intp[2];
    coord_dims[0] = (int)nvidx;
    coord_dims[1] = (int)_ndims;
    PyArrayObject* coord_array = (PyArrayObject*)PyArray_SimpleNew( 2, coord_dims, NPY_LONG );
    for ( auto it=_voxel_list.begin(); it!=_voxel_list.end(); it++ ) {
      int vidx = it->second;
      const std::array<int,3>& coord = it->first;
      for (int j=0; j<_ndims; j++)
        *((long*)PyArray_GETPTR2( coord_array, (int)vidx, j)) = (long)coord[j];
    }
    std::cout << "  made coord array" << std::endl;

    // voxel feature array: charge on planes, taking mean
    npy_intp* feat_dims = new npy_intp[2];
    feat_dims[0] = (int)nvidx;
    feat_dims[1] = (int)3;
    PyArrayObject* feat_array = (PyArrayObject*)PyArray_SimpleNew( 2, feat_dims, NPY_FLOAT );
    for ( auto it=_voxel_list.begin(); it!=_voxel_list.end(); it++ ) {
      int vidx = it->second; // voxel index

      // ave plane charge of spacepoints associated to this voxel
      std::vector<int>& tripidx_v = _voxelidx_to_tripidxlist[vidx]; // index of triplet
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
    for ( auto it=_voxel_list.begin(); it!=_voxel_list.end(); it++ ) {
      int vidx = it->second;
      int truthlabel = 0.;

      if ( has_truth ) {
        // is there a true pixel?
        std::vector<int>& tripidx_v = _voxelidx_to_tripidxlist[vidx];
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
    std::cout << "  made truth array" << std::endl;    

    // the triplet to voxel index array
    npy_intp* trip2vidx_dims = new npy_intp[1];
    trip2vidx_dims[0] = (int)_trip2voxelidx.size();
    PyArrayObject* trip2vidx_array = (PyArrayObject*)PyArray_SimpleNew( 1, trip2vidx_dims, NPY_LONG );
    for (int itriplet=0; itriplet<(int)_trip2voxelidx.size(); itriplet++ ) {
      *((long*)PyArray_GETPTR1( trip2vidx_array, itriplet )) = (long)_trip2voxelidx[itriplet];
    }
    std::cout << "  made triplet-to-voxelindex array" << std::endl;    

    // finally the list of triplet indices for each voxel
    PyObject* tripidx_pylist = PyList_New( nvidx );
    for ( int vidx=0; vidx<nvidx; vidx++ ) {

      // make the array
      npy_intp* tidx_dims = new npy_intp[1];
      std::vector<int>& tripidx_v = _voxelidx_to_tripidxlist[vidx];
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
    std::cout << "  made voxel-index to triplet-list list" << std::endl;        

    // the dictionary
    PyObject *d = PyDict_New();
    PyObject *key_coord     = Py_BuildValue("s", "voxcoord" );
    PyObject *key_feat      = Py_BuildValue("s", "voxfeat" );    
    PyObject *key_label     = Py_BuildValue("s", "voxlabel" );
    PyObject *key_trip2vidx = Py_BuildValue("s", "trip2vidx" );
    PyObject *key_vox2trips = Py_BuildValue("s", "vox2trips_list" );

    PyDict_SetItem( d, key_coord, (PyObject*)coord_array );
    PyDict_SetItem( d, key_feat, (PyObject*)feat_array );    
    PyDict_SetItem( d, key_label, (PyObject*)vlabel_array );
    PyDict_SetItem( d, key_trip2vidx, (PyObject*)trip2vidx_array );
    PyDict_SetItem( d, key_vox2trips, (PyObject*)tripidx_pylist );

    std::cout << "  dereference" << std::endl;
    Py_DECREF( key_coord );
    Py_DECREF( key_feat );    
    Py_DECREF( key_label );
    Py_DECREF( key_trip2vidx );
    Py_DECREF( key_vox2trips );

    Py_DECREF( coord_array );
    Py_DECREF( feat_array );    
    Py_DECREF( vlabel_array );
    Py_DECREF( trip2vidx_array );
    Py_DECREF( tripidx_pylist );
    
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
    return make_voxeldata_dict( _triplet_maker );
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
      _triplet_maker.make_truth_vector( ev_larflow->Image2DArray() );
    }

    make_voxeldata( _triplet_maker );

    // persistancy:

    // save larflow3dhits for visualization
    // save class in ttree data for training
    
  }

  /**
   * @brief get voxel indices from non-t0 corrected space point coordinate
   *
   */
  std::vector<int> VoxelizeTriplets::get_voxel_indices( const std::vector<float>& xyz ) const
  {
    std::vector<int> indices(3,0);
    for (int i=0; i<3; i++) {

      if ( xyz[i]-_origin[i]<0 || xyz[i]>=_origin[i]+_len[i] ) {
        std::stringstream ss;
        ss << "[VoxelizeTriplets::get_voxel_indices.L" << __LINE__ << "] "
           << " Space point provided (" << xyz[0] << "," << xyz[1] << "," << xyz[2] << ") is out of bounds"
           << " for axis=" << i << " bounds=[" << _origin[i] << "," << _origin[i]+_len[i] << ")"
           << std::endl;
        throw std::runtime_error( ss.str() );
      }
      indices[i] = get_axis_voxel( i, xyz[i] );
    }
    return indices;
  }

  /**
   * @brief make voxel labels for ssnet
   */
  PyObject* VoxelizeTriplets::make_ssnet_voxel_labels( const larflow::keypoints::LoaderKeypointData& data )
  {

    // ok now we can make the arrays
    if ( !_setup_numpy ) {
      std::cout << "[VoxelizeTriplets::" << __FUNCTION__ << ".L" << __LINE__ << "] setup numpy" << std::endl;
      import_array1(0);
      _setup_numpy = true;
    }       

    int nvidx = (int)_voxel_set.size();
    
    // voxel ssnet label array: charge on planes, taking mean
    npy_intp* ssnet_dims = new npy_intp[2];
    ssnet_dims[0] = (int)nvidx;
    ssnet_dims[1] = (int)1;
    PyArrayObject* ssnet_array = (PyArrayObject*)PyArray_SimpleNew( 2, ssnet_dims, NPY_LONG );
    for ( auto it=_voxel_list.begin(); it!=_voxel_list.end(); it++ ) {
      int vidx = it->second; // voxel index

      // find class label with most (background) triplets
      std::vector<int>& tripidx_v = _voxelidx_to_tripidxlist[vidx]; // index of triplet
      std::vector<int> nclass( larflow::prep::PrepSSNetTriplet::kNumClasses, 0 );      
      for ( auto const& tripidx : tripidx_v ) {
	int triplet_label = data.ssnet_label_v->at(tripidx);
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

      if (max_class>0 && max_class_n>=nclass[0] )
	*((long*)PyArray_GETPTR2( ssnet_array, (int)vidx, 0)) = max_class;
      else
	*((long*)PyArray_GETPTR2( ssnet_array, (int)vidx, 0)) = 0;
      
    }

    std::cout << "  made ssnet truth array" << std::endl;
    return (PyObject*)ssnet_array;
    
  }

  /**
   * @brief get full label set for voxels
   * @param data Class holding the triplet data and truth labels loaded inside
   * @return dictionary with numpy arrays
   */
  PyObject* VoxelizeTriplets::get_full_voxel_labelset_dict( const larflow::keypoints::LoaderKeypointData& data )
  {
    // get larmatch voxels and truth labels
    PyObject* larmatch_dict = VoxelizeTriplets::make_voxeldata_dict( data.triplet_v->at(0) );
    
    PyObject *ssnet_label_key = Py_BuildValue("s", "ssnet_labels" );
    PyObject* ssnet_array = make_ssnet_voxel_labels( data );
    PyDict_SetItem(larmatch_dict, ssnet_label_key, (PyObject*)ssnet_array);

    PyArrayObject* larmatch_labels = (PyArrayObject*)PyDict_GetItemString( larmatch_dict, "voxlabel" );
    PyArrayObject* kplabel  = nullptr;
    PyArrayObject* kpweight = nullptr;
    make_kplabel_arrays( data, larmatch_labels, kplabel, kpweight );
    PyObject *kp_label_key = Py_BuildValue("s", "kplabel" );
    PyDict_SetItem(larmatch_dict, kp_label_key, (PyObject*)kplabel );
    //PyDict_SetItem(larmatch_dict, kp_label_key, (PyObject*)kplabel );    

    Py_DECREF(ssnet_label_key);
    Py_DECREF(ssnet_array);
    Py_DECREF(kp_label_key);
    Py_DECREF(kplabel);

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
  int VoxelizeTriplets::make_kplabel_arrays( const larflow::keypoints::LoaderKeypointData& data,
					     PyArrayObject* larmatch_label_array,
					     PyArrayObject*& kplabel_array,
					     PyArrayObject*& kplabel_weight )
  {

    // ok now we can make the arrays
    if ( !_setup_numpy ) {
      std::cout << "[VoxelizeTriplets::" << __FUNCTION__ << ".L" << __LINE__ << "] setup numpy" << std::endl;
      import_array1(0);
      _setup_numpy = true;
    }       

    int nvidx = (int)_voxel_set.size();
    int nclasses = 6; //number of keypoint classes    
    
    // voxel ssnet label array: charge on planes, taking mean
    npy_intp* kplabel_dims = new npy_intp[2];
    kplabel_dims[0] = (int)nvidx;
    kplabel_dims[1] = (int)nclasses;
    kplabel_array = (PyArrayObject*)PyArray_SimpleNew( 2, kplabel_dims, NPY_FLOAT );

    /// ------- ///
    
    float sigma = 10.0; // cm
    float sigma2 = sigma*sigma; // cm^2

    std::vector<int> npos(nclasses,0);
    std::vector<int> nneg(nclasses,0);

    for ( auto it=_voxel_list.begin(); it!=_voxel_list.end(); it++ ) {
      int vidx = it->second; // voxel index, index in the array we are filling
      const std::array<int,3>& arr_index = it->first; // index in the dense 3D array

      std::vector<float> vox_center(3,0);
      for (int i=0; i<3; i++)
	vox_center[i] = ((float)arr_index[i]+0.5)*get_voxel_size() + get_origin()[i]; // position of voxel center

      //std::cout << "vox-center: (" << vox_center[0] << "," << vox_center[1] << "," << vox_center[2] << ")" << std::endl;

      long larmatch_truth_label = 1.0;
      if ( larmatch_label_array!=NULL )
	larmatch_truth_label = *((long*)PyArray_GETPTR1(larmatch_label_array,vidx));
	  
      // for each class, calculate distance to closest true keypoint
      // use smallest label
      for (int c=0; c<6; c++) {

	if ( larmatch_truth_label==0 ) {
	  *((float*)PyArray_GETPTR2(kplabel_array,vidx,c)) = 0.0;
	  nneg[c]++;
	  continue;
	}
	
	const std::vector< std::vector<float> >& pos_v = *(data.kppos_v[c]);
	int nkp = pos_v.size();
	float min_dist_kp = 1.0e9;
	int   max_kp_idx = -1;
	for (int ikp=0; ikp<nkp; ikp++) {
	  const std::vector<float>& pos = pos_v[ikp];
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
	    *((float*)PyArray_GETPTR2(kplabel_array,vidx,c)) = labelscore;
	    npos[c]++;	    
	  }
	  else {
	    *((float*)PyArray_GETPTR2(kplabel_array,vidx,c)) = 0.0;
	    nneg[c]++;	    
	  }
	}
	else {
	  *((float*)PyArray_GETPTR2(kplabel_array,vidx,c)) = 0.0;
	  nneg[c]++;
	}
      }
    }//end of voxel list
	
    // weights to balance positive and negative examples
    // int kpweight_nd = 2;
    // npy_intp kpweight_dims[] = { (long)pos_match_index.size(), nclasses };
    // if ( !_exclude_neg_examples )
    //   kpweight_dims[0] = num_max_samples;
    // kplabel_weight = (PyArrayObject*)PyArray_SimpleNew( kpweight_nd, kpweight_dims, NPY_FLOAT );

    // for (int c=0; c<nclasses; c++ ) {
    //   float w_pos = (npos[c]) ? float(npos[c]+nneg[c])/float(npos[c]) : 0.0;
    //   float w_neg = (nneg[c]) ? float(npos[c]+nneg[c])/float(nneg[c]) : 0.0;
    //   float w_norm = w_pos*npos[c] + w_neg*nneg[c];

    //   //std::cout << "Keypoint class[" << c << "] WEIGHT: W(POS)=" << w_pos/w_norm << " W(NEG)=" << w_neg/w_norm << std::endl;
    
    //   for (int i=0; i<kpweight_dims[0]; i++ ) {
    //     // sample array index
    //     int idx = (_exclude_neg_examples) ? pos_match_index[i] : i;
    //     // triplet index
    //     long index = *((long*)PyArray_GETPTR2(match_array,idx,index_col));
    //     // hard label
    //     long label = kplabel_v[c]->at( index )[0];
    //     if (label==1) {
    //       *((float*)PyArray_GETPTR2(kplabel_weight,i,c)) = w_pos/w_norm;
    //     }
    //     else {
    //       *((float*)PyArray_GETPTR2(kplabel_weight,i,c))  = w_neg/w_norm;
    //     }
    //   }
    // }//end of class loop

    return 0;
  }
  
}
}
