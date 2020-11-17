#include "Prep3DSpatialEmbed.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "DataFormat/larflow3dhit.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "larflow/Reco/ShowerLikelihoodBuilder.h"

namespace larflow {
namespace spatialembed {

  /**
   * static variable to track if numpy environment has been setup
   */
  bool Prep3DSpatialEmbed::_setup_numpy = false;

  /**
   * @brief constructor where input files have been specified
   *
   */
  Prep3DSpatialEmbed::Prep3DSpatialEmbed( const std::vector<std::string>& input_root_files )
    : larcv::larcv_base("Prep3DSpatialEmbed"),
    _filter_by_instance_image(false),
    _tree(nullptr),
    _kowner(true),
    _in_pvid_row(nullptr),
    _in_pvid_col(nullptr),
    _in_pvid_depth(nullptr),
    _in_pinstance_id(nullptr),      
    _in_pq_u(nullptr),
    _in_pq_v(nullptr),
    _in_pq_y(nullptr)
  {
    TChain* chain = new TChain("s3dembed");
    for ( auto const& input_file : input_root_files ) {
      chain->Add( input_file.c_str() );
    }
    loadTreeBranches( (TTree*)chain );
  }
  
  
  /**
   * @brief convert larmatch info voxel data list, including truth
   */
  Prep3DSpatialEmbed::VoxelDataList_t
  Prep3DSpatialEmbed::process( larcv::IOManager& iolcv,
                               larlite::storage_manager& ioll,
                               bool make_truth_if_available )
  {

    larlite::event_larflow3dhit* ev_lfhit_v
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "larmatch" );

    larcv::EventImage2D* ev_adc_v
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _adc_image_treename );

    if ( ev_adc_v->as_vector().size()==0 ) {
      LARCV_CRITICAL() << "No ADC images!" << std::endl;
    }
    else
      LARCV_INFO() << "Number of adc images: " << ev_adc_v->as_vector().size() << std::endl;

    VoxelDataList_t data = process_larmatch_hits( *ev_lfhit_v, ev_adc_v->as_vector(), 0.5 );

    if ( _filter_by_instance_image ) {
      larcv::EventImage2D* ev_instance_v
        = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "instance" );
      if ( ev_instance_v->as_vector().size()>0 ) {
        VoxelDataList_t filtered_data = filterVoxelsByInstanceImage( data, ev_instance_v->as_vector() );
        if ( filtered_data.size()<data.size() ) {
          std::swap( filtered_data, data );
        }
      }
    }


    if ( make_truth_if_available )
      generateTruthLabels( iolcv, ioll, data );

    if ( _filter_out_non_nu_pixels ) {
      VoxelDataList_t filtered;
      for ( auto& voxel : data ) {
        if ( voxel.truth_instance_index>=0 ) {
          filtered.push_back(voxel);
        }
      }
      std::swap( filtered, data );
    }
    
    
    return data;
  }

  /**
   * @brief function that converts the larmatch points
   */
  Prep3DSpatialEmbed::VoxelDataList_t
  Prep3DSpatialEmbed::process_larmatch_hits( const larlite::event_larflow3dhit& ev_lfhit_v,
                                             const std::vector<larcv::Image2D>& adc_v,
                                             const float larmatch_threshold )
  {

    VoxelDataList_t voxel_v;
    voxel_v.reserve( ev_lfhit_v.size() );
    std::map< std::vector<int>, int > voxcoord_2_voxel_v; ///< map from (row,col,depth) to position in voxel_v

    int nhits_used = 0;
    for ( auto const& hit : ev_lfhit_v ) {
      std::vector<float> xyz = { hit[0], hit[1], hit[2] };
      std::vector<int> voxid = _voxelizer.get_voxel_indices( xyz );
      float lm_score = hit.track_score;

      if ( lm_score<larmatch_threshold )
        continue;

      nhits_used++;

      // find charges on wireplanes
      std::vector<float> q_v(3,0);
      for (int p=0; p<3; p++) {
        auto const& meta = adc_v[p].meta();
        int row = meta.row( hit.tick );
        int col = meta.col( hit.targetwire[p] );
        for (int dr=-2; dr<=2; dr++) {
          int r = row+dr;
          if ( r<=0 || r>=(int)meta.rows()) continue;
          for ( int dc=-2; dc<=2; dc++) {
            int c = col+dc;
            if ( c<0 || c>=(int)meta.cols()) continue;
            float q = adc_v[p].pixel(r,c,__FILE__,__LINE__);
            if ( q>10.0 ) {
              q_v[p] += q;
            }
          }
        }
      }//end of loop over planes

      auto it = voxcoord_2_voxel_v.find( voxid );
      if ( it==voxcoord_2_voxel_v.end() ) {
        // new voxeldata struct
        VoxelData_t data;
        data.voxel_index = voxid;
        data.feature_v.resize(3,0);
        data.ave_xyz_v.resize(3,0);
        for (int p=0; p<3; p++) {
          data.feature_v[p] = fabs(lm_score)*q_v[p];
          data.ave_xyz_v[p] = fabs(lm_score)*xyz[p];
        }
        data.npts = 1;
        data.totw = fabs(lm_score);
        voxel_v.emplace_back( std::move(data) );
        int idx = (int)voxel_v.size()-1;
        voxcoord_2_voxel_v[ voxid ] = idx;
      }
      else {
        // append to voxel data
        auto& data = voxel_v[it->second];
        for (int p=0; p<3; p++) {
          data.feature_v[p] += fabs(lm_score)*q_v[p];
          data.ave_xyz_v[p] += fabs(lm_score)*xyz[p];
        }
        data.npts++;
        data.totw += fabs(lm_score);
      }
      
    }//end of loop over larmatch hits

    // take averages
    // use average position to assign a pixel location for each voxel
    for ( auto& voxel : voxel_v ) {
      voxel.imgcoord_v.resize(4,0);
      if ( voxel.totw>0 ) {
        std::vector<double> xyz(3,0);
        for (int i=0; i<3; i++) {
          voxel.feature_v[i] /= voxel.totw;
          voxel.ave_xyz_v[i] /= voxel.totw;
          xyz[i] = voxel.ave_xyz_v[i];
        }

        for (int p=0; p<3; p++) {
          voxel.imgcoord_v[p] = larutil::Geometry::GetME()->WireCoordinate( xyz, p );
        }
        voxel.imgcoord_v[3] = xyz[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
      }
    }
    
    LARCV_NORMAL() << "Number of voxels created, " << voxel_v.size() << ", "
                   << "from " << nhits_used << " used hits "
                   << "(total " << ev_lfhit_v.size() << ")"
                   << std::endl;
      
    return voxel_v;
  }
  

  void Prep3DSpatialEmbed::bindVariablesToTree( TTree* atree )
  {
    if ( !_tree ) _tree = atree;
    _tree->Branch( "vidrow", &vid_row );
    _tree->Branch( "vidcol", &vid_col );
    _tree->Branch( "viddepth", &vid_depth );
    _tree->Branch( "q_u", &q_u );
    _tree->Branch( "q_v", &q_v );
    _tree->Branch( "q_y", &q_y );
    _tree->Branch( "instanceid", &instance_id );
  }

  void Prep3DSpatialEmbed::fillTree( const Prep3DSpatialEmbed::VoxelDataList_t& data )
  {

    vid_row.resize(data.size());
    vid_col.resize(data.size());
    vid_depth.resize(data.size());
    instance_id.resize(data.size());
    q_u.resize(data.size());
    q_v.resize(data.size());
    q_y.resize(data.size());

    for (size_t i=0; i<data.size(); i++) {
      auto const& d = data[i];
      vid_row[i]   = d.voxel_index[0];
      vid_col[i]   = d.voxel_index[1];
      vid_depth[i] = d.voxel_index[2];
      if ( d.totw>0) {
        q_u[i] = d.feature_v[0];
        q_v[i] = d.feature_v[1];
        q_y[i] = d.feature_v[2];
      }
      else {
        q_u[i] = 0.;
        q_v[i] = 0.;
        q_y[i] = 0.;
      }
      instance_id[i] = d.truth_instance_index+1;
    }

  }

  /**
   * @brief return a python dictionary with numpy arrays for single event
   *
   * contents of dictionary:
   * "coord_t":coordinate tensor
   * "feat_t":feature tensor
   */  
  PyObject* Prep3DSpatialEmbed::makeTrainingDataDict( const VoxelDataList_t& voxeldata ) const
  {
    std::vector<VoxelDataList_t> voxeldata_v;
    voxeldata_v.push_back(voxeldata);
    return makeTrainingDataDict( voxeldata_v );
  }
  
  /**
   * @brief return a python dictionary with numpy arrays for batch of events
   *
   * contents of dictionary:
   * "coord_t":coordinate tensor
   * "feat_t":feature tensor
   */
  PyObject* Prep3DSpatialEmbed::makeTrainingDataDict( const std::vector<VoxelDataList_t>& voxeldata_v ) const
  {

    if ( !Prep3DSpatialEmbed::_setup_numpy ) {
      import_array1(0);
      Prep3DSpatialEmbed::_setup_numpy = true;
    }

    size_t nbatches = voxeldata_v.size();
    size_t nvoxels_tot = 0;
    LARCV_INFO() << "Converting data for " << nbatches << " of voxels" << std::endl;
    for (size_t ibatch=0; ibatch<nbatches; ibatch++) {
      auto const& voxeldata = voxeldata_v[ibatch];
      size_t nvoxels = voxeldata.size();
      nvoxels_tot += nvoxels;
    }
    LARCV_INFO() << "Converting data for " << nvoxels_tot << " total voxels into numpy arrays" << std::endl;

    // DECLARE TENSORS and dict keys
    
    // coord tensor
    npy_intp coord_t_dim[] = { (long int)nvoxels_tot, 4 };
    PyArrayObject* coord_t = (PyArrayObject*)PyArray_SimpleNew( 2, coord_t_dim, NPY_LONG );
    PyObject *coord_t_key = Py_BuildValue("s", "coord_t");        

    // feature tensor
    npy_intp feat_t_dim[] = { (long int)nvoxels_tot, 3 };
    PyArrayObject* feat_t = (PyArrayObject*)PyArray_SimpleNew( 2, feat_t_dim, NPY_FLOAT );
    PyObject *feat_t_key = Py_BuildValue("s", "feat_t");    

    // instance tensor
    npy_intp instance_t_dim[] = { (long int)nvoxels_tot };
    PyArrayObject* instance_t = (PyArrayObject*)PyArray_SimpleNew( 1, instance_t_dim, NPY_LONG );
    PyObject *instance_t_key = Py_BuildValue("s", "instance_t");

    // FILL TENSORS
    size_t nvoxels_filled = 0;
    for ( size_t ibatch=0; ibatch<nbatches; ibatch++ ) {

      auto const& voxeldata = voxeldata_v[ibatch];
      size_t nvoxels = voxeldata.size();

      // fill coord tensor
      for (size_t i=0; i<nvoxels; i++ ) {
	auto const& voxel = voxeldata[i];
	for (size_t j=0; j<3; j++)
	  *((long*)PyArray_GETPTR2(coord_t,nvoxels_filled+i,j)) = voxel.voxel_index[j];
	*((long*)PyArray_GETPTR2(coord_t,nvoxels_filled+i,3))   = ibatch;
      }
      
      // fill feat tensor
      for (size_t i=0; i<nvoxels; i++ ) {
	auto const& voxel = voxeldata[i];
	if ( voxel.totw>0 ) {
	  for (size_t j=0; j<3; j++)
	    *((float*)PyArray_GETPTR2(feat_t,nvoxels_filled+i,j)) = voxel.feature_v[j];
	}
	else {
	  for (size_t j=0; j<3; j++)
	    *((float*)PyArray_GETPTR2(feat_t,nvoxels_filled+i,j)) = 0.;
	}
      }

      // fill instance tensor
      for (size_t i=0; i<nvoxels; i++ ) {
	auto const& voxel = voxeldata[i];
	*((long*)PyArray_GETPTR1(instance_t,nvoxels_filled+i)) = voxel.truth_instance_index+1;
      }

      nvoxels_filled += nvoxels;
    }
    
    // Create and fill dictionary
    PyObject *d = PyDict_New();
    PyDict_SetItem(d, coord_t_key,    (PyObject*)coord_t);
    PyDict_SetItem(d, feat_t_key,     (PyObject*)feat_t);
    PyDict_SetItem(d, instance_t_key, (PyObject*)instance_t);     
    
    Py_DECREF(coord_t_key);
    Py_DECREF(feat_t_key);
    Py_DECREF(instance_t_key);
    
    return d;
    
  }

    /**
   * @brief return a python dictionary with numpy arrays
   *
   * contents of dictionary:
   * "coord_t":coordinate tensor
   * "feat_t":feature tensor
   */
  PyObject* Prep3DSpatialEmbed::getTrainingDataBatch(int batch_size)
  {


    std::vector< VoxelDataList_t > data_batch;
    data_batch.reserve(batch_size);

    int ntries = 0;
    while ( ntries<batch_size*10 && data_batch.size()<batch_size ) {
      try {
	auto data = getTreeEntry(_current_entry);
	_current_entry++;
	if (data.size()>0) {
	  data_batch.emplace_back( std::move(data) );
	}
      }
      catch (...) {
	// reset entry index and try again
	_current_entry = 0;
      }
      ntries++;
    }

    if ( data_batch.size()==batch_size ) {
      return makeTrainingDataDict( data_batch );
    }

    Py_INCREF(Py_None);
    return Py_None;
  }

  void Prep3DSpatialEmbed::loadTreeBranches( TTree* atree )
  {
    if ( !_tree ) _tree = atree;
    _tree->SetBranchAddress("vidrow",&_in_pvid_row);
    _tree->SetBranchAddress("vidcol",&_in_pvid_col);
    _tree->SetBranchAddress("viddepth",&_in_pvid_depth);
    _tree->SetBranchAddress("instanceid",&_in_pinstance_id);    
    _tree->SetBranchAddress("q_u",&_in_pq_u);
    _tree->SetBranchAddress("q_v",&_in_pq_v);
    _tree->SetBranchAddress("q_y",&_in_pq_y);
  }

  Prep3DSpatialEmbed::VoxelDataList_t Prep3DSpatialEmbed::getTreeEntry(int entry)
  {
    
    if ( !_tree ) {
      LARCV_ERROR() << "Tree not loaded. Did you call loadTreeBranches(...)?" << std::endl;
      throw std::runtime_error("Tree not loaded");
    }
    
    Prep3DSpatialEmbed::VoxelDataList_t data;
    unsigned long bytes = _tree->GetEntry(entry);
    if ( !bytes ) {
      throw std::runtime_error("out of file-bounds");
    }
    _current_entry = entry;
    
    // fill container with vector values
    size_t nvoxels = _in_pvid_row->size();
    data.reserve( nvoxels );

    for ( size_t i=0; i<nvoxels; i++ ) {
      VoxelData_t voxel;
      voxel.voxel_index = { (*_in_pvid_row)[i], (*_in_pvid_col)[i], (*_in_pvid_depth)[i] };
      voxel.feature_v   = { (*_in_pq_u)[i], (*_in_pq_v)[i], (*_in_pq_y)[i] };
      voxel.npts = 1;
      voxel.totw = 1.0;
      voxel.truth_realmatch = -1;
      voxel.truth_instance_index = (*_in_pinstance_id)[i]-1;
      data.emplace_back( std::move(voxel) );
    }

    return data;
  }

  PyObject* Prep3DSpatialEmbed::getTreeEntryDataAsArray( int entry )
  {
    try {
      auto data = getTreeEntry(entry);
      if (data.size()>0)
	return makeTrainingDataDict( data );
    }
    catch (...){
    }

    Py_INCREF(Py_None);
    return Py_None;
  }

  /**
   * @brief get next entry in file. loops around
   * 
   */
  PyObject* Prep3DSpatialEmbed::getNextTreeEntryDataAsArray()
  {
    try {
      auto data = getTreeEntry(_current_entry);
      _current_entry++;          
      if (data.size()>0) {
	return makeTrainingDataDict( data );
      }
    }
    catch (...) {
      _current_entry = 0;
    }
    // try again
    try {
      auto data2 = getTreeEntry(_current_entry);
      _current_entry++;
      if (data2.size()>0)
	return makeTrainingDataDict( data2 );
    }
    catch (...){
    }

    // problems
    Py_INCREF(Py_None);
    return Py_None;
  }
  
  /**
   * @brief generate true labels for the voxels. use truth larflow points
   *
   */
  void Prep3DSpatialEmbed::generateTruthLabels( larcv::IOManager& iolcv,
                                                larlite::storage_manager& ioll,
                                                Prep3DSpatialEmbed::VoxelDataList_t& voxel_v )
  {

    larcv::EventImage2D* ev_adc_v
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _adc_image_treename );
    auto const& adc_v = ev_adc_v->as_vector();
    
    // build particle graph and assign pixels
    ublarcvapp::mctools::MCPixelPGraph mcpg;
    mcpg.set_adc_treename(_adc_image_treename);    
    mcpg.buildgraph( iolcv, ioll );
    LARCV_INFO() << "pre-shower builder graph" << std::endl;
    //mcpg.printAllNodeInfo();
    mcpg.printGraph();

    // additional algorithm to fix the shower instances
    larflow::reco::ShowerLikelihoodBuilder mcshowerbuilder;
    mcshowerbuilder.set_wire_tree_name( _adc_image_treename );
    mcshowerbuilder.process( iolcv, ioll );
    mcshowerbuilder.updateMCPixelGraph( mcpg, iolcv );
    LARCV_INFO() << "post-shower builder graph" << std::endl;
    mcpg.printGraph();
    std::vector<ublarcvapp::mctools::MCPixelPGraph::Node_t*> node_v = mcpg.getNeutrinoParticles();

    
    std::map<int,int> instance_2_index;
    std::vector<int> instance_v;
    
    // now do a dumb loop first
    size_t nvoxels = voxel_v.size();
    
    for ( size_t ivoxel=0; ivoxel<nvoxels; ivoxel++ ) {
      auto& voxeldata = voxel_v[ivoxel];

      std::vector<double> xyz = { (double)voxeldata.ave_xyz_v[0],
                                  (double)voxeldata.ave_xyz_v[1],
                                  (double)voxeldata.ave_xyz_v[2] };

      // default set instance to -1
      voxeldata.truth_instance_index = -1;
      voxeldata.truth_realmatch = 0;
      
      // now we determine if voxel is a part of instance 3d cluster, sigh

      // loop over all instance pixels
      int max_match_inode = -1;
      int max_num_matched = 0;
      for ( size_t inode=0; inode<node_v.size(); inode++ ) {
        ublarcvapp::mctools::MCPixelPGraph::Node_t* pnode = node_v[inode];

        int num_matched = 0;
        std::vector<int> plane_matched(3,0);
        
        for (int p=0; p<3; p++) {
          auto const& pix_v = pnode->pix_vv[p];
          int npix = pix_v.size()/2;
          for (int ipix=0; ipix<npix; ipix++) {
            int pixtick = pix_v[2*ipix];
            int pixwire = pix_v[2*ipix+1];
            // near the voxel?
            float dtick = fabs(pixtick-voxeldata.imgcoord_v[3]);
            float dwire = fabs(pixwire-voxeldata.imgcoord_v[p]);
            if ( dwire<2.5 && dtick < 2.5*adc_v[p].meta().pixel_height() ) {
              num_matched++;
              plane_matched[p] = 1;
            }
          }
        }
        int num_plane_matched = 0;
        for (int p=0; p<3; p++)
          num_plane_matched += plane_matched[p];

        if ( num_plane_matched>=2 && num_matched>max_num_matched ) {
          max_num_matched = num_matched;
          max_match_inode = inode;
        }
        
      }//loop over nodes

      if ( max_num_matched > 0 ) {
        // voxel matched to an instance!
        // get an instance id
        auto it = instance_2_index.find( max_match_inode );
        int instance_index = 0 ;
        if ( it==instance_2_index.end() ) {
          // new index
          instance_index = instance_v.size();
          instance_v.push_back( max_match_inode );
          instance_2_index[max_match_inode] = instance_index;
        }
        else {
          instance_index = it->second;
        }

        // all of the above just for this index ...
        voxeldata.truth_instance_index = instance_index;
      }//end of if a matching instance cluster found for voxel
      
    }//end of voxel loop


    LARCV_INFO() << "We matched " << instance_v.size() << " truth clusters to voxels" << std::endl;
    
  }

  /**
   * @brief select a subset of the voxels based on which ones land on instance image pixels
   *
   */
  Prep3DSpatialEmbed::VoxelDataList_t
  Prep3DSpatialEmbed::filterVoxelsByInstanceImage( const Prep3DSpatialEmbed::VoxelDataList_t& voxel_v,
                                                   const std::vector<larcv::Image2D>& instance_v )
  {
    Prep3DSpatialEmbed::VoxelDataList_t filtered_v;
    filtered_v.reserve( voxel_v.size() );

    int nrows = (int)instance_v.front().meta().rows();
    
    for (auto const& voxel : voxel_v ) {

      if ( voxel.imgcoord_v[3]<=instance_v.front().meta().min_y()
           || voxel.imgcoord_v[3]>=instance_v.front().meta().max_y() )
        continue;
      
      int voxelrow = instance_v.front().meta().row( voxel.imgcoord_v[3], __FILE__, __LINE__ );

      std::vector<int> found_in_plane(3,0);
      int num_instance_pixels = 0;
      
      // for each voxel check if lands on an instance image pixel on at least two planes
      for (int dr=-2; dr<=2; dr++) {
        int row = voxelrow + dr;
        if ( row<0 || row>=nrows ) continue;
        
        for (int p=0; p<3; p++) {
          int voxelcol = instance_v[p].meta().col( voxel.imgcoord_v[p] );
          int ncols = (int)instance_v[p].meta().cols();
     
          for (int dc=-2; dc<=2; dc++) {
            int col = voxelcol + dc;
            if ( col<0 || col>=ncols ) continue;
            int instanceid = instance_v[p].pixel( row, col, __FILE__, __LINE__ );
            if (instanceid>0) {
              num_instance_pixels++;
              found_in_plane[p] = 1;
            }
          }
        }//end of plane loop
        
      }//end of dr loop

      int num_planes_with_instancepix = 0;
      for (int p=0; p<3; p++)
        num_planes_with_instancepix += found_in_plane[p];

      if ( num_planes_with_instancepix>=2 && num_instance_pixels>0 ) {
        filtered_v.push_back( voxel );
      }
    }//end of voxel loop

    return filtered_v;
  }
  

  /**
   * @brief make 'perfect network output' for loss testing
   *
   * contents of dictionary:
   * "embed_t":(nvoxels,4)
   * "seed_t":(nvoxels,1)
   */
  PyObject* Prep3DSpatialEmbed::makePerfectNetOutput( const VoxelDataList_t& voxeldata,
                                                      const std::vector<int>& nvoxels_dim ) const
  {

    if ( !Prep3DSpatialEmbed::_setup_numpy ) {
      import_array1(0);
      Prep3DSpatialEmbed::_setup_numpy = true;
    }

    size_t nvoxels = voxeldata.size();
    //const std::vector<int>& nvoxels_dim = _voxelizer.get_nvoxels();    
    LARCV_INFO() << "Converting data for " << nvoxels << " voxels into numpy arrays" << std::endl;

    // first we calculate the centroid of each instance.
    int max_instance_id = -1;
    for (auto const& voxel : voxeldata ) {
      if ( voxel.truth_instance_index>max_instance_id )
        max_instance_id = voxel.truth_instance_index;
    }

    std::vector<int> num_instance_voxels(max_instance_id+1,0);
    std::vector< std::vector<float> > centroid(max_instance_id+1);
    for (int i=0; i<max_instance_id+1; i++) {
      centroid[i] = std::vector<float>(3,0);
    }

    for (auto const& voxel : voxeldata ) {
      if ( voxel.truth_instance_index>=0 ) {
        for (int i=0; i<3; i++) {
          centroid[voxel.truth_instance_index][i] += float(voxel.voxel_index[i]);
        }
        num_instance_voxels[voxel.truth_instance_index]++;
      }
    }

    for (int id=0; id<max_instance_id+1; id++) {
      if ( num_instance_voxels[id]>0 ) {
        std::cout << "centroid[" << id << "]: (";
        for (int j=0; j<3; j++) {
          centroid[id][j] /= float(num_instance_voxels[id]);
          std::cout << centroid[id][j];
          if (j+1<3 ) std::cout << ",";
        }
        std::cout << ") "
                  << "from " << num_instance_voxels[id]
                  << std::endl;

      }
    }
    

    // embed tensor
    npy_intp embed_t_dim[] = { (long int)nvoxels, 4 };
    PyArrayObject* embed_t = (PyArrayObject*)PyArray_SimpleNew( 2, embed_t_dim, NPY_FLOAT );

    // seed tensor
    npy_intp seed_t_dim[] = { (long int)nvoxels, 1 };
    PyArrayObject* seed_t = (PyArrayObject*)PyArray_SimpleNew( 2, seed_t_dim, NPY_FLOAT );

    std::vector<float> max_dist_from_centroid(num_instance_voxels.size(),0);

    // loop over voxel
    for (size_t i=0; i<nvoxels; i++ ) {
      auto const& voxel = voxeldata[i];

      // set normalized shift in embed tensor
      if ( voxel.truth_instance_index>=0 ) {
        // in instance
        int iid = voxel.truth_instance_index;

        // calc shift
        float voxel_diff = 0.;
        for (size_t j=0; j<3; j++) {
          float dv = centroid[iid][j]-float(voxel.voxel_index[j]);
          *((float*)PyArray_GETPTR2(embed_t,i,j)) = dv/float(nvoxels_dim[j]);
          voxel_diff += dv*dv;
        }
        voxel_diff = sqrt(voxel_diff);
        if ( voxel_diff>max_dist_from_centroid[iid] ) {
          max_dist_from_centroid[iid] = voxel_diff;
        }
        
        // set sigma in embed tensor
        *((float*)PyArray_GETPTR2(embed_t,i,3)) = 100.0;

        // set seed map
        *((float*)PyArray_GETPTR2(seed_t,i,0)) = 1.0;
      }
      else {
        for (size_t j=0; j<3; j++) {        
          *((float*)PyArray_GETPTR2(embed_t,i,j)) = 0.;
        }
        *((float*)PyArray_GETPTR2(embed_t,i,3)) = 0.01;
        *((float*)PyArray_GETPTR2(seed_t,i,0)) = 0.0;
      }

    }

    for (size_t iid=0; iid<max_dist_from_centroid.size(); iid++)
      LARCV_NORMAL() << "instance[" << iid << "] max voxel diff: " << max_dist_from_centroid[iid] << std::endl; 
    
    PyObject *embed_t_key = Py_BuildValue("s", "embed_t");    
    PyObject *seed_t_key = Py_BuildValue("s", "seed_t");


    PyObject *d = PyDict_New();
    PyDict_SetItem(d, embed_t_key,    (PyObject*)embed_t);
    PyDict_SetItem(d, seed_t_key,     (PyObject*)seed_t);
    
    Py_DECREF(embed_t_key);
    Py_DECREF(seed_t_key);
    
    return d;
    
    
  }
  
}
}
