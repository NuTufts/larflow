#include "Prep3DSpatialEmbed.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "TRandom3.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "DataFormat/larflow3dhit.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/NeutrinoVertex.h"
//#include "larflow/Reco/ShowerLikelihoodBuilder.h"

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
    _current_entry(0),
    _num_entries(0),
    _kowner(true),
    _adc_image_treename("wire"),
    _truth_image_treename("segment"),    
    _in_pvid_row(nullptr),
    _in_pvid_col(nullptr),
    _in_pvid_depth(nullptr),
    _in_pinstance_id(nullptr),
    _in_pancestor_id(nullptr),
    _in_pparticle_id(nullptr),          
    _in_pq_u(nullptr),
    _in_pq_v(nullptr),
    _in_pq_y(nullptr),
    _in_ptriplet_idx_v(nullptr),    
    _in_psubcluster_id(nullptr),
    _rand(nullptr)
  {
    TChain* chain = new TChain("s3dembed");
    for ( auto const& input_file : input_root_files ) {
      chain->Add( input_file.c_str() );
    }
    loadTreeBranches( (TTree*)chain );
  }
  
  Prep3DSpatialEmbed::~Prep3DSpatialEmbed()
  {
    if (_kowner) delete (TChain*)_tree;
    if ( _rand)  delete _rand;
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
        = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _truth_image_treename );
      if ( ev_instance_v->as_vector().size()>0 ) {
        VoxelDataList_t filtered_data = filterVoxelsByInstanceImage( data, ev_instance_v->as_vector() );
        if ( filtered_data.size()<data.size() ) {
          std::swap( filtered_data, data );
        }
      }
    }

    // if ( make_truth_if_available )
    //   generateTruthLabels( iolcv, ioll, data );

    // if ( _filter_out_non_nu_pixels ) {
    //   VoxelDataList_t filtered;
    //   for ( auto& voxel : data ) {
    //     if ( voxel.truth_instance_index>=0 ) {
    //       filtered.push_back(voxel);
    //     }
    //   }
    //   std::swap( filtered, data );
    // }
    
    
    return data;
  }

  /**
   * @brief we make voxels from true larflow hits
   */
  Prep3DSpatialEmbed::VoxelDataList_t
  Prep3DSpatialEmbed::process_from_truelarflowhits( larcv::IOManager& iolcv,
                                                    larlite::storage_manager& ioll )
  {

    _triplet_maker.clear();
    _triplet_maker.process( iolcv, _adc_image_treename, _adc_image_treename, 10.0, true );
    _triplet_maker.process_truth_labels( iolcv, _adc_image_treename );
    _triplet_truth_fixer.calc_reassignments( _triplet_maker, iolcv, ioll );

    larcv::EventImage2D* ev_adc_v
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _adc_image_treename );
    auto const& adc_v = ev_adc_v->as_vector();
    
    larlite::event_larflow3dhit* ev_prep =
      (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, "prepspembed");

    // we make larflow hit vector
    // we keep only those hits on neutrino pixels, using the segment image to filter out cosmic pixels
    ev_prep->reserve( _triplet_maker._triplet_v.size() );
    for ( int itriplet=0; itriplet<_triplet_maker._triplet_v.size(); itriplet++ ) {
      int truth  = _triplet_maker._truth_v[itriplet];
      int origin = _triplet_maker._origin_v[itriplet];

      // check: is true hit
      if ( truth==0 ) continue;
      // check: is of neutrino origin
      if ( _filter_out_non_nu_pixels && origin!=1 ) continue;

      auto const& triplet = _triplet_maker._triplet_v[itriplet];      
      std::vector<int> imgcoord = _triplet_maker.get_triplet_imgcoord_rowcol( itriplet );      
      auto const& pos = _triplet_maker._pos_v[itriplet];
      int row = triplet[3];
      int tick = adc_v.front().meta().pos_y(row);

      //std::cout << "imgcoord[" << itriplet << "]: " << imgcoord[0] << "," << imgcoord[1] << "," << imgcoord[2] << "," << imgcoord[3] << std::endl;      
      // check, out of image
      bool inside_image = true;
      for (int p=0; p<3; p++) {
        if ( imgcoord[p]<0 || imgcoord[p]>=(int)adc_v[p].meta().cols() )
          inside_image = false;
      }
      if ( !inside_image )
        continue;
      
      // check: non-neutrino
      int segid = _triplet_maker._pdg_v[itriplet];
      if ( _filter_out_non_nu_pixels && segid<=0 ) {
        continue;
      }
      
      // make the hit      
      larlite::larflow3dhit lfhit;
      lfhit.resize(3);
      lfhit.targetwire.resize(3,0);
      for (int i=0; i<3; i++ ) {
        lfhit.targetwire[i] = imgcoord[i];
        lfhit[i] = pos[i];
      }
      lfhit.tick = tick;
      lfhit.track_score = 1.0;
      lfhit.idxhit = itriplet;

      ev_prep->emplace_back( std::move(lfhit) );
    }

    VoxelDataList_t data = process_larmatch_hits( *ev_prep, adc_v, 0.5 );

    // loop over triplet, assigning instance ids
    generateTruthLabels( iolcv, ioll, _triplet_maker, data );

    _generate_subcluster_labels( data, ioll, true );

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
        int row = meta.row( hit.tick, __FILE__, __LINE__ );
        int col = meta.col( hit.targetwire[p], __FILE__, __LINE__ );
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
        data.tripletidx_v.clear();
        data.tripletidx_v.push_back( hit.idxhit );
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
        data.tripletidx_v.push_back( hit.idxhit );        
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

      // set truth labels to default:
      voxel.truth_instance_index = 0;
      voxel.truth_ancestor_index = 0;
      voxel.truth_realmatch = 0;      
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
    _tree->Branch( "ancestorid", &ancestor_id );    
    _tree->Branch( "particleid", &particle_id );
    _tree->Branch( "triplet_idx_v", &triplet_idx_v );
    _tree->Branch( "subclusterid",  &subcluster_id );    
  }

  void Prep3DSpatialEmbed::fillTree( const Prep3DSpatialEmbed::VoxelDataList_t& data )
  {

    vid_row.resize(data.size());
    vid_col.resize(data.size());
    vid_depth.resize(data.size());
    instance_id.resize(data.size());
    ancestor_id.resize(data.size());    
    particle_id.resize(data.size());    
    q_u.resize(data.size());
    q_v.resize(data.size());
    q_y.resize(data.size());
    triplet_idx_v.resize( data.size() );
    subcluster_id.resize( data.size() );

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

      instance_id[i] = d.truth_instance_index;
      ancestor_id[i] = d.truth_ancestor_index;
      if ( d.truth_pid>=2 )
        particle_id[i] = d.truth_pid-2;
      else
        particle_id[i] = 0;
      
      int ntrip = (int)d.tripletidx_v.size();
      triplet_idx_v[i].resize(ntrip);
      for (int ii=0; ii<ntrip; ii++)
        triplet_idx_v[i][ii] = d.tripletidx_v[ii];

      subcluster_id[i] = d.subclusterid;
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

    // class tensor
    npy_intp class_t_dim[] = { (long int)nvoxels_tot };
    PyArrayObject* class_t = (PyArrayObject*)PyArray_SimpleNew( 1, class_t_dim, NPY_LONG );
    PyObject *class_t_key = Py_BuildValue("s", "class_t");

    // subcluster tensor
    npy_intp subcluster_t_dim[] = { (long int)nvoxels_tot };
    PyArrayObject* subcluster_t = (PyArrayObject*)PyArray_SimpleNew( 1, subcluster_t_dim, NPY_LONG );
    PyObject *subcluster_t_key = Py_BuildValue("s", "subcluster_t");
        
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
	*((long*)PyArray_GETPTR1(instance_t,nvoxels_filled+i)) = voxel.truth_instance_index;
      }

      // fill class tensor
      for (size_t i=0; i<nvoxels; i++ ) {
	auto const& voxel = voxeldata[i];
	*((long*)PyArray_GETPTR1(class_t,nvoxels_filled+i)) = voxel.truth_pid;
      }

      // fill subcluster tensor
      for (size_t i=0; i<nvoxels; i++ ) {
	auto const& voxel = voxeldata[i];
	*((long*)PyArray_GETPTR1(subcluster_t,nvoxels_filled+i)) = voxel.subclusterid;
      }
      
      
      nvoxels_filled += nvoxels;
    }

    // set own data flag
    PyArray_ENABLEFLAGS(coord_t,      NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS(feat_t,       NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS(instance_t,   NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS(class_t,      NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS(subcluster_t, NPY_ARRAY_OWNDATA);    

    PyObject* tripletmap_list = PyList_New(0);
    PyObject* tripletmapweight_list = PyList_New(0);
    PyObject *tm_t_key  = Py_BuildValue("s", "tripletmap_t");
    PyObject *tmw_t_key = Py_BuildValue("s", "tripletmapweight_t");
    for ( size_t ibatch=0; ibatch<nbatches; ibatch++ ) {
      // triplet to voxel map tensor
      auto const& voxeldata = voxeldata_v[ibatch];
      size_t nvoxels = voxeldata.size();
      
      npy_intp tripletmap_t_dim[] = { (long int)nvoxels, (long int)_kMaxTripletPerVoxel };
      PyArrayObject* tripletmap_t = (PyArrayObject*)PyArray_SimpleNew( 2, tripletmap_t_dim, NPY_LONG );

      npy_intp tripletmapweight_t_dim[] = { (long int)nvoxels, (long int)_kMaxTripletPerVoxel };
      PyArrayObject* tripletmapweight_t = (PyArrayObject*)PyArray_SimpleNew( 2, tripletmapweight_t_dim, NPY_FLOAT );
      
      // fill triplet map
      // we use the previous two tensors to make an average sum of the triplet feature tensors
      for (size_t i=0; i<nvoxels; i++ ) {
	auto const& voxel = voxeldata[i];
        int ntrips = (int)voxel.tripletidx_v.size();
        if ( ntrips>_kMaxTripletPerVoxel )
          ntrips = _kMaxTripletPerVoxel;

        float weight = 0;
        if ( ntrips>0 )
          weight = 1.0/float(ntrips);
        
        for (int j=0; j<ntrips; j++) {
          *((long*)PyArray_GETPTR2(tripletmap_t,i,j)) = voxel.tripletidx_v[j];
          *((float*)PyArray_GETPTR2(tripletmapweight_t,i,j)) = weight;
        }
        for (int j=ntrips;j<_kMaxTripletPerVoxel;j++) {
          *((long*)PyArray_GETPTR2(tripletmap_t,i,j)) = 0;
          *((float*)PyArray_GETPTR2(tripletmapweight_t,i,j)) = 0;
        }
      }

      PyList_Append(tripletmap_list,       (PyObject*)tripletmap_t);
      PyList_Append(tripletmapweight_list, (PyObject*)tripletmapweight_t);
      Py_DECREF(tripletmap_t);
      Py_DECREF(tripletmapweight_t);
    }
    
    
    // Create and fill dictionary
    PyObject *d = PyDict_New();
    PyDict_SetItem(d, coord_t_key,      (PyObject*)coord_t);
    PyDict_SetItem(d, feat_t_key,       (PyObject*)feat_t);
    PyDict_SetItem(d, instance_t_key,   (PyObject*)instance_t);
    PyDict_SetItem(d, class_t_key,      (PyObject*)class_t);
    PyDict_SetItem(d, subcluster_t_key, (PyObject*)subcluster_t);    
    PyDict_SetItem(d, tm_t_key,         (PyObject*)tripletmap_list);
    PyDict_SetItem(d, tmw_t_key,        (PyObject*)tripletmapweight_list);
    
    Py_DECREF(coord_t_key);
    Py_DECREF(feat_t_key);
    Py_DECREF(instance_t_key);
    Py_DECREF(class_t_key);
    Py_DECREF(subcluster_t_key);    
    Py_DECREF(tm_t_key);
    Py_DECREF(tmw_t_key);        
    // do i need to do this?
    Py_DECREF(coord_t);
    Py_DECREF(feat_t);
    Py_DECREF(instance_t);
    Py_DECREF(class_t);
    Py_DECREF(subcluster_t);
    Py_DECREF(tripletmap_list);
    Py_DECREF(tripletmapweight_list);
    
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

    if ( _shuffle && _num_entries==0 ) {
      _num_entries = _tree->GetEntries();
      _rand = new TRandom3(0);
    }

    int ntries = 0;
    while ( ntries<batch_size*10 && data_batch.size()<batch_size ) {

      if ( !_shuffle ) {
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
      }
      else {
	// shuffle, dumb
	try {
	  _current_entry = _rand->Integer(_num_entries);
	  auto data = getTreeEntry(_current_entry);
	  if (data.size()>0) {
	    data_batch.emplace_back( std::move(data) );
	  }
	}
	catch (...) {
	  _current_entry = 0;
	}
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
    _tree->SetBranchAddress("particleid",&_in_pparticle_id);
    _tree->SetBranchAddress("ancestorid",&_in_pancestor_id);            
    _tree->SetBranchAddress("q_u",&_in_pq_u);
    _tree->SetBranchAddress("q_v",&_in_pq_v);
    _tree->SetBranchAddress("q_y",&_in_pq_y);
    _tree->SetBranchAddress("triplet_idx_v",&_in_ptriplet_idx_v);
    _tree->SetBranchAddress("subclusterid",&_in_psubcluster_id);
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
      voxel.truth_realmatch = 0;
      voxel.truth_instance_index = (*_in_pinstance_id)[i];
      voxel.truth_pid = (*_in_pparticle_id)[i];
      voxel.truth_ancestor_index = (*_in_pancestor_id)[i];
      voxel.tripletidx_v = (*_in_ptriplet_idx_v)[i];
      voxel.subclusterid = (*_in_psubcluster_id)[i];      
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
                                                larflow::prep::PrepMatchTriplets& triplet_maker,
                                                Prep3DSpatialEmbed::VoxelDataList_t& voxel_v )
  {

    larcv::EventImage2D* ev_adc_v
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _adc_image_treename );
    auto const& adc_v = ev_adc_v->as_vector();

    larcv::EventImage2D* ev_instance_v
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "instance" );
    auto const& instance_img_v = ev_instance_v->as_vector();
    
    
    // loop over voxels and take votes as to instance and class of pixels
    // also if true
    size_t nvoxels = voxel_v.size();
    
    for ( size_t ivoxel=0; ivoxel<nvoxels; ivoxel++ ) {
      auto& voxeldata = voxel_v[ivoxel];

      auto const& hitidx_v = voxeldata.tripletidx_v;
      int istrue = 0;
      std::map<int,float> instance_votes;
      std::map<int,int> instance2segment;
      
      for ( auto const& idx : hitidx_v ) {
        
        if ( triplet_maker._truth_v[idx]==1 )
          istrue = 1;

        int segid = triplet_maker._pdg_v[idx];
        int iid   = triplet_maker._instance_id_v[idx];
        std::vector<int> imgcoord = triplet_maker.get_triplet_imgcoord_rowcol( idx );
        float totalq = 0.;
        for (int p=0; p<3; p++ ) {
          totalq += adc_v[p].pixel(imgcoord[3],imgcoord[p],__FILE__,__LINE__);
        }
        if ( instance_votes.find(iid)==instance_votes.end() )
          instance_votes[iid] = 0.;       
        instance_votes[iid]+=totalq;

        if ( instance2segment.find(iid)!=instance2segment.end() ) {
          if ( instance2segment[iid]!=segid )  {
            throw std::runtime_error( "instance id associated to two different segment ids!" );            
          }
        }
        instance2segment[iid] = segid;
        
      }

      // set label by vote
      int maxiid = -1;
      float maxq_iid = 0.;
      for ( auto it=instance_votes.begin(); it!=instance_votes.end(); it++ ) {
        if ( maxq_iid<it->second ) {
          maxq_iid = it->second;
          maxiid = it->first;
        }
      }

      // get an instance id
      voxeldata.truth_instance_index = maxiid;
      voxeldata.truth_realmatch = istrue;
      voxeldata.truth_pid = instance2segment[maxiid];
      voxeldata.truth_ancestor_index = 0;
      
    }//end of voxel loop


    // --- data conditioning -----------------------------
    // we cull instances with small numbers of voxels
    // we dont want the network getting penalized for small little depositions
    
    std::map<int,int> shower_instance_count;
    std::map<int,int> track_instance_count;
    for ( auto& voxel : voxel_v ) {
      if ( voxel.truth_pid==3 || voxel.truth_pid==4 ) {
        // electron or gamma
        if ( voxel.truth_instance_index>0
             && shower_instance_count.find( voxel.truth_instance_index )==shower_instance_count.end() ) {
          shower_instance_count[voxel.truth_instance_index] = 0;
        }
        shower_instance_count[voxel.truth_instance_index] += 1;
      }
      else {
        // track particles
        if ( voxel.truth_instance_index>0
             && track_instance_count.find( voxel.truth_instance_index )==track_instance_count.end() ) {
          track_instance_count[voxel.truth_instance_index] = 0;
        }
        track_instance_count[voxel.truth_instance_index] += 1;        
      }
    }

    // getting sparse voxel clusters where the showerlikelihood builder does not have hits
    // that go into voxels with shower spacepoints that have unrecognized ids.
    // use shower-likelihood builder elements to absorb those?
    struct IDCount_t {
      int idx;
      int ncounts;
      int shape;
      int operator<( IDCount_t& rhs ) {
        if ( ncounts<rhs.ncounts ) return true;
        return false;
      };
    };

    std::vector< IDCount_t > count_v;
    for ( auto it=shower_instance_count.begin(); it!=shower_instance_count.end(); it++ ) {
      IDCount_t idcnt;
      idcnt.idx = it->first;
      idcnt.ncounts = it->second;
      idcnt.shape = 0;
      count_v.push_back( idcnt );
    }

    int max_nhits_track = 0;
    for ( auto it=track_instance_count.begin(); it!=track_instance_count.end(); it++ ) {
      IDCount_t idcnt;
      idcnt.idx = it->first;
      idcnt.ncounts = it->second;
      idcnt.shape = 1;
      if (max_nhits_track<idcnt.ncounts)
        max_nhits_track = idcnt.ncounts;
      count_v.push_back( idcnt );
    }
    std::sort( count_v.begin(), count_v.end() );
    std::cout << "///////// INSTANCE COUNTS //////////////////////////" << std::endl;
    for ( auto& idcnt : count_v ) {
      std::cout << "[" << idcnt.idx << "] shape=" << idcnt.shape << " counts=" << idcnt.ncounts << std::endl;
    }
    std::cout << "////////////////////////////////////////////////////" << std::endl;
    

    // zero out shower instance ids
    int nzerod = 0;
    for ( auto it=shower_instance_count.begin(); it!=shower_instance_count.end(); it++ ) {
      if ( it->second<10 ) {
        // set to background instance id, 0
        std::cout << "zero out small cluster shower index [" << it->first << "] counts=" << it->second << std::endl;        
        for ( auto& voxel : voxel_v ) {
          if ( voxel.truth_instance_index==it->first ) {            
            voxel.truth_instance_index = 0;
          }
        }
        nzerod++;
      }
    }
    std::cout << "num zero'd: " << nzerod << std::endl;

    // assign track ids
    // set threshold at number of hits of largest track
    // as long as threshold is above 10
    float threshold = 25.0;
    if ( max_nhits_track<(int)threshold+1 )
      threshold = (float)max_nhits_track-1;
    if ( threshold<10.0 )
      threshold = 10.0;
    std::cout << "max_nhits_track=" << max_nhits_track << " set hit threshold @ " << threshold << std::endl;
    
    _reassignSmallTrackClusters( voxel_v, instance_img_v, track_instance_count, threshold );
    

    // reassign id: we want instance ids from 1 to N instances, sequential
    std::map<int,int> reassign_index;
    std::vector<int>  newindex_v;
    for ( auto& voxel : voxel_v ) {
      if ( voxel.truth_instance_index!=0 ) {
        if ( reassign_index.find( voxel.truth_instance_index )==reassign_index.end() ) {
          // new index
          newindex_v.push_back( voxel.truth_instance_index );            
          reassign_index[voxel.truth_instance_index] = (int)newindex_v.size();
          std::cout << "re-assign [" << voxel.truth_instance_index << "] -> [" << reassign_index[voxel.truth_instance_index] << "]" << std::endl;
        }
        int newindex = reassign_index[voxel.truth_instance_index];
        voxel.truth_instance_index = newindex;
      }
    }
     
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
                                                      const std::vector<int>& nvoxels_dim,
                                                      const int nsigma,
                                                      const float seed_sigma ) const
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
    std::map<int,int> voxel_class;
    for (int i=0; i<max_instance_id+1; i++) {
      centroid[i] = std::vector<float>(3,0);
    }

    for (auto const& voxel : voxeldata ) {
      if ( voxel.truth_instance_index>=0 ) {
        for (int i=0; i<3; i++) {
          centroid[voxel.truth_instance_index][i] += float(voxel.voxel_index[i]);
        }
        num_instance_voxels[voxel.truth_instance_index]++;
        
        if ( voxel.truth_instance_index>0 && voxel.truth_pid>0 && voxel_class.find(voxel.truth_instance_index)==voxel_class.end() )
          voxel_class[voxel.truth_instance_index] = voxel.truth_pid;
      }
    }

    for (int id=0; id<max_instance_id+1; id++) {
      if ( num_instance_voxels[id]>0 ) {
        std::vector<float> fcoord(3,0);
        std::cout << "centroid[" << id << "] nvoxels=" << num_instance_voxels[id];
        std::cout << "  class[" << voxel_class[id] << "]: (";
        for (int j=0; j<3; j++) {
          centroid[id][j] /= float(num_instance_voxels[id]);
          std::cout << centroid[id][j];
          if (j+1<3 ) std::cout << ",";
          fcoord[j] = centroid[id][j]/float(nvoxels_dim[j]);
        }
        std::cout << "normalized-centroid[" << id << "] ( ";
        for (int j=0; j<3; j++) std::cout << fcoord[j] << "  ";
        std::cout << ")" << std::endl;
      }
    }
    

    // coord tensor
    npy_intp coord_t_dim[] = { (long int)nvoxels, 4 };
    PyArrayObject* coord_t = (PyArrayObject*)PyArray_SimpleNew( 2, coord_t_dim, NPY_LONG );
    PyObject *coord_t_key = Py_BuildValue("s", "coord_t");        
    
    // embed tensor
    npy_intp embed_t_dim[] = { (long int)nvoxels, 3+nsigma };
    PyArrayObject* embed_t = (PyArrayObject*)PyArray_SimpleNew( 2, embed_t_dim, NPY_FLOAT );

    // seed tensor: 7 classes
    npy_intp seed_t_dim[] = { (long int)nvoxels, 7 };
    PyArrayObject* seed_t = (PyArrayObject*)PyArray_SimpleNew( 2, seed_t_dim, NPY_FLOAT );

    std::vector<float> max_dist_from_centroid(num_instance_voxels.size(),0);

    // loop over voxel
    std::map<int,int> instance_voxel_closest_2_centroid;
    std::map<int,float> instance_mindist2_centroid;
    for (size_t i=0; i<nvoxels; i++ ) {
      auto const& voxel = voxeldata[i];

      for (size_t j=0; j<3; j++)
        *((long*)PyArray_GETPTR2(coord_t,i,j)) = voxel.voxel_index[j];
      *((long*)PyArray_GETPTR2(coord_t,i,3))   = 0;
      
      // set normalized shift in embed tensor
      if ( voxel.truth_instance_index>0 ) {
        // in instance
        int iid = voxel.truth_instance_index;
	int pid = voxel.truth_pid;

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

        if ( instance_mindist2_centroid.find( voxel.truth_instance_index )==instance_mindist2_centroid.end() ) {
          instance_mindist2_centroid[voxel.truth_instance_index ] = 1e6;
          instance_voxel_closest_2_centroid[ voxel.truth_instance_index ] = i;
        }

        if ( instance_mindist2_centroid[voxel.truth_instance_index ] > voxel_diff ) {
          instance_mindist2_centroid[voxel.truth_instance_index ] = voxel_diff;
          instance_voxel_closest_2_centroid[ voxel.truth_instance_index ] = i;
        }
        
        // set sigma in embed tensor (sharp)
        for (int isig=0; isig<nsigma; isig++)
          *((float*)PyArray_GETPTR2(embed_t,i,3+isig)) = 10.0;

        // set seed map, first zero out        
	for (int c=0; c<7; c++)
	  *((float*)PyArray_GETPTR2(seed_t,i,c)) = 0.0;
        // then set proper class entry with target seed value
	if ( pid>0 ) {
          //std::cout << "[" << i << "," << pid-1 << "] " << exp(-0.5*voxel_diff*voxel_diff/(seed_sigma*seed_sigma) ) << " " << voxel_diff << std::endl;
	  //*((float*)PyArray_GETPTR2(seed_t,i,pid-1)) = exp(-0.5*voxel_diff*voxel_diff/(seed_sigma*seed_sigma) );
          *((float*)PyArray_GETPTR2(seed_t,i,pid-1)) = 1.0;
        }
      }
      else {
        // not part of any true instance
        for (size_t j=0; j<3; j++) {        
          *((float*)PyArray_GETPTR2(embed_t,i,j)) = 0.;
        }
        *((float*)PyArray_GETPTR2(embed_t,i,3)) = 0.01;
	for (int c=0; c<7; c++)
	  *((float*)PyArray_GETPTR2(seed_t,i,c)) = 0.0;
      }
      
    }

    for (size_t iid=0; iid<max_dist_from_centroid.size(); iid++)
      LARCV_NORMAL() << "instance[" << iid << "] max voxel diff: " << max_dist_from_centroid[iid] << std::endl;

    for (auto it=instance_voxel_closest_2_centroid.begin(); it!=instance_voxel_closest_2_centroid.end(); it++) {
      LARCV_NORMAL() << "instance[" << it->first << "] class[" << voxel_class[it->first] << "] "
                     << "instance_mindist2_centroid=" << instance_mindist2_centroid[it->first] << " voxelid=" << it->second << std::endl;
      if ( it->first>0 ) {
        *((float*)PyArray_GETPTR2(seed_t,(int)it->second, (int)voxel_class[it->first]-1) ) = 1.0;
      }
    }
    
    PyObject *embed_t_key = Py_BuildValue("s", "embed_t");    
    PyObject *seed_t_key = Py_BuildValue("s", "seed_t");


    PyObject *d = PyDict_New();
    PyDict_SetItem(d, embed_t_key,    (PyObject*)embed_t);
    PyDict_SetItem(d, seed_t_key,     (PyObject*)seed_t);
    PyDict_SetItem(d, coord_t_key,    (PyObject*)coord_t);    
    
    Py_DECREF(embed_t_key);
    Py_DECREF(seed_t_key);
    Py_DECREF(coord_t_key);    
    
    return d;
    
    
  }

  /**
   * @brief use neighboring pixels to reassign track clusters with small number of voxels
   *
   * These are often secondary protons or pions created by proton/pion reinteractions.
   * Small proton/pion will usually have some larger track with a proper instance nearby
   *
   */
  void Prep3DSpatialEmbed::_reassignSmallTrackClusters( Prep3DSpatialEmbed::VoxelDataList_t& voxel_v,
                                                        const std::vector< larcv::Image2D >& instanceimg_v,
                                                        std::map<int,int>& track_instance_count,
                                                        const float threshold )
  {

    const int dvoxel = 4;
    const int nrows = instanceimg_v.front().meta().rows();
    const int ncols = instanceimg_v.front().meta().cols();    
    auto const& meta = instanceimg_v.front().meta(); // assuming metas the same for all planes
    
    struct ReplacementTally_t {
      int orig_instanceid;
      std::set<int> replacement_ids;
      ReplacementTally_t()
        : orig_instanceid(0) {};
      ReplacementTally_t(int id)
        : orig_instanceid(id) {};
    };
    std::map<int,ReplacementTally_t> replacement_tally_v;
    
    for ( auto& voxel : voxel_v ) {
      auto it=track_instance_count.find( voxel.truth_instance_index );
      if ( it==track_instance_count.end() )
        continue; /// unexpected (should throw an error)
      
      if ( it->second>threshold )
        continue; // above threshold for reassignment
      
      auto it_t = replacement_tally_v.find( voxel.truth_instance_index );
      if ( it_t==replacement_tally_v.end() )
        replacement_tally_v[voxel.truth_instance_index] = ReplacementTally_t(voxel.truth_instance_index);

      auto& tally = replacement_tally_v[voxel.truth_instance_index];
      
      for (int dr=-dvoxel; dr<=dvoxel; dr++) {
        int row = (int)meta.row( voxel.imgcoord_v[3] ); // tick to row
        if (row<=0 || row>=nrows ) continue;
        
        for (int p=0; p<3; p++) {
          for (int dc=-dvoxel; dc<=dvoxel; dc++) {
            int col = voxel.imgcoord_v[p]+dc;
            if (col<0 || col>=ncols ) continue;

            int iid = instanceimg_v[p].pixel(row,col,__FILE__,__LINE__);

            // ignore own instanceid
            if ( iid==voxel.truth_instance_index || iid<0)
              continue;

            if ( track_instance_count.find(iid)!=track_instance_count.end() )
              tally.replacement_ids.insert( iid );
            else {
              //std::cout << "trying to replace id=" << voxel.truth_instance_index << " but id=" << iid << " not in track count dict" << std::endl;
            }
          }//end of col loop
        }//end of plane loop
      }//end of row loop
    }
    
    // choose the replacement id
    // we pick the id associated to the largest track cluster
    std::map<int,int> replace_trackid;
    for ( auto it=track_instance_count.begin(); it!=track_instance_count.end(); it++ ) {
      if ( it->second>threshold)
        continue;
      
      auto& tally = replacement_tally_v[it->first];
      int max_nhits = 0;
      int max_replacement_id = 0;
      for (auto& id : tally.replacement_ids ) {
        if ( track_instance_count[id]>max_nhits ) {
          max_nhits = track_instance_count[id];
          max_replacement_id = id;
        }
      }
      replace_trackid[it->first] = max_replacement_id;
      if ( max_replacement_id>0 ) {
        std::cout << "successfully replace trackid=" << it->first
                  << "( w/ " << it->second << " counts)"
                  << " with " << max_replacement_id << " (w/ " << max_nhits << " counts)"
                  << std::endl;
        track_instance_count[max_replacement_id] += it->second;
        it->second = 0;
        // need to propagate this reassignment to past reassignments
        for ( auto it_r=replace_trackid.begin(); it_r!=replace_trackid.end(); it_r++ ) {
          if ( it_r->second==it->first )
            it_r->second = max_replacement_id;
        }
      }
    }
    
    // execute the replacement
    for ( auto& voxel : voxel_v ) {
      if ( replace_trackid.find( voxel.truth_instance_index )!=replace_trackid.end() ) {
        voxel.truth_instance_index = replace_trackid[voxel.truth_instance_index];
      }
    }
    
  }

  /**
   * @brief use dbscan to form subclusters
   *
   * we use dbscan to form subclusters of voxels.
   * the goal of this is to then use a network to split the subcluster
   * into components from different particles.
   *
   * limit clustering algorithms to this case, so that algorithm not trying to cluster
   * the shower fragments together, which might have to much spatial variation.
   *
   * instead if we can split a cluster into correct particle pieces, we can use this
   * to seed the final particle reconstruction.
   *
   * we use the truth flag to limit the voxels considered.
   * eventually, we would want to replace this with the output of larmatch.
   *
   */
  void Prep3DSpatialEmbed::_generate_subcluster_labels( Prep3DSpatialEmbed::VoxelDataList_t& data,
                                                        larlite::storage_manager& ioll,
                                                        bool use_only_true_voxels )
  {
    // params (need to make class attribute and provide set functions
    const float maxdist = 1.5;
    const int minsize = 5;
    const int maxkd = 50;      
    const int _kcheckradius = 3.0;
    
    // collect points
    std::vector< std::vector<float> > pos_v;
    std::vector<int> point2voxelidx_v; // map from position in pos_v to original index in VoxelDataList_t data
    pos_v.reserve( data.size() );
    point2voxelidx_v.reserve( data.size() );
    
    for ( size_t ivoxel=0; ivoxel<data.size(); ivoxel++ ) {
      auto & voxel = data[ivoxel];
      voxel.subclusterid = 0; // initialize to zero
      if ( use_only_true_voxels && voxel.truth_realmatch!=1 )
        continue;

      pos_v.push_back( voxel.ave_xyz_v );
      point2voxelidx_v.push_back( ivoxel );
    }

    std::vector< larflow::reco::cluster_t > cluster_v;
    larflow::reco::cluster_sdbscan_spacepoints( pos_v, cluster_v, maxdist, minsize, maxkd );
    LARCV_INFO() << "dbscan produced " << cluster_v.size() << " clusters" << std::endl;
    for ( auto& cluster : cluster_v ) {
      larflow::reco::cluster_bbox( cluster );
    }
    
    // now we find the vertex
    std::vector<float> vtxpos_w_tick = ublarcvapp::mctools::NeutrinoVertex::getPos3DwSCE( ioll, _triplet_truth_fixer.getSCE() );
    float cluster_min_dist2 = 1.0e9;
    int cluster_min_cidx = -1;
    
    for (int cidx=0; cidx<(int)cluster_v.size(); cidx++) {
      auto& cluster = cluster_v[cidx];
      float dist2box = larflow::reco::cluster_dist_to_bbox( cluster, vtxpos_w_tick );
      if ( dist2box<=_kcheckradius ) {
        // sigh, have to check
        float mindist = 1.0e9;
        for ( auto& hitpos : cluster.points_v ) {
          float testdist = 0;
          for ( int i=0; i<3; i++)
            testdist += ( hitpos[i]-vtxpos_w_tick[i] )*( hitpos[i]-vtxpos_w_tick[i] );
          if (testdist<mindist )
            mindist = testdist;
        }
        if ( cluster_min_dist2>mindist ) {
          cluster_min_cidx = cidx;
          cluster_min_dist2 = mindist;
        }
      }
    }
      
    // ok, we find the cluster with the vertex in it.
    // we will label this cluster with the 1 label.
    // then we label the rest by size
    struct ClusterRank_t {
      int cidx;
      int count;
      bool operator<( const ClusterRank_t& rhs ) {
        if ( count>rhs.count )
          return true;
        return false;
      };
      ClusterRank_t()
        : cidx(-1), count(-1)
      {};
    };
    std::vector<ClusterRank_t> rank_v(cluster_v.size());
    for (int cidx=0; cidx<(int)cluster_v.size(); cidx++) {
      rank_v[cidx].cidx = cidx;
      rank_v[cidx].count = (int)cluster_v[cidx].points_v.size();
    }
    std::sort( rank_v.begin(), rank_v.end() );

    // now we can add the labels finally
    int currentlabel = 1;
    bool found_vtx_cluster = false;
    float checkr2 = _kcheckradius*_kcheckradius;
    if ( cluster_min_cidx>=0 && cluster_min_dist2<checkr2 ) {
      auto& cluster = cluster_v[cluster_min_cidx];
      for ( auto& hitidx : cluster.hitidx_v ) {
        int orig_idx = point2voxelidx_v[hitidx];
        data[orig_idx].subclusterid = currentlabel;
      }
      found_vtx_cluster = true;
      LARCV_INFO() << "Found vertex cluster. closest cluster index=" << cluster_min_cidx << " mindist=" << sqrt(cluster_min_dist2) << std::endl;
      currentlabel++;
    }
    
    for (auto&  crank : rank_v ) {
      if ( found_vtx_cluster && crank.cidx==cluster_min_cidx )
        continue;
      auto& cluster = cluster_v[crank.cidx];
      for ( auto& hitidx : cluster.hitidx_v ) {
        int orig_idx = point2voxelidx_v[hitidx];
        data[orig_idx].subclusterid = currentlabel;
      }
      currentlabel++;
    }

    // done!
  }
  
}
}
