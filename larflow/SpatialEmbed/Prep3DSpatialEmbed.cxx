#include "Prep3DSpatialEmbed.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "DataFormat/larflow3dhit.h"
#include "larcv/core/DataFormat/EventImage2D.h"

namespace larflow {
namespace spatialembed {

  /**
   * static variable to track if numpy environment has been setup
   */
  bool Prep3DSpatialEmbed::_setup_numpy = false;
  
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
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );

    VoxelDataList_t data = process_larmatch_hits( *ev_lfhit_v, ev_adc_v->as_vector(), 0.5 );
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
            if ( c<=0 || c>=(int)meta.cols()) continue;
            float q = adc_v[p].pixel(r,c);
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
        for (int p=0; p<3; p++)
          data.feature_v[p] = lm_score*q_v[p];
        data.npts = 1;
        data.totw = lm_score;
        voxel_v.emplace_back( std::move(data) );
        int idx = (int)voxel_v.size()-1;
        voxcoord_2_voxel_v[ voxid ] = idx;
      }
      else {
        // append to voxel data
        auto& data = voxel_v[it->second];
        for (int p=0; p<3; p++)
          data.feature_v[p] += lm_score*q_v[p];
        data.npts++;
        data.totw += lm_score;
      }
      
    }//end of loop over larmatch hits

    LARCV_NORMAL() << "Number of voxels created, " << voxel_v.size() << ", "
                   << "from " << nhits_used << " used hits "
                   << "(total " << ev_lfhit_v.size() << ")"
                   << std::endl;
      
    return voxel_v;
  }
  

  void Prep3DSpatialEmbed::bindVariablesToTree( TTree* atree )
  {
    _tree = atree;
    _tree->Branch( "vidrow", &vid_row );
    _tree->Branch( "vidcol", &vid_col );
    _tree->Branch( "viddepth", &vid_depth );
    _tree->Branch( "q_u", &q_u );
    _tree->Branch( "q_v", &q_v );
    _tree->Branch( "q_y", &q_y );
  }

  void Prep3DSpatialEmbed::fillTree( const Prep3DSpatialEmbed::VoxelDataList_t& data )
  {

    vid_row.resize(data.size());
    vid_col.resize(data.size());
    vid_depth.resize(data.size());
    q_u.resize(data.size());
    q_v.resize(data.size());
    q_y.resize(data.size());

    for (size_t i=0; i<data.size(); i++) {
      auto const& d = data[i];
      vid_row[i]   = d.voxel_index[0];
      vid_col[i]   = d.voxel_index[1];
      vid_depth[i] = d.voxel_index[2];
      if ( d.totw>0) {
        q_u[i] = d.feature_v[0]/d.totw;
        q_v[i] = d.feature_v[1]/d.totw;
        q_y[i] = d.feature_v[2]/d.totw;
      }
      else {
        q_u[i] = 0.;
        q_v[i] = 0.;
        q_y[i] = 0.;
      }
    }

  }

  /**
   * @brief return a python dictionary with numpy arrays
   *
   * contents of dictionary:
   * "coord_t":coordinate tensor
   * "feat_t":feature tensor
   */
  PyObject* Prep3DSpatialEmbed::makeTrainingDataDict( const VoxelDataList_t& voxeldata ) const
  {

    if ( !Prep3DSpatialEmbed::_setup_numpy ) {
      import_array1(0);
      Prep3DSpatialEmbed::_setup_numpy = true;
    }

    size_t nvoxels = voxeldata.size();

    // coord tensor
    npy_intp coord_t_dim[] = { (long int)nvoxels, 4 };
    PyArrayObject* coord_t = (PyArrayObject*)PyArray_SimpleNew( 2, coord_t_dim, NPY_LONG );
    for (size_t i=0; i<nvoxels; i++ ) {
      auto const& voxel = voxeldata[i];
      for (size_t j=0; j<3; j++)
        *((long*)PyArray_GETPTR2(coord_t,i,j)) = voxel.voxel_index[j];
      *((long*)PyArray_GETPTR2(coord_t,i,3))   = 0;
    }
    PyObject *coord_t_key = Py_BuildValue("s", "coord_t");    

    // feature tensor
    npy_intp feat_t_dim[] = { (long int)nvoxels, 3 };
    PyArrayObject* feat_t = (PyArrayObject*)PyArray_SimpleNew( 2, feat_t_dim, NPY_FLOAT );
    for (size_t i=0; i<nvoxels; i++ ) {
      auto const& voxel = voxeldata[i];
      if ( voxel.totw>0 ) {
        for (size_t j=0; j<3; j++)
          *((long*)PyArray_GETPTR2(feat_t,i,j)) = voxel.feature_v[j]/voxel.totw;
      }
      else {
        for (size_t j=0; j<3; j++)
          *((long*)PyArray_GETPTR2(feat_t,i,j)) = 0.;
      }
    }
    PyObject *feat_t_key = Py_BuildValue("s", "feat_t");    


    PyObject *d = PyDict_New();
    PyDict_SetItem(d, coord_t_key, (PyObject*)coord_t);
    PyDict_SetItem(d, feat_t_key,  (PyObject*)feat_t); 
    
    Py_DECREF(coord_t_key);
    Py_DECREF(feat_t_key);    
    
    return d;
    
    
  }

  void Prep3DSpatialEmbed::loadTreeBranches( TTree* atree )
  {
    _tree = atree;
    _tree->SetBranchAddress("vidrow",&_in_pvid_row);
    _tree->SetBranchAddress("vidcol",&_in_pvid_col);
    _tree->SetBranchAddress("viddepth",&_in_pvid_depth);
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
    if ( !bytes )
      return data; // return empty container

    // fill container with vector values
    size_t nvoxels = _in_pvid_row->size();
    data.reserve( nvoxels );

    for ( size_t i=0; i<nvoxels; i++ ) {
      VoxelData_t voxel;
      voxel.voxel_index = { (*_in_pvid_row)[i], (*_in_pvid_col)[i], (*_in_pvid_depth)[i] };
      voxel.feature_v   = { (*_in_pq_u)[i], (*_in_pq_v)[i], (*_in_pq_y)[i] };
      voxel.npts = 1;
      voxel.totw = 1.0;
      data.emplace_back( std::move(voxel) );
    }

    return data;
  }

  PyObject* Prep3DSpatialEmbed::getTreeEntryDataAsArray( int entry )
  {
    return makeTrainingDataDict( getTreeEntry(entry) );
  }
  
}
}
