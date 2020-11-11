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

    if ( make_truth_if_available )
      generateTruthLabels( iolcv, ioll, data );
    
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
    for ( auto& voxel : voxel_v ) {
      if ( voxel.totw>0 ) {
        for (int i=0; i<3; i++) {
          voxel.feature_v[i] /= voxel.totw;
          voxel.ave_xyz_v[i] /= voxel.totw;
        }
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
    _tree = atree;
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
    LARCV_INFO() << "Converting data for " << nvoxels << " voxels into numpy arrays" << std::endl;

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
          *((float*)PyArray_GETPTR2(feat_t,i,j)) = voxel.feature_v[j];
      }
      else {
        for (size_t j=0; j<3; j++)
          *((float*)PyArray_GETPTR2(feat_t,i,j)) = 0.;
      }
    }
    PyObject *feat_t_key = Py_BuildValue("s", "feat_t");

    // instance tensor
    npy_intp instance_t_dim[] = { (long int)nvoxels };
    PyArrayObject* instance_t = (PyArrayObject*)PyArray_SimpleNew( 1, instance_t_dim, NPY_LONG );
    for (size_t i=0; i<nvoxels; i++ ) {
      auto const& voxel = voxeldata[i];
      *((long*)PyArray_GETPTR1(instance_t,i)) = voxel.truth_instance_index+1;
    }
    PyObject *instance_t_key = Py_BuildValue("s", "instance_t");
    

    PyObject *d = PyDict_New();
    PyDict_SetItem(d, coord_t_key,    (PyObject*)coord_t);
    PyDict_SetItem(d, feat_t_key,     (PyObject*)feat_t);
    PyDict_SetItem(d, instance_t_key, (PyObject*)instance_t);     
    
    Py_DECREF(coord_t_key);
    Py_DECREF(feat_t_key);
    Py_DECREF(instance_t_key);
    
    return d;
    
    
  }

  void Prep3DSpatialEmbed::loadTreeBranches( TTree* atree )
  {
    _tree = atree;
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
      voxel.truth_realmatch = -1;
      voxel.truth_instance_index = (*_in_pinstance_id)[i]-1;
      data.emplace_back( std::move(voxel) );
    }

    return data;
  }

  PyObject* Prep3DSpatialEmbed::getTreeEntryDataAsArray( int entry )
  {
    return makeTrainingDataDict( getTreeEntry(entry) );
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
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_adc_v->as_vector();
    
    // build particle graph and assign pixels
    ublarcvapp::mctools::MCPixelPGraph mcpg;
    mcpg.buildgraph( iolcv, ioll );
    mcpg.set_adc_treename("wire");
    LARCV_INFO() << "pre-shower builder graph" << std::endl;
    mcpg.printGraph();

    // additional algorithm to fix the shower instances
    larflow::reco::ShowerLikelihoodBuilder mcshowerbuilder;
    mcshowerbuilder.set_wire_tree_name( "wire" );
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

      // get tick,wire coordinates of ave space point position inside voxel
      std::vector<float> imgcoord_v(4,0);
      for (int p=0; p<3; p++) {
        imgcoord_v[p] = larutil::Geometry::GetME()->WireCoordinate( xyz, p );
      }
      imgcoord_v[3] = xyz[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
      
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
            float dtick = fabs(pixtick-imgcoord_v[3]);
            float dwire = fabs(pixwire-imgcoord_v[p]);
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
  
  
  
}
}
