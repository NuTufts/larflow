#include "SimChTripletLabelMaker.h"

#include "larlite/LArUtil/TimeService.h"
#include "larlite/LArUtil/Geometry.h"
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/DataFormat/simch.h"

#include "larcv/core/DataFormat/EventImage2D.h"

#include <highfive/H5Easy.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>

#include "ublarcvapp/MCTools/MCPos2ImageUtils.h"

#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"
#include "larflow/PrepFlowMatchData/TripletTruthFixer.h"
#include "larflow/PrepFlowMatchData/ConvertMatchTripletsToEventTriplets.h"
#include "larflow/RecoUtils/cluster_functions.h" 

namespace larflow {
namespace prep {

  /**
   * @brief Helper function to write compressed HDF5 datasets
   *
   * @tparam T Data type (e.g., std::vector<int>, std::vector<std::vector<float>>)
   * @param file HDF5 file reference
   * @param dataset_path Full path to dataset in HDF5 file
   * @param data Data to write
   * @param compression_level Deflate compression level 0-9 (default: 6)
   * @param chunk_size Number of elements per chunk (default: 10000)
   */
  template<typename T>
  void SimChTripletLabelMaker::dump_compressed(
    HighFive::File& file,
    const std::string& dataset_path,
    const T& data,
    int compression_level,
    size_t chunk_size )
  {

    // HighFive::Chunking requires std::vector<hsize_t>
    std::vector<hsize_t> chunk_dims;
    std::vector<size_t> dims;

    // Create dataset properties with compression
    HighFive::DataSetCreateProps props;
    props.add(HighFive::Deflate(compression_level));

    // Determine chunk dimensions based on data structure
    auto dataspace = HighFive::DataSpace::From(data);
    dims = dataspace.getDimensions();

    // Handle empty data - HDF5 chunk dimensions must be > 0
    if (dims.empty() || dims[0] == 0) {
      LARCV_WARNING() << "Skipping empty dataset: " << dataset_path
                      << " (dims[0]=0)" << std::endl;
      // Write without compression/chunking for empty datasets
      file.createDataSet(dataset_path, data);
      return;
    }

    try {

      if (dims.size() == 1) {
        // 1D data: chunk along the single dimension
        chunk_dims = {static_cast<hsize_t>(std::min(chunk_size, dims[0]))};
      } else if (dims.size() == 2) {
        // 2D data: chunk rows, keep all columns
        chunk_dims = {static_cast<hsize_t>(std::min(chunk_size, dims[0])),
                      static_cast<hsize_t>(dims[1])};
      } else {
        // Higher dimensions: chunk first dimension
        for (auto d : dims) {
          chunk_dims.push_back(static_cast<hsize_t>(d));
        }
        chunk_dims[0] = static_cast<hsize_t>(std::min(chunk_size, dims[0]));
      }

      props.add(HighFive::Chunking(chunk_dims));

      // Create dataset and write data in one call
      // HighFive infers type and dataspace from data, uses our props for chunking/compression
      file.createDataSet(dataset_path, data, props);

    } catch (const HighFive::Exception& e) {
      LARCV_CRITICAL() << "Failed to write compressed dataset '"
                       << dataset_path << "': " << e.what() << std::endl
                       << " ndims=" << dims.size()
                       << std::endl;
          
      throw;
    }
  }

  SimChTripletLabelMaker::SimChTripletLabelMaker()
  : larcv::larcv_base("SimChTripletLabelMaker"),
    _psce(nullptr),
    _hdf_file(nullptr),
    _save_weights_to_hdf(false),
    _save_truth_triplet_info(false),
    _adc_treename("wiremc"),
    _is_mc(true),
    _process_ub_mcc9(false)
  {

    // utility for moving real position to apparent position
    _psce = new larutil::SpaceChargeMicroBooNE(larutil::SpaceChargeMicroBooNE::kMCC9_Forward);

  };

  SimChTripletLabelMaker::~SimChTripletLabelMaker()
  {
    if ( _hdf_file )
      close_hdf_file();

    delete _psce;
    _psce = nullptr;
  }

  void SimChTripletLabelMaker::process( 
    larlite::storage_manager& ioll, 
    larcv::IOManager& iolcv )
  {
    
    // clear algorithms and containers
    _mcpgraph.clear();
    _mcpixelmaker.clear();
    _tripletmaker.clear();
    _mckpmaker.clear();
    _ev_reco_triplets.clear();
    _final_keypoint_list.clear();

    if ( _is_mc ) {
      // if simulated data, use the truth information to make a graph
      //  of most particles tracked by geant4 in the event.
      // the graph encodes the mother-daugher relationships between particles.
      _mcpgraph.buildgraph(ioll);
    }

    make_reco_triplets(iolcv);

    if ( _is_mc ) {
      if ( _process_ub_mcc9 ) {
        // get truth from 2D images and assign them back to the 3D points
        _tripletmaker.process_truth_labels( iolcv, ioll, "wire" );
        // create class that adjusts for deficiencies in old truth data
        larflow::prep::TripletTruthFixer mcc9_truthfixer;
        mcc9_truthfixer.calc_reassignments( _tripletmaker, iolcv, ioll );
        // extract triplet points with truth labels into EventMCPixelLabels class in _mcpixelmaker
        larflow::prep::ConvertMatchTripletsToEventTriplets::convert( _mcpixelmaker, _tripletmaker );
      } else {
        _mcpixelmaker.make_truthlabels_fromsimch( _adc_treename, ioll, iolcv, _mcpgraph, _psce );
      }
    }

    if ( _is_mc ) {
      label_reco_triplets( _mcpixelmaker._pixels_v, _ev_reco_triplets );

      _mckpmaker.set_mcparticle_graph( &_mcpgraph );
      _mckpmaker.set_spacecharge_instance( _psce );
      _mckpmaker.setADCimageTreeName( _adc_treename );
      _mckpmaker.clear();
      //_mckpmaker.set_verbosity(larcv::msg::kINFO);
      _mckpmaker.process( iolcv, ioll, &_mcpixelmaker );

      adjust_keypoints( 
        _mckpmaker.getMCKeypoint(),
        _ev_reco_triplets, 
        _mcpgraph );

      make_keypoint_labels( 3.0, 0.01 );

      make_ssnet_labels( _mcpgraph );

      _shower_fragment_maker.clear();
      // MCC9 path stuffs ADC pixval into edep (see
      // ConvertMatchTripletsToEventTriplets::convert), so the ShowerFragmentOriginMaker
      // edep thresholds need to be in ADC units rather than MeV.
      // Per-point ADC of 0.5 lets through almost everything above zero;
      // per-cluster sum of 5 ADC gates out clusters with no real charge.
      float cluster_edep_thresh = _process_ub_mcc9 ? 5.0  : 1.0;  // sum-edep cut
      int   cluster_size_thresh = 5;
      float point_edep_thresh   = _process_ub_mcc9 ? 0.5  : 0.5;  // per-point cut
      _shower_fragment_maker.build_shower_fragments(
        _mckpmaker.getMCKeypoint(),
        _mcpgraph,
        _ev_reco_triplets,
        cluster_edep_thresh, cluster_size_thresh, point_edep_thresh );
    }

  }

  void SimChTripletLabelMaker::make_reco_triplets( 
      larcv::IOManager& iolcv )
  {

    LARCV_INFO() << "start" << std::endl;
    _ev_reco_triplets.clear();
    _tripletmaker.clear();

    std::string wireplane_tree_name = ( _is_mc ) ? _adc_treename : "wire";

    // make reco triplets
    _tripletmaker.process( iolcv, wireplane_tree_name, wireplane_tree_name, 10.0, true );

    // copy over triplets
    larcv::EventImage2D* ev_img 
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,wireplane_tree_name);

    auto const& img_v = ev_img->as_vector();
    int nplanes = (int)img_v.size();
    auto const& meta0 = img_v.at(0).meta();

    for (size_t itrip=0; itrip<_tripletmaker._triplet_v.size(); itrip++ ) {

        TripletLabels_t trip;

        trip.index = (long)_ev_reco_triplets._triplets_v.size();

        int row = -1;
        for (int ip=0; ip<nplanes; ip++) {
            auto const& imgpix = _tripletmaker._sparseimg_vv.at(ip).at( _tripletmaker._triplet_v[itrip][ip] );
            trip.imgcoord[ip] = imgpix.col;
            if ( ip==0 )
                row = imgpix.row;
        }
        int tick = meta0.pos_y( row );
        trip.imgcoord[3] = row;
        trip.imgcoord[4] = tick;

        // std::cout << "(" << itrip << ") "
        //           << trip.imgcoord[0] << " "
        //           << trip.imgcoord[1] << " "
        //           << trip.imgcoord[2] << " "
        //           << trip.imgcoord[3] << " "
        //           << trip.imgcoord[4] << " "
        //           << std::endl;

        if ( tick < (int)meta0.min_y() || tick>(int)meta0.max_y() ) 
          continue;

        std::array<int,4> imgcoord;
        imgcoord[0] =  trip.imgcoord[0];
        imgcoord[1] =  trip.imgcoord[1];
        imgcoord[2] =  trip.imgcoord[2];
        imgcoord[3] =  trip.imgcoord[3];

        trip.pos[0] = 0.0;
        trip.pos[1] = 0.0;
        trip.pos[2] = 0.0;
        trip.pos_reco[0] = _tripletmaker._pos_v[itrip][0];
        trip.pos_reco[1] = _tripletmaker._pos_v[itrip][1];
        trip.pos_reco[2] = _tripletmaker._pos_v[itrip][2];  

        trip.edep = std::array<double,3>{0.0,0.0,0.0};
        for (int ip=0; ip<nplanes; ip++)
            trip.pixval[ip] = img_v.at(ip).pixel(row,imgcoord[ip]);

        _ev_reco_triplets._imgcoord_to_tripindex[imgcoord] = trip.index;
        _ev_reco_triplets._triplets_v.push_back( trip );

        

    }

    LARCV_INFO() << "Made triplets from image. Num image triplets:" 
                 << _ev_reco_triplets._triplets_v.size() << std::endl;


  }

  void SimChTripletLabelMaker::label_reco_triplets(
      ublarcvapp::mctools::EventMCPixelLabels& truth_triplets,
      larflow::prep::EventTriplets_t& reco_triplets
  )
  {
    // we have two methods to label points
    // (1) matching the imgcoord index
    // (2) matching by distances

    LARCV_INFO() << "start" << std::endl;

    size_t num_reco_labeled = 0;

    for ( auto it=reco_triplets._imgcoord_to_tripindex.begin(); 
          it!=reco_triplets._imgcoord_to_tripindex.end(); it++ ) 
    {

        auto& index = it->first;

        // look for index in the truth triplets

        auto it_truth = truth_triplets._imgcoord_to_tripindex.find( index );

        if ( it_truth == truth_triplets._imgcoord_to_tripindex.end() )
          continue;

        //std::cout << "reco_index=" << it->second << "  truth_index=" << it_truth->second << std::endl;

        // found match: transfer info
        auto& truth_trip = truth_triplets._triplets_v.at( it_truth->second );
        auto& reco_trip  = reco_triplets._triplets_v.at( it->second );
        transfer_truth_to_reco( truth_trip, reco_trip );
        num_reco_labeled++;

    }

    LARCV_INFO() << "Number of reco triplets label-matched with simch: " << num_reco_labeled << std::endl;


    // match by distance ... (too slow)
    // size_t num_matched_by_dist = 0;
    // for ( auto& reco_trip : _ev_reco_triplets._triplets_v ) {

    //     long ireco = reco_trip.index;

    //     if ( ireco>0 && ireco%10000==0)
    //       LARCV_INFO() << "  process reco[" << ireco << "]" << std::endl;

    //     std::vector<float> pos(3,0);
    //     for (int i=0; i<3; i++)
    //         pos[i] = reco_trip.pos_reco[i];

    //     float mindist = 1e9;

    //     long truth_index = -1;

    //     for ( auto& truth_trip : _ev_triplets._triplets_v ) {
    //       std::vector<float> truth_pos_sce(3,0);
    //       for (int i=0; i<3; i++)
    //         truth_pos_sce[i] = truth_trip.pos_reco[i];
    //       float dist = 0.;
    //       for (int i=0; i<3; i++)
    //         dist += (truth_pos_sce[i]-pos[i])*(truth_pos_sce[i]-pos[i]);
    //       if (dist<mindist) {
    //         truth_index = truth_trip.index;
    //       }
    //     }

    //     if ( mindist<0.5 ) {
    //       auto& truth_trip = _ev_triplets._triplets_v.at( truth_index );
    //       transfer_truth_to_reco( truth_trip, reco_trip );
    //       num_matched_by_dist++;
    //     }

    // }
    // LARCV_INFO() << "Number of reco triplets label-matched by dist to simch: " << num_matched_by_dist << std::endl;
    
  }

  void SimChTripletLabelMaker::transfer_truth_to_reco( 
      ublarcvapp::mctools::MCPixelLabels& truth_trip, 
      TripletLabels_t& reco_trip )
  {

    for (size_t i=0; i<3; i++) {
      reco_trip.pos[i]   = truth_trip.pos[i];
      reco_trip.edep[i]  = truth_trip.edep[i];
    }
    reco_trip.trackids = truth_trip.trackids;
    reco_trip.aids     = truth_trip.aids;
    reco_trip.pids     = truth_trip.pids;
    reco_trip.origin   = truth_trip.origin;
    reco_trip.hasmatch = 1;

  }

  /**
   * @brief move keypoints to near-by reconstructable spacepoints
   * 
   * This is to prevent keypoint labels pointing into empty space.
   * 
   */
  void SimChTripletLabelMaker::adjust_keypoints( 
    const std::vector< larflow::prep::MCKeypoint >& mckeypoints,
    larflow::prep::EventTriplets_t& labeled_reco_triplets,
    ublarcvapp::mctools::MCParticleGraph& mcpg )
  {

    float edep_point_threshold   = 0.01;
    float edep_cluster_threshold = 1.0;
    _final_keypoint_list.clear();

    // narrow the list of triplets to ones with truth-matches
    std::vector< TripletLabels_t* > _true_triplets_v;
    _true_triplets_v.reserve( labeled_reco_triplets._triplets_v.size() );
    for ( auto& triplet : labeled_reco_triplets._triplets_v ) {
        if ( triplet.hasmatch==1 )
          _true_triplets_v.push_back( &triplet );
    }

    for ( auto const& mckp : _mckpmaker.getMCKeypoint() ) {

      // for each keypoint
      //   1. collect true-spacepoints near the keypoint
      //   2. cluster the spacepoints
      //   3. use the momentum to define rough time-axis
      //   4. pick the nearest pt within a qualifying cluster
      std::vector<float> kppos = mckp.keypt_appear;

      // we just pass nu keypoints
      if ( mckp.kptype==larflow::prep::MCKeypoint::kNuVertex ) {
        _final_keypoint_list.push_back( mckp );
        continue;
      }

      std::vector< std::vector<float> > points_v;
      std::vector< std::vector<float> > edep_vv;
      for ( auto& ptriplet : _true_triplets_v ) {
        auto it_tid = ptriplet->trackids.find( mckp.trackid );
        if ( it_tid==ptriplet->trackids.end() )
          continue;

        if ( mckp.kptype==larflow::prep::MCKeypoint::kTrackStart
          || mckp.kptype==larflow::prep::MCKeypoint::kTrackEnd) {

          // for track particles, limit distance from keypoint
          float dist = 0;
          for (int i=0; i<3; i++) {
            dist += (kppos[i]-ptriplet->pos_reco[i])*(kppos[i]-ptriplet->pos_reco[i]);
          }
          dist = sqrt(dist);
          if (dist>50.0) {
            continue;
          }
        }

        std::vector<float> trip_pos(3,0);
        std::vector<float> trip_edep(3,0);
        int nabove_threshold = 0;
        for (int i=0; i<3; i++) {
          trip_pos[i]  = ptriplet->pos_reco[i];
          trip_edep[i] = ptriplet->edep[i];
          if ( trip_edep[i]>edep_point_threshold)
            nabove_threshold++;
        }

        // fallback for MCC9 data where edep is not available:
        // use pixval (ADC) with a 10.0 threshold instead
        if ( nabove_threshold==0 ) {
          for (int i=0; i<3; i++) {
            if ( ptriplet->pixval[i] > 10.0 )
              nabove_threshold++;
          }
        }

        if ( nabove_threshold>=2 ) {
          points_v.push_back( trip_pos );
          edep_vv.push_back( trip_edep );
        }
      }//end of loop over truth-labeled triplet spacepoints

      // cluster pts
      if ( points_v.size()==0 )
        continue;

      // get momentum dir
      auto pnode = mcpg.findTrackID( mckp.trackid );
      if ( pnode == nullptr )
        continue;

      std::vector<float> mom_dir(3,0);
      std::vector<float> orig_pt(3,0);
      std::vector<float> start_pt = pnode->start;

      float pnorm = 0;
      for (int i=0; i<3; i++) {
        mom_dir[i] = pnode->mom4[1+i];
        pnorm += mom_dir[i]*mom_dir[i];
        if ( mckp.kptype==larflow::prep::MCKeypoint::kTrackEnd ) {
          // reverse direction for track end
          mom_dir[i] *= -1.0;
        }
      }
      pnorm = sqrt(pnorm);
      if ( pnorm>0 ) {
        for (int i=0; i<3; i++)
          mom_dir[i] /= pnorm;
      }
      else {
        continue;
      }
        
      // now we cluster these points using dbscan
      float maxdist = 3.0;
      float minsize = 4;
      int maxkd = 10;
      std::vector< larflow::recoutils::cluster_t > cluster_v;
      larflow::recoutils::cluster_sdbscan_spacepoints( points_v, cluster_v, maxdist, minsize, maxkd);
      int nclusters = cluster_v.size(); // skip the last cluster which are noise points
      
      std::vector<float> most_upstream_pt(3,0);
      std::vector<float> most_upstream_edep(3,0);
      float min_s = 1e9;
      bool found_qualifying_pt = false;

      LARCV_DEBUG() << " keypoint[tid=" << pnode->tid << "] "
        << " num points=" << points_v.size() 
        << " num clusters=" << nclusters
        << std::endl;
      LARCV_DEBUG() << "    mom4=" << mom_dir[0] << ", "
                   << mom_dir[1] << ", "
                   << mom_dir[2]
                   << std::endl;
      
      for ( int icluster=0; icluster<nclusters; icluster++ ){
        std::vector<float> edep_planesum(3,0.0);
        auto const& cluster = cluster_v.at(icluster);
        for (int ihit=0; ihit<(int)cluster.hitidx_v.size(); ihit++) {
          auto hitidx = cluster.hitidx_v.at(ihit);
          auto const& hitedep = edep_vv.at(hitidx);
          for (int i=0; i<3; i++) {
            edep_planesum[i] += hitedep[i];
          }
        }
        int nabove_threshold_planes = 0;
        if ( cluster.hitidx_v.size()>0 ) {
          for (int i=0; i<3; i++) {
            if ( edep_planesum[i]>edep_cluster_threshold) {
              nabove_threshold_planes++;
            }
          }
        }

        LARCV_DEBUG() << "   cluster edep: " 
          <<  edep_planesum[0] << ", "
          <<  edep_planesum[1] << ", "
          <<  edep_planesum[2] << " MeV"
          << " nabove=" << nabove_threshold_planes
          << std::endl;

        if ( nabove_threshold_planes>=2 ) {
          // qualifying cluster, get most upstream position
          std::vector<float> forwardpt(3,0);
          for (int i=0; i<3; i++)
            forwardpt[i] = orig_pt[i] + mom_dir[i]*10.0;

          for ( auto& testpt : cluster.points_v ) {
            float s = larflow::recoutils::pointRayProjection3f( kppos, mom_dir, testpt );
            float r = larflow::recoutils::pointLineDistance3f( kppos, forwardpt, testpt );
            if ( s < min_s && s>0.0 && r/s<0.5) {
              min_s = s;
              most_upstream_pt = testpt;
              found_qualifying_pt = true;
              most_upstream_edep = edep_planesum;
            }
          }
        }

      }

      // make a copy
      larflow::prep::MCKeypoint kpd = mckp;

      if ( found_qualifying_pt ) {
        LARCV_DEBUG() << "Adjust keypoint" << std::endl;
        LARCV_DEBUG() << "  from: (" << kpd.keypt_appear[0] << ", " 
          << kpd.keypt_appear[1] << ", "
          << kpd.keypt_appear[2] << ")" << std::endl;
        LARCV_DEBUG() << "  to: (" << most_upstream_pt[0] << ", "
          << most_upstream_pt[1] << ", "
          << most_upstream_pt[2] << ")"
          << std::endl;
        LARCV_DEBUG() << "  edep: " << most_upstream_edep[0] << ", "
          << most_upstream_edep[1] << ", "
          << most_upstream_edep[2] << " MeV"
          << std::endl;
        kpd.keypt_appear = most_upstream_pt;
      }

      _final_keypoint_list.emplace_back( std::move(kpd) );
    }

  }

  /**
   * @brief make keypoint scores for each triplet point
   */
  void SimChTripletLabelMaker::make_keypoint_labels( 
    float kp_sigma, 
    float score_threshold ) 
  {

    const int nkptypes = larflow::prep::MCKeypoint::kNumKPTypes;

    for (auto& triplet : _ev_reco_triplets._triplets_v ) {
      std::vector< float > min_dist_to_kptype( nkptypes, 1e9 );
      // find closest distance to keypoint of each type
      for (auto& kp : _mckpmaker.getMCKeypoint() ) {
        float dist = 0;
        for (int i=0; i<3; i++){
          float dx = kp.keypt_appear[i]-triplet.pos_reco[i];
          dist += dx*dx;
        }
        dist = sqrt(dist);
        int kptype = (int)kp.kptype;
        if ( dist < min_dist_to_kptype[kptype] ) {
          min_dist_to_kptype[kptype] = dist;
        }
      }//end of keypt loops

      std::vector< float > kp_scores( nkptypes, 0.0 );
      for (int i=0; i<nkptypes; i++) {
        float sig_dist = min_dist_to_kptype[i]/kp_sigma;
        float score = exp( -0.5*sig_dist*sig_dist );
        if ( score < score_threshold )
          score = 0.;
        kp_scores[i] = score;
      }

      triplet.kpdist   = min_dist_to_kptype;
      triplet.kpscores = kp_scores;
    }

  }

  void SimChTripletLabelMaker::make_ssnet_labels(
    ublarcvapp::mctools::MCParticleGraph& mcpg )
  {
    // we reduce a map to labeled triplets only
    std::map< std::array<int,4>, unsigned long > matchedtripletmap;

    for( auto it=_ev_reco_triplets._imgcoord_to_tripindex.begin();
        it!=_ev_reco_triplets._imgcoord_to_tripindex.end();
        it++ ) {
      
      auto& triplet  = _ev_reco_triplets._triplets_v.at( it->second );
      if ( triplet.hasmatch==1 ) {
        matchedtripletmap[ it->first ] = it->second;
      }

    }

    std::map<int,unsigned long> class_counts;

    for ( auto it=_ev_reco_triplets._imgcoord_to_tripindex.begin();
      it!=_ev_reco_triplets._imgcoord_to_tripindex.end();
      it++ ) {

      auto& triplet  = _ev_reco_triplets._triplets_v.at( it->second );
      if ( triplet.hasmatch==0 ) {
        triplet.ssnetlabel = 0; // label as background
      }
      else {
        // we set the pid by the most energetic trackid
        float maxKE = 0.0;
        int pid_maxKE = 0;
        for (auto& tid : triplet.trackids ) {
          auto pnode = mcpg.findTrackID( tid );
          if ( pnode!=nullptr ) {
            float E = pnode->mom4[0];
            float pnorm = 0;
            for (int i=0; i<3; i++) {
              pnorm += pnode->mom4[i+1]*pnode->mom4[i+1];
            }
            float m = sqrt( E*E - pnorm );
            float KE = E-m;
            if ( KE>=maxKE ) {
              maxKE = KE;
              pid_maxKE = pnode->pid;
            }
          }
        }
        triplet.ssnetlabel = get_ssnet_class_label( pid_maxKE );
        get_refined_shower_class_label( mcpg, triplet );
      }
      
      // add to class counts for weight calculation
      auto it_count=class_counts.find( triplet.ssnetlabel );
      if ( it_count==class_counts.end() ) {
        class_counts[ triplet.ssnetlabel ] = 0;
        it_count = class_counts.find( triplet.ssnetlabel );
      }
      it_count->second += 1;
    }

    make_low_energy_deposit_labels( 15 );

    LARCV_INFO() << "Class Counts" << std::endl;
    for (int i=0; i<7; i++)
      LARCV_INFO() << "  ssnet[" << i << "] " << class_counts[i] << std::endl;

    // finished labels and counts
    // now we assign boundary flag and class weights
    unsigned long total = 0;
    for ( auto it_cc=class_counts.begin(); it_cc!=class_counts.end(); it_cc++)
    {
      if ( it_cc->first!=0 ) {
        total += it_cc->second;
      }
    }
    float ftotal = (float)total;

    for ( auto it=_ev_reco_triplets._imgcoord_to_tripindex.begin();
      it!=_ev_reco_triplets._imgcoord_to_tripindex.end();
      it++ ) {

      auto& triplet  = _ev_reco_triplets._triplets_v.at( it->second );
      if ( triplet.ssnetlabel==0 )
        continue;

      // store count weight
      if ( total>0 ) {
        float fcount = class_counts[ triplet.ssnetlabel ];
        if ( fcount>0.0 ) {
          triplet.ssnet_classcount_weight = 1.0/fcount;
        }
        else {
          triplet.ssnet_classcount_weight = 0.;
        }
      }

      int num_diff_pid = 0;

      for (int du=-1; du<=1; du++) {
      for (int dv=-1; dv<=1; dv++) {
      for (int dy=-1; dy<=1; dy++) {
      for (int dr=-1; dr<=1; dr++) {

        if ( du==0 && dv==0 && dy==0 && dr==0)
          continue;

        std::array<int,4> modindex = it->first;
        modindex[0] += du;
        modindex[1] += dv;
        modindex[2] += dy;
        modindex[3] += dr;

        auto it_neighbor = matchedtripletmap.find( modindex );
        if ( it_neighbor==matchedtripletmap.end() ) {
          continue;
        }

        auto& neighbor = _ev_reco_triplets._triplets_v.at( it_neighbor->second );
        if ( neighbor.ssnetlabel!=triplet.ssnetlabel ) {
          num_diff_pid++;
        }

      }
      }
      }
      }

      triplet.ssnetboundary = num_diff_pid;

    }

  }

  /**
   * @brief Refine shower-labeled points based on creation process
   * 
   * For initial ssnet labels, 'electron' and 'photon', we refine their
   * class label into 
   *  - primary electron from nu interaction or cosmic generation
   *  - electron from muon decay (Michel electron), so always secondary
   *  - photon that is primary or from some kind of decay
   *  - delta ray showers (regardless of electron or photon)
   *  - "LED": low energy deposits (usually products from brem photons from showers).
   * 
   * We can get almost all labels from mcparticlegraph object. 
   * For low energy deposits, we should probably cluster true spacepoints and for spatially small
   *   clusters, maybe measured by 1st principal component length, mark them as "LED".
  */
  int SimChTripletLabelMaker::get_refined_shower_class_label( 
    ublarcvapp::mctools::MCParticleGraph& mcpg,
    larflow::prep::TripletLabels_t& triplet )
  {

    // in each triplet, which represents the label for a candidate spacepoint,
    // we may should have one or more trackids associated to it.

    // first we count number of non-shower and shower pdg codes associated to this triplet
    int n_shower = 0;
    int n_nonshower = 0;
    int n_nonlabels = 0;
    for ( auto& pdgcode : triplet.pids ) {
      if ( pdgcode==0 || pdgcode==-1 ) {
        n_nonlabels++;
      }
      else if ( abs(pdgcode)==11 || abs(pdgcode)==22 ) {
        n_shower++;
      }
      else {
        n_nonshower++;
      }
    }

    // if we have one or more non-shower and non-null label, then we do nothing.
    if (n_nonshower>=1 ) {
      return triplet.ssnetlabel;
    }

    // if no shower pdg codes found, also do nothing -- this is not a shower
    if ( n_shower==0 )
      return triplet.ssnetlabel;

    // now we try to infer the creation process of this spacepoint using the mcparticlegraph
    // lets loop over the trackids
    int current_shower_process_class = -1;
    int current_process_pdg = -1;

    // 0: shower
    // 1: michel
    // 2: delta
    for ( auto& trackid : triplet.trackids ) {
      ublarcvapp::mctools::MCPGNode* node = mcpg.findTrackID( trackid );
      if ( node==nullptr ) {
        //LARCV_NORMAL() << "triplet_label=" << triplet.ssnetlabel << " no trackid for " << trackid << std::endl;
        continue;
      }

      ublarcvapp::mctools::MCPGNode* mothernode   = mcpg.findTrackID( node->mtid );
      ublarcvapp::mctools::MCPGNode* ancestornode = mcpg.findTrackID( node->aid );
      std::string process = node->process;

      // we have a node, determine shower process class for this trackid
      int shower_process_class = -1;
      int mother_pdg= -1;
      int ancestor_pdg = -1;
      if ( mothernode )
        mother_pdg = mothernode->pid;
      if ( ancestornode )
        ancestor_pdg = ancestornode->pid;

      // look for michel electrons vs. deltas from muons
      if ( process=="primary" ) {
        shower_process_class = 0; // primary
      }
      else if ( process=="Decay" && node->pid==22 ) {
        shower_process_class = 0; // primary
      }
      else if ( (mothernode && abs(mothernode->pid)==13) || (ancestornode && abs(ancestornode->pid)==13) ) {
        // mother is a muon or ancestor is a muon
        if ( process=="Decay" || process=="muMinusCaptureAtRest")
          shower_process_class = 1;
        else
          shower_process_class = 2;
      }
      else if ( process=="muIoni" || process=="muBrems"  || process=="muPairProd" || process=="eBrem" || process=="muBrem") {
        shower_process_class = 2;
      }
      else {
        // everything else
        shower_process_class = -1; // dont do anything
      }

      // update with label priority
      if ( shower_process_class>=0 && (current_shower_process_class==-1 || shower_process_class<current_shower_process_class) ) {
        current_shower_process_class = shower_process_class;
        current_process_pdg = node->pid;
      }

      // LARCV_NORMAL() << "trackid=" << trackid << "  mother_pdg=" << mother_pdg << " ancestor_pdg=" << ancestor_pdg 
      //                 << " process=" << process << std::endl;
      // LARCV_NORMAL() << "  shower_process_class=" << shower_process_class << " current=" << current_shower_process_class << std::endl;

    }


    //LARCV_NORMAL() << "  final process class=" << current_shower_process_class << std::endl;

    if ( current_shower_process_class==0 ) {
      // do nothing, accept PID-based label
      if ( current_process_pdg==22 )
        triplet.ssnetlabel = 2;
      else
        triplet.ssnetlabel = 1;
    }
    else if ( current_shower_process_class==1 ) {
      // michel electron shower
      triplet.ssnetlabel = 6;
    }
    else if ( current_shower_process_class==2 ) {
      // delta shower
      triplet.ssnetlabel = 7;
    }
    else if ( current_shower_process_class==-1 ) {
      // likely a fragment mc backtracker didn't associate
      triplet.ssnetlabel = 8;
    }
    else {
      triplet.ssnetlabel = -1;
    }
      
    return triplet.ssnetlabel;
  }

  /**
   * @brief return class index for different particle types
   * 
   * The labels we define:
   * @verbatim embed:rst:leading-asterisk
   *  * [0]: background (larcv::kROIUnknown)
   *  * [1]: electron (larcv::kROIEminus)
   *  * [2]: gamma (larcv::kROIGamma)
   *  * [3]: muon (larcv::kROIMuminus)
   *  * [4]: proton (larcv::kROIProton)
   *  * [5]: pion (larcv::kROIPiminus)
   *  * [6]: michel electron
   *  * [7]: delta shower
   *  * [8]: low energy deposit
   *  * [9]: other (the rest of the labels)
   */
  int SimChTripletLabelMaker::get_ssnet_class_label( int pid ) 
  {
    int ssnet_label = 0;
    switch( pid ) {
    case 11:
    case -11:
      ssnet_label = 1; // electron shower
      break;
    case 22:
      ssnet_label = 2; // photon shower
      break;
    case 13:
    case -13:
      ssnet_label = 3; // muon
      break;
    case 2212:
    case 2112:
      // note: edep labeled as neutrons are protons
      // created by neutron interactions
      ssnet_label = 4; // proton
      break;
    case 211:
    case -211:
    case 321:
    case -321:
      ssnet_label = 5; // pion/meson
      break;
    default:
      ssnet_label = 9; // other
      break;
    };
    return ssnet_label;
  }

  /**
   * @brief cluster all true shower deposits and assign small clusters with "low energy deposit" label
   * 
   */
  void SimChTripletLabelMaker::make_low_energy_deposit_labels( int npts_threshold )
  {
    std::vector< std::vector<float> > true_shower_pts;
    std::vector< int > reco_triplet_vector_index;

    true_shower_pts.reserve( _ev_reco_triplets._triplets_v.size() );
    reco_triplet_vector_index.reserve( _ev_reco_triplets._triplets_v.size() );

    int index = 0;
    for ( auto& reco_triplet : _ev_reco_triplets._triplets_v ) {
      int has_shower_pid = 0;
      for ( auto& pdgcode : reco_triplet.pids ) {
        if ( abs(pdgcode)==11 || abs(pdgcode)==22 ) {
          has_shower_pid++;
        }
      }
      if ( has_shower_pid>0 ) {
        std::vector<float> pos(3,0);
        for (int i=0; i<3; i++)
          pos[i] = reco_triplet.pos_reco[i];
        true_shower_pts.push_back( pos );
        reco_triplet_vector_index.push_back(index);
      } 
      index++;
    }

    float maxdist = 1.0;
    int minsize = 3;
    int maxkd = 10;
    std::vector< larflow::recoutils::cluster_t > cluster_v;
    larflow::recoutils::cluster_sdbscan_spacepoints( true_shower_pts, cluster_v, maxdist, minsize, maxkd);
    int nclusters = cluster_v.size(); // skip the last cluster which are noise points

    std::vector<int> above_thresh( reco_triplet_vector_index.size(), 0 );

    for (size_t icluster=0; icluster<cluster_v.size(); icluster++) {
      auto& cluster = cluster_v.at(icluster);
      int npts = cluster.points_v.size();
      if ( npts>=npts_threshold ) {
        // mark hits which are part of above threshold clusters
        for (auto& hitidx : cluster.hitidx_v ) {
          above_thresh.at(hitidx) = 1;
        }
      }
    }

    for ( size_t ihit=0; ihit<reco_triplet_vector_index.size(); ihit++ ) {
      if ( above_thresh[ihit]==0 ) {
        int hitidx = reco_triplet_vector_index[ihit];
        _ev_reco_triplets._triplets_v.at( hitidx ).ssnetlabel = 8; // assign LED Label
      }
    }

  }

  /**
   * @brief Save MC particle tree information to HDF5 file
   *
   * Writes the full MCParticleGraph tree structure including true and
   * SCE-corrected start positions, parentage, and daughter relationships.
   * This information is used downstream for shower origin prediction.
   *
   * @param file HDF5 file reference
   * @param groupname_prefix Prefix for HDF5 group path (e.g. "entry_0")
   */
  void SimChTripletLabelMaker::save_mc_particle_tree(
    HighFive::File& file,
    std::string groupname_prefix )
  {

    std::string mc_groupname = "/mc_particle_tree";
    if ( groupname_prefix != "" ) {
      mc_groupname = groupname_prefix + mc_groupname;
    }
    LARCV_INFO() << "create group: " << mc_groupname << std::endl;
    file.createGroup(mc_groupname);

    size_t nnodes = _mcpgraph.node_v.size();

    // Per-particle 1D arrays
    std::vector<int>   mc_trackid(nnodes, 0);
    std::vector<int>   mc_pid(nnodes, 0);
    std::vector<int>   mc_parent_trackid(nnodes, -1);
    std::vector<int>   mc_origin(nnodes, 0);
    std::vector<float> mc_energy_mev(nnodes, 0.0f);
    std::vector<int>   mc_process_code(nnodes, -1);

    // Per-particle 2D arrays (nnodes x 3)
    std::vector< std::vector<float> > mc_start_pos(nnodes);
    std::vector< std::vector<float> > mc_start_pos_sce(nnodes);

    // Daughter relationship arrays
    std::vector<int> mc_num_daughters(nnodes, 0);
    std::vector<int> mc_daughter_start_indices(nnodes, 0);
    std::vector<int> mc_daughter_trackids; // flattened

    // Process string to integer mapping
    // Common Geant4 processes in LArTPC simulation
    auto map_process = [](const std::string& proc) -> int {
      if      (proc == "primary")              return 0;
      else if (proc == "Decay")                return 1;
      else if (proc == "compt")                return 2;
      else if (proc == "conv")                 return 3;
      else if (proc == "phot")                 return 4;
      else if (proc == "eBrem")                return 5;
      else if (proc == "eIoni")                return 6;
      else if (proc == "muIoni")               return 7;
      else if (proc == "muBrems" || proc == "muBrem") return 8;
      else if (proc == "muPairProd")           return 9;
      else if (proc == "hIoni")                return 10;
      else if (proc == "hadElastic")           return 11;
      else if (proc == "neutronInelastic")     return 12;
      else if (proc == "protonInelastic")      return 13;
      else if (proc == "pi+Inelastic")         return 14;
      else if (proc == "pi-Inelastic")         return 15;
      else if (proc == "muMinusCaptureAtRest") return 16;
      else if (proc == "nCapture")             return 17;
      else if (proc == "annihil")              return 18;
      else if (proc == "CoulombScat")          return 19;
      else if (proc == "photonNuclear")        return 20;
      else if (proc == "null")                 return -1;
      else                                     return 99; // other/unknown
    };

    int daughter_offset = 0;
    for (size_t inode = 0; inode < nnodes; inode++) {
      auto const& node = _mcpgraph.node_v[inode];

      mc_trackid[inode]        = node.tid;
      mc_pid[inode]            = node.pid;
      mc_parent_trackid[inode] = node.mtid;
      mc_origin[inode]         = node.origin;
      mc_energy_mev[inode]     = node.E_MeV;
      mc_process_code[inode]   = map_process(node.process);

      // True start position (first 3 elements of start vector)
      std::vector<float> spos(3, 0.0f);
      if (node.start.size() >= 3) {
        spos[0] = node.start[0];
        spos[1] = node.start[1];
        spos[2] = node.start[2];
      }
      mc_start_pos[inode] = spos;

      // SCE-corrected start position
      std::vector<float> spos_sce(3, 0.0f);
      if (node.start.size() >= 3 && _psce != nullptr) {
        double x = (double)node.start[0];
        double y = (double)node.start[1];
        double z = (double)node.start[2];
        std::vector<double> offsets = _psce->GetPosOffsets(x, y, z);
        spos_sce[0] = (float)(x - offsets[0] + 0.7);
        spos_sce[1] = (float)(y + offsets[1]);
        spos_sce[2] = (float)(z + offsets[2]);
      }
      mc_start_pos_sce[inode] = spos_sce;

      // Daughter relationships
      mc_num_daughters[inode] = (int)node.daughter_v.size();
      mc_daughter_start_indices[inode] = daughter_offset;
      for (size_t idau = 0; idau < node.daughter_v.size(); idau++) {
        if (node.daughter_v[idau] != nullptr) {
          mc_daughter_trackids.push_back(node.daughter_v[idau]->tid);
        } else if (idau < node.daughter_idx_v.size()) {
          // Fallback: use daughter_idx_v to look up trackid
          int didx = node.daughter_idx_v[idau];
          if (didx >= 0 && didx < (int)_mcpgraph.node_v.size()) {
            mc_daughter_trackids.push_back(_mcpgraph.node_v[didx].tid);
          } else {
            mc_daughter_trackids.push_back(-1);
          }
        } else {
          mc_daughter_trackids.push_back(-1);
        }
      }
      daughter_offset += (int)node.daughter_v.size();
    }

    // Write per-particle data
    dump_compressed(file, mc_groupname + "/trackid",             mc_trackid);
    dump_compressed(file, mc_groupname + "/pid",                 mc_pid);
    dump_compressed(file, mc_groupname + "/parent_trackid",      mc_parent_trackid);
    dump_compressed(file, mc_groupname + "/origin",              mc_origin);
    dump_compressed(file, mc_groupname + "/start_pos",           mc_start_pos);
    dump_compressed(file, mc_groupname + "/start_pos_sce",       mc_start_pos_sce);
    dump_compressed(file, mc_groupname + "/energy_mev",          mc_energy_mev);
    dump_compressed(file, mc_groupname + "/process_code",        mc_process_code);

    // Write daughter relationship data
    dump_compressed(file, mc_groupname + "/num_daughters",          mc_num_daughters);
    dump_compressed(file, mc_groupname + "/daughter_start_indices", mc_daughter_start_indices);
    if (mc_daughter_trackids.size() > 0) {
      dump_compressed(file, mc_groupname + "/daughter_trackids",   mc_daughter_trackids);
    } else {
      // Write empty vector
      std::vector<int> empty_vec;
      file.createDataSet(mc_groupname + "/daughter_trackids", empty_vec);
    }

    // Write neutrino vertex info if available
    if (_mcpgraph._nu_vertices_v.size() > 0) {
      std::vector< std::vector<float> > nu_vertices;
      nu_vertices.reserve(_mcpgraph._nu_vertices_v.size());
      for (auto const& vtx : _mcpgraph._nu_vertices_v) {
        std::vector<float> vtx3(3, 0.0f);
        if (vtx.size() >= 3) {
          vtx3[0] = vtx[0];
          vtx3[1] = vtx[1];
          vtx3[2] = vtx[2];
        }
        nu_vertices.push_back(vtx3);
      }
      dump_compressed(file, mc_groupname + "/nu_vertices", nu_vertices);
    } else {
      // Write empty 2D array placeholder
      std::vector< std::vector<float> > empty_vtx;
      file.createDataSet(mc_groupname + "/nu_vertices", empty_vtx);
    }

    LARCV_INFO() << "Wrote mc_particle_tree: " << nnodes
                 << " nodes, " << mc_daughter_trackids.size()
                 << " daughter entries, "
                 << _mcpgraph._nu_vertices_v.size()
                 << " neutrino vertices" << std::endl;
  }

  void SimChTripletLabelMaker::export_as_hdf( std::string hdf_outfile )
  {

// """
//         "matchtriplet",
//         "match_weight",
//         "spacepoints",
//         #"positive_indices",                                                                                                                                                                  
//         "ssnet_label",
//         "ssnet_top_weight",
//         "ssnet_class_weight",
//         "kplabel",
//         "kplabel_weight",
//         #"kpshift",                                                                                                                                                                           
//         "paf_label",
//         "paf_weight",
//         #"origin_label",                                                                                                                                                                      
//         "keypoint_truth_kptype_pdg_trackid",
//         "keypoint_truth_pos",
//         "wireimage_plane0",
//         "wireimage_plane1",
//         "wireimage_plane2"]

// """

    LARCV_INFO() << "export to " << hdf_outfile << std::endl;

    HighFive::File file(hdf_outfile, HighFive::File::Overwrite);

    save_entry_truetriplets( file, "" );
    save_entry_sparseimg( file, "" );
    save_entry_to_hdf( file, "" );
    save_mc_particle_tree( file, "" );

    file.flush();

  }

  void SimChTripletLabelMaker::save_entry_to_hdf( 
    HighFive::File& file,
    std::string groupname_prefix )
  {

    size_t n_reco_triplets = _ev_reco_triplets._triplets_v.size();
    std::vector< std::vector<float> > reco_pos(n_reco_triplets);
    std::vector< std::vector<float> > reco_edep(n_reco_triplets);
    std::vector< std::vector<float> > reco_pixval(n_reco_triplets);
    std::vector<int>   reco_uwire(n_reco_triplets,0);
    std::vector<int>   reco_vwire(n_reco_triplets,0);
    std::vector<int>   reco_ywire(n_reco_triplets,0);
    std::vector<int>   reco_tick(n_reco_triplets,0);
    std::vector<int>   reco_hasmatch(n_reco_triplets,0);

    std::vector<long>  reco_trackid(n_reco_triplets,0);
    std::vector<int>   reco_pid(n_reco_triplets,0);
    std::vector<int>   reco_aid(n_reco_triplets,0);
    std::vector<int>   reco_origin(n_reco_triplets,0);

    std::vector< std::vector<float> > reco_kpscores(n_reco_triplets);

    std::vector<int>   reco_ssnet_label(n_reco_triplets,0);
    std::vector<int>   reco_ssnet_boundary(n_reco_triplets,0);
    std::vector<float> reco_ssnet_weight(n_reco_triplets,0);

    for (size_t idx=0; idx<n_reco_triplets; idx++) {

      auto const& tripinfo = _ev_reco_triplets._triplets_v.at(idx);

      for ( auto& tid : tripinfo.trackids ) {
          reco_trackid[idx] = tid;
          if (reco_trackid[idx]!=-1)
              break;
      }

      for ( auto& xpid : tripinfo.pids ) {
          reco_pid[idx]     = xpid;
          if (reco_pid[idx]!=-1)
              break;
      }

      for ( auto& xaid : tripinfo.aids ) {
          reco_aid[idx]     = xaid;
          if (reco_aid[idx]!=-1)
              break;
      }

      for ( auto& xorigin : tripinfo.origin ) {
          reco_origin[idx]  = xorigin;
          if (reco_origin[idx]!=-1)
              break;
      }

      std::vector<float> rpos = {
        tripinfo.pos_reco[0],
        tripinfo.pos_reco[1],
        tripinfo.pos_reco[2]
      };

      std::vector<float> edep = {
        (float)tripinfo.edep[0],
        (float)tripinfo.edep[1],
        (float)tripinfo.edep[2]
      };

      std::vector<float> pixval = {
        (float)tripinfo.pixval[0],
        (float)tripinfo.pixval[1],
        (float)tripinfo.pixval[2]
      };

      reco_pos[idx]    = rpos;
      reco_edep[idx]   = edep;
      reco_pixval[idx] = pixval;
      reco_uwire[idx]  = tripinfo.imgcoord[0];
      reco_vwire[idx]  = tripinfo.imgcoord[1];
      reco_ywire[idx]  = tripinfo.imgcoord[2];
      reco_tick[idx]   = tripinfo.imgcoord[4];
      reco_hasmatch[idx] = tripinfo.hasmatch;

      reco_kpscores[idx] = tripinfo.kpscores;

      reco_ssnet_label[idx]    = tripinfo.ssnetlabel;
      reco_ssnet_boundary[idx] = tripinfo.ssnetboundary;
      reco_ssnet_weight[idx]   = tripinfo.ssnet_classcount_weight;

    }

    std::string recotriplet_groupname = "/triplet_data";
    if ( groupname_prefix!="" ) {
      recotriplet_groupname = groupname_prefix+recotriplet_groupname;
    }
    LARCV_INFO() << "create group: " << recotriplet_groupname << std::endl;
    file.createGroup(recotriplet_groupname);

    // Write datasets with compression (level 6, chunk size 10000)
    dump_compressed( file, recotriplet_groupname+"/pos",      reco_pos );
    dump_compressed( file, recotriplet_groupname+"/edep",     reco_edep );
    dump_compressed( file, recotriplet_groupname+"/pixval",   reco_pixval );
    dump_compressed( file, recotriplet_groupname+"/uwire",    reco_uwire );
    dump_compressed( file, recotriplet_groupname+"/vwire",    reco_vwire );
    dump_compressed( file, recotriplet_groupname+"/ywire",    reco_ywire );
    dump_compressed( file, recotriplet_groupname+"/tick",     reco_tick  );

    if ( _is_mc ) {
      dump_compressed( file, recotriplet_groupname+"/pid",      reco_pid);
      dump_compressed( file, recotriplet_groupname+"/aid",      reco_aid);
      dump_compressed( file, recotriplet_groupname+"/origin",   reco_origin);
      dump_compressed( file, recotriplet_groupname+"/trackid",  reco_trackid);
      dump_compressed( file, recotriplet_groupname+"/hasmatch", reco_hasmatch  );
      dump_compressed( file, recotriplet_groupname+"/kpscores", reco_kpscores );
      dump_compressed( file, recotriplet_groupname+"/ssnet_label",    reco_ssnet_label );
      dump_compressed( file, recotriplet_groupname+"/ssnet_boundary", reco_ssnet_boundary );
      if ( _save_weights_to_hdf )
        dump_compressed( file, recotriplet_groupname+"/ssnet_weight",   reco_ssnet_weight );

      //_mckpmaker.save_entry_to_hdf(file,"");
      std::string kp_groupname = "/mckeypoints";
      if ( groupname_prefix!="" ) {
        kp_groupname = groupname_prefix + "/mckeypoints";
      }
      LARCV_INFO() << "create groupname: " << kp_groupname << std::endl;
      file.createGroup(kp_groupname);

      // export different arrays for export
      int nkeypoints = _final_keypoint_list.size();

      std::vector< std::vector<float> > pos_appear(nkeypoints);
      std::vector< std::vector<float> > startpos_appear(nkeypoints);
      std::vector< std::vector<int> >   imgcoord(nkeypoints);
      std::vector< int > kptype(nkeypoints);
      std::vector< int > kppid(nkeypoints);
      std::vector< int > kptrackid(nkeypoints);

      int ikp=0;
      for ( auto const& kpd : _final_keypoint_list ) {
        pos_appear[ikp] = kpd.keypt_appear;
        imgcoord[ikp]   = kpd.imgcoord;
        kptype[ikp]     = kpd.kptype;
        kppid[ikp]      = kpd.pid;
        kptrackid[ikp]  = kpd.trackid;
        startpos_appear[ikp] = kpd.startpt_appear;
        ikp++;
      }

      dump_compressed( file, kp_groupname+"/pos",      pos_appear);
      dump_compressed( file, kp_groupname+"/imgcoord", imgcoord);
      dump_compressed( file, kp_groupname+"/kptype",   kptype);
      dump_compressed( file, kp_groupname+"/pid",      kppid);
      dump_compressed( file, kp_groupname+"/trackid",  kptrackid);
      dump_compressed( file, kp_groupname+"/startpos", startpos_appear);
    }
  }

  /**
   * @brief Save current entry's image2d and triplet image maps
   */
  void SimChTripletLabelMaker::save_entry_truetriplets( 
    HighFive::File& file, std::string groupname_prefix )
  {
    if ( _hdf_file==nullptr ) {
      std::stringstream errmsg;
      errmsg << "Saving entry without first creating HDF file." << std::endl;
      errmsg << "Call open_hdf_file( std::string ) first." << std::endl;
      throw std::runtime_error( errmsg.str() );
    }

    std::string truetriplet_groupname = "/triplet_truth";
    if ( groupname_prefix!="" ) {
      truetriplet_groupname = groupname_prefix+truetriplet_groupname;
    }
    LARCV_INFO() << "create group: " << truetriplet_groupname << std::endl;
    file.createGroup(truetriplet_groupname);


    // export different arrays for export
    ublarcvapp::mctools::EventMCPixelLabels& pixel3d = _mcpixelmaker._pixels_v;
    int ntriplets = pixel3d._triplets_v.size();

    std::vector< std::vector<float> > pos_true_v(ntriplets);
    std::vector< std::vector<float> > pos_reco_v(ntriplets);

    std::vector< std::array<double,3> > edep(ntriplets);
    std::vector<long>  trackid(ntriplets,0);
    std::vector<int>   pid(ntriplets,0);
    std::vector<int>   aid(ntriplets,0);
    std::vector<int>   origin(ntriplets,0);
    std::vector<int>   uwire(ntriplets,0);
    std::vector<int>   vwire(ntriplets,0);
    std::vector<int>   ywire(ntriplets,0);
    std::vector<int>   tick(ntriplets,0);
    std::vector<int>   row(ntriplets,0);

    for (auto const& triplet : pixel3d._triplets_v ) {
        long idx = triplet.index;

        std::vector<float> pos_true = {
          triplet.pos[0],
          triplet.pos[1],
          triplet.pos[2]
        };
        pos_true_v[idx] = pos_true;

        std::vector<float> pos_reco = {
          triplet.pos_reco[0],
          triplet.pos_reco[1],
          triplet.pos_reco[2] 
        };
        pos_reco_v[idx] = pos_reco;

        edep[idx] = std::array<double,3>{0,0,0};
        for (int i=0; i<3; i++)
          edep[idx][i] = triplet.edep[i];

        for ( auto& tid : triplet.trackids ) {
            trackid[idx] = tid;
            if (trackid[idx]!=-1)
                break;
        }

        for ( auto& xpid : triplet.pids ) {
            pid[idx]     = xpid;
            if (pid[idx]!=-1)
                break;
        }

        for ( auto& xaid : triplet.aids ) {
            aid[idx]     = xaid;
            if (aid[idx]!=-1)
                break;
        }

        for ( auto& xorigin : triplet.origin ) {
            origin[idx]  = xorigin;
            if (origin[idx]!=-1)
                break;
        }

        uwire[idx]   = triplet.imgcoord[0];
        vwire[idx]   = triplet.imgcoord[1];
        ywire[idx]   = triplet.imgcoord[2];
        tick[idx]    = triplet.imgcoord[4];
        row[idx]     = triplet.imgcoord[3];
    }

    dump_compressed( file, truetriplet_groupname+"/pos",      pos_true_v);
    dump_compressed( file, truetriplet_groupname+"/pos_reco", pos_reco_v);

    dump_compressed( file, truetriplet_groupname+"/edep",    edep);
    dump_compressed( file, truetriplet_groupname+"/trackid", trackid);
    dump_compressed( file, truetriplet_groupname+"/pid",     pid);
    dump_compressed( file, truetriplet_groupname+"/aid",     aid);
    dump_compressed( file, truetriplet_groupname+"/origin",  origin);
    dump_compressed( file, truetriplet_groupname+"/uwire",   uwire);
    dump_compressed( file, truetriplet_groupname+"/vwire",   vwire);
    dump_compressed( file, truetriplet_groupname+"/ywire",   ywire);
    dump_compressed( file, truetriplet_groupname+"/tick",    tick);
    dump_compressed( file, truetriplet_groupname+"/row",     row);

  }

  /**
   * @brief Save current entry's image2d and triplet image maps
   */
  void SimChTripletLabelMaker::save_entry_sparseimg( HighFive::File& file, std::string groupname_prefix )
  {
    if ( _hdf_file==nullptr ) {
      std::stringstream errmsg;
      errmsg << "Saving entry without first creating HDF file." << std::endl;
      errmsg << "Call open_hdf_file( std::string ) first." << std::endl;
      throw std::runtime_error( errmsg.str() );
    }

    LARCV_INFO() << "Save to image info to hdf file. group prefix=" << groupname_prefix << std::endl;
    
    std::string img_group_name = "/image_data";
    if ( groupname_prefix!="" ) {
      img_group_name = groupname_prefix + "/image_data";
    }
    LARCV_INFO() << "create group: " << img_group_name << std::endl;
    file.createGroup(img_group_name);

    auto& imgpixels_vv = _tripletmaker._sparseimg_vv;

    LARCV_INFO() << "Number of Sparse Images: " << imgpixels_vv.size() << std::endl;

    for (size_t iplane=0; iplane<imgpixels_vv.size(); iplane++ ) {

      std::stringstream ss_plane_group;
      ss_plane_group << img_group_name << "/plane" << iplane;
      LARCV_INFO() << "create group: " << ss_plane_group.str() << std::endl;
      file.createGroup( ss_plane_group.str() );

      auto& meta = _tripletmaker._imgmeta_v.at(iplane);

      std::vector<int>   dims    = { (int)meta.cols(), (int)meta.rows() };
      std::vector<float> origin  = { (float)meta.min_x(), (float)meta.min_y() };
      std::vector<float> pixsize = { (float)meta.pixel_width(), (float)meta.pixel_height() };

      auto& imgpixels_v = imgpixels_vv.at(iplane);
      std::vector< std::vector<int> > pixcoords_v;
      std::vector< float > pixfeat_v;
      pixcoords_v.reserve( imgpixels_v.size() );
      pixfeat_v.reserve( imgpixels_v.size() );
      for ( auto& pixdata : imgpixels_v ) {
        std::vector<int> pixcoord = { pixdata.col, pixdata.row  };
        pixcoords_v.push_back( pixcoord );
        pixfeat_v.push_back( pixdata.val );
      }
      
      LARCV_INFO() << "save coord and feat to group: " << ss_plane_group.str() << std::endl;
      // Use higher compression for image data (often largest datasets)
      dump_compressed( file, ss_plane_group.str()+"/coord",   pixcoords_v, 7 );
      dump_compressed( file, ss_plane_group.str()+"/feat",    pixfeat_v, 7 );
      // ImageMeta information (small, use default compression)
      dump_compressed( file, ss_plane_group.str()+"/dims",    dims );
      dump_compressed( file, ss_plane_group.str()+"/origin",  origin );
      dump_compressed( file, ss_plane_group.str()+"/pixsize", pixsize );
    }

    LARCV_INFO() << "save triplet to: " << img_group_name + "/triplet_imgpix_index" << std::endl;
    dump_compressed( file, img_group_name + "/triplet_imgpix_index", _tripletmaker._triplet_v );

  }

  /**
   * @brief Save current data to class member HDF file
   */
  void SimChTripletLabelMaker::save_entry( std::string groupname_prefix )
  {
    if ( _hdf_file==nullptr ) {
      std::stringstream errmsg;
      errmsg << "Saving entry without first creating HDF file." << std::endl;
      errmsg << "Call open_hdf_file( std::string ) first." << std::endl;
      throw std::runtime_error( errmsg.str() );
    }

    LARCV_INFO() << "Save to hdf file. group prefix=" << groupname_prefix << std::endl;
    
    LARCV_INFO() << "create group: " << groupname_prefix << std::endl;
    _hdf_file->createGroup(groupname_prefix);
    
    save_entry_sparseimg( *_hdf_file, groupname_prefix );
    save_entry_to_hdf( *_hdf_file, groupname_prefix );
    save_mc_particle_tree( *_hdf_file, groupname_prefix );
    if ( _save_truth_triplet_info )
      save_entry_truetriplets( *_hdf_file, groupname_prefix );

    _shower_fragment_maker.save_entry_to_hdf( *_hdf_file, groupname_prefix );

    _hdf_file->flush();

  }

  /**
   * @brief Create class member HDF file to save entries
   */
  void SimChTripletLabelMaker::open_hdf_file( std::string hdf_outfile )
  {
    LARCV_INFO() << "output file = " << hdf_outfile << std::endl;
    _hdf_file = new HighFive::File( hdf_outfile, HighFive::File::Overwrite);
  }

  /**
   * @brief Create class member HDF file to save entries
   */
 void SimChTripletLabelMaker::close_hdf_file()
  {
    if ( _hdf_file ) {
      LARCV_INFO() << "Flush and close file." << std::endl;
      _hdf_file->flush();
      delete _hdf_file;
      _hdf_file = nullptr;
    }
    return;
  }



}
}
