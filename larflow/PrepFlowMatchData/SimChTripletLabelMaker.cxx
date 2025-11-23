#include "SimChTripletLabelMaker.h"

#include "larlite/LArUtil/TimeService.h"
#include "larlite/LArUtil/Geometry.h"
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/DataFormat/simch.h"

#include "larcv/core/DataFormat/EventImage2D.h"

#include <highfive/H5Easy.hpp>

#include "ublarcvapp/MCTools/MCPos2ImageUtils.h"

#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"
#include "larflow/RecoUtils/cluster_functions.h" 

namespace larflow {
namespace prep {

  SimChTripletLabelMaker::SimChTripletLabelMaker()
  : larcv::larcv_base("SimChTripletLabelMaker"),
    _psce(nullptr),
    _hdf_file(nullptr)
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

    _mcpgraph.buildgraph(ioll);

    _mcpixelmaker.make_truthlabels_fromsimch( "wiremc", ioll, iolcv, _mcpgraph, _psce );

    make_reco_triplets(iolcv);

    label_reco_triplets( _mcpixelmaker._pixels_v, _ev_reco_triplets );

    _mckpmaker.set_mcparticle_graph( &_mcpgraph );
    _mckpmaker.set_spacecharge_instance( _psce );
    _mckpmaker.setADCimageTreeName( "wiremc" );
    _mckpmaker.clear();
    _mckpmaker.process( iolcv, ioll );

    adjust_keypoints( _mckpmaker.getMCKeypoint(), 
      _ev_reco_triplets,
      _mcpgraph );

    make_keypoint_labels( 3.0, 0.01 );

  }

  void SimChTripletLabelMaker::make_reco_triplets( 
      larcv::IOManager& iolcv )
  {

    LARCV_INFO() << "start" << std::endl;
    _ev_reco_triplets.clear();

    // make reco triplets
    larflow::prep::PrepMatchTriplets reco_triplet_maker;
    reco_triplet_maker.process( iolcv, "wiremc", "wiremc", 10.0, true );

    // copy over triplets
    larcv::EventImage2D* ev_img = 
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wiremc");

    auto const& img_v = ev_img->as_vector();
    int nplanes = (int)img_v.size();
    auto const& meta0 = img_v.at(0).meta();

    for (size_t itrip=0; itrip<reco_triplet_maker._triplet_v.size(); itrip++ ) {

        TripletLabels_t trip;

        trip.index = (long)_ev_reco_triplets._triplets_v.size();

        int row = -1;
        for (int ip=0; ip<nplanes; ip++) {
            auto const& imgpix = reco_triplet_maker._sparseimg_vv.at(ip).at( reco_triplet_maker._triplet_v[itrip][ip] );
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
        trip.pos_reco[0] = reco_triplet_maker._pos_v[itrip][0];
        trip.pos_reco[1] = reco_triplet_maker._pos_v[itrip][1];
        trip.pos_reco[2] = reco_triplet_maker._pos_v[itrip][2];  

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

      LARCV_INFO() << " keypoint[tid=" << pnode->tid << "] "
        << " num points=" << points_v.size() 
        << " num clusters=" << nclusters
        << std::endl;
      LARCV_INFO() << "    mom4=" << mom_dir[0] << ", "
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

        LARCV_INFO() << "   cluster edep: " 
          <<  edep_planesum[0] << ", "
          <<  edep_planesum[1] << ", "
          <<  edep_planesum[2] << " MeV"
          << " nabove=" << nabove_threshold_planes
          << std::endl;

        if ( nabove_threshold_planes>=2 ) {
          // qualifying cluster, get most upstream position
          for ( auto& testpt : cluster.points_v ) {
            float s = larflow::recoutils::pointRayProjection3f( kppos, mom_dir, testpt );
            if ( s < min_s ) {
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
        LARCV_NORMAL() << "Adjust keypoint" << std::endl;
        LARCV_NORMAL() << "  from: (" << kpd.keypt_appear[0] << ", " 
          << kpd.keypt_appear[1] << ", "
          << kpd.keypt_appear[2] << ")" << std::endl;
        LARCV_NORMAL() << "  to: (" << most_upstream_pt[0] << ", "
          << most_upstream_pt[1] << ", "
          << most_upstream_pt[2] << ")"
          << std::endl;
        LARCV_NORMAL() << "  edep: " << most_upstream_edep[0] << ", "
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

  void SimChTripletLabelMaker::export_as_hdf( std::string hdf_outfile )
  {

    LARCV_INFO() << "export to " << hdf_outfile << std::endl;

    HighFive::File file(hdf_outfile, HighFive::File::Overwrite);

    save_entry_to_hdf( file, "" );

  }

  void SimChTripletLabelMaker::save_entry_to_hdf( 
    HighFive::File& file,
    std::string groupname_prefix )
  {

    // export different arrays for export
    ublarcvapp::mctools::EventMCPixelLabels& pixel3d = _mcpixelmaker._pixels_v;
    int ntriplets = pixel3d._triplets_v.size();

    std::vector<float> pos_x(ntriplets,0);
    std::vector<float> pos_y(ntriplets,0);
    std::vector<float> pos_z(ntriplets,0);

    std::vector<float> pos_x_reco(ntriplets,0);
    std::vector<float> pos_y_reco(ntriplets,0);
    std::vector<float> pos_z_reco(ntriplets,0);

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

        pos_x[idx] = triplet.pos[0];
        pos_y[idx] = triplet.pos[1];
        pos_z[idx] = triplet.pos[2];

        pos_x_reco[idx] = triplet.pos_reco[0];
        pos_y_reco[idx] = triplet.pos_reco[1];
        pos_z_reco[idx] = triplet.pos_reco[2];

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

    std::string truetriplet_groupname = "/triplet_truth";
    if ( groupname_prefix!="" ) {
      truetriplet_groupname = groupname_prefix+truetriplet_groupname;
    }
    LARCV_INFO() << "create group: " << truetriplet_groupname << std::endl;
    file.createGroup(truetriplet_groupname);

    H5Easy::dump( file, truetriplet_groupname+"/pos_x", pos_x);
    H5Easy::dump( file, truetriplet_groupname+"/pos_y", pos_y);
    H5Easy::dump( file, truetriplet_groupname+"/pos_z", pos_z);

    H5Easy::dump( file, truetriplet_groupname+"/pos_x_reco", pos_x_reco);
    H5Easy::dump( file, truetriplet_groupname+"/pos_y_reco", pos_y_reco);
    H5Easy::dump( file, truetriplet_groupname+"/pos_z_reco", pos_z_reco);

    H5Easy::dump( file, truetriplet_groupname+"/edep",    edep);
    H5Easy::dump( file, truetriplet_groupname+"/trackid", trackid);
    H5Easy::dump( file, truetriplet_groupname+"/pid",     pid);
    H5Easy::dump( file, truetriplet_groupname+"/aid",     aid);
    H5Easy::dump( file, truetriplet_groupname+"/origin",  origin);
    H5Easy::dump( file, truetriplet_groupname+"/uwire",   uwire);
    H5Easy::dump( file, truetriplet_groupname+"/vwire",   vwire);
    H5Easy::dump( file, truetriplet_groupname+"/ywire",   ywire);
    H5Easy::dump( file, truetriplet_groupname+"/tick",    tick);
    H5Easy::dump( file, truetriplet_groupname+"/row",     row);

    size_t n_reco_triplets = _ev_reco_triplets._triplets_v.size();
    std::vector<float> reco_pos_x(n_reco_triplets,0);
    std::vector<float> reco_pos_y(n_reco_triplets,0);
    std::vector<float> reco_pos_z(n_reco_triplets,0);
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

        reco_pos_x[idx] = tripinfo.pos_reco[0];
        reco_pos_y[idx] = tripinfo.pos_reco[1];
        reco_pos_z[idx] = tripinfo.pos_reco[2];
        reco_uwire[idx] = tripinfo.imgcoord[0];
        reco_vwire[idx] = tripinfo.imgcoord[1];
        reco_ywire[idx] = tripinfo.imgcoord[2];
        reco_tick[idx]  = tripinfo.imgcoord[4];
        reco_hasmatch[idx] = tripinfo.hasmatch;

        reco_kpscores[idx] = tripinfo.kpscores;
    }

    std::string recotriplet_groupname = "/triplet_data";
    if ( groupname_prefix!="" ) {
      recotriplet_groupname = groupname_prefix+recotriplet_groupname;
    }
    LARCV_INFO() << "create group: " << recotriplet_groupname << std::endl;
    file.createGroup(recotriplet_groupname);
    
    H5Easy::dump( file, recotriplet_groupname+"/pos_x",    reco_pos_x );
    H5Easy::dump( file, recotriplet_groupname+"/pos_y",    reco_pos_y );
    H5Easy::dump( file, recotriplet_groupname+"/pos_z",    reco_pos_z );
    H5Easy::dump( file, recotriplet_groupname+"/uwire",    reco_uwire );
    H5Easy::dump( file, recotriplet_groupname+"/vwire",    reco_vwire );
    H5Easy::dump( file, recotriplet_groupname+"/ywire",    reco_ywire );
    H5Easy::dump( file, recotriplet_groupname+"/tick",     reco_tick  );
    H5Easy::dump( file, recotriplet_groupname+"/pid",      reco_pid);
    H5Easy::dump( file, recotriplet_groupname+"/aid",      reco_aid);
    H5Easy::dump( file, recotriplet_groupname+"/origin",   reco_origin);
    H5Easy::dump( file, recotriplet_groupname+"/trackid",  reco_trackid);
    H5Easy::dump( file, recotriplet_groupname+"/hasmatch", reco_hasmatch  );
    H5Easy::dump( file, recotriplet_groupname+"/kpscores", reco_kpscores );
    
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
      ikp++;
    }

    H5Easy::dump( file, kp_groupname+"/pos",      pos_appear);
    H5Easy::dump( file, kp_groupname+"/imgcoord", imgcoord);
    H5Easy::dump( file, kp_groupname+"/kptype",   kptype);
    H5Easy::dump( file, kp_groupname+"/pid",      kppid);
    H5Easy::dump( file, kp_groupname+"/trackid",  kptrackid);

    file.flush();

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
    
    save_entry_to_hdf( *_hdf_file, groupname_prefix );

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