#include "SimChTripletLabelMaker.h"

#include "larlite/LArUtil/TimeService.h"
#include "larlite/LArUtil/Geometry.h"
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/DataFormat/simch.h"

#include "larcv/core/DataFormat/EventImage2D.h"

#include <highfive/H5Easy.hpp>

#include "ublarcvapp/MCTools/MCPos2ImageUtils.h"

#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"

namespace larflow {
namespace prep {

  void SimChTripletLabelMaker::process( 
    larlite::storage_manager& ioll, 
    larcv::IOManager& iolcv )
  {
    
    // moving real position to apparent position
    larutil::SpaceChargeMicroBooNE* psce = 
      new larutil::SpaceChargeMicroBooNE(larutil::SpaceChargeMicroBooNE::kMCC9_Forward);

    _mcpgraph.clear();
    _mcpgraph.buildgraph(ioll);

    _mcpixelmaker.make_truthlabels_fromsimch( "wiremc", ioll, iolcv, _mcpgraph, psce );

    make_reco_triplets(iolcv);

    label_reco_triplets( _mcpixelmaker._pixels_v, _ev_reco_triplets );

    _mckpmaker.set_mcparticle_graph( &_mcpgraph );
    _mckpmaker.set_spacecharge_instance( psce );
    _mckpmaker.setADCimageTreeName( "wiremc" );
    _mckpmaker.clear();
    _mckpmaker.process( iolcv, ioll );

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

  void SimChTripletLabelMaker::export_as_hdf( std::string hdf_outfile )
  {

    LARCV_INFO() << "export to " << hdf_outfile << std::endl;

    HighFive::File file(hdf_outfile, HighFive::File::Overwrite);

    file.createGroup("/triplet_data");

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

    H5Easy::dump( file, "/triplet_truth/pos_x", pos_x);
    H5Easy::dump( file, "/triplet_truth/pos_y", pos_y);
    H5Easy::dump( file, "/triplet_truth/pos_z", pos_z);

    H5Easy::dump( file, "/triplet_truth/pos_x_reco", pos_x_reco);
    H5Easy::dump( file, "/triplet_truth/pos_y_reco", pos_y_reco);
    H5Easy::dump( file, "/triplet_truth/pos_z_reco", pos_z_reco);

    H5Easy::dump( file, "/triplet_truth/edep",    edep);
    H5Easy::dump( file, "/triplet_truth/trackid", trackid);
    H5Easy::dump( file, "/triplet_truth/pid",     pid);
    H5Easy::dump( file, "/triplet_truth/aid",     aid);
    H5Easy::dump( file, "/triplet_truth/origin",  origin);
    H5Easy::dump( file, "/triplet_truth/uwire",   uwire);
    H5Easy::dump( file, "/triplet_truth/vwire",   vwire);
    H5Easy::dump( file, "/triplet_truth/ywire",   ywire);
    H5Easy::dump( file, "/triplet_truth/tick",    tick);
    H5Easy::dump( file, "/triplet_truth/row",     row);

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
    }
    
    H5Easy::dump( file, "/triplet_data/pos_x", reco_pos_x );
    H5Easy::dump( file, "/triplet_data/pos_y", reco_pos_y );
    H5Easy::dump( file, "/triplet_data/pos_z", reco_pos_z );
    H5Easy::dump( file, "/triplet_data/uwire", reco_uwire );
    H5Easy::dump( file, "/triplet_data/vwire", reco_vwire );
    H5Easy::dump( file, "/triplet_data/ywire", reco_ywire );
    H5Easy::dump( file, "/triplet_data/tick",  reco_tick  );
    H5Easy::dump( file, "/triplet_data/hasmatch", reco_hasmatch  );
    H5Easy::dump( file, "/triplet_data/trackid", reco_trackid);
    H5Easy::dump( file, "/triplet_data/pid",     reco_pid);
    H5Easy::dump( file, "/triplet_data/aid",     reco_aid);
    H5Easy::dump( file, "/triplet_data/origin",  reco_origin);

    _mckpmaker.save_entry_to_hdf(file,"");

    file.flush();

  }


}
}