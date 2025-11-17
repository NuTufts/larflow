#include "SimChTripletLabelMaker.h"

#include "larlite/LArUtil/TimeService.h"
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"
#include "larlite/LArUtil/Geometry.h"
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/DataFormat/simch.h"

#include "larcv/core/DataFormat/EventImage2D.h"

#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "ublarcvapp/MCTools/MCPos2ImageUtils.h"

namespace larflow {
namespace prep {

  void SimChTripletLabelMaker::process( 
    larlite::storage_manager& ioll, 
    larcv::IOManager& iolcv )
  {
    
    // utility to go from simulated electronics TDC to 
    // ticks (tdcs after readout trigger)
    const larutil::TimeService* timeservice = larutil::TimeService::GetME();

    // geometry
    const larutil::Geometry* geom = larutil::Geometry::GetME();

    // moving real position to apparent position
    larutil::SpaceChargeMicroBooNE* psce = 
      new larutil::SpaceChargeMicroBooNE(larutil::SpaceChargeMicroBooNE::kMCC9_Forward);

    // drift velocity
    float driftv = larutil::LArProperties::GetME()->DriftVelocity();

    // get the simch product we need
    larlite::event_simch* ev_simch = 
      (larlite::event_simch*)ioll.get_data(larlite::data::kSimChannel,"largeant");

    // get images
    larcv::EventImage2D* ev_img = 
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wiremc");

    ublarcvapp::mctools::MCPixelPGraph mcpg;
    mcpg.buildgraphonly(ioll);

    auto const& img_v = ev_img->as_vector();
    int nplanes = (int)img_v.size();

    auto const& meta0 = img_v.at(0).meta();

    // loop over simch information, making TripletLabels_t
    size_t nsimch = ev_simch->size();
    size_t ide_w_no_t0 = 0;
    size_t ide_outofimg = 0;
    size_t ide_w_badwire = 0;
    size_t num_ide_used = 0;

    for (size_t isimch=0; isimch<nsimch; isimch++) {
        auto& simch = ev_simch->at(isimch);
        auto chid = simch.Channel();
        auto& idcmap = simch.TDCIDEMap();
        for ( auto it=idcmap.begin(); it!=idcmap.end(); it++ ) {
            long tdc = it->first;
            int tick = int(timeservice->TPCTDC2Tick(tdc));
            size_t nide = it->second.size();
            for (auto& ide : it->second ) {

                std::vector<double> pos = { ide.x, ide.y, ide.z };
                long tid = ide.trackID;
                long xtid = (tid>=0) ? tid : -tid;
                double edep = ide.energy;

                // get wire coordinates for these positions
                std::vector<int> wire_v(nplanes,0);
                bool bad_wire = false;
                for (int iplane=0; iplane<nplanes; iplane++) {
                    try {
                        UInt_t wireid = geom->NearestWire( pos, iplane );
                        wire_v[iplane] = wireid;
                    }
                    catch (...){
                        bad_wire = true;
                    }
                }
                if ( bad_wire ) {
                    ide_w_badwire++;
                    continue;
                }

                // replace low-energy shower trackid label with mother of the shower
                long mtid = mcpg.getShowerMotherID( tid );
                if (mtid>0) {
                    xtid = mtid;
                }

                auto pnode_t = mcpg.findTrackID( xtid );
                if ( pnode_t==nullptr ) {
                    // don't have an alternative for this right now
                    ide_w_no_t0++;
                    continue;
                }

                long aid = mcpg.getAncestorID( xtid );
                int pid  = pnode_t->pid;
                int origin = pnode_t->origin;

                double t0 = pnode_t->start.at(3);

                bool applied = false;
                std::vector<double> pos_sce = psce->ApplySpaceChargeEffect( pos[0], pos[1], pos[2], applied );

                // get (u,v,y,tick)
                std::vector<float> imgpos = 
                    ublarcvapp::mctools::MCPos2ImageUtils::Get()->truepos_to_imagepos( pos[0],
                        pos[1],
                        pos[2],
                        t0,
                        true );

                float tick = imgpos[3];
                if ( tick<(float)meta0.min_y() || tick>=(float)meta0.max_y() ) {
                    ide_outofimg++;
                    continue;
                }

                int row = meta0.row( imgpos[3] );

                std::array<int,4> imgindex = { 
                    (int)imgpos[0], 
                    (int)imgpos[1], 
                    (int)imgpos[2], 
                    row };

                auto it_index = _imgcoord_to_tripindex.find( imgindex );
                if ( it_index==_imgcoord_to_tripindex.end() ) {
                    TripletLabels_t trip;
                    trip.index = (long)_triplets_v.size();
                    trip.imgcoord[0] = imgindex[0];
                    trip.imgcoord[1] = imgindex[1];
                    trip.imgcoord[2] = imgindex[2];
                    trip.imgcoord[3] = (int)tick;
                    trip.imgcoord[4] = row;
                    trip.pos[0] = pos[0];
                    trip.pos[1] = pos[1];
                    trip.pos[2] = pos[2];
                    trip.pos_reco[0] = (tick-3200)*0.5/driftv;
                    trip.pos_reco[1] = pos_sce[1];
                    trip.pos_reco[2] = pos_sce[2];
                    _imgcoord_to_tripindex[imgindex] = trip.index;
                    _triplets_v.emplace_back( std::move(trip) );     
                    it_index = _imgcoord_to_tripindex.find( imgindex );
                }

                auto& tripinfo = _triplets_v.at(it_index->second);
                tripinfo.edep += edep;
                tripinfo.trackids.insert(xtid);
                tripinfo.aids.insert(aid);
                tripinfo.pids.insert(pid);
                tripinfo.origin.insert(origin);
                num_ide_used++;

            }

        }
    }

    LARCV_INFO() << "Number of Triplets Created: " << _triplets_v.size() << std::endl;
    LARCV_INFO() << "  IDEs with no track ID match and t0: " << ide_w_no_t0 << std::endl;
    LARCV_INFO() << "  IDEs out-of-image: " << ide_outofimg << std::endl;
    LARCV_INFO() << "  IDEs with no nearby-wire: " << ide_w_badwire << std::endl;
    LARCV_INFO() << "  IDEs used: " << num_ide_used << std::endl; 


  }


}
}