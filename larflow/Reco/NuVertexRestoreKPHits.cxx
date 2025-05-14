#include "larflow/Reco/NuVertexRestoreKPHits.h"
#include "larlite/DataFormat/track.h"
#include "larflow/RecoUtils/geofuncs.h"

namespace larflow {
namespace reco {

    void NuVertexRestoreKPHits::process( std::vector<larflow::reco::NuVertexCandidate>& nuvtx_v, 
                                         larlite::storage_manager& ioll, 
                                         larcv::IOManager& iolcv )
    {
        
        LARCV_INFO() << "start. input=" << _input_kpvetoed_hit_treename << std::endl;

        // get KP-vetoed hits
        larlite::event_larflow3dhit* p_ev_kpvetoed
         = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, 
                                                        _input_kpvetoed_hit_treename );

        LARCV_INFO() << "number of hits: " << p_ev_kpvetoed->size() << std::endl;

        for ( auto& nuvtx : nuvtx_v ) {
            restoreVertexHits( nuvtx, *p_ev_kpvetoed );
        }
    }

    void NuVertexRestoreKPHits::restoreVertexHits( larflow::reco::NuVertexCandidate& nuvtx,
                                                   larlite::event_larflow3dhit& ev_kpvetoed )
    {

        std::vector<larlite::larflow3dhit> nearby_hits_v
         =  gatherKPVetoedHitsNearVertex( nuvtx, ev_kpvetoed, _collection_radius_cm );

        if ( nearby_hits_v.size()==0 )
            return;

        int npts = nearby_hits_v.size();

        // count number of track and shower primary prongs and get indices
        int nprongs = 0;
        std::vector<int> prim_track_indices;
        prim_track_indices.reserve( nuvtx.track_v.size());
        std::vector<int> prim_shower_indices;
        prim_shower_indices.reserve( nuvtx.shower_v.size());

        for (size_t itrack=0; itrack<nuvtx.track_v.size(); itrack++) {
            if ( itrack < nuvtx.track_isSecondary_v.size() && nuvtx.track_isSecondary_v[itrack]==0 ) {
                nprongs++;
                prim_track_indices.push_back(itrack);
            }
        }

        for (size_t ishower=0; ishower<nuvtx.shower_v.size(); ishower++) {
            if ( ishower < nuvtx.shower_isSecondary_v.size() && nuvtx.shower_isSecondary_v[ishower]==0 ) {
                nprongs++;
                prim_shower_indices.push_back(ishower);
            }
        }

        LARCV_INFO() << "number of prongs in vertex: " << nprongs 
                    << " ntrack=" << prim_track_indices.size() 
                    << " nshower=" << prim_shower_indices.size()
                    << std::endl;

        if ( nprongs==0 )
            return;

        KPDistArray_t dist_data( nearby_hits_v.size(), nprongs );

        LARCV_INFO() << "KPDistArray_t made (" << nearby_hits_v.size() << "," << nprongs << ")" << std::endl;

        int iprong = 0;
        for (auto const& trackidx : prim_track_indices )  {
            auto const& track = nuvtx.track_v.at(trackidx);
            LARCV_INFO() << "track_dir_v.size()=" << nuvtx.track_dir_v.size() << std::endl;
            LARCV_INFO() << "track ntrajpts=" << track.NumberTrajectoryPoints() << std::endl;

            std::vector<float> track_start(3,0);
            for (size_t i=0; i<3; i++)
                track_start[i] = track.LocationAtPoint(0)[i];

            //std::vector<float> track_dir(3,0);
            // int itrackpt = 3;
            // if ( itrackpt >= track.NumberTrajectoryPoints() )
            //     itrackpt = track.NumberTrajectoryPoints()-1;
            // float tracklen = 0.;
            // for (int i=0; i<3; i++) {
            //     track_dir[i] = track.LocationAtPoint(itrackpt)[i]-track_start[i];
            //     tracklen += track_dir[i]*track_dir[i];
            // }
            // tracklen = sqrt(tracklen);
            // if ( tracklen>0 ) {
            //     for (int i=0; i<3; i++)
            //         track_dir[i] /= tracklen;
            // }
	    std::vector<float> track_dir = nuvtx.track_dir_v.at(trackidx);

            std::vector<float> prong_dists 
                = getHitDistancesFromProngEnds( nuvtx.pos, track_start, track_dir, nearby_hits_v );
            for (int ipt=0; ipt<(int)prong_dists.size(); ipt++)
                dist_data.set( ipt, iprong, prong_dists[ipt] );
            iprong++;
        }

        for (auto const& showeridx : prim_shower_indices )  {

            const larlite::track& shower_trunk = nuvtx.shower_trunk_v.at(showeridx);

            std::vector<float> shower_start(3,0);
            std::vector<float> shower_dir(3,0);
            float trunklen = 0.;

            for (size_t i=0; i<3; i++) {
                shower_start[i] = shower_trunk.LocationAtPoint(0)[i];
                shower_dir[i]   = shower_trunk.LocationAtPoint(1)[i]-shower_start[i];
                trunklen += shower_dir[i]*shower_dir[i];
            }

            trunklen = sqrt(trunklen);
            if ( trunklen>1.0e-10 ) {
                for (int i=0; i<3; i++)
                    shower_dir[i] /= trunklen;
            }

            LARCV_INFO() << "get hit distances for shower" << std::endl;
            LARCV_INFO() << "  start=("
                        << shower_start[0] << "," 
                        << shower_start[1] << ","
                        << shower_start[2] << ")"
                        << std::endl;
            LARCV_INFO() << "  start_dir=("
                        << shower_dir[0] << "," 
                        << shower_dir[1] << ","
                        << shower_dir[2] << ")"
                        << std::endl;

            std::vector<float> prong_dists 
                = getHitDistancesFromProngEnds( nuvtx.pos, shower_start, shower_dir, nearby_hits_v );
            for (int ipt=0; ipt<(int)prong_dists.size(); ipt++)
                dist_data.set( ipt, iprong, prong_dists[ipt] );
            iprong++;
        }

        // we filled the distance array, now do assignments
        std::vector<int> number_added_to_prong_v( nprongs, 0 );
        for (int ipt=0; ipt<npts; ipt++) {
            auto& hit = nearby_hits_v.at(ipt);
            int min_prong_index = dist_data.get_closest_prong( ipt );
            float min_dist = dist_data.get( ipt, min_prong_index );
            if ( min_prong_index < (int)prim_track_indices.size() ) {
                if ( min_dist<_collection_radius_cm ) {
                    nuvtx.track_hitcluster_v.at( prim_track_indices[min_prong_index] ).push_back( hit );
                    number_added_to_prong_v[ min_prong_index ]++;
                }
            }
            else {
                if ( min_dist<_collection_radius_cm ) {
                    int min_showerprong_index = min_prong_index-(int)prim_track_indices.size();
                    nuvtx.shower_v.at( prim_shower_indices[min_showerprong_index]).push_back(hit);
                    number_added_to_prong_v[ min_prong_index ]++;
                }
            }
        }

        LARCV_INFO() << "Number of veto hits added to NuVertexCandidate primary prongs: " << std::endl;
        for (int iprong=0; iprong<nprongs; iprong++ ) {
            if ( iprong<(int)prim_track_indices.size() ) {
                LARCV_INFO() << "  track[" << prim_track_indices[iprong] << "]: " << number_added_to_prong_v[iprong] << std::endl;
            }
            else {
                int shower_index = iprong-(int)prim_track_indices.size();
                LARCV_INFO() << "  shower[" << prim_shower_indices[shower_index] << "]: " << number_added_to_prong_v[iprong] << std::endl;
            }
        }
    }

    std::vector<larlite::larflow3dhit> 
    NuVertexRestoreKPHits::gatherKPVetoedHitsNearVertex( larflow::reco::NuVertexCandidate& nuvtx,
                                                         const larlite::event_larflow3dhit& ev_kpvetoed,
                                                         const float collection_radius_cm  )
    {
        std::vector<larlite::larflow3dhit> nearby_hits;
        for ( auto& hit : ev_kpvetoed ) {
            float dist = 0.;
            for (int i=0; i<3; i++) {
                dist += (hit[i]-nuvtx.pos[i])*(hit[i]-nuvtx.pos[i]);
            }
            dist = sqrt(dist);
            if ( dist < collection_radius_cm ) {
                nearby_hits.push_back(hit);
            }
        }
        return nearby_hits;
    }

    std::vector<float> 
    NuVertexRestoreKPHits::getHitDistancesFromProngEnds( const std::vector<float>& vtxpos,
                                                         const std::vector<float>& prong_start, 
                                                         const std::vector<float>& prong_dir,
                                                         const std::vector<larlite::larflow3dhit>& nearby_kpvetoed_hits_v )
    {
        std::vector<float> prong_end(3,0);
        for (int i=0; i<3; i++) {
            prong_end[i] = prong_start[i] + 10.0*prong_dir[i];
        }

        std::vector<float> dist_v( nearby_kpvetoed_hits_v.size(), 9999.0 );

	float dist2vtx = 0.;
	for (int i=0; i<3; i++) {
	  dist2vtx += ( prong_start[i]-vtxpos[i] )*( prong_start[i]-vtxpos[i] );
	}
	dist2vtx = sqrt( dist2vtx );

	if ( dist2vtx>1.5*_collection_radius_cm ) {
	  // dont absorb for this cluster - its' too far from the vertex
	  return dist_v;
	}

	float min_s_hit = 1.0e9;
	float max_s_hit = 0.0;
	float s_vtx = larflow::recoutils::pointRayProjection( prong_start, prong_dir, vtxpos );
	s_vtx = fabs(s_vtx);
	
        for (int ipt=0; ipt<(int)nearby_kpvetoed_hits_v.size(); ipt++ ) {
            auto const& hit = nearby_kpvetoed_hits_v.at(ipt);
            std::vector<float> hitpos = { hit[0], hit[1], hit[2] };
            float d = larflow::recoutils::pointLineDistance3f( prong_start, prong_end, hitpos );
            float s_hit = larflow::recoutils::pointRayProjection( prong_start, prong_dir, hitpos );
            //std::cout << "[" << ipt << "] d=" << d << " s_hit=" << s_hit << " s_vtx=" << s_vtx << std::endl;
            // only update the distance if its closer than the vertex
	    s_hit = fabs(s_hit);
            if ( d<1.0 && (fabs(s_hit) < fabs(s_vtx)) ) {
                dist_v[ipt] =  d;
		if ( s_hit < min_s_hit )
		  min_s_hit = s_hit;
		if ( s_hit > max_s_hit )
		  max_s_hit = s_hit;
	    }
        }

	// we want to make sure we fill in prongs that actually are missing the trunk along the track
	if ( min_s_hit > _max_s_hit_gap_cm ) {
	  LARCV_INFO() << "gap between vtx and hits: " << min_s_hit << " > " << _max_s_hit_gap_cm << std::endl;
	  // absorb nothing: reject results and return no matches
	  return std::vector<float>( nearby_kpvetoed_hits_v.size(), 9999.0 );
	}

        return dist_v;
    }

}
}
