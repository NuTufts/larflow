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

	        if ( track.NumberTrajectoryPoints()>=2 ) {
	          std::vector<float> track_start(3,0);
	          for (size_t i=0; i<3; i++)
                    track_start[i] = track.LocationAtPoint(0)[i];
    
	          std::vector<float> track_dir = nuvtx.track_dir_v.at(trackidx);
              // for debug
	          //   std::cout << "track_dir=(" << track_dir[0] << ", " << track_dir[1] << "," << track_dir[2] << ")" << std::endl;
	          //   std::cout << "pos=(" << nuvtx.pos[0] << "," << nuvtx.pos[1] << "," << nuvtx.pos[2] << ")" << std::endl;
	          //   std::cout << "pos=(" << track_start[0] << "," << track_start[1] << "," << track_start[2] << ")" << std::endl;
	          //   std::cout << "nearby_hits_v.size()=" << nearby_hits_v.size() << std::endl;
	          std::vector<float> prong_dists;
	          try {
		        prong_dists
		          = getHitDistancesFromProngEnds( nuvtx.pos, track_start, track_dir, nearby_hits_v );
	          }
	          catch ( std::exception& e ) {
		        LARCV_ERROR() << e.what() << std::endl;
		        throw std::runtime_error("error calling: getHitDistancesFromProngEnds");
	          }
	          std::cout << "prong_dists.size()=" << prong_dists.size() << std::endl;
	          for (int ipt=0; ipt<(int)prong_dists.size(); ipt++)
                    dist_data.set( ipt, iprong, prong_dists[ipt] );
	          iprong++;
	        }
        }

        for (auto const& showeridx : prim_shower_indices )  {

            const larlite::track& shower_trunk = nuvtx.shower_trunk_v.at(showeridx);
            LARCV_INFO() << "shower[idx=" << showeridx << "] of nuvtx.shower_trunk_v.size()="  << nuvtx.shower_trunk_v.size() << std::endl;

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

        // get original number of hits: helps us know which hits are new
	    std::vector<int> original_prong_num_hits_v( nprongs, 0 );
        for (int ii=0; ii<nprongs; ii++ ) {
            if ( ii<(int)prim_track_indices.size() ) {
                original_prong_num_hits_v[ii] = nuvtx.track_hitcluster_v.at( prim_track_indices[ii] ).size();
            }
            else {
                int shower_index = ii-(int)prim_track_indices.size();
                original_prong_num_hits_v[ii] = nuvtx.shower_v.at( prim_shower_indices[shower_index] ).size();	    
            }
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
        for (int ii=0; ii<nprongs; ii++ ) {

	    // we need to extend certain objects
	    // for track, need to add first step to larlite::track
	    // for shower, need to mod first point
	  
            if ( ii<(int)prim_track_indices.size() ) {
		        int trackidx = prim_track_indices[ii];
                LARCV_INFO() << "  track[" << prim_track_indices[ii] << "]: " << number_added_to_prong_v[ii] << std::endl;
                if ( number_added_to_prong_v[ii]>0 ) {
                    larlite::track extended_track;
                    bool was_extended = _extendTrack( nuvtx.track_v.at(trackidx),
                                  nuvtx.track_dir_v.at(trackidx),
                                  nuvtx.track_hitcluster_v.at(trackidx),
                                  extended_track,
                                  original_prong_num_hits_v[ii], ii );
                    if (was_extended) {
                        std::swap( nuvtx.track_v.at( trackidx ), extended_track );
                    }
                }
            }
            else {
                int shower_index = ii-(int)prim_track_indices.size();
                LARCV_INFO() << "  shower[" << prim_shower_indices[shower_index] << "]: " << number_added_to_prong_v[ii] << std::endl;
                if ( number_added_to_prong_v[ii]>0 )  {
                    larlite::track extended_track;
                    bool was_extended = _extendShower( nuvtx.shower_v.at(shower_index), // hits
                        nuvtx.shower_trunk_v.at(shower_index), // trunk dir inform of larlite::track (why)
                        extended_track,
                        original_prong_num_hits_v[ii],  ii);
                    if (was_extended) {
                        std::swap( nuvtx.shower_trunk_v.at(shower_index), extended_track );
                    }
                }
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

    bool NuVertexRestoreKPHits::_extendShower( const larlite::larflowcluster& hitcluster, 
                                               const larlite::track& trunk, 
                                               larlite::track& extended_trunk,
                                               int num_orig, int prongidx )
    {
        // loop over new hits. find the firstest projection s
        TVector3 start = trunk.LocationAtPoint(0);
        TVector3 end = trunk.LocationAtPoint(1);
        TVector3 diff = end-start;
        double mag = diff.Mag();
        if ( mag>1.0e-10 ) {
            for (int i=0; i<3; i++)
                diff[i] /= mag;
        }

        std::vector<float> trunk_dir = { (float)diff[0], (float)diff[1], (float)diff[2] };
        std::vector<float> fstart = { (float)start[0], (float)start[1], (float)start[2] };
        std::vector<float> fend   = { (float)end[0], (float)end[1], (float)end[2] };
        float min_s = 1.0e9;
        for (int ihit=num_orig; ihit<(int)hitcluster.size(); ihit++ ) {
            auto& hit = hitcluster.at(ihit);
            std::vector<float> testpt = { (float)hit[0], (float)hit[1], (float)hit[2] };
            float s = larflow::recoutils::pointRayProjection3f( fstart, trunk_dir, testpt );
            // we expect s to be negative since we added points back to the vertex
            if ( s < min_s )
                min_s = s;
        }
        std::vector<float> fnewstart(3,0);
        if ( min_s<0 ) {
            for (int v=0; v<3; v++) {
                fnewstart[v] = fstart[v] + min_s*trunk_dir[v];
            }
            // replace
            extended_trunk.clear_data();
            extended_trunk.set_track_id( prongidx );
            TVector3 tv3start( fnewstart[0], fnewstart[1], fnewstart[2] );
            TVector3 tv3end  = end;
            extended_trunk.add_vertex( tv3start );
            extended_trunk.add_direction( diff );
            extended_trunk.add_vertex( tv3end );
            extended_trunk.add_direction( diff );
            return true;
        }

        return false;
    }


    bool NuVertexRestoreKPHits::_extendTrack( const larlite::track& orig,
                                            const std::vector<float>& track_dir,
                                            const larlite::larflowcluster& hitcluster,
                                            larlite::track& extended_track,
                                            int num_orig_hits, int trackidx )
    {
        TVector3 start = orig.LocationAtPoint(0);
        TVector3 vdir( track_dir[0], track_dir[1], track_dir[2] );
        std::vector<float> fstart = { (float)start[0], (float)start[1], (float)start[2] };
        float min_s = 1.0e9;
        for (int ihit=num_orig_hits; ihit<(int)hitcluster.size(); ihit++ ) {
            auto& hit = hitcluster.at(ihit);
            std::vector<float> testpt = { (float)hit[0], (float)hit[1], (float)hit[2] };
            float s = larflow::recoutils::pointRayProjection3f( fstart, track_dir, testpt );
            // we expect s to be negative since we added points back to the vertex
            if ( s < min_s )
                min_s = s;
        }

        if ( min_s<0 ) {
            // rebuild track. yay.
            TVector3 newstart = start + min_s*vdir;
            extended_track.clear_data();
            extended_track.set_track_id( trackidx );
            int npts = orig.NumberTrajectoryPoints()+1;
            extended_track.reserve( npts );
            extended_track.add_vertex( newstart );
            extended_track.add_direction( vdir );
            for (int i=1; i<npts; i++) {
                extended_track.add_vertex( orig.LocationAtPoint(i-1) );
                extended_track.add_direction( orig.DirectionAtPoint(i-1));
            }
            return true;
        }
    
        return false;
    }

}
}
