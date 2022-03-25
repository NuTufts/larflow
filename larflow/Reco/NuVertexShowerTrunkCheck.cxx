#include "NuVertexShowerTrunkCheck.h"

#include "larlite/LArUtil/Geometry.h"
#include "larlite/LArUtil/LArProperties.h"

#include "larcv/core/DataFormat/EventImage2D.h"

#include "geofuncs.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  /**
   * @brief check the candidate prongs
   * 
   * absorb tracks that are just a core within shower
   * 
   * @param[inout] nuvtx Neutrino candidate vertex to modify.
   */
  void NuVertexShowerTrunkCheck::checkNuCandidateProngs( larflow::reco::NuVertexCandidate& nuvtx )
  {

    LARCV_DEBUG() << "start" << std::endl;
    
    std::vector<int> trunk_absorbed( nuvtx.track_v.size(), 0 ); // mark if we absorbed the track prong
    int num_absorbed = 0;

    TVector3 vtx( nuvtx.pos[0], nuvtx.pos[1], nuvtx.pos[2] );
    LARCV_DEBUG() << "--- checking for vertex (" << vtx[0] << "," << vtx[1] << "," << vtx[2] << ") ------" << std::endl;

    // loop over showers
    for (int j=0; j<(int)nuvtx.shower_v.size(); j++) {
      
      auto& shower = nuvtx.shower_v[j];
      auto& shower_trunk = nuvtx.shower_trunk_v[j];
      auto& shower_pca   = nuvtx.shower_pcaxis_v[j];

      LARCV_DEBUG() << "shower[" << j << "] check for tracks to absorb" << std::endl;
      LARCV_DEBUG() << "shower[" << j << "] trunk start=("
		    << shower_trunk.LocationAtPoint(0)[0] << ","
		    << shower_trunk.LocationAtPoint(0)[1] << ","
		    << shower_trunk.LocationAtPoint(0)[2] << ")"	
		    << std::endl;
      LARCV_DEBUG() << "shower[" << j << "] pca start=("      
		    << shower_pca.getEigenVectors()[3][0] << ","
		    << shower_pca.getEigenVectors()[3][1] << ","
		    << shower_pca.getEigenVectors()[3][2] << ")"
		    << std::endl;
      LARCV_DEBUG() << "shower[" << j << "] pca end=("      
		    << shower_pca.getEigenVectors()[4][0] << ","
		    << shower_pca.getEigenVectors()[4][1] << ","
		    << shower_pca.getEigenVectors()[4][2] << ")"
		    << std::endl;
    
      // loop over tracks, see if we should absorb it into the shower
      for (int i=0; i<(int)nuvtx.track_v.size(); i++) {

        if ( trunk_absorbed[i]==1 ) {
	  LARCV_DEBUG() << "  already absorbing track[" << i << "]" << std::endl;
          continue;
	}
              
        auto& track = nuvtx.track_v[i];
        auto& track_hits = nuvtx.track_hitcluster_v[i];
        
        // decide to absorb or not
        // (1) cosine direction
        // (2) max distance from vertex
        // (3) min distance from vertex
        
        // get track length
        float tracklen = 0.;
        float maxvtxdist = 0;
        float minvtxdist = 1e9;
        int npts = (int)track.NumberTrajectoryPoints();
        for (int ipt=0; ipt<(int)track.NumberTrajectoryPoints()-1;ipt++) {
          tracklen += ( track.LocationAtPoint(ipt)-track.LocationAtPoint(ipt+1) ).Mag();
          float ptdist = (track.LocationAtPoint(ipt)-vtx).Mag();
          if ( ptdist > maxvtxdist ) maxvtxdist = ptdist;
          if ( ptdist < minvtxdist ) minvtxdist = ptdist;
        }
        TVector3 start_to_end_dir = track.LocationAtPoint( npts-1 )-track.LocationAtPoint(0);
        float s2edist = start_to_end_dir.Mag();
        for (int v=0; v<3; v++)
          start_to_end_dir[v] /= s2edist;

        float cos_track = 0.;
        for (int v=0; v<3; v++) {
          cos_track += start_to_end_dir[v]*shower_trunk.DirectionAtPoint(0)[v];
        }
        
        cos_track = fabs(cos_track);
        float ang_track = acos(cos_track)*180.0/3.14159;

        LARCV_DEBUG() << "  track[" << i << "] min=" << minvtxdist << " max=" << maxvtxdist
                      << " ang=" << ang_track
                      << " s2edist=" << s2edist
                      << " dirlen=" << shower_trunk.DirectionAtPoint(0).Mag()
                      << std::endl;
	
        if ( minvtxdist<0.3 && (ang_track>30.0 || maxvtxdist<1.5) ) {
	  LARCV_DEBUG() << "  close to vtx or too wide an angle" << std::endl;
          continue;
	}

            
        float frac_path = 0.;
        float frac_core = 0.;
        bool within_shower = isTrackTrunkOfShower( nuvtx.pos,
                                                   track,
                                                   track_hits,
                                                   shower_trunk,
                                                   shower,
                                                   shower_pca,
                                                   frac_path,
                                                   frac_core );

        LARCV_DEBUG() << "  track[" << i << "] frac_core=" << frac_core << "  within_shower=" << within_shower << std::endl;
	if ( within_shower ) {
	  if ( frac_path<0.95 && ang_track>15.0 ) {
	    LARCV_DEBUG() << "angle is wide too wide for partial path coverage" << std::endl;
	    continue;
	  }
	}
	
        if ( within_shower ) {
          trunk_absorbed[i] = 1;
          num_absorbed++;
          if ( frac_core>frac_path ) {
            _addTrackToCore( nuvtx.pos,
                             track,
                             track_hits,
                             shower_trunk,
                             shower,
                             shower_pca );
	  }
          else {
            _addTrackAsNewTrunk( nuvtx.pos,
                                 track,
                                 track_hits,
                                 shower_trunk,
                                 shower,
                                 shower_pca );
          }
        }
      }//end of track loop
    }//end of shower loop

    if ( num_absorbed>0 ) {
      LARCV_DEBUG() << "Zero out " << num_absorbed << " tracks" << std::endl;
      // for (int ii=0; ii<(int)nuvtx.track_v.size(); ii ) {
      //   if ( trunk_absorbed[ii]==0 ) {
      //     keep_track_v.emplace_back( std::move(nuvtx.track_v[ii]) );
      //     keep_cluster_v.emplace_back( std::move(nuvtx.track_hitcluster_v[ii]) );
      //   }
      // }
      
      // resort track vector, removing absorbed tracks
      std::vector< larlite::track > keep_track_v;
      std::vector< larlite::larflowcluster > keep_cluster_v;
      std::vector<float>           keep_track_len_v;
      std::vector< std::vector<float> > keep_track_dir_v;
      keep_track_v.reserve(   (int)nuvtx.track_v.size() - num_absorbed );
      keep_cluster_v.reserve( (int)nuvtx.track_hitcluster_v.size() - num_absorbed );
      keep_track_len_v.reserve( nuvtx.track_len_v.size() );
      keep_track_dir_v.reserve( nuvtx.track_dir_v.size() );
      
      for (int ii=0; ii<(int)nuvtx.track_v.size(); ii++ ) {
        if ( trunk_absorbed[ii]==0 ) {
          keep_track_v.emplace_back( std::move(nuvtx.track_v[ii]) );
          keep_cluster_v.emplace_back( std::move(nuvtx.track_hitcluster_v[ii]) );
	  if ( ii<(int)nuvtx.track_len_v.size() )
	    keep_track_len_v.emplace_back( std::move(nuvtx.track_len_v[ii]) );
	  if ( ii<(int)nuvtx.track_dir_v.size() )
	    keep_track_dir_v.emplace_back( std::move(nuvtx.track_dir_v[ii]) );
        }
      }
      std::swap( nuvtx.track_v, keep_track_v );
      std::swap( nuvtx.track_hitcluster_v, keep_cluster_v );
      std::swap( nuvtx.track_len_v, keep_track_len_v );
      std::swap( nuvtx.track_dir_v, keep_track_dir_v );
      
    }


  }

  void NuVertexShowerTrunkCheck::checkNuCandidateProngsForMissingCharge( larflow::reco::NuVertexCandidate& nuvtx,
                                                                         larcv::IOManager& iolcv,
                                                                         larlite::storage_manager& ioll )
  {

    larcv::EventImage2D* ev_img
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wire");
    const std::vector<larcv::Image2D>& adc_v = ev_img->as_vector();
        
    // now run missing shower trunk charge algo
    for ( size_t ishower=0; ishower<nuvtx.shower_v.size(); ishower++ ) {
      LARCV_DEBUG() << "Check Shower[" << ishower << "] for missing trunk hits" << std::endl;
      auto& shower = nuvtx.shower_v[ishower];
      auto& shower_trunk = nuvtx.shower_trunk_v[ishower];
      auto& shower_pca   = nuvtx.shower_pcaxis_v[ishower];
      LARCV_DEBUG() << "trunk length: " << (shower_trunk.LocationAtPoint(1)-shower_trunk.LocationAtPoint(0)).Mag() << " cm" << std::endl;
      float max_gapdist = 10.;
      larlite::larflowcluster add_hits_v =
        makeMissingTrunkHits( nuvtx, adc_v,
                              shower_trunk,
                              shower,
                              shower_pca, max_gapdist );
      
      if ( add_hits_v.size()>3 && max_gapdist<10.0 ) {
        // merge the shower trunk
        _mergeNewTrunkHits( nuvtx.pos,
                            add_hits_v,
                            shower_trunk,
                            shower,
                            shower_pca);

	// check objects
	LARCV_DEBUG() << "After merger: " << std::endl;
	LARCV_DEBUG() << "Trunk: npts=" << shower_trunk.NumberTrajectoryPoints() << std::endl;
	LARCV_DEBUG() << "Trunk: ("
		      << shower_trunk.LocationAtPoint(0)[0] << "," << shower_trunk.LocationAtPoint(0)[1] << "," << shower_trunk.LocationAtPoint(0)[2] << ") to ("
		      << shower_trunk.LocationAtPoint(1)[0] << "," << shower_trunk.LocationAtPoint(1)[1] << "," << shower_trunk.LocationAtPoint(1)[2] << ")"
		      << std::endl;
	LARCV_DEBUG() << "PCA num eigenvectors: " << shower_pca.getEigenVectors().size() << std::endl;
      }
    }
    
  }    

  /**
   * @brief asks what fraction of track is near trunk of shower
   *
   */
  bool NuVertexShowerTrunkCheck::isTrackTrunkOfShower( const std::vector<float>& vtxpos,
                                                       larlite::track& track,
                                                       larlite::larflowcluster& track_hitcluster,
                                                       larlite::track&          shower_trunk,
                                                       larlite::larflowcluster& shower_hitcluster,
                                                       larlite::pcaxis& shower_pcaxis,
                                                       float& frac_path, float& frac_core )
  {

    frac_path = 0.;
    frac_core = 0.;

    // we define a line segment between the vertex and shower
    TVector3 vstart; // vertex position
    for (int i=0; i<3; i++)
      vstart[i] = vtxpos[i];

    // closest point to the shower
    float dist[2] = { 0, 0 };
    for (int iend=0; iend<2; iend++) {
      dist[iend] = (shower_trunk.LocationAtPoint(iend)-vstart).Mag();
    }

    std::vector<float> shrstart(3,0); // shower trunk start
    std::vector<float> shrend(3,0);   // shower trunk end
    if ( dist[0]<dist[1] ) {
      for (int i=0; i<3; i++) {
        shrstart[i] = shower_trunk.LocationAtPoint(0)[i];
        shrend[i]   = shower_trunk.LocationAtPoint(1)[i];
      }
    }
    else {
      for (int i=0; i<3; i++) {
        shrstart[i] = shower_trunk.LocationAtPoint(1)[i];
        shrend[i]   = shower_trunk.LocationAtPoint(0)[i];        
      }
    }


    std::vector<float> showerdir(3,0);
    std::vector<float> pathdir(3,0);
    std::vector<float> pcastart(3,0);
    std::vector<float> pcaend(3,0);    
    float len = 0.;
    float pcalen = 0.;
    float pcadx = 0.;
    float pathlen = 0.;
    for (int i=0; i<3; i++) {
      showerdir[i] = shrend[i]-shrstart[i];
      len += showerdir[i]*showerdir[i];
      pcadx = ( shower_pcaxis.getEigenVectors()[3][i] - shower_pcaxis.getEigenVectors()[4][i] );
      pcalen += pcadx*pcadx;

      pathdir[i] = shrstart[i]-vtxpos[i];
      pathlen += pathdir[i]*pathdir[i];
      pcastart[i] = shower_pcaxis.getEigenVectors()[3][i];
      pcaend[i]   = shower_pcaxis.getEigenVectors()[4][i];
    }
    len = sqrt(len);
    pcalen = sqrt(pcalen);
    pathlen = sqrt(pathlen);
    for (int i=0; i<3; i++) {
      if (len>0)
	showerdir[i] /= len;
      if (pathlen>0)
	pathdir[i] /= pathlen;
    }

    int nhits_within_startpath = 0.;
    int nhits_startpath = 0.;    
    int nhits_within_shower = 0.;
    std::vector<int> within(track_hitcluster.size(),0);
    for (size_t ihit=0; ihit<track_hitcluster.size(); ihit++) {
      auto const& hit = track_hitcluster[ihit];
      //within shower
      float r1 = larflow::reco::pointLineDistance3f( shrstart, shrend, hit );
      float r2 = larflow::reco::pointLineDistance3f( pcastart, pcaend, hit );
      float r = (r1<r2 ) ? r1 : r2;
      float s_start = larflow::reco::pointRayProjection3f( shrstart, showerdir, hit );
      float r_cone = 0.466*s_start;
      if ( r_cone<2.0 )
	r_cone = 2.0;
      if ( r<r_cone && s_start>-0.5 && s_start<pcalen ) {
        nhits_within_shower++;
	within[ihit] = 1;
      }

      // along path
      r = larflow::reco::pointLineDistance3f( vtxpos, shrstart, hit );
      s_start = larflow::reco::pointRayProjection3f( vtxpos, pathdir, hit );
      if ( s_start>-0.5 && s_start<pathlen ) {
	if (r<3.5) {
	  nhits_within_startpath++;
	  within[ihit] += 2;	  
	}
	nhits_startpath++;
      }
      
    }

    int nclaimed = 0;
    for (int ihit=0; ihit<(int)within.size(); ihit++) {
      if ( within[ihit]>0 )
	nclaimed++;
    }
    float frac_within = 0;
    if ( within.size()>0 )
      frac_within = (float)nclaimed/(float)within.size();

    frac_path = 0.;
    if ( nhits_startpath>0 )
      frac_path = (float)nhits_within_startpath/(float)nhits_startpath;
    frac_core = 0.;
    if ( track_hitcluster.size()>0 )
      frac_core = (float)nhits_within_shower/(float)track_hitcluster.size();    
    LARCV_DEBUG() << " fraction of track hits within cylinder of vtx-to-shower start path: " << frac_path << std::endl;
    LARCV_DEBUG() << " fraction of track hits within cylinder of shower core: " << frac_core << std::endl;
    LARCV_DEBUG() << " fraction of track hits within core or path (frac_within): " << frac_within << std::endl;        

    if ( frac_within>0.9 )
      return true;
    
    return false;
  }

  /**
   * @brief supplement missing shower trunk hits
   */
  larlite::larflowcluster
  NuVertexShowerTrunkCheck::makeMissingTrunkHits( larflow::reco::NuVertexCandidate& nuvtx,
                                                  const std::vector<larcv::Image2D>& adc_v,
                                                  larlite::track& shower_trunk,                                                  
                                                  larlite::larflowcluster& shower_hitcluster,
                                                  larlite::pcaxis& shower_pcaxis,
                                                  float& max_gapdist )
  {

    const std::vector<float>& vtxpos = nuvtx.pos;
    max_gapdist = -1;
    
    larlite::larflowcluster added_hit_v;

    // we define a line segment between the vertex and shower
    TVector3 vstart; // vertex position
    for (int i=0; i<3; i++)
      vstart[i] = vtxpos[i];

    // closest point to the shower
    float dist[2] = { 0, 0 };
    for (int iend=0; iend<2; iend++) {
      dist[iend] = (shower_trunk.LocationAtPoint(iend)-vstart).Mag();
    }

    // define shower trunk start    
    std::vector<float> shrstart(3,0);
    std::vector<float> shrend(3,0);
    if ( dist[0]<dist[1] ) {
      for (int i=0; i<3; i++) {
        shrstart[i] = shower_trunk.LocationAtPoint(0)[i];
        shrend[i]   = shower_trunk.LocationAtPoint(1)[i];
      }
    }
    else {
      for (int i=0; i<3; i++) {
        shrstart[i] = shower_trunk.LocationAtPoint(1)[i];
        shrend[i]   = shower_trunk.LocationAtPoint(0)[i];
      }
    }
    LARCV_DEBUG() << "search between "
                  << " start=(" << vstart[0] << "," << vstart[1] << "," << vstart[2] << ")"
                  << " end=(" << shrstart[0] << "," << shrstart[1] << "," << shrstart[2] << ")"
                  << std::endl;

    // define shower dir
    std::vector<double> showerdir(3,0);
    float len = 0.;
    for (int i=0; i<3; i++) {
      showerdir[i] = shrstart[i]-vtxpos[i];
      len += showerdir[i]*showerdir[i];
    }
    len = sqrt(len);
    for (int i=0; i<3; i++)
      showerdir[i] /= len;

    // if too far, don't bother
    if ( len>20.0 ) {
      LARCV_DEBUG() << "vertex-shower start gap is " << len << " which is too big. don't add hits to this gap" << std::endl;
      return added_hit_v;
    }

    // step through, save unique wire combinations
    const float max_stepsize = 0.3;
    int nsteps = len/max_stepsize+1;
    double stepsize = (double)len/(double)nsteps;

    LARCV_DEBUG() << "num steps=" << nsteps << " stepsize=" << stepsize << std::endl;

    auto const& meta = adc_v.front().meta();    
    std::set< std::vector<int> > past_hits;

    // load up past hits from the current shower and track
    std::vector<const larlite::larflowcluster*> vtx_cluster_v;    
    for ( auto const& vtxshower : nuvtx.shower_v )
      vtx_cluster_v.push_back( &vtxshower );
    for ( auto const& vtxtrack : nuvtx.track_hitcluster_v )
      vtx_cluster_v.push_back( &vtxtrack );

    for ( auto& pcluster : vtx_cluster_v ) {
      for (auto const& lfhit : *pcluster ) {
        if ( lfhit.tick<=meta.min_y() || lfhit.tick>=meta.max_y() )
          continue;
        std::vector<int> hitcoord(4);
        for (int p=0; p<3; p++)
          hitcoord[p] = lfhit.targetwire[p];      
        hitcoord[3] = meta.row(lfhit.tick,__FILE__,__LINE__);
        past_hits.insert(hitcoord);
      }
    }//end of cluster loop

    TVector3 last_addition( vtxpos[0], vtxpos[1], vtxpos[2] );
    max_gapdist = -1.0;
    
    for (int istep=0; istep<=nsteps; istep++) {
      
      std::vector<double> pos(3,0);
      for (int i=0; i<3; i++)
        pos[i] = (double)vtxpos[i] + (istep*stepsize)*showerdir[i];


      float tick = pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5 + 3200.0;
      if ( tick<=meta.min_y() || tick>=meta.max_y() )
        continue;
      int row = meta.row(tick,__FILE__,__LINE__);

      int nplanes_w_charge = 0;
      std::vector<int> imgcoord(4,0);
      for (int p=0; p<3; p++) {
        imgcoord[p] = (int)larutil::Geometry::GetME()->WireCoordinate( pos, p );
        int npix = 0;
        for (int dc=-2; dc<=2; dc++) {
          if ( npix>0 )
            break;
          int col = imgcoord[p] + dc;
          if ( col<0 || col>=(int)adc_v[p].meta().cols() )
            continue;
          float pixval = adc_v[p].pixel( row, col, __FILE__, __LINE__ );
          if ( pixval>10.0 )
            npix++;
        }
        if (npix>0)
          nplanes_w_charge++;
      }//end of plane

      bool added_pt = false;
      if ( nplanes_w_charge>=2 ) {
      
        imgcoord[3] = row;
        if ( past_hits.find( imgcoord )==past_hits.end() ) {
          // create a hit
          larlite::larflow3dhit lfhit;
          lfhit.resize(19,0);
          for (int v=0; v<3; v++)
            lfhit[v] = pos[v];
          lfhit.tick = tick;
          lfhit.targetwire.resize(3,0);
          for (int p=0; p<3; p++)
            lfhit.targetwire[p] = imgcoord[p];
          // fake larmatch score
          lfhit[9] = 1.0; 
          lfhit.track_score = 1.0;
          // ssnet score
          lfhit.renormed_shower_score = 1.0;
          added_hit_v.emplace_back( std::move(lfhit) );
          past_hits.insert( imgcoord );
          added_pt = true;
        }// if added
      }//end of if nplanes with charge        
      
      if ( added_pt || istep==nsteps ) {
        // added hit, calc gap dist since last addition.
        // update last position
        TVector3 addedpos( pos[0], pos[1], pos[2] );
        float gapdist = (addedpos-last_addition).Mag();

        // wait to update gap dist after first 1 cm from vertex?
        if ( istep*stepsize>0.0 && (max_gapdist<gapdist || max_gapdist<0) )
          max_gapdist = gapdist;
      }//end of if added, then update last added pos and calc gapdist

      
    }//end of step loop

    LARCV_DEBUG() << "proposed " << added_hit_v.size() << " hits with max gap dist of " << max_gapdist << " cm" << std::endl;
    
    return added_hit_v;
  }

  void NuVertexShowerTrunkCheck::_addTrackAsNewTrunk( const std::vector<float>& vtxpos,
                                                      larlite::track& track,
                                                      larlite::larflowcluster& track_hitcluster,
                                                      larlite::track&          shower_trunk,
                                                      larlite::larflowcluster& shower_hitcluster,
                                                      larlite::pcaxis& shower_pcaxis )
  {

    LARCV_DEBUG() << "start" << std::endl;
    
    // we will need to make a cluster in order to update the pca
    // then replace the shower trunk with start and end of track

    larflow::reco::cluster_t cluster;
    cluster.points_v.reserve( track_hitcluster.size()+shower_hitcluster.size() );
    cluster.hitidx_v.reserve( track_hitcluster.size()+shower_hitcluster.size() );

    int iidx = 0;
    for ( auto& hit : shower_hitcluster ) {
      std::vector<float> pt = { hit[0], hit[1], hit[2] };
      cluster.points_v.push_back( pt );
      cluster.hitidx_v.push_back(iidx);
      iidx++;
    }
    iidx = 0;
    for ( auto& hit : track_hitcluster ) {
      std::vector<float> pt = { hit[0], hit[1], hit[2] };
      cluster.points_v.push_back( pt );
      cluster.hitidx_v.push_back( -1-iidx );
      iidx++;
    }

    LARCV_DEBUG() << "stored " << cluster.points_v.size() << " combined hits. run pca." << std::endl;
    larflow::reco::cluster_pca( cluster );

    float dist[2] = { 0, 0 };
    for (int iend=0; iend<2; iend++) {
      for (int i=0; i<3; i++) {
        dist[iend] += ( cluster.pca_ends_v[iend][i] - vtxpos[i] )*( cluster.pca_ends_v[iend][i] - vtxpos[i] );
      }
    }

    int istart=0;
    int iend = (int)cluster.points_v.size();
    int dii = 1;
    if ( dist[1]<dist[0] ) {
      istart = (int)cluster.points_v.size() - 1;
      iend = 0;
      dii = -1;
    }

    larlite::larflowcluster lfcluster;
    lfcluster.reserve( cluster.points_v.size() );

    LARCV_DEBUG() << "start point transfer: start=" << istart << " iend=" << iend << " dii=" << dii << std::endl;
    
    for (int ii=istart; ii!=iend; ii += dii ) {
      int iidx = cluster.ordered_idx_v[ii];
      int ipt  = cluster.hitidx_v[iidx];      
      // LARCV_DEBUG() << "  [" << ii << "] iidx=" << iidx << " of " << cluster.hitidx_v.size()
      //               << " ipt="
      //               << ipt
      //               << " (of " << shower_hitcluster.size() << " or " << track_hitcluster.size() << ")" 
      //               << std::endl;
      if ( ipt>=0 ) {
        lfcluster.push_back( shower_hitcluster[ipt] );
      }
      else {
        ipt += 1; // remove offset
        ipt *= -1; // make positive
        lfcluster.push_back( track_hitcluster[ipt] );
      }
    }//end of loop over pca order

    LARCV_DEBUG() << "make new pca" << std::endl;
    larlite::pcaxis new_pca = larflow::reco::cluster_make_pcaxis( cluster );
    
    LARCV_DEBUG() << "make new trunk with " << cluster.points_v.size() << " hits" << std::endl;    
    larlite::track new_trunk = larflow::reco::cluster_make_trunk( cluster, vtxpos );
    LARCV_DEBUG() << "new trunk start: ("
		  << track.LocationAtPoint(0)[0] << ","
		  << track.LocationAtPoint(0)[1] << ","
		  << track.LocationAtPoint(0)[2] << ")"
		  << std::endl;
    LARCV_DEBUG() << "new trunk end: ("
		  << track.LocationAtPoint(1)[0] << ","
		  << track.LocationAtPoint(1)[1] << ","
		  << track.LocationAtPoint(1)[2] << ")"
		  << std::endl;

    LARCV_DEBUG() << "swap new shower objects" << std::endl;        
    std::swap( shower_hitcluster, lfcluster );
    std::swap( shower_trunk, new_trunk );
    std::swap( shower_pcaxis, new_pca );
    
  }

  void NuVertexShowerTrunkCheck::_addTrackToCore(  const std::vector<float>& vtxpos,
                                                   larlite::track& track,
                                                   larlite::larflowcluster& track_hitcluster,
                                                   larlite::track&          shower_trunk,
                                                   larlite::larflowcluster& shower_hitcluster,
                                                   larlite::pcaxis& shower_pcaxis )
  {

    LARCV_DEBUG() << "start" << std::endl;

    larflow::reco::cluster_t cluster;
    cluster.points_v.reserve( track_hitcluster.size()+shower_hitcluster.size() );
    cluster.hitidx_v.reserve( track_hitcluster.size()+shower_hitcluster.size() );

    int iidx = 0;
    for ( auto& hit : shower_hitcluster ) {
      std::vector<float> pt = { hit[0], hit[1], hit[2] };
      cluster.points_v.push_back( pt );
      cluster.hitidx_v.push_back(iidx);
      iidx++;
    }
    iidx = 0;
    for ( auto& hit : track_hitcluster ) {
      std::vector<float> pt = { hit[0], hit[1], hit[2] };
      cluster.points_v.push_back( pt );
      cluster.hitidx_v.push_back( -1-iidx );
      iidx++;
    }

    larflow::reco::cluster_pca( cluster );

    float dist[2] = { 0, 0 };
    for (int iend=0; iend<2; iend++) {
      for (int i=0; i<3; i++) {
        dist[iend] += ( cluster.pca_ends_v[iend][i] - vtxpos[i] )*( cluster.pca_ends_v[iend][i] - vtxpos[i] );
      }
    }        

    int istart=0;
    int iend = (int)cluster.points_v.size();
    int dii = 1;
    if ( dist[1]<dist[0] ) {
      istart = (int)cluster.points_v.size() - 1;
      iend = 0;
      dii = -1;
    }
    
    larlite::larflowcluster lfcluster;
    lfcluster.reserve( cluster.points_v.size() );

    std::vector< std::vector<float> > newtrunk_hits_v;
    newtrunk_hits_v.reserve( 200 );
    for (int ii=istart; ii!=iend; ii += dii ) {
      int iidx = cluster.ordered_idx_v[ii];
      int ipt  = cluster.hitidx_v[iidx];
      // LARCV_DEBUG() << "  [" << ii << "] iidx=" << iidx << " of " << cluster.hitidx_v.size()
      //               << " ipt="
      //               << ipt
      //               << " (of " << shower_hitcluster.size() << " or " << track_hitcluster.size() << ")" 
      //               << std::endl;
      if ( ipt>=0 ) {
        lfcluster.push_back( shower_hitcluster[ipt] );
      }
      else {
        ipt++; // remove offset
        ipt *= -1; // make positive
        lfcluster.push_back( track_hitcluster[ipt] );
      }

      if ( fabs(cluster.pca_proj_v[istart]-cluster.pca_proj_v[ii]) < 10.0 ) {
	std::vector<float> pt = cluster.points_v.at(iidx);
	newtrunk_hits_v.push_back(pt);	  
      }
    }//end of loop over pca order

    // make new pca axis
    larlite::pcaxis new_pca = larflow::reco::cluster_make_pcaxis( cluster );
    
    larlite::track shower_newtrunk = larflow::reco::cluster_make_trunk( cluster, vtxpos );
    
    std::swap( shower_hitcluster, lfcluster );
    std::swap( shower_pcaxis, new_pca );
    std::swap( shower_trunk, shower_newtrunk );
    
  }
  
  void NuVertexShowerTrunkCheck::_mergeNewTrunkHits( const std::vector<float>& vtxpos,
                                                     const larlite::larflowcluster& newhits,
                                                     larlite::track& shower_trunk,                                                  
                                                     larlite::larflowcluster& shower_hitcluster,
                                                     larlite::pcaxis& shower_pcaxis )
  {

    // we have to start by gathering all the hits
    // we also need to define a new trunk
    // we define it as the pca of the line from the vertex to the other end of the shower_trunk line
    
    larlite::larflowcluster combined;
    larlite::larflowcluster new_trunk;
    combined.reserve( newhits.size()+shower_hitcluster.size() );

    const float max_dist2 =  100.0;

    // need original shower start, using trunk line ends
    float shower_end_dist[2] = {0};
    int npts = (int)shower_trunk.NumberTrajectoryPoints();
    for (int i=0; i<3; i++) {
      shower_end_dist[0] += ( shower_trunk.LocationAtPoint(0)[i] - vtxpos[i] )*( shower_trunk.LocationAtPoint(0)[i] - vtxpos[i] );
      shower_end_dist[1] += ( shower_trunk.LocationAtPoint(npts-1)[i] - vtxpos[i] )*( shower_trunk.LocationAtPoint(npts-1)[i] - vtxpos[i] );
    }
    int shower_startpt_index = ( shower_end_dist[0]<shower_end_dist[1] ) ? 0 : npts-1;
    int shower_endpt_index   = ( shower_end_dist[0]<shower_end_dist[1] ) ? npts-1 : 0;
    TVector3 tvtx( vtxpos[0], vtxpos[1], vtxpos[2] );
    TVector3 shower_startpt = shower_trunk.LocationAtPoint( shower_startpt_index );
    TVector3 shower_endpt   = shower_trunk.LocationAtPoint( shower_endpt_index );
    TVector3 newtrunk_dir = (shower_endpt - tvtx);
    std::vector< float > fend(3,0);
    std::vector< float > fdir(3,0);    
    float trunk_len = newtrunk_dir.Mag();
    if ( trunk_len>0 ) {
      trunk_len = sqrt(trunk_len);
      for (int i=0; i<3; i++) {
        newtrunk_dir[i] /= trunk_len;
        fend[i] = shower_endpt[i];
        fdir[i] = newtrunk_dir[i];
      }
    }
    else {
      // this is weird, do nothing
      return;
    }

    // absorb hits for combined and for trunk
    // use hits from original cluster and from generated points
    for ( auto& hit : newhits ) {
      std::vector<float> fhit = { hit[0], hit[1], hit[2] };
      float r = larflow::reco::pointLineDistance3f( vtxpos, fend, fhit );
      float s = larflow::reco::pointRayProjection3f( vtxpos, fdir, fhit );

      if ( r<1.5 && s>0.0 && s<10.0 ) {
        new_trunk.push_back( hit );
      }
      combined.emplace_back( std::move(hit) );      
    }

    for ( auto& hit : shower_hitcluster ) {
      std::vector<float> fhit = { hit[0], hit[1], hit[2] };
      float r = larflow::reco::pointLineDistance3f( vtxpos, fend, fhit );
      float s = larflow::reco::pointRayProjection3f( vtxpos, fdir, fhit );
      if ( r<1.5 && s>0.0 && s<10.0 ) {
        new_trunk.push_back( hit );
      }
      combined.emplace_back( std::move(hit) );      
    }
    
    // cluster: will calc pca for us
    larflow::reco::cluster_t cluster_all   = larflow::reco::cluster_from_larflowcluster( combined );
    larflow::reco::cluster_t cluster_trunk;
    bool trunk_has_pca = false;
    try {
      cluster_trunk = larflow::reco::cluster_from_larflowcluster( new_trunk );
      trunk_has_pca = true;
    }
    catch (...) {
      trunk_has_pca = false;
    }

    // check the ends
    if ( cluster_trunk.pca_ends_v.size()!=2 ) {
      trunk_has_pca = false;
    }
    else {
      if( cluster_trunk.pca_ends_v[0].size()!=3
	  || cluster_trunk.pca_ends_v[1].size()!=3 ) {
	trunk_has_pca = false;
      }
      else {
	for (int i=0; i<3; i++) {
	  if ( std::isnan( cluster_trunk.pca_ends_v[0][i] ) )
	    trunk_has_pca = false;
	  if( std::isnan( cluster_trunk.pca_ends_v[1][i] ) )
	    trunk_has_pca = false;
	}
      }
    }
    
    if ( trunk_has_pca ) {
      LARCV_DEBUG() << "Trunk made ok PC-axis" << std::endl;
      std::stringstream ss_end1;
      std::stringstream ss_end2;
      for (int i=0; i<3; i++) {
	ss_end1 << cluster_trunk.pca_ends_v[0][i] << " ";
	ss_end2 << cluster_trunk.pca_ends_v[1][i] << " ";
      }
      LARCV_DEBUG() << "pca-end[0] (" << ss_end1.str() << ")" << std::endl;
      LARCV_DEBUG() << "pca-end[1] (" << ss_end2.str() << ")" << std::endl;
    }
    else {
      LARCV_DEBUG() << "Trunk cluster could not make good PC-axis" << std::endl;
    }


    // resort points from closest to vertex to furthest
    larlite::larflowcluster combined_sorted;

    // determine order based on pca ends
    int closest_end = larflow::reco::cluster_closest_pcaend( cluster_all, vtxpos );
    int index_start = 0;
    int index_end = (int)cluster_all.points_v.size()-1;
    int index_dir = 1;    
    if ( closest_end==1 ) {
      index_start = (int)cluster_all.points_v.size()-1;
      index_end = 0;
      index_dir = -1;
    }

    // make final combined cluster
    for ( int i=index_start; i!=index_end; i += index_dir ) {
      combined_sorted.emplace_back( std::move(combined[cluster_all.ordered_idx_v[i]]));
    }

    // make replacement trunk and pc axis
    larlite::track llnewtrunk;
    llnewtrunk.reserve(2);
    
    larlite::pcaxis newtrunk_pca;

    if ( trunk_has_pca ) {
      // use pca as new trunk    
      int closest_trunkend = larflow::reco::cluster_closest_pcaend( cluster_trunk, vtxpos );
      int other_end = (closest_trunkend==0) ? 1 : 0;
      TVector3 newtrunk_start;
      TVector3 newtrunk_end;
      for (int i=0; i<3; i++) {
        newtrunk_start[i] = cluster_trunk.pca_ends_v[closest_trunkend][i];
        newtrunk_end[i]   = cluster_trunk.pca_ends_v[other_end][i];      
      }
      newtrunk_dir = newtrunk_end-newtrunk_start;
      float len = newtrunk_dir.Mag();
      if ( len>0 )
        for (int i=0; i<3; i++)
          newtrunk_dir[i] /= len;
      
      llnewtrunk.add_vertex( newtrunk_start );
      llnewtrunk.add_vertex( newtrunk_end );
      llnewtrunk.add_direction( newtrunk_dir );
      llnewtrunk.add_direction( newtrunk_dir );

      newtrunk_pca = larflow::reco::cluster_make_pcaxis_wrt_point( cluster_trunk, vtxpos );
    }
    else {
      // make simple trunk
      llnewtrunk.add_vertex( tvtx );
      llnewtrunk.add_vertex( shower_endpt );
      llnewtrunk.add_direction( newtrunk_dir );
      llnewtrunk.add_direction( newtrunk_dir );

      // use old pca
      LARCV_DEBUG() << "Use old PCA-axis" << std::endl;
      std::stringstream ss_end1;
      std::stringstream ss_end2;
      for (int i=0; i<3; i++) {
	ss_end1 << shower_pcaxis.getEigenVectors()[3][i] << " ";
	ss_end2 << shower_pcaxis.getEigenVectors()[4][i] << " ";
      }
      LARCV_DEBUG() << "pca-end[0] (" << ss_end1.str() << ")" << std::endl;
      LARCV_DEBUG() << "pca-end[1] (" << ss_end2.str() << ")" << std::endl;
    }
    
    std::swap( combined_sorted, shower_hitcluster ); // exchange new cluster
    std::swap( llnewtrunk, shower_trunk );           // exchange new trunk track
    if( trunk_has_pca )
      std::swap( shower_pcaxis, newtrunk_pca ); // exchange new pca if could calculate it
    
  }

  
}
}
