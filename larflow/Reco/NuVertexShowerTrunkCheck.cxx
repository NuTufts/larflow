#include "NuVertexShowerTrunkCheck.h"

#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"

#include "larcv/core/DataFormat/EventImage2D.h"

#include "geofuncs.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  void NuVertexShowerTrunkCheck::checkNuCandidateProngs( larflow::reco::NuVertexCandidate& nuvtx )
  {

    std::vector<int> trunk_absorbed( nuvtx.track_v.size(), 0 );
    int num_absorbed = 0;
    for (int i=0; i<(int)nuvtx.track_v.size(); i++) {
      auto& track = nuvtx.track_v[i];
      auto& track_hits = nuvtx.track_hitcluster_v[i];

      // get track length
      float tracklen = 0.;
      for (int ipt=0; ipt<(int)track.NumberTrajectoryPoints()-1;ipt++) {
        tracklen += ( track.LocationAtPoint(ipt)-track.LocationAtPoint(ipt+1) ).Mag();
      }

      if ( tracklen<7.0 )
        continue;

      for (int j=0; j<(int)nuvtx.shower_v.size(); j++) {

        if ( trunk_absorbed[i]==1 )
          continue;
        
        auto& shower = nuvtx.shower_v[j];
        auto& shower_trunk = nuvtx.shower_trunk_v[j];
        auto& shower_pca   = nuvtx.shower_pcaxis_v[j];
        float frac_path = 0.;
        float frac_core = 0.;
        bool add_to_shower = isTrackTrunkOfShower( nuvtx.pos,
                                                   track,
                                                   track_hits,
                                                   shower_trunk,
                                                   shower,
                                                   shower_pca,
                                                   frac_path,
                                                   frac_core );

        if ( add_to_shower ) {
          trunk_absorbed[i] = 1;
          num_absorbed++;
          if ( frac_path>0.8 ) {

            _addTrackAsNewTrunk( nuvtx.pos,
                                 track,
                                 track_hits,
                                 shower_trunk,
                                 shower,
                                 shower_pca );
          }
          else if ( frac_core>0.8 )
            _addTrackToCore( nuvtx.pos,
                             track,
                             track_hits,
                             shower_trunk,
                             shower,
                             shower_pca );
        }
      }//end of shower loop
    }//end of track loop

        
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
      keep_track_v.reserve(   (int)nuvtx.track_v.size() - num_absorbed );
      keep_cluster_v.reserve( (int)nuvtx.track_hitcluster_v.size() - num_absorbed );
      for (int ii=0; ii<(int)nuvtx.track_v.size(); ii++ ) {
        if ( trunk_absorbed[ii]==0 ) {
          keep_track_v.emplace_back( std::move(nuvtx.track_v[ii]) );
          keep_cluster_v.emplace_back( std::move(nuvtx.track_hitcluster_v[ii]) );
        }
      }
      std::swap( nuvtx.track_v, keep_track_v );
      std::swap( nuvtx.track_hitcluster_v, keep_cluster_v );
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
      float max_gapdist = 10.;
      larlite::larflowcluster add_hits_v =
        makeMissingTrunkHits( nuvtx, adc_v,
                              shower_trunk,
                              shower,
                              shower_pca, max_gapdist );

      if ( add_hits_v.size()>3 && max_gapdist<3.0) {
        // merge the shower trunk
        _mergeNewTrunkHits( nuvtx.pos,
                            add_hits_v,
                            shower_trunk,
                            shower,
                            shower_pca);
      }
    }
    
  }    

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
    if ( dist[0]<dist[1] ) {
      for (int i=0; i<3; i++)
        shrstart[i] = shower_trunk.LocationAtPoint(0)[i];
    }
    else {
      for (int i=0; i<3; i++)
        shrstart[i] = shower_trunk.LocationAtPoint(1)[i];
    }


    std::vector<float> showerdir(3,0);
    float len = 0.;
    float pcalen = 0.;
    float dx = 0.;
    for (int i=0; i<3; i++) {
      showerdir[i] = shrstart[i]-vtxpos[i];
      len += showerdir[i]*showerdir[i];
      dx = (shower_pcaxis.getEigenVectors()[3][i]-shower_pcaxis.getEigenVectors()[4][i]);
      pcalen += dx*dx;
    }
    len = sqrt(len);
    for (int i=0; i<3; i++)
      showerdir[i] /= len;
    pcalen = sqrt(pcalen);

    int nhits_within_startpath = 0.;
    int nhits_within_shower = 0.;
    for (size_t ihit=0; ihit<track_hitcluster.size(); ihit++) {
      auto const& hit = track_hitcluster[ihit];
      float r = larflow::reco::pointLineDistance3f( vtxpos, shrstart, hit );
      float s_start = larflow::reco::pointRayProjection3f( vtxpos, showerdir, hit );
      if ( r<2.0 ) {
        if ( s_start>-2.0 && s_start<len )
          nhits_within_startpath++;
        else if ( s_start>len && s_start<len+pcalen+5.0 )
          nhits_within_shower++;
      }
    }

    frac_path = (float)nhits_within_startpath/(float)track_hitcluster.size();
    frac_core = (float)nhits_within_shower/(float)track_hitcluster.size();    
    LARCV_DEBUG() << " fraction of track hits within cylinder of vtx-to-shower start path: " << frac_path << std::endl;
    LARCV_DEBUG() << " fraction of track hits within cylinder of shower core: " << frac_core << std::endl;    

    if ( frac_path>0.8 || frac_core>0.8 )
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
                  << " start=(" << shrstart[0] << "," << shrstart[1] << "," << shrstart[2] << ")"
                  << " end=(" << shrend[0] << "," << shrend[1] << "," << shrend[2] << ")"
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
          lfhit.targetwire.resize(2,0);          
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
        dist[i] += ( cluster.pca_ends_v[iend][i] - vtxpos[i] )*( cluster.pca_ends_v[iend][i] - vtxpos[i] );
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

    LARCV_DEBUG() << "make new trunk" << std::endl;    
    larlite::track new_trunk;
    new_trunk.reserve( 2 );
    new_trunk.add_vertex( track.LocationAtPoint(0) );
    new_trunk.add_direction( track.DirectionAtPoint(0) );
    new_trunk.add_vertex( track.LocationAtPoint( (int)track.NumberTrajectoryPoints()-1 ) );
    new_trunk.add_direction( track.DirectionAtPoint( (int)track.NumberTrajectoryPoints()-1 ) );

    LARCV_DEBUG() << "make new pca" << std::endl;
    larlite::pcaxis new_pca = larflow::reco::cluster_make_pcaxis( cluster );

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
        dist[i] += ( cluster.pca_ends_v[iend][i] - vtxpos[i] )*( cluster.pca_ends_v[iend][i] - vtxpos[i] );
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
    }//end of loop over pca order


    larlite::pcaxis new_pca = larflow::reco::cluster_make_pcaxis( cluster );

    std::swap( shower_hitcluster, lfcluster );
    std::swap( shower_pcaxis, new_pca );
    
  }
  
  void NuVertexShowerTrunkCheck::_mergeNewTrunkHits( const std::vector<float>& vtxpos,
                                                     const larlite::larflowcluster& newhits,
                                                     larlite::track& shower_trunk,                                                  
                                                     larlite::larflowcluster& shower_hitcluster,
                                                     larlite::pcaxis& shower_pcaxis )
  {
    
    larlite::larflowcluster combined;
    larlite::larflowcluster new_trunk;
    combined.reserve( newhits.size()+shower_hitcluster.size() );
    for ( auto& hit : newhits ) {
      float dist = 0.;
      for (int i=0; i<3; i++)
        dist += ( hit[i]-vtxpos[i] )*( hit[i]-vtxpos[i] );
      if ( dist<25.0 )
        new_trunk.push_back( hit );
      combined.emplace_back( std::move(hit) );      
    }

    // need shower start
    float shower_end_dist[2] = {0};
    int npts = (int)shower_trunk.NumberTrajectoryPoints();
    for (int i=0; i<3; i++) {
      shower_end_dist[0] += ( shower_trunk.LocationAtPoint(0)[i] - vtxpos[i] )*( shower_trunk.LocationAtPoint(0)[i] - vtxpos[i] );
      shower_end_dist[1] += ( shower_trunk.LocationAtPoint(npts-1)[i] - vtxpos[i] )*( shower_trunk.LocationAtPoint(npts-1)[i] - vtxpos[i] );
    }
    int shower_endpt_index = ( shower_end_dist[0]<shower_end_dist[1] ) ? 0 : npts-1;
    
    std::vector<float> shower_startpt(3,0);
    for (int i=0; i<3; i++)
      shower_startpt[i] = shower_trunk.LocationAtPoint(shower_endpt_index)[i];
    
    for ( auto& hit : shower_hitcluster ) {
      // fill trunk container
      float dist = 0.;
      for (int i=0; i<3; i++)
        dist += ( hit[i]-shower_startpt[i] )*( hit[i]-shower_startpt[i] );
      if ( dist<25.0 )
        new_trunk.push_back( hit );

      // fill all
      combined.emplace_back( std::move(hit) );      
    }

    // cluster: will calc pca for us
    larflow::reco::cluster_t cluster_all   = larflow::reco::cluster_from_larflowcluster( combined );
    larflow::reco::cluster_t cluster_trunk = larflow::reco::cluster_from_larflowcluster( new_trunk );    

    // resort points
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
      
    
    for ( int i=index_start; i!=index_end; i += index_dir ) {
      combined_sorted.emplace_back( std::move(combined[cluster_all.ordered_idx_v[i]]));
    }

    // replace hit cluster
    std::swap( combined_sorted, shower_hitcluster );

    // make replace trunk and pc axis
    int closest_trunkend = larflow::reco::cluster_closest_pcaend( cluster_trunk, vtxpos );
    int other_end = (closest_trunkend==0) ? 1 : 0;
    larlite::track llnewtrunk;
    llnewtrunk.reserve(2);
    TVector3 newtrunk_start;
    TVector3 newtrunk_end;
    for (int i=0; i<3; i++) {
      newtrunk_start[i] = cluster_trunk.pca_ends_v[closest_trunkend][i];
      newtrunk_end[i]   = cluster_trunk.pca_ends_v[closest_trunkend][i];      
    }
    TVector3 newtrunk_dir = newtrunk_end-newtrunk_start;
    float len = newtrunk_dir.Mag();
    if ( len>0 )
      for (int i=0; i<3; i++)
        newtrunk_dir[i] /= len;

    llnewtrunk.add_vertex( newtrunk_start );
    llnewtrunk.add_vertex( newtrunk_end );
    llnewtrunk.add_direction( newtrunk_dir );
    llnewtrunk.add_direction( newtrunk_dir );
    
    larlite::pcaxis newtrunk_pca = larflow::reco::cluster_make_pcaxis_wrt_point( cluster_trunk, vtxpos );

    std::swap( shower_trunk, llnewtrunk );
    std::swap( shower_pcaxis, newtrunk_pca );
    
  }

  
}
}
