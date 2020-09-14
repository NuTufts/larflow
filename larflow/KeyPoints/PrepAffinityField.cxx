#include "PrepAffinityField.h"

#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "larflow/Reco/geofuncs.h"

namespace larflow {
namespace keypoints {

  /** @brief default constructor */
  PrepAffinityField::PrepAffinityField()
    : larcv::larcv_base("PrepAffinityField")
  {
    psce = new larutil::SpaceChargeMicroBooNE;
  }

  /** @brief default destructor */  
  PrepAffinityField::~PrepAffinityField()
  {
    delete psce;
  }
  
  /**
   * @brief goal is to define a direction vector for every triplet proposal
   *
   * for each triplet spacepoint, if good, then we project into
   * image, get instance ID. Then if track, we find the truth mcstep segment the point
   * is closest to to extract the direction. For showers, we find the initial 
   * detector profile direction and use that for the label.
   *
   * for many pixels/triplets, we will probably fail to produce a ground truth
   * value. we weight the contribution of these to the loss function
   * as zero and prepare a weight in addition to the ground truth label.
   *
   * @param[in] iolcv LArCV IOManager containing the event data
   * @param[in] ioll  larlite storage_manager containing the event data
   * @param[in] match_proposals copy of larflow::prep::PrepMatchTriplets containing the proposed space points for the event, 
                                along with labels for which points are true or are ghosts.
   */
  void PrepAffinityField::process( larcv::IOManager& iolcv,
                                   larlite::storage_manager& ioll,
                                   const larflow::prep::PrepMatchTriplets& match_proposals )
  {

    larcv::EventImage2D* ev_instance =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "instance" );

    larlite::event_mctrack* ev_mctrack =
      (larlite::event_mctrack*)ioll.get_data( larlite::data::kMCTrack, "mcreco" );
    larlite::event_mcshower* ev_mcshower =
      (larlite::event_mcshower*)ioll.get_data( larlite::data::kMCShower, "mcreco" );


    _match_labels_v.clear();
    _match_labels_v.reserve( match_proposals._triplet_v.size() );

    std::cout << "Make labels for " << match_proposals._triplet_v.size() << " triplet proposals" << std::endl;
    std::cout << "  plane[" << 0 <<"] pixels: " << match_proposals._sparseimg_vv[0].size() << std::endl;
    std::cout << "  plane[" << 1 <<"] pixels: " << match_proposals._sparseimg_vv[1].size() << std::endl;
    std::cout << "  plane[" << 2 <<"] pixels: " << match_proposals._sparseimg_vv[2].size() << std::endl;    
    
    for ( size_t itriplet=0; itriplet<match_proposals._triplet_v.size(); itriplet++ ) {
      auto const& triplet = match_proposals._triplet_v.at(itriplet);
      int true_triplet = match_proposals._truth_v[itriplet];
      const std::vector<float>& spacepoint = match_proposals._pos_v[itriplet];
      // std::cout << "(" << itriplet << ") [" << triplet[0] << "," << triplet[1] << "," << triplet[2] << "," << triplet[3] << "] "
      //           << " true=" << true_triplet << std::endl;

      std::vector<float> label_v;
      float weight = 0.;
      if ( true_triplet==1 ) {
        std::vector< std::vector<int> > pixlist_v(ev_instance->Image2DArray().size());
        for (size_t p=0; p<3; p++) {
          int row = match_proposals._sparseimg_vv[p].at( triplet[p] ).row;
          int col = match_proposals._sparseimg_vv[p].at( triplet[p] ).col;
          pixlist_v[p] = std::vector<int>( { row, col } );
        }
      
        _determine_triplet_labels( pixlist_v,
                                   spacepoint,
                                   ev_instance->Image2DArray(),
                                   *ev_mctrack,
                                   *ev_mcshower,
                                   label_v, weight );
      }
      _match_labels_v.push_back( label_v );
    }
    
  }


  /**
   * @brief make truth direction vector for all space points
   *
   * For each space point we create a vector<float> with the following contents
   * \verbatim embed:rst:leading-asterisk
   *  * `[0-2]`: 3D direction
   *  * `[3-8]`: 3 planes x 2D direction (projected direction in the wire planes)
   *  * `[9]`: instance id
   * \endverbatim
   *
   * @param[in] pixlist_v list of pixel coordinates (row,col) for each image (outer vector) for pixels with values above threshold
   * @param[in] spacepoint_v given 3D space point we are making the direction label for
   * @param[in] instance_v vector of larcv::Image2D, one for each plane, where each pixel stores the MC ID of the particle that made the most charge within that pixel.
   * @param[in] ev_mctrack_v container holding larlite::mctrack objects, which store the trajectory of the simulated track-like particles
   * @param[in] ev_mcshower_v container holding larlite::mcshower objects, which store the true start direction of simulated shower-like particles
   * @param[out] label_v the 3D direction for the space point
   * @param[out] weight the weight calculated for this point. [unused]
   *
   */
  void PrepAffinityField::_determine_triplet_labels( const std::vector< std::vector<int> >& pixlist_v,
                                                     const std::vector<float>& spacepoint_v,
                                                     const std::vector< larcv::Image2D >& instance_v,
                                                     const larlite::event_mctrack& ev_mctrack_v,
                                                     const larlite::event_mcshower& ev_mcshower_v,
                                                     std::vector<float>& label_v,
                                                     float& weight )
  {

    weight = 0.0;
    
    // [0-2]: 3D direction
    // [3-8]: 3 planes x 2D direction
    // [9]: instance id

    std::map< int, int > instance_counts;
    for ( size_t p=0; p<pixlist_v.size(); p++ ) {
      //std::cout << "pixel[plane " << p << "] (" << pixlist_v[p][0] << "," << pixlist_v[p][1] << ")" << std::endl;
      int instance_id = (int)instance_v[p].pixel( pixlist_v[p][0], pixlist_v[p][1] );
      //std::cout << "  instanceid=" << instance_id << std::endl;      
      if ( instance_id<=0 )
        continue;

      auto it = instance_counts.find(instance_id);
      if ( it==instance_counts.end() ) {
        instance_counts[instance_id] = 0;
        it = instance_counts.find(instance_id);
      }

      it->second += 1;

    }

    int max_id = -1;
    int max_count = 0;
    for ( auto it=instance_counts.begin(); it!=instance_counts.end(); it++ ) {
      if ( it->second>max_count ) {
        max_count = it->second;
        max_id = it->first;
      }
    }
    if ( max_id<=0 || max_count==0 ) {
      return;
    }

    //std::cout << "max track-id counts: " << max_id << " counts=" << max_count << std::endl;

    int is_track = 0;
    int idx = -1;

    for ( int i=0; i<(int)ev_mctrack_v.size(); i++ ) {
      if ( (int)ev_mctrack_v[i].TrackID()==max_id ) {
        is_track = 1;
        idx = i;
        break;
      }
    }

    if ( is_track<1 ) {
      for (int i=0; i<(int)ev_mcshower_v.size(); i++ ) {
        if ( (int)ev_mcshower_v[i].TrackID()==max_id  ) {
          is_track = 0;
          idx = i;
        }
      }
    }

    if ( idx<0 ) {
      return;
    }

    label_v.resize(10,0.0);
    for (int i=0; i<10; i++) label_v[i] = 0.;
    label_v[9] = (float)max_id;
      
    std::vector<double> dspacepoint = { (double)spacepoint_v[0],
                                        (double)spacepoint_v[1],
                                        (double)spacepoint_v[2] };
    
    if ( is_track==1 ) {
      _get_track_direction( ev_mctrack_v[idx], dspacepoint, instance_v, label_v, weight );
    }
    else {
      _get_shower_direction( ev_mcshower_v[idx], dspacepoint, instance_v, label_v, weight );
    }
      
    return;
    
  }

  /**
   * @brief use truth trajectory information from the simulation to get the direction label for a space point
   *
   * @param[in] track larlite::mctrack instance with the true trajectory of the track particle
   * @param[in] pt the space point for which we want to know the assign a ground truth direction
   * @param[in] img_v vector of wire plane images
   * @param[out] label_v Assigned 3D direction
   * @param[out] weight  Assigned weight for this space point. [unused]
   * 
   */
  bool PrepAffinityField::_get_track_direction( const larlite::mctrack& track,
                                                const std::vector<double>& pt,
                                                const std::vector<larcv::Image2D>& img_v,
                                                std::vector<float>& label_v,
                                                float& weight )
  {

    int nsteps = track.size();

    if ( nsteps==0 ) {
      label_v.clear();
      return false;
    }
    
    std::vector<double> start(3,0);
    std::vector<double> end(3,0);
    std::vector<double> stepdir(3,0);
    double steplen = 0.;

    int best_step = -1;
    double best_step_r = -1;
    std::vector<double> best_stepdir(3,0);
    std::vector<double> best_start(3,0);
    std::vector<double> best_end(3,0);

    //std::cout << "  track size: " << nsteps << std::endl;
    
    // set start
    for (int i=0; i<3; i++ ) start[i] = track.front().Position()[i];
    std::vector<double> offsets = psce->GetPosOffsets( (double)start[0], (double)start[1], (double)start[2] );
    start[0] -= offsets[0] + 0.7;
    start[1] += offsets[1];
    start[2] += offsets[2];
    
    for (int istep=0; istep+1<nsteps; istep++) {
      // we get the end point, apply the space charge effect
      for (int i=0; i<3; i++ ) end[i] = track[istep+1].Position()[i];
      offsets = psce->GetPosOffsets( (double)end[0], (double)end[1], (double)end[2] );
      end[0] -= offsets[0] + 0.7;
      end[1] += offsets[1];
      end[2] += offsets[2];

      // std::cout << "step[" << istep << "] "
      //           << "start=(" << start[0] << "," << start[1] << "," << start[2] << ") -> "
      //           << "end=(" << end[0] << "," << end[1] << "," << end[2] << ")"
      //           << std::endl;

      steplen = 0.;
      for (int i=0; i<3; i++ ) {
        stepdir[i] = (end[i]-start[i]);
        steplen += stepdir[i]*stepdir[i];
      }
      if ( steplen>0 ) {
        steplen = sqrt(steplen);
        for (int i=0; i<3; i++ )
          stepdir[i] /= steplen;
      }

      if ( steplen>0 ) {
        double projs = larflow::reco::pointRayProjection<double>( start, stepdir, pt );
        if ( projs>0 && projs<steplen ) {
          double r = larflow::reco::pointLineDistance<double>( start, end, pt );
          if ( best_step==-1 || best_step_r>r ) {
            best_step = istep;
            best_step_r = r;
            best_stepdir = stepdir;
            best_start = start;
            best_end = end;
          }
        }
      }

      // replace the starting step
      start = end;
    }

    if ( best_step==-1 ) {
      // no label generated
      label_v.clear();
      return false;
    }

    // copy 3d direction
    for (int i=0; i<3; i++ ) {
      label_v[i] = best_stepdir[i];
    }

    // calculate 2d projected direction, by projecting into image, making direction
    // should use plane directions to do this, but projection into plane is the dumb thing to do for now
    float start_tick = 3200 + best_start[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
    float end_tick   = 3200 + best_end[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
    float row_dir = (end_tick-start_tick)/img_v.front().meta().pixel_height();
    for (size_t p=0; p<3; p++ ) {
      float wire_start = larutil::Geometry::GetME()->WireCoordinate( best_start, p );
      float wire_end   = larutil::Geometry::GetME()->WireCoordinate( best_end, p );
      float col_dir = wire_end-wire_start;
      float len2d = 0.;
      len2d = row_dir*row_dir + col_dir*col_dir;
      len2d = sqrt(len2d);
      if ( len2d>0.0 ) {
        label_v[ 3+2*p ]   = row_dir/len2d;
        label_v[ 3+2*p+1 ] = col_dir/len2d;
      }
    }
    
    return true;
  }

  /**
   * @brief use truth information from the simulation to get the direction label for a showering particle space point
   *
   * @param[in] shower larlite::mcshower instance with the truth information about the shower
   * @param[in] pt the space point for which we want to know the assign a ground truth direction
   * @param[in] img_v vector of wire plane images
   * @param[out] label_v Assigned 3D direction
   * @param[out] weight  Assigned weight for this space point. [unused]
   * 
   */
  bool PrepAffinityField::_get_shower_direction( const larlite::mcshower& shower,
                                                 const std::vector<double>& pt,
                                                 const std::vector<larcv::Image2D>& img_v,
                                                 std::vector<float>& label_v,
                                                 float& weight )
  {

    // use det profile direction for 3D label
    float len = 0.;
    for (int i=0; i<3; i++) {
      label_v[i] = shower.DetProfile().Momentum()[i];
      len += label_v[i]*label_v[i];
    }
    len = sqrt(len);

    if (len==0) {
      label_v.clear();
      return false;
    }

    for (int i=0; i<3; i++) label_v[i] /= len;

    // make step
    std::vector<double> start(3,0);
    std::vector<double> end(3,0);
    for (int i=0; i<3; i++) {
      end[i] = start[i] + 1.0*label_v[i];
    }

    // sce correct step
    std::vector<double> offset_start = psce->GetPosOffsets( start[0], start[1], start[2] );
    std::vector<double> offset_end   = psce->GetPosOffsets( end[0], end[1], end[2] );

    start[0] = start[0] - offset_start[0] + 0.7;
    start[1] = start[1] + offset_start[1];
    start[2] = start[2] + offset_start[2];

    end[0] = end[0] - offset_end[0] + 0.7;
    end[1] = end[1] + offset_end[1];
    end[2] = end[2] + offset_end[2];
    
    // calculate the final direction
    len = 0.;
    for (int i=0; i<3; i++ ) {
      label_v[i] = end[i]-start[i];
      len += label_v[i]*label_v[i];
    }
    len = sqrt(len);

    if ( len>0 ) {
      for (int i=0; i<3; i++)
        label_v[i] /= len;
    }
    
    // calculate 2d projected direction, by projecting into image, making direction
    // should use plane directions to do this, but projection into plane is the dumb thing to do for now
    float start_tick = 3200 + start[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
    float end_tick   = 3200 + end[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
    float row_dir = (end_tick-start_tick)/img_v.front().meta().pixel_height();
    for (size_t p=0; p<3; p++ ) {
      float wire_start = larutil::Geometry::GetME()->WireCoordinate( start, p );
      float wire_end   = larutil::Geometry::GetME()->WireCoordinate( end, p );
      float col_dir = wire_end-wire_start;
      float len2d = 0.;
      len2d = row_dir*row_dir + col_dir*col_dir;
      len2d = sqrt(len2d);
      if ( len2d>0.0 ) {
        label_v[ 3+2*p ]   = row_dir/len2d;
        label_v[ 3+2*p+1 ] = col_dir/len2d;
      }
    }

    return true;
  }

  /**
   * @brief make the output tree where will store the labels
   */
  void PrepAffinityField::defineAnaTree()
  {
    _label_tree = new TTree("AffinityFieldTree", "Affinity Field Label Tree");
    _label_tree->Branch( "run",     &_run,    "run/I" );
    _label_tree->Branch( "subrun",  &_subrun, "subrun/I" );
    _label_tree->Branch( "event",   &_event,  "event/I" );
    _label_tree->Branch( "label_v", &_match_labels_v );
  }

  /** @brief fill the output tree with the values for the current entry */
  void PrepAffinityField::fillAnaTree()
  {
    if (_label_tree) _label_tree->Fill();
  }

  /** @brief write output tree to the output file */
  void PrepAffinityField::writeAnaTree()
  {
    if ( _label_tree ) _label_tree->Write();
  }
}
}
