#include "makesimpleflowhits.h"

#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"

#include "larcv/core/DataFormat/ImageMeta.h"

namespace larflow {

  /**
   * make hits from flow data using very simple algo.
   *
   * we use the data in the ConturFlowMatchDict(s) to make hits using
   * a very simple algorithm. we do not attempt a sophisticated
   * method to correct flow predictions using the contour-matches.
   *
   * @param[in] adc_full_v Image2D with ADC values over a full image (not expecting crops)
   * @param[in] contours   ContourClusterAlgo which holds the contours found in each plane
   * @param[in] matchdict  ContorFlowMatchDict_t for each flow direction
   *
   */
  std::vector<larlite::larflow3dhit> makeSimpleFlowHits( const std::vector<larcv::Image2D>& adc_full_v,
                                                         const ublarcvapp::ContourClusterAlgo& contours,
                                                         const std::vector<ContourFlowMatchDict_t>& matchdict_v )
  {

    // we loop through the source contours
    // and choose the best flow for each source pixel.
    // we choose based on distance of source pixel to center.
    // also, we try to pix one which flows into a target contour

    const larcv::ImageMeta& src_full_meta = adc_full_v.at(2).meta();
    auto const& src_ctr_v = contours.m_plane_atomics_v.at(2);

    std::vector<larlite::larflow3dhit> flowhits_v;

    for ( size_t src_ctr_idx=0; src_ctr_idx<src_ctr_v.size(); src_ctr_idx++ ) {

      // get the source contour
      auto const& src_ctr = src_ctr_v.at(src_ctr_idx);

      // for every unique source pixel(index), we store a larflowhit
      // note: index = row*ncols + col
      std::map<int,larlite::larflow3dhit> srcpix_hitmap;

      // loop through the possible contours
      for (int iflowdir=0; iflowdir<2; iflowdir++ ) {
        for ( auto const& it_mapdict : matchdict_v[iflowdir] ) {

          if ( it_mapdict.first.first==src_ctr_idx ) {
            // found src contour match

            // now loop over the flow information in it
            for ( auto const& it_flowinfo : it_mapdict.second.matchingflow_map ) {
              int srcpix_index = it_flowinfo.first;
              const std::vector< ContourFlowMatch_t::FlowPixel_t >& flowinfo_v = it_flowinfo.second;

              // what is the row and column for this hit?
              int src_col = srcpix_index%src_full_meta.cols();
              int src_row = srcpix_index/src_full_meta.cols();

              // get a hit (or make one)
              auto it_hit = srcpix_hitmap.find( srcpix_index );
              if ( it_hit==srcpix_hitmap.end() ) {
                srcpix_hitmap.insert( std::pair<int,larlite::larflow3dhit>( srcpix_index, larlite::larflow3dhit() ) );
                it_hit = srcpix_hitmap.find( srcpix_index );
              }

              larlite::larflow3dhit& hit = it_hit->second;

              // std::cout << "src-contour[" << src_ctr_idx << "] "
              //           << "src-tar-pair(" << src_ctr_idx << "," << it_mapdict.first.second << ") "
              //           << " sourcepixel[" << srcpix_index << "] "
              //           << "@(" << src_row << "," << src_col << ") "
              //           << "has " << flowinfo_v.size() << " flow-pixels" << std::endl;

              // now fill in the details
              // ------------------------
              
              if ( hit.size()!=3 ) {
                hit.resize(3,0.0);
                hit.targetwire.resize(2,0);
              }

              for ( auto& info : flowinfo_v ) {

                // determine if we replace the values for the hit
                bool replace = false;
                if ( hit.matchquality==larlite::larflow3dhit::kNoMatch ) {
                  replace = true;
                }
                else if ( hit.matchquality<=larlite::larflow3dhit::kClosestC ) {
                  // if there was a value ...
                  // replace if we're closer to a contour
                  if ( fabs(hit.renormed_track_score) > fabs(info.pred_miss) ) {
                    replace = true;
                  }
                  // replace if same distance, but closer to center of y-plane
                  else if ( fabs(hit.renormed_track_score)==fabs(info.pred_miss) && hit.center_y_dist>info.dist2cropcenter ) {
                    replace = true;
                  }
                }

                if ( replace ) {
                  
                  // we fill the hit
                  const larutil::Geometry* geo       = larutil::Geometry::GetME();
                  const larutil::LArProperties* larp = larutil::LArProperties::GetME();

                  // do the channels intersect
                  Double_t y, z;
                  geo->IntersectionPoint( info.src_wire, info.tar_wire, 2, iflowdir, y, z );
                  hit[1] = y;
                  hit[2] = z;
                  
                  // tick and x
                  hit.tick = info.tick;
                  hit[0] = (info.tick-3200.0)*0.5*larp->DriftVelocity();
                  
                  // wires
                  hit.srcwire = info.src_wire;
                  hit.targetwire[ iflowdir ] = info.tar_wire;

                  if ( info.pred_miss==0 )
                    hit.matchquality = larlite::larflow3dhit::kQandCmatch;
                  else if ( info.pred_miss<10.0 )
                    hit.matchquality = larlite::larflow3dhit::kCmatch;
                  else
                    hit.matchquality = larlite::larflow3dhit::kClosestC;

                  // flow-dir
                  hit.flowdir = (larlite::larflow3dhit::FlowDirection_t)iflowdir;

                  // truth matching
                  hit.truthflag = larlite::larflow3dhit::kNoTruthMatch;

                  // center to distance
                  hit.center_y_dist = info.dist2cropcenter;

                  // consistency
                  hit.consistency3d = larlite::larflow3dhit::kNoValue; // do later
                  hit.dy = 0;
                  hit.dz = 0;

                  // using to store closest distance to contour
                  hit.renormed_track_score = fabs(info.pred_miss);
                  
                  // ssnet scores
                  hit.endpt_score = 0.;
                  hit.track_score = 0.;
                  hit.shower_score = 0.;
                  hit.renormed_shower_score = 0.;

                  // infill
                  hit.src_infill = 0;
                  hit.tar_infill.clear();
                  
                  // truth
                  hit.X_truth.clear();
                  hit.trackid = 0.;
                  hit.dWall = 0.;
                  
                }//if replace

              }//loop over flowinfo
                
            }//loop over pixelflow info stored in each contour pair

          }//end of if we matched to source contour index

        }//end of loop over contour flow match dictionary contact
        
      }//end of flow-direction loop

      // move hits into final container
      for ( auto it_srcpixhits : srcpix_hitmap ) {
        flowhits_v.emplace_back( std::move(it_srcpixhits.second) );
      }

    }//end of loop over source contour indices

    
    return flowhits_v;
  }
  
  


}
