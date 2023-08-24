#include "FlowTriples.h"
#include "WireOverlap.h"
#include "LArUtil/GeometryHelper.h"

#include <ctime>
#include <sstream>
#include <algorithm>
#include <map>

namespace larflow {
namespace prep {

  /**
   * @brief generate list of possible combinations of (U,V,Y) intersections given source to target plane matching.
   *
   * this version takes image2d as input
   *
   * @param[in] source The source plane index, where we start the match with a pixel above threshold.
   * @param[in] target The target plane index, where we match to pixels above threshold
   * @param[in] adc_v  Vector of image pixel values
   * @param[in] badch_v Vector of images labeling bad channels
   * @param[in] threshold Pixels that we consider must be above this value
   * @param[in] save_index If true, we save the index of the pixel in the sparse matrix representation.
   *                       If false, we save the image col posiiton.
   *                       Saving the col position requires searching the sparse matrix entries, but one can append to matrix.
   *                       Saving the index requires no searching, but look-up is broken if matrix is changed.
   *
   */
  FlowTriples::FlowTriples( int source, int target,
                            const std::vector<larcv::Image2D>& adc_v,
                            const std::vector<larcv::Image2D>& badch_v,
                            float threshold, bool save_index ) {
    
    _sparseimg_vv = make_initial_sparse_image( adc_v, threshold );
    _makeTriples( source, target, adc_v, badch_v, _sparseimg_vv, threshold, save_index );
    
  }

  /**
   * @brief generate list of possible combinations of (U,V,Y) intersections given source to target plane matching.
   *
   * this version takes the sparse matrix representation of the image, skipping its generation
   *
   * @param[in] source The source plane index, where we start the match with a pixel above threshold.
   * @param[in] target The target plane index, where we match to pixels above threshold
   * @param[in] adc_v  Vector of image pixel values
   * @param[in] badch_v Vector of images labeling bad channels
   * @param[in] sparseimg_vv Sparse representation (i.e. list of pixels) for each plane.
   * @param[in] threshold Pixels that we consider must be above this value
   * @param[in] save_index If true, we save the index of the pixel in the sparse matrix representation.
   *                       If false, we save the image col posiiton.
   *                       Saving the col position requires searching the sparse matrix entries, but one can append to matrix.
   *                       Saving the index requires no searching, but look-up is broken if matrix is changed.
   *
   */  
  FlowTriples::FlowTriples( int source, int target,
                            const std::vector<larcv::Image2D>& adc_v,
                            const std::vector<larcv::Image2D>& badch_v,
                            const std::vector< std::vector<PixData_t> >& sparseimg_vv,
                            float threshold, bool save_index ) {

    _makeTriples( source, target, adc_v, badch_v, sparseimg_vv, threshold, save_index );
    
  }

  /**
   * @brief using the list of pixels in the sparse matrix representation, build the list of possible
   *  three-plane triplets
   *
   * @param[in] source The source plane index, where we start the match with a pixel above threshold.
   * @param[in] target The target plane index, where we match to pixels above threshold
   * @param[in] adc_v  Vector of image pixel values
   * @param[in] badch_v Vector of images labeling bad channels
   * @param[in] sparseimg_vv Sparse representation (i.e. list of pixels) for each plane.
   * @param[in] threshold Pixels that we consider must be above this value
   * @param[in] save_index If true, we save the index of the pixel in the sparse matrix representation.
   *                       If false, we save the image col posiiton.
   *                       Saving the col position requires searching the sparse matrix entries, but one can append to matrix.
   *                       Saving the index requires no searching, but look-up is broken if matrix is changed.
   *
   */    
  void FlowTriples::_makeTriples( int source, int target,
                                  const std::vector<larcv::Image2D>& adc_v,
                                  const std::vector<larcv::Image2D>& badch_v,
                                  const std::vector< std::vector<PixData_t> >& sparseimg_vv,                                  
                                  float threshold, bool save_index ) {


    std::clock_t start_ = std::clock();

    _source_plane = source;
    _target_plane = target;
    _other_plane  = larflow::LArFlowConstants::getOtherPlane( source, target );
    
    // allocate space for triples
    _triple_v.reserve( sparseimg_vv[_source_plane].size() );

    int ndeadch_added = 0;
    std::map< std::pair<int,int>, int > _added_dead_channel_pixels[3];
    _deadch_to_add.resize( adc_v.size() );

    // turn this on for a full dump
    bool for_debug = false;

    // make possible triples based on going from source pixel to target pixels
    for ( auto& srcpix : sparseimg_vv[_source_plane] ) {

      // activate conditional debug
      // we puke for pixels above threshold, but sitting in dead channel.
      // this is to trace effect of INFILL
      // float badchval = badch_v[_source_plane].pixel( srcpix.row, srcpix.col );
      // if ( badchval>0 )
      //   for_debug = true;
      // else
      //   for_debug = false;
      
      // we get all the 'target' and 'other' plane wires this pixel overlaps with
      std::vector< std::vector<int> > overlap = larflow::prep::WireOverlap::getOverlappingWires( _source_plane, _target_plane, srcpix.col );

      if ( for_debug ) {
        std::cout << "FlowTriples[" << _source_plane << "," << _target_plane << "," << _other_plane << "] "
                  << " srcwire=" << srcpix.col << " row=" << srcpix.row 
                  << " tar-overlap=[" << overlap[0].front() << "," << overlap[0].back() << "] "
                  << " oth-overlap=[" << overlap[1].front() << "," << overlap[1].back() << "] "
                  << std::endl;
      }
      
      if ( overlap[0].size()==0 ) {
        if ( for_debug )
          std::cout << " -- target no overlap" << std::endl;
        continue;
      }

      // get iterator to overlap[0] vector: contains pixels in the target wire, the source wire overlaps with
      auto it_overlap0 = overlap[0].begin();
      
      // get the lowerbound pixel for (row,col) in target sparse image container (assumes sorted properly sort worked)
      PixData_t lb( srcpix.row, (int)overlap[0][0], 0.0 );
      auto it_target = std::lower_bound( sparseimg_vv[_target_plane].begin(), sparseimg_vv[_target_plane].end(), lb );

      if ( for_debug ) {
        if ( it_target!=sparseimg_vv[_target_plane].end() )
          std::cout << " target_pix_lb(row,col)=(" << it_target->row << "," << it_target->col << ") ";
        std::cout << " other=[" << overlap[1].front() << "," << overlap[1].back() << "]"
                  << std::endl;
      }

      // iterator forward in target wire with charge, until we go to another row (different time)
      if (for_debug) std::cout << " ... start of target pixel loop ..." << std::endl;
      while ( it_target!=sparseimg_vv[_target_plane].end() && it_target->row==srcpix.row ) {

        // break if we out of range over the target wire
        if ( it_target->col>overlap[0].back() || it_target->row!=srcpix.row ) break;

        // find position in overlap[0] in order to get overlap[1] element, i.e. the other wire the source+target wires intersect
        it_overlap0 = std::lower_bound( it_overlap0, overlap[0].end(), it_target->col );
        if ( it_overlap0==overlap[0].end() ) {
          if ( for_debug ) std::cout << " target col not in list: break" << std::endl;
          break; // didnt find the target column in the list
        }

        // scan up until matches
        int ivec = -1;
        while ( *it_overlap0<=it_target->col && it_overlap0!=overlap[0].end()) {
          ivec = it_overlap0-overlap[0].begin();
          it_overlap0++;
        }
        if ( for_debug )
          std::cout << " .. target pixel: row=" << it_target->row << " col=" << it_target->col << " ivec=" << ivec << std::endl;

        int otherwire = overlap[1][ivec];
        if ( for_debug )
          std::cout << "  ... search for otherwire=" << otherwire << " ivec=" << ivec << std::endl;
        
        // now find the other plane pixel in the sparse matrix.
        // we allow for a little slop
        // first search for pixel in other plane sparseimg vector (that is above threshold)
        auto it_other = std::lower_bound( sparseimg_vv[_other_plane].begin(),
                                          sparseimg_vv[_other_plane].end(),
                                          PixData_t( srcpix.row, otherwire-2, 0.0) );

        if ( for_debug ) {
          if ( it_other!=sparseimg_vv[_other_plane].end() ) {
            std::cout << " ... otherlb(r,c)=" << it_other->row << "," << it_other->col << ")" << std::endl;
          }
        }

        // now we scan through the pixels in the other plane
        bool found = false;        
        if ( it_other!=sparseimg_vv[_other_plane].end() && it_other->row==srcpix.row && it_other->col<=otherwire ) {
          // pixels to search
          while ( it_other!=sparseimg_vv[_other_plane].end() && it_other->row==srcpix.row ) {
            // if a pixel in the other plane is found close to the one we searched for,
            //  we know it has charge, so we store the triple
            //valid triple
            if ( for_debug ) {
              std::cout << " .... looking for row=" << srcpix.row << " otherwire=" << otherwire
                        << " cols=(" << srcpix.col << "," << it_target->col << "," << it_other->col << ")"
                        << std::endl;
            }
            
            if ( abs(it_other->col-otherwire)<=1 ) {
              if (for_debug) {
                std::cout << "  ... found triple with charge!!!" << std::endl;
              }
              if (save_index) {
                std::vector<int> trip = { srcpix.idx, it_target->idx, it_other->idx, srcpix.row }; // store positions in sparsematrix
                _triple_v.push_back( trip );
              }
              else {
                std::vector<int> trip = { srcpix.col, it_target->col, it_other->col, srcpix.row }; // store positions in sparsematrix
                _triple_v.push_back( trip );                
              }
              found = true;
              break;
            }
            it_other++;
          }
        }

        // if we did not find a pixel in the other plane, then we check if it is in a bad region
        if ( !found && badch_v[ _other_plane ].pixel( srcpix.row, otherwire ) > 0 ) {
          // badchannel, other wire lands in dead channel
          
          // check we have this pixel
          if ( for_debug ) {
            std::cout << " ... looking for badch row=" << srcpix.row
                      << " cols=(" << srcpix.col << "," << it_target->col << "," << otherwire << ")"
                      << std::endl;          
            std::cout << "  ... found triple with dead channel" << std::endl;
          }

          // store this dead pixel to add to the sparsematrix after we complete the search method
          auto it_dead_other = _added_dead_channel_pixels[_other_plane].find( std::pair<int,int>(srcpix.row,otherwire) );
          int otherplane_index = 0;

          if ( it_dead_other==_added_dead_channel_pixels[_other_plane].end() ) {
            // was not in the creation set, create this pixel in the other plane sparse image
            otherplane_index = sparseimg_vv[_other_plane].size() + _deadch_to_add[_other_plane].size();
            PixData_t badpix( srcpix.row, otherwire, 0.0 );
            badpix.idx = otherplane_index;
            // this screws up the search, but at the end, so won't matter?
            _deadch_to_add[_other_plane].push_back( badpix );
            _added_dead_channel_pixels[_other_plane][ std::pair<int,int>(srcpix.row,otherwire) ]  = otherplane_index;
            ndeadch_added++;
          }
          else {
            otherplane_index = it_dead_other->second;
          }

          if (save_index) {
            // store position in sparsematrix, omit last plane            
            std::vector<int> trip = { srcpix.idx, it_target->idx, otherplane_index, srcpix.row }; 
            _triple_v.push_back( trip );
          }
          else {
            // store position in image
            std::vector<int> trip = { srcpix.col, it_target->col, otherwire, srcpix.row }; 
            _triple_v.push_back( trip );            
          }
        }
        else {

          if ( for_debug ) {
            std::cout << "looking for row=" << srcpix.row
                      << " cols=(" << srcpix.col << "," << it_target->col << "," << otherwire << ")"
                      << std::endl;                    
          
            if ( it_other==sparseimg_vv[_other_plane].end() )
              std::cout << "  ... no triple" << std::endl;
            else
              std::cout << "  ... lowerbound found, but no match? lowerbound otherpix=(" <<it_other->row << "," << it_other->col << ")" <<  std::endl;
          }
          
        }
        
        // iterate the target pixel
        it_target++;
      }//target while loop
    }

    // for ( auto& deadpix : _deadch_to_add[_other_plane] ) {
    //   sparseimg_vv[_other_plane].push_back( deadpix );
    // }
    
    _triple_v.shrink_to_fit();

    std::clock_t end_ = std::clock();
    
    std::cout << "[FlowTriples] for flow source[" << _source_plane << "] "
              << "to target[" << _target_plane << "] planes "
              << "found " << _triple_v.size() << " triples "
              << "ndeadch-added=" << ndeadch_added << " "
              << "elasped=" << float(end_-start_)/float(CLOCKS_PER_SEC)
              << std::endl;
    
    
  }


  /**
   * @brief use a th2d to plot the projected locations of the triplets in each of the planes
   *
   * utility function to help with visual debugging
   *
   * @param[in] adc_v  Vector of image pixel values
   * @param[in] sparseimg_vv Sparse representation (i.e. list of pixels) for each plane.
   * @param[in] hist_stem_name  Stem of histogram name generated.
   *
   */    
  std::vector<TH2D> FlowTriples::plot_triple_data( const std::vector<larcv::Image2D>& adc_v,
                                                   const std::vector< std::vector<PixData_t> >& sparseimg_vv,
                                                   std::string hist_stem_name ) {

    std::vector<TH2D> out_v;
    
    for ( int p=0; p<(int)adc_v.size(); p++ ) {
      std::stringstream ss;
      ss << "htriples_plane" << p << "_" << hist_stem_name;
      auto const& meta = adc_v[p].meta();
      TH2D hist( ss.str().c_str(), "",
                 meta.cols(), meta.min_x(), meta.max_x(),
                 meta.rows(), meta.min_y(), meta.max_y() );
      out_v.emplace_back(std::move(hist));
    }

    std::vector<int> pl = { _source_plane, _target_plane, _other_plane };
    for ( auto const& trip : _triple_v ) {
      for (size_t i=0; i<3; i++ ) {
        auto const& pix = sparseimg_vv[ pl[i] ][ trip[i] ];
        out_v[ pl[i] ].SetBinContent( pix.col+1, pix.row+1, pix.val+1 );
      }
    }

    return out_v;
  }

  /**
   * @brief make a TH2D in order to visualize the sparse matrix data
   *
   * utility function to help with visual debugging
   *
   * @param[in] adc_v  Vector of image pixel values
   * @param[in] sparseimg_vv Sparse representation (i.e. list of pixels) for each plane.
   * @param[in] hist_stem_name  Stem of histogram name generated.
   *
   */      
  std::vector<TH2D> FlowTriples::plot_sparse_data( const std::vector<larcv::Image2D>& adc_v,
                                                   const std::vector< std::vector<PixData_t> >& sparseimg_vv,
                                                   std::string hist_stem_name ) {

    std::vector<TH2D> out_v;
    
    for ( int p=0; p<(int)adc_v.size(); p++ ) {
      std::stringstream ss;
      ss << "htriples_plane" << p << "_" << hist_stem_name;
      auto const& meta = adc_v[p].meta();
      TH2D hist( ss.str().c_str(), "",
                 meta.cols(), meta.min_x(), meta.max_x(),
                 meta.rows(), meta.min_y(), meta.max_y() );

      for ( auto const& pix : sparseimg_vv[p] ) {
        if (pix.val>=10 )
          hist.SetBinContent( pix.col+1, pix.row+1, pix.val );
      }
      
      out_v.emplace_back(std::move(hist));
    }


    return out_v;
  }

  
  std::vector<TH2D> FlowTriples::plot_cropped_sparse_data( int rowSpan, int colSpan,
                                        const std::vector< std::vector<PixData_t> >& sparseimg_vv,
                                        std::string hist_stem_name ) {

    std::vector<TH2D> out_v;
    
    for ( int p=0; p<(int)sparseimg_vv.size(); p++ ) {
      std::stringstream ss;
      ss << "htriples_plane" << p << "_" << hist_stem_name;
      TH2D hist( ss.str().c_str(), "",
                 colSpan, 0, colSpan,
                 rowSpan, 0, rowSpan );

      for ( auto const& pix : sparseimg_vv[p] ) {
        if (pix.val>=10 )
          hist.SetBinContent( pix.col, pix.row, pix.val );
      }
      
      out_v.emplace_back(std::move(hist));
    }


    return out_v;
  }


  std::vector<TH2D> FlowTriples::plot_cropped_sparse_data( int rowSpan, int colSpan,
                                        const std::vector< std::vector<CropPixData_t> >& sparseimg_vv,
                                        std::string hist_stem_name ) {

    std::vector<TH2D> out_v;
    
    for ( int p=0; p<(int)sparseimg_vv.size(); p++ ) {
      std::stringstream ss;
      ss << "htriples_plane" << p << "_" << hist_stem_name;
      TH2D hist( ss.str().c_str(), "",
                 colSpan, 0, colSpan,
                 rowSpan, 0, rowSpan );

      for ( auto const& pix : sparseimg_vv[p] ) {
        if (pix.val>=10 )
          hist.SetBinContent( pix.col, pix.row, pix.val );
      }
      
      out_v.emplace_back(std::move(hist));
    }


    return out_v;
  }

  
  /**
   * @brief convert the wire image data into a sparse represntation
   *
   * we convert the image into a vector of PixData_t objects.
   *
   * @param[in] adc_v  Vector of image pixel values
   * @param[in] threshold Keep only pixels with value above this threshold
   * @return    Vector of images in sparse representation (i.e. a list of pixels above threshold)
   */        
  std::vector< std::vector<FlowTriples::PixData_t> >
  FlowTriples::make_initial_sparse_image( const std::vector<larcv::Image2D>& adc_v,
                                          float threshold ) {

    // sparsify planes: pixels must be above threshold
    std::vector< std::vector<FlowTriples::PixData_t> > sparseimg_vv(adc_v.size());
    
    for ( size_t p=0; p<adc_v.size(); p++ ) {
      sparseimg_vv[p].reserve( (int)( 0.1 * adc_v[p].as_vector().size() ) );

      for ( size_t r=0; r<adc_v[p].meta().rows(); r++ ) {
        for ( size_t c=0; c<adc_v[p].meta().cols(); c++ ) {
          float val = adc_v[p].pixel(r,c);
          if ( val>=threshold ) {
            sparseimg_vv[p].push_back( PixData_t((int)r,(int)c, val) );
          }
        }
      }
      // should be sorted in (r,c). do i pull the trigger and sort?
      // std::sort( sparseimg_vv[p].begin(), sparseimg_vv[p].end() );
      int idx=0;
      for ( auto& pix : sparseimg_vv[p] ) {
        pix.idx = idx;
        idx++;
      }
      std::cout << "[FlowTriples] plane[" << p << "] has " << sparseimg_vv[p].size() << " (above threshold) pixels" << std::endl;
    }

    return sparseimg_vv;
  }

  /**
   * @brief convert the wire image data into a sparse represntation of input prong
   *
   * we convert the image into a vector of PixData_t objects.
   * ignore all pixels not belonging to the input trackid using truth
   *
   * @param[in] adc_v  Vector of image pixel values
   * @param[in] mcpg  MCParticlePGraph (used to select prong pixels)
   * @param[in] ioll  larlite storage manager (used to get truth info for prong start/end)
   * @param[in] trackid  trackid for selected prong
   * @param[in] threshold  Keep only pixels with value above this threshold
   * @param[in] rowSpan  number of rows in cropped image
   * @param[in] colSpan  number of columns in cropped image
   * @param[in] shower  ensure beginning (end) of shower (track) is in image if true (false)
   * @return    Vector of images in sparse representation (i.e. a list of pixels above threshold)
   */        
  std::vector< std::vector<FlowTriples::PixData_t> >
  FlowTriples::make_cropped_initial_sparse_prong_image_truth( const std::vector<larcv::Image2D>& adc_v,
                                                              ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                                              larlite::storage_manager& ioll, 
                                                              int trackid, float threshold,
                                                              int rowSpan, int colSpan,
                                                              bool shower ) {

    // sparsify planes: pixels must be above threshold
    std::vector< std::vector<FlowTriples::PixData_t> > sparseimg_vv(adc_v.size()*2);
    
    // pixels must belong to particle with input trackid - get corresponding rows, columns
    const auto partPix_vv = mcpg.getPixelsFromParticleAndDaughters(trackid);

    std::vector< std::vector<int> > prongBounds;
    for ( size_t p=0; p<adc_v.size(); p++ ) {
      std::vector<int> planeProngBounds{9999999,-9999999,9999999,-9999999};
      for ( unsigned int iP = 0; iP < partPix_vv[p].size()/2; iP++ ) {
        int row = (partPix_vv[p][2*iP] - 2400)/6;
        int col = partPix_vv[p][2*iP+1];
        if(row < planeProngBounds[0]) planeProngBounds[0] = row;
        if(row > planeProngBounds[1]) planeProngBounds[1] = row;
        if(col < planeProngBounds[2]) planeProngBounds[2] = col;
        if(col > planeProngBounds[3]) planeProngBounds[3] = col;
      }
      prongBounds.push_back(planeProngBounds);
    }

    bool reCenterAny = false;
    std::vector<bool> reCenter;
    std::vector< std::vector<int> > imgBounds;
    for ( size_t p=0; p<adc_v.size(); p++ ) {
      std::vector<int> imgPlaneBounds{0, 0, 0, 0};
      if( (prongBounds[p][1] - prongBounds[p][0]) < rowSpan && 
          (prongBounds[p][3] - prongBounds[p][2]) < colSpan    ){
        reCenter.push_back(false);
        imgPlaneBounds[0] = (prongBounds[p][0] + prongBounds[p][1])/2 - rowSpan/2;
        imgPlaneBounds[2] = (prongBounds[p][2] + prongBounds[p][3])/2 - colSpan/2;
      } else{
        reCenterAny = true;
        reCenter.push_back(true);
      }
      imgBounds.push_back(imgPlaneBounds);
    }

    if(reCenterAny){

      float centerX, centerY, centerZ;
      if(shower){
        larlite::event_mcshower* ev_mcshower
          = (larlite::event_mcshower*)ioll.get_data(larlite::data::kMCShower,"mcreco");
        for(const auto& mcshower : *ev_mcshower){
          if(mcshower.TrackID() == trackid){
            centerX = mcshower.Start().X();
            centerY = mcshower.Start().Y();
            centerZ = mcshower.Start().Z();
            break;
          }
        }
      } else{
        larlite::event_mctrack* ev_mctrack
          = (larlite::event_mctrack*)ioll.get_data(larlite::data::kMCTrack,"mcreco");
        for(const auto& mctrack : *ev_mctrack){
          if(mctrack.TrackID() == trackid){
            centerX = mctrack.End().X();
            centerY = mctrack.End().Y();
            centerZ = mctrack.End().Z();
            break;
          }
        }
      }

      for ( size_t p=0; p<adc_v.size(); p++ ) {
       if(!reCenter[p]) continue;
       auto center2D = larutil::GeometryHelper::GetME()->Point_3Dto2D(centerX,centerY,centerZ,p);
       int center2Dr = (int)((center2D.t/larutil::GeometryHelper::GetME()->TimeToCm() + 800.)/6.);
       int center2Dc = (int)(center2D.w/larutil::GeometryHelper::GetME()->WireToCm());
       imgBounds[p][0] = center2Dr - rowSpan/2;
       imgBounds[p][2] = center2Dc - colSpan/2;
      }

    }

    for ( size_t p=0; p<adc_v.size(); p++ ) {
      if(imgBounds[p][0] < 0) imgBounds[p][0] = 0;
      if(imgBounds[p][0] + rowSpan > adc_v[p].meta().rows())
        imgBounds[p][0] = adc_v[p].meta().rows() - rowSpan;
      imgBounds[p][1] = imgBounds[p][0] + rowSpan;
      if(imgBounds[p][2] < 0) imgBounds[p][2] = 0;
      if(imgBounds[p][2] + colSpan > adc_v[p].meta().cols())
        imgBounds[p][2] = adc_v[p].meta().cols() - colSpan;
      imgBounds[p][3] = imgBounds[p][2] + colSpan;
    }

    for ( size_t p=0; p<adc_v.size(); p++ ) {

      sparseimg_vv[p].reserve( (int)( 0.1 * adc_v[p].as_vector().size() ) );
      sparseimg_vv[p+3].reserve( (int)( 0.1 * adc_v[p].as_vector().size() ) );

      for ( unsigned int iP = 0; iP < partPix_vv[p].size()/2; iP++ ) {
        // TO DO: REPLACE HARD-CODED VALUES!!!
        int row = (partPix_vv[p][2*iP] - 2400)/6;
        int col = partPix_vv[p][2*iP+1];
        float val = adc_v[p].pixel(row, col);
          if ( val>=threshold && 
               row >= imgBounds[p][0] && row < imgBounds[p][1] &&
               col >= imgBounds[p][2] && col < imgBounds[p][3] ) {
            sparseimg_vv[p].push_back( PixData_t(row - imgBounds[p][0],
                                                 col - imgBounds[p][2], val) );
          }
      }

      for ( size_t row=0; row<adc_v[p].meta().rows(); row++ ) {
        for ( size_t col=0; col<adc_v[p].meta().cols(); col++ ) {
          float val = adc_v[p].pixel(row,col);
          if ( val>=threshold && 
               (int)row >= imgBounds[p][0] && (int)row < imgBounds[p][1] &&
               (int)col >= imgBounds[p][2] && (int)col < imgBounds[p][3] ) {
            sparseimg_vv[p+3].push_back( PixData_t((int)row - imgBounds[p][0],
                                                   (int)col - imgBounds[p][2], val) );
          }
        }
      }

      // should be sorted in (r,c). do i pull the trigger and sort?
      // std::sort( sparseimg_vv[p].begin(), sparseimg_vv[p].end() );
      int idx=0;
      for ( auto& pix : sparseimg_vv[p] ) {
        pix.idx = idx;
        idx++;
      }
      idx=0;
      for ( auto& pix : sparseimg_vv[p+3] ) {
        pix.idx = idx;
        idx++;
      }
      //std::cout << "[FlowTriples] plane[" << p << "] has " << sparseimg_vv[p].size() << " (above threshold) pixels" << std::endl;
    }

    return sparseimg_vv;
  }

  /**
   * @brief convert the wire image data into a sparse represntation of input prong
   *
   * we convert the image into a vector of CropPixData_t objects.
   * ignore all pixels not belonging to the input reco cluster
   *
   * @param[in] adc_v  Vector of image pixel values
   * @param[in] thrumu_v  Vector of cosmic image pixel values
   * @param[in] prong  larflowcluster object with reco prong hits
   * @param[in] cropCenter  point to center crop if input prong exceeds rowSpan or colSpan
   * @param[in] threshold  Keep only pixels with value above this threshold
   * @param[in] rowSpan  number of rows in cropped image
   * @param[in] colSpan  number of columns in cropped image
   * @return    Vector of images in sparse representation (i.e. a list of pixels above threshold)
   */        
  std::vector< std::vector<FlowTriples::CropPixData_t> >
  FlowTriples::make_cropped_initial_sparse_prong_image_reco( const std::vector<larcv::Image2D>& adc_v,
                                                             const std::vector<larcv::Image2D>& thrumu_v,
                                                             const larlite::larflowcluster& prong,
                                                             const TVector3& cropCenter, 
                                                             float threshold, int rowSpan, int colSpan ) {

    // sparsify planes: pixels must be above threshold
    std::vector< std::vector<FlowTriples::CropPixData_t> > sparseimg_vv(adc_v.size()*2);
    for ( size_t p=0; p<adc_v.size(); p++ ) {
      sparseimg_vv[p].reserve( (int)( 0.1 * adc_v[p].as_vector().size() ) );
      sparseimg_vv[p+3].reserve( (int)( 0.1 * adc_v[p].as_vector().size() ) );
    }

    std::vector< std::vector<int> > imgBounds;
    getRecoImageBounds(imgBounds, adc_v, prong, cropCenter, rowSpan, colSpan);

    fillProngImagesFromReco(sparseimg_vv, threshold, adc_v, thrumu_v, prong, imgBounds);
    fillContextImages(sparseimg_vv, threshold, adc_v, thrumu_v, imgBounds);

    return sparseimg_vv;
  }

  /**
   * @brief convert the wire image data into a sparse represntation of input prong
   *
   * we convert the image into a vector of CropPixData_t objects.
   * ignore all pixels not belonging to the input reco cluster
   *
   * @param[in] adc_v  Vector of image pixel values
   * @param[in] thrumu_v  Vector of cosmic image pixel values
   * @param[in] prong  larflowcluster object with reco prong hits
   * @param[in] cropCenter  point to center crop if input prong exceeds rowSpan or colSpan
   * @param[in] threshold  Keep only pixels with value above this threshold
   * @param[in] rowSpan  number of rows in cropped image
   * @param[in] colSpan  number of columns in cropped image
   * @param[in] mcpg  MCPixelPGraph used to remove specified particle
   * @param[in] trackid_rm  track id of particle to remove from context images
   * @param[in] trackid_rm2  optional: track id of 2nd particle to remove from context images
   * @return    Vector of images in sparse representation (i.e. a list of pixels above threshold)
   */        
  std::vector< std::vector<FlowTriples::CropPixData_t> >
  FlowTriples::make_cropped_initial_sparse_prong_image_reco_rmContextPart(
                                                             const std::vector<larcv::Image2D>& adc_v,
                                                             const std::vector<larcv::Image2D>& thrumu_v,
                                                             const larlite::larflowcluster& prong,
                                                             const TVector3& cropCenter, 
                                                             float threshold, int rowSpan, int colSpan,
                                                             ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                                             int trackid_rm, int trackid_rm2 ) {

    // sparsify planes: pixels must be above threshold
    std::vector< std::vector<FlowTriples::CropPixData_t> > sparseimg_vv(adc_v.size()*2);
    for ( size_t p=0; p<adc_v.size(); p++ ) {
      sparseimg_vv[p].reserve( (int)( 0.1 * adc_v[p].as_vector().size() ) );
      sparseimg_vv[p+3].reserve( (int)( 0.1 * adc_v[p].as_vector().size() ) );
    }

    std::vector< std::vector<int> > imgBounds;
    getRecoImageBounds(imgBounds, adc_v, prong, cropCenter, rowSpan, colSpan);

    fillProngImagesFromReco(sparseimg_vv, threshold, adc_v, thrumu_v, prong, imgBounds);
    fillPartRmContextImages(sparseimg_vv, threshold, adc_v, thrumu_v, imgBounds, mcpg, trackid_rm, trackid_rm2);

    return sparseimg_vv;
  }

  /**
   * @brief convert the wire image data into a sparse represntation of input prong
   *
   * we convert the image into a vector of CropPixData_t objects.
   * ignore all pixels not belonging to the input reco cluster
   *
   * @param[in] adc_v_reco  Vector of image pixel values in reco input
   * @param[in] adc_v_sim  Vector of image pixel values in simulated prong for substitution
   * @param[in] thrumu_v  Vector of cosmic image pixel values
   * @param[in] prong  larflowcluster object with reco prong hits
   * @param[in] cropCenter  point to center crop if input prong exceeds rowSpan or colSpan
   * @param[in] threshold  Keep only pixels with value above this threshold
   * @param[in] rowSpan  number of rows in cropped image
   * @param[in] colSpan  number of columns in cropped image
   * @param[in] mcpg  MCPixelPGraph used to remove specified particle
   * @param[in] trackid_rm  track id of particle to remove from context images
   * @param[in] trackid_rm2  optional: track id of 2nd particle to remove from context images
   * @return    Vector of images in sparse representation (i.e. a list of pixels above threshold)
   */        
  std::vector< std::vector<FlowTriples::CropPixData_t> >
  FlowTriples::make_cropped_initial_sparse_prong_image_reco_subContextPart(
                                                             const std::vector<larcv::Image2D>& adc_v_reco,
                                                             const std::vector<larcv::Image2D>& adc_v_sim, 
                                                             const std::vector<larcv::Image2D>& thrumu_v,
                                                             const larlite::larflowcluster& prong,
                                                             const TVector3& cropCenter, 
                                                             float threshold, int rowSpan, int colSpan,
                                                             ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                                             int trackid_rm, int trackid_rm2 ) {

    // sparsify planes: pixels must be above threshold
    std::vector< std::vector<FlowTriples::CropPixData_t> > sparseimg_vv(adc_v_reco.size()*2);
    for ( size_t p=0; p<adc_v_reco.size(); p++ ) {
      sparseimg_vv[p].reserve( (int)( 0.1 * adc_v_reco[p].as_vector().size() ) );
      sparseimg_vv[p+3].reserve( (int)( 0.1 * adc_v_reco[p].as_vector().size() ) );
    }

    std::vector< std::vector<int> > imgBounds;
    getRecoImageBounds(imgBounds, adc_v_reco, prong, cropCenter, rowSpan, colSpan);

    fillProngImagesFromReco(sparseimg_vv, threshold, adc_v_reco, thrumu_v, prong, imgBounds);
    fillPartRmContextImages(sparseimg_vv, threshold, adc_v_reco, thrumu_v, imgBounds, mcpg, trackid_rm, trackid_rm2);
    fillContextImages(sparseimg_vv, threshold, adc_v_sim, imgBounds);

    return sparseimg_vv;
  }

  /**
   * @brief convert the wire image data into a sparse represntation of input prong
   *
   * we convert the image into a vector of CropPixData_t objects.
   * ignore all pixels not belonging to the input reco cluster
   *
   * @param[in] adc_v_reco  Vector of image pixel values in reco input
   * @param[in] adc_v_sim  Vector of image pixel values in simulated prong for substitution
   * @param[in] thrumu_v  Vector of cosmic image pixel values
   * @param[in] prong  larflowcluster object with reco prong hits
   * @param[in] cropCenter  point to center crop if input prong exceeds rowSpan or colSpan
   * @param[in] threshold  Keep only pixels with value above this threshold
   * @param[in] rowSpan  number of rows in cropped image
   * @param[in] colSpan  number of columns in cropped image
   * @param[in] mcpg  MCPixelPGraph used to remove specified particle
   * @param[in] trackid_rm  track id of particle to remove from context images
   * @param[in] trackid_rm2  optional: track id of 2nd particle to remove from context images
   * @return    Vector of images in sparse representation (i.e. a list of pixels above threshold)
   */        
  std::vector< std::vector<FlowTriples::CropPixData_t> >
  FlowTriples::make_cropped_initial_sparse_prong_image_reco_truthProngSub(
                                                             const std::vector<larcv::Image2D>& adc_v_reco,
                                                             const std::vector<larcv::Image2D>& adc_v_sim, 
                                                             const std::vector<larcv::Image2D>& thrumu_v,
                                                             const larlite::larflowcluster& prong,
                                                             const TVector3& cropCenter, 
                                                             float threshold, int rowSpan, int colSpan,
                                                             ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                                             int trackid_rm, int trackid_rm2 ) {

    // sparsify planes: pixels must be above threshold
    std::vector< std::vector<FlowTriples::CropPixData_t> > sparseimg_vv(adc_v_reco.size()*2);
    for ( size_t p=0; p<adc_v_reco.size(); p++ ) {
      sparseimg_vv[p].reserve( (int)( 0.1 * adc_v_reco[p].as_vector().size() ) );
      sparseimg_vv[p+3].reserve( (int)( 0.1 * adc_v_reco[p].as_vector().size() ) );
    }

    std::vector< std::vector<int> > imgBounds;
    getRecoImageBounds(imgBounds, adc_v_reco, prong, cropCenter, rowSpan, colSpan);

    fillProngImagesFromTruth(sparseimg_vv, threshold, adc_v_sim, imgBounds);
    fillPartRmContextImages(sparseimg_vv, threshold, adc_v_reco, thrumu_v, imgBounds, mcpg, trackid_rm, trackid_rm2);

    return sparseimg_vv;
  }



  void FlowTriples::getRecoImageBounds( std::vector< std::vector<int> >& imgBounds,
                                        const std::vector<larcv::Image2D>& adc_v, 
                                        const larlite::larflowcluster& prong,
                                        const TVector3& cropCenter, int rowSpan, int colSpan ) {

    std::vector< std::vector<int> > prongBounds;
    for ( size_t p=0; p<adc_v.size(); p++ ) {
      std::vector<int> planeProngBounds{9999999,-9999999,9999999,-9999999};
      for( const auto& hit : prong ){
        int row = (hit.tick - 2400)/6;
        int col = hit.targetwire[p];
        if(row < planeProngBounds[0]) planeProngBounds[0] = row;
        if(row > planeProngBounds[1]) planeProngBounds[1] = row;
        if(col < planeProngBounds[2]) planeProngBounds[2] = col;
        if(col > planeProngBounds[3]) planeProngBounds[3] = col;
      }
      prongBounds.push_back(planeProngBounds);
    }

    bool reCenterAny = false;
    std::vector<bool> reCenter;
    for ( size_t p=0; p<adc_v.size(); p++ ) {
      std::vector<int> imgPlaneBounds{0, 0, 0, 0};
      if( (prongBounds[p][1] - prongBounds[p][0]) < rowSpan && 
          (prongBounds[p][3] - prongBounds[p][2]) < colSpan    ){
        reCenter.push_back(false);
        imgPlaneBounds[0] = (prongBounds[p][0] + prongBounds[p][1])/2 - rowSpan/2;
        imgPlaneBounds[2] = (prongBounds[p][2] + prongBounds[p][3])/2 - colSpan/2;
      } else{
        reCenterAny = true;
        reCenter.push_back(true);
      }
      imgBounds.push_back(imgPlaneBounds);
    }

    if(reCenterAny){

      for ( size_t p=0; p<adc_v.size(); p++ ) {
       if(!reCenter[p]) continue;
       auto center2D = larutil::GeometryHelper::GetME()->Point_3Dto2D(cropCenter.X(),cropCenter.Y(),cropCenter.Z(),p);
       int center2Dr = (int)((center2D.t/larutil::GeometryHelper::GetME()->TimeToCm() + 800.)/6.);
       int center2Dc = (int)(center2D.w/larutil::GeometryHelper::GetME()->WireToCm());
       imgBounds[p][0] = center2Dr - rowSpan/2;
       imgBounds[p][2] = center2Dc - colSpan/2;
      }

    }

    for ( size_t p=0; p<adc_v.size(); p++ ) {
      if(imgBounds[p][0] < 0) imgBounds[p][0] = 0;
      if(imgBounds[p][0] + rowSpan > adc_v[p].meta().rows())
        imgBounds[p][0] = adc_v[p].meta().rows() - rowSpan;
      imgBounds[p][1] = imgBounds[p][0] + rowSpan;
      if(imgBounds[p][2] < 0) imgBounds[p][2] = 0;
      if(imgBounds[p][2] + colSpan > adc_v[p].meta().cols())
        imgBounds[p][2] = adc_v[p].meta().cols() - colSpan;
      imgBounds[p][3] = imgBounds[p][2] + colSpan;
    }

    return;
  }



  void FlowTriples::fillProngImagesFromReco(std::vector< std::vector<FlowTriples::CropPixData_t> >& sparseimg_vv,
                                            const float& threshold,
                                            const std::vector<larcv::Image2D>& adc_v,
                                            const std::vector<larcv::Image2D>& thrumu_v,
                                            const larlite::larflowcluster& prong,
                                            const std::vector< std::vector<int> >& imgBounds) {

    for ( size_t p=0; p<adc_v.size(); p++ ) {

      for( const auto& hit : prong ){
        // TO DO: REPLACE HARD-CODED VALUES!!!
        int row = (hit.tick - 2400)/6;
        int col = hit.targetwire[p];
        float val = adc_v[p].pixel(row, col);
        float val_cosmic = thrumu_v[p].pixel(row, col);
          if ( val >= threshold && val_cosmic < threshold &&
               row >= imgBounds[p][0] && row < imgBounds[p][1] &&
               col >= imgBounds[p][2] && col < imgBounds[p][3] ) {
            CropPixData_t cropPixData(row - imgBounds[p][0], col - imgBounds[p][2], row, col, val);
            if( std::find(sparseimg_vv[p].begin(), sparseimg_vv[p].end(), cropPixData) == 
                sparseimg_vv[p].end() )
              sparseimg_vv[p].push_back(cropPixData);
          }
      }

      int idx=0;
      for ( auto& pix : sparseimg_vv[p] ) {
        pix.idx = idx;
        idx++;
      }

    }

    return;
  }


  void FlowTriples::fillProngImagesFromTruth(std::vector< std::vector<FlowTriples::CropPixData_t> >& sparseimg_vv,
                                             const float& threshold,
                                             const std::vector<larcv::Image2D>& adc_v,
                                             const std::vector< std::vector<int> >& imgBounds){

    for ( size_t p=0; p<adc_v.size(); p++ ) {

      for ( size_t row=0; row<adc_v[p].meta().rows(); row++ ) {
        for ( size_t col=0; col<adc_v[p].meta().cols(); col++ ) {
          float val = adc_v[p].pixel(row,col);
          if ( val>=threshold && 
               (int)row >= imgBounds[p][0] && (int)row < imgBounds[p][1] &&
               (int)col >= imgBounds[p][2] && (int)col < imgBounds[p][3] ) {
            sparseimg_vv[p].push_back( CropPixData_t((int)row - imgBounds[p][0],
                                                     (int)col - imgBounds[p][2], (int)row, (int)col, val) );
            sparseimg_vv[p+3].push_back( CropPixData_t((int)row - imgBounds[p][0],
                                                       (int)col - imgBounds[p][2], (int)row, (int)col, val) );
          }
        }
      }

      int idx=0;
      for ( auto& pix : sparseimg_vv[p] ) {
        pix.idx = idx;
        idx++;
      }
    }

    return;
  }


  void FlowTriples::fillContextImages(std::vector< std::vector<FlowTriples::CropPixData_t> >& sparseimg_vv,
                                      const float& threshold,
                                      const std::vector<larcv::Image2D>& adc_v,
                                      const std::vector<larcv::Image2D>& thrumu_v,
                                      const std::vector< std::vector<int> >& imgBounds){

    for ( size_t p=0; p<adc_v.size(); p++ ) {

      for ( size_t row=0; row<adc_v[p].meta().rows(); row++ ) {
        for ( size_t col=0; col<adc_v[p].meta().cols(); col++ ) {
          float val = adc_v[p].pixel(row,col);
          float val_cosmic = thrumu_v[p].pixel(row, col);
          if ( val >= threshold && val_cosmic < threshold &&
               (int)row >= imgBounds[p][0] && (int)row < imgBounds[p][1] &&
               (int)col >= imgBounds[p][2] && (int)col < imgBounds[p][3] ) {
            sparseimg_vv[p+3].push_back( CropPixData_t((int)row - imgBounds[p][0],
                                                       (int)col - imgBounds[p][2], (int)row, (int)col, val) );
          }
        }
      }

      int idx=0;
      for ( auto& pix : sparseimg_vv[p+3] ) {
        pix.idx = idx;
        idx++;
      }
    }

    return;
  }


  void FlowTriples::fillContextImages(std::vector< std::vector<FlowTriples::CropPixData_t> >& sparseimg_vv,
                                      const float& threshold,
                                      const std::vector<larcv::Image2D>& adc_v,
                                      const std::vector< std::vector<int> >& imgBounds){

    for ( size_t p=0; p<adc_v.size(); p++ ) {

      for ( size_t row=0; row<adc_v[p].meta().rows(); row++ ) {
        for ( size_t col=0; col<adc_v[p].meta().cols(); col++ ) {
          float val = adc_v[p].pixel(row,col);
          if ( val >= threshold &&
               (int)row >= imgBounds[p][0] && (int)row < imgBounds[p][1] &&
               (int)col >= imgBounds[p][2] && (int)col < imgBounds[p][3] ) {
            sparseimg_vv[p+3].push_back( CropPixData_t((int)row - imgBounds[p][0],
                                                       (int)col - imgBounds[p][2], (int)row, (int)col, val) );
          }
        }
      }

      int idx=0;
      for ( auto& pix : sparseimg_vv[p+3] ) {
        pix.idx = idx;
        idx++;
      }
    }

    return;
  }


  void FlowTriples::fillPartRmContextImages(std::vector< std::vector<FlowTriples::CropPixData_t> >& sparseimg_vv,
                                            const float& threshold,
                                            const std::vector<larcv::Image2D>& adc_v,
                                            const std::vector<larcv::Image2D>& thrumu_v,
                                            const std::vector< std::vector<int> >& imgBounds,
                                            ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                            const int& trackid_rm, const int& trackid_rm2){

    //get simulated particle pixels to remove from context
    const auto partPix_vv = mcpg.getPixelsFromParticleAndDaughters(trackid_rm);
    std::vector< std::vector<int> > partPix_vv_2;
    if(trackid_rm2 >= 0) partPix_vv_2 = mcpg.getPixelsFromParticleAndDaughters(trackid_rm2);

    for ( size_t p=0; p<adc_v.size(); p++ ) {

      //sort simulated particle pixels into map for logarithmic lookup
      std::map<int, std::vector<int> > partPix_map;
      for ( unsigned int iP = 0; iP < partPix_vv[p].size()/2; iP++ ) {
        int partRow = (partPix_vv[p][2*iP] - 2400)/6;
        int partCol = partPix_vv[p][2*iP+1];
        if(partPix_map.find(partRow) != partPix_map.end()){
          partPix_map[partRow].push_back(partCol);
        }
        else{
          std::vector<int> colVec;
          colVec.reserve(10);
          colVec.push_back(partCol);
          partPix_map[partRow] = colVec;
        }
      }
      for ( auto& rowEntries : partPix_map ) {
        std::sort(rowEntries.second.begin(), rowEntries.second.end());
      }
      if(trackid_rm2 >= 0){
        for ( unsigned int iP = 0; iP < partPix_vv_2[p].size()/2; iP++ ) {
          int partRow = (partPix_vv_2[p][2*iP] - 2400)/6;
          int partCol = partPix_vv_2[p][2*iP+1];
          if(partPix_map.find(partRow) != partPix_map.end()){
            bool pixInMap = std::binary_search(partPix_map[partRow].begin(), partPix_map[partRow].end(), partCol);
            if(!pixInMap) partPix_map[partRow].push_back(partCol);
          }
          else{
            std::vector<int> colVec;
            colVec.reserve(10);
            colVec.push_back(partCol);
            partPix_map[partRow] = colVec;
          }
        }
        for ( auto& rowEntries : partPix_map ) {
          std::sort(rowEntries.second.begin(), rowEntries.second.end());
        }
      }

      for ( size_t row=0; row<adc_v[p].meta().rows(); row++ ) {
        bool rowInPixMap = false;
        bool firstRow = true;
        for ( size_t col=0; col<adc_v[p].meta().cols(); col++ ) {
          float val = adc_v[p].pixel(row,col);
          float val_cosmic = thrumu_v[p].pixel(row, col);
          if ( val >= threshold && val_cosmic < threshold &&
               (int)row >= imgBounds[p][0] && (int)row < imgBounds[p][1] &&
               (int)col >= imgBounds[p][2] && (int)col < imgBounds[p][3] ) {
            if(firstRow){
              rowInPixMap = (partPix_map.find((int)row) != partPix_map.end());
              firstRow = false;
            }
            if( rowInPixMap && std::binary_search(partPix_map[(int)row].begin(),
                                                  partPix_map[(int)row].end(), (int)col) ) continue;
            sparseimg_vv[p+3].push_back( CropPixData_t((int)row - imgBounds[p][0],
                                                       (int)col - imgBounds[p][2], (int)row, (int)col, val) );
          }
        }
      }

      int idx=0;
      for ( auto& pix : sparseimg_vv[p+3] ) {
        pix.idx = idx;
        idx++;
      }
    }

    return;
  }


}
}
