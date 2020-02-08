#include "FlowTriples.h"
#include "WireOverlap.h"

#include <ctime>
#include <sstream>

namespace larflow {

  /**
   * generate list of possible combinations of (U,V,Y) intersections given source to target plane matching.
   *
   * this version takes image2d as input
   *
   * @param[in] source The source plane index, where we start the match with a pixel above threshold.
   * @param[in] target The target plane index, where we match to pixels above threshold
   * @param[in] adc_v  Vector of image pixel values
   * @param[in] badc_v Vector of images labeling bad channels
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
   * generate list of possible combinations of (U,V,Y) intersections given source to target plane matching.
   *
   * this version takes the sparse matrix representation of the image, skipping its generation
   *
   * @param[in] source The source plane index, where we start the match with a pixel above threshold.
   * @param[in] target The target plane index, where we match to pixels above threshold
   * @param[in] adc_v  Vector of image pixel values
   * @param[in] badc_v Vector of images labeling bad channels
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
   * using the list of pixels in the sparse matrix representation, build the list of possible
   *  three-plane triplets
   *
   * @param[in] source The source plane index, where we start the match with a pixel above threshold.
   * @param[in] target The target plane index, where we match to pixels above threshold
   * @param[in] adc_v  Vector of image pixel values
   * @param[in] badc_v Vector of images labeling bad channels
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

    // make possible triples based on going from source pixel to target pixels
    for ( auto& srcpix : sparseimg_vv[_source_plane] ) {

      // we get all the target plane wires this pixel overlaps with
      std::vector< std::vector<int> > overlap = larflow::WireOverlap::getOverlappingWires( _source_plane, _target_plane, srcpix.col );
      // std::cout << "FlowTriples[" << _source_plane << "," << _target_plane << "," << _other_plane << "] "
      //           << " srcwire=" << srcpix.col << " row=" << srcpix.row 
      //           << " tar-overlap=[" << overlap[0].front() << "," << overlap[0].back() << "] "
      //           << " oth-overlap=[" << overlap[1].front() << "," << overlap[1].back() << "] "
      //           << std::endl;
      
      if ( overlap[0].size()==0 ) {
        //std::cout << " -- target no overlap" << std::endl;
        continue;
      }

      // get iterator to overlap[0] vector
      auto it_overlap0 = overlap[0].begin();
      
      // get the lowerbound pixel for (row,col) in target image
      PixData_t lb( srcpix.row, (int)overlap[0][0], 0.0 );
      auto it_target = std::lower_bound( sparseimg_vv[_target_plane].begin(), sparseimg_vv[_target_plane].end(), lb );
      //if ( it_target!=sparseimg_vv[_target_plane].end() )
      //std::cout << " tarlb(row,col)=(" << it_target->row << "," << it_target->col << ") ";

      // std::cout << " other=[" << overlap[1].front() << "," << overlap[1].back() << "]"
      //           << std::endl;

      // iterator forward in target wire with charge, until we go to another row (different time)
      while ( it_target!=sparseimg_vv[_target_plane].end() && it_target->row==srcpix.row ) {

        // break if we out of range over the target wire
        if ( it_target->col>overlap[0].back() || it_target->row!=srcpix.row ) break;

        // find position in overlap[0] in order to get overlap[1] element, i.e. the other wire the source+target wires intersect
        it_overlap0 = std::lower_bound( it_overlap0, overlap[0].end(), it_target->col );
        if ( it_overlap0==overlap[0].end() ) {
          //std::cout << " column not in list: break" << std::endl;
          break; // didnt find the target column in the list
        }

        // scan up until matches
        int ivec = -1;
        while ( *it_overlap0<=it_target->col && it_overlap0!=overlap[0].end()) {
          ivec = it_overlap0-overlap[0].begin();
          it_overlap0++;
        }
        //std::cout << " ivec=" << ivec << std::endl;

        int otherwire = overlap[1][ivec];
        //std::cout << "  ... search for otherwire=" << otherwire << " ivec=" << ivec << std::endl;
        
        // now find the other plane pixel in the sparse matrix.
        // we allow for a little slop
        // first search for pixel in other plane sparseimg vector (that is above threshold)
        auto it_other = std::lower_bound( sparseimg_vv[_other_plane].begin(),
                                          sparseimg_vv[_other_plane].end(),
                                          PixData_t( srcpix.row, otherwire-2, 0.0) );

        // for debug
        // if ( it_other!=sparseimg_vv[_other_plane].end() ) {
        //   std::cout << " ... otherlb(r,c)=" << it_other->row << "," << it_other->col << ")" << std::endl;
        // }

        // now we scan through the pixels in the other plane
        bool found = false;        
        if ( it_other!=sparseimg_vv[_other_plane].end() && it_other->row==srcpix.row && it_other->col<=otherwire ) {
          // pixels to search
          while ( it_other!=sparseimg_vv[_other_plane].end() && it_other->row==srcpix.row ) {
            // if a pixel in the other plane is found close to the one we searched for,
            //  we know it has charge, so we store the triple
            //valid triple
            // std::cout << " .... looking for row=" << srcpix.row << " otherwire=" << otherwire
            //           << " cols=(" << srcpix.col << "," << it_target->col << "," << it_other->col << ")"
            //           << std::endl;
            
            if ( abs(it_other->col-otherwire)<=1 ) {
              //std::cout << "  ... found triple with charge" << std::endl;
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
          // std::cout << " ... looking for badch row=" << srcpix.row
          //           << " cols=(" << srcpix.col << "," << it_target->col << "," << otherwire << ")"
          //           << std::endl;          
          // std::cout << "  ... found triple with dead channel" << std::endl;

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

          // std::cout << "looking for row=" << srcpix.row
          //           << " cols=(" << srcpix.col << "," << it_target->col << "," << otherwire << ")"
          //           << std::endl;                    
          
          // if ( it_other==sparseimg_vv[_other_plane].end() )
          //   std::cout << "  ... no triple" << std::endl;
          // else
          //   std::cout << "  ... lowerbound found, but no match? lowerbound otherpix=(" <<it_other->row << "," << it_other->col << ")" <<  std::endl;
          
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
   * use a th2d to plot the projected locations of the triplets in each of the planes
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
   * make a TH2D in order to visualize the sparse matrix data
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

  
  /**
   * convert the wire image data into a sparse represntation
   *
   * @param[in] adc_v  Vector of image pixel values
   * @param[in] hist_stem_name  Stem of histogram name generated.
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
  
}
