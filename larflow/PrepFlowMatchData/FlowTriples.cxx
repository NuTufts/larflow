#include "FlowTriples.h"
#include "WireOverlap.h"

#include <ctime>

namespace larflow {

  FlowTriples::FlowTriples( int source, int target,
                            const std::vector<larcv::Image2D>& adc_v,
                            const std::vector<larcv::Image2D>& badch_v,
                            float threshold ) {

    std::clock_t start_ = std::clock();

    _source_plane = source;
    _target_plane = target;
    switch ( source ) {
    case 0:
      _other_plane = ( target==1 ) ? 2 : 1;
      break;
    case 1:
      _other_plane = ( target==0 ) ? 2 : 0;
      break;
    case 2:
      _other_plane = ( target==2 ) ? 1 : 2;
      break;
    default:
      char msg[100];
      sprintf( msg, "Unrecognized combination of source[%d] and target[%d] planes", source, target );
      throw std::runtime_error( msg );
      break;
    }

    // sparsify planes
    _sparseimg_vv.resize(adc_v.size());
    for ( size_t p=0; p<adc_v.size(); p++ ) {
      _sparseimg_vv.reserve( (int)0.1 * adc_v[p].as_vector().size() );

      for ( size_t r=0; r<adc_v[p].meta().rows(); r++ ) {
        for ( size_t c=0; c<adc_v[p].meta().cols(); c++ ) {
          float val = adc_v[p].pixel(r,c);
          if ( val>=threshold ) {
            _sparseimg_vv[p].push_back( PixData_t((int)r,(int)c, val) );
          }
        }
      }
      // should be sorted in (r,c). do i pull the trigger and sort?
      // std::sort( _sparseimg_vv[p].begin(), _sparseimg_vv[p].end() );
      int idx=0;
      for ( auto& pix : _sparseimg_vv[p] ) {
        pix.idx = idx;
        idx++;
      }
      std::cout << "[FlowTriples] plane[" << p << "] has " << _sparseimg_vv[p].size() << " (above threshold) pixels" << std::endl;
    }

    // make possible triples
    _triple_v.reserve( _sparseimg_vv[_source_plane].size() );
    
    for ( auto& srcpix : _sparseimg_vv[_source_plane] ) {

      // we get the wires this pixel overlaps with
      std::vector< std::vector<int> > overlap = larflow::WireOverlap::getOverlappingWires( _source_plane, _target_plane, srcpix.col );
      //std::cout << "  target overlap size for sourcepixel=" << srcpix.col << ": " << overlap[0].size() << std::endl;
      if ( overlap[0].size()==0 ) continue;

      // get iterator to overlap[0] vector
      auto it_overlap0 = overlap[0].begin();
      
      // get the lowerbound
      PixData_t lb( srcpix.row, (int)overlap[0][0], 0.0 );
      auto it_target = std::lower_bound( _sparseimg_vv[_target_plane].begin(), _sparseimg_vv[_target_plane].end(), lb );

      // for debug
      // std::cout << "  src(" << srcpix.row << "," << srcpix.col << ") target bounds=[" << overlap[0].front() << "," << overlap[0].back() << "]"
      //           << " pos=" << it_target-_sparseimg_vv[_target_plane].begin() << "/" << _sparseimg_vv[_target_plane].size();
      // if ( it_target!=_sparseimg_vv[_target_plane].end() )
      //   std::cout << " target pixel=(" << it_target->row << "," << it_target->col << ")";
      // else
      //   std::cout << " not found.";
      // std::cout << std::endl;
      
      while ( it_target!=_sparseimg_vv[_target_plane].end() && it_target->row==srcpix.row ) {

        // break if we out of range over the target wire
        if ( it_target->col>overlap[0].back() || it_target->row!=srcpix.row ) break;

        // find position in overlap[0] in order to get overlap[1] element, i.e. the other the source+target wires intersect
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
        
        // now find the other plane
        // first search for pixel in sparseimg vector
        auto it_other = std::lower_bound( _sparseimg_vv[_other_plane].begin(),
                                          _sparseimg_vv[_other_plane].end(),
                                          PixData_t( srcpix.row, overlap[1][ivec],0.0) );
        if ( it_other!=_sparseimg_vv[_other_plane].end() && it_other->col==overlap[1][ivec] && it_other->row==srcpix.row ) {
          // valid triple: found pixel in other plane sparse data, so much charge
          std::vector<int> trip = { srcpix.idx, it_target->idx, it_other->idx, srcpix.row }; // store position in sparsematrix
          _triple_v.push_back( trip );
        }
        else if ( badch_v[ _other_plane ].pixel( srcpix.row, overlap[1][ivec] ) > 0 ) {
          // check badchannel
          std::vector<int> trip = { srcpix.idx, it_target->idx, -1, srcpix.row }; // store position in sparsematrix, omit last plane
          _triple_v.push_back( trip );          
        }
        
        // iterate the target pixel
        it_target++;
      }
    }
    
    _triple_v.shrink_to_fit();

    std::clock_t end_ = std::clock();
    
    std::cout << "[FlowTriples] for flow source[" << _source_plane << "] "
              << "to target[" << _target_plane << "] planes "
              << "found " << _triple_v.size() << " triples "
              << "elasped=" << float(end_-start_)/float(CLOCKS_PER_SEC)
              << std::endl;
    
  }


}
