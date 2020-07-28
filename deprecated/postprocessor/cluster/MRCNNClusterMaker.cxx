#include "MRCNNClusterMaker.h"

namespace larflow {


  /**
   * simply creates clusters by including hits within the neighborhood of a mask
   *
   * @param[in] masks_v vector of ClusterMask objects
   * @param[in] hits_v vector of larflow3dhits from post-processor
   * @param[in] adc_v  vector of whole-view ADC images
   */
  std::vector< larlite::larflowcluster > MRCNNClusterMaker::makeSimpleClusters( const std::vector<larcv::ClusterMask>&  masks_v,
                                                                                const std::vector<larlite::larflow3dhit>& hits_v,
                                                                                const std::vector<larcv::Image2D>& adc_v )
  {

    int dn = 1;
    std::vector< larlite::larflowcluster > cluster_v;

    // we first create a lookup table for the hits for faster searching
    const larcv::ImageMeta& src_meta = adc_v.at(2).meta();
    int* hitidx_map = new int[src_meta.cols()*src_meta.rows()];
    memset( hitidx_map, 0, sizeof(int)*src_meta.cols()*src_meta.rows() );
    for (size_t ihit=0; ihit<hits_v.size(); ihit++) {
      const larlite::larflow3dhit& hit = hits_v[ihit];
      int row = src_meta.row(hit.tick);
      int col = src_meta.col(hit.srcwire);
      //std::cout << "hit[" << ihit << "](r,c)=(" << row << "," << col << ")" << std::endl;
      *(hitidx_map+src_meta.rows()*col + row) = (int)(ihit+1);
    }

    std::vector<bool> used_once( hits_v.size(), false);
    
    for ( auto const& mask : masks_v ) {
      int row_offset = mask.box.min_y();
      int col_offset = mask.box.min_x();
      std::cout << "mask offset(r,c)=(" << row_offset << "," << col_offset << ") "
                << "from box (y,x)=(" << mask.box.min_y() << "," << mask.box.min_x() << ")"
                << std::endl;

      larlite::larflowcluster cluster;
      std::vector<int> used_v( hits_v.size(), 0 );
      for ( auto const& pt2d : mask.points_v ) {
        int row=row_offset + (int)pt2d.y;
        int col=col_offset + (int)pt2d.x;

        for ( int dr=-dn; dr<=dn; dr++ ) {
          int r = row+dr;
          if ( r<0 || r>=(int)src_meta.rows()) continue;
          
          for ( int dc=-dn; dc<=dn; dc++ ) {            
            int c = col+dc;
            if ( c<0 || c>=(int)src_meta.cols()) continue;
            
            int idx = int(src_meta.rows())*c + r;
            int hitidx = *(hitidx_map+idx);
            //std::cout << "  hitidx@idx=" << idx << ": " << hitidx << std::endl;
            if (hitidx>0 && used_v[hitidx-1]==0) {
              used_v[hitidx-1] = 1;
              cluster.push_back( hits_v.at(hitidx-1) );
              used_once[hitidx-1] = true;
            }
          }//end of loop over col neighborhood
        }//end of loop over row neighborhood
        
      }//end of loop over points in mask

      cluster_v.emplace_back( std::move(cluster) );
    }//end of loop over masks


    // make cluster of hits not in any mask
    larlite::larflowcluster leftover_cluster;
    for ( size_t ihit=0; ihit<hits_v.size(); ihit++ ) {
      if ( !used_once[ihit] ) {
        leftover_cluster.push_back( hits_v[ihit] );
      }
    }
    cluster_v.emplace_back( std::move(leftover_cluster) );

    std::cout << "Produced " << cluster_v.size() << " clusters" << std::endl;
    
    return cluster_v;
  }
  

  
}
