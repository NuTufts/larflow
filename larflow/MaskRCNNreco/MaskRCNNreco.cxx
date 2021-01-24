#include "MaskRCNNreco.h"

#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "larcv/core/DataFormat/EventClusterMask.h"
#include "larcv/core/DataFormat/EventImage2D.h"

namespace larflow {
namespace mrcnnreco {

  Mask_t::Mask_t( const larcv::ClusterMask& mask, const larcv::ImageMeta& embedding_img_meta )
    : img( embedding_img_meta )
  {
    img.paint(0);
    auto const& meta = img.meta();
    const std::vector<float>& box = mask.as_vector_box_no_convert(); // (colmin,rowmin,colmax,rowmax,class)

    float xmin[2] = { box[0], box[2] };
    float ymin[2] = {  (float)meta.pos_y((int)meta.rows()-box[3]-1), (float)meta.pos_y((int)meta.rows()-box[1]-1) };
    if ( xmin[1]>=meta.max_x() ) xmin[1] -= 1.0;

    col_range[0] = (int)meta.col( xmin[0], __FILE__, __LINE__ );
    col_range[1] = (int)meta.col( xmin[1], __FILE__, __LINE__ );
    row_range[0] = (int)meta.row( ymin[0], __FILE__, __LINE__ );
    row_range[1] = (int)meta.row( ymin[1], __FILE__, __LINE__ );

    pixrow.reserve( mask.points_v.size() );
    pixcol.reserve( mask.points_v.size() );
    for ( size_t ipt=0; ipt<mask.points_v.size(); ipt++ ) {
      float x = xmin[0]+mask.points_v.at(ipt).x;
      float y = (float)meta.rows() - (box[1]+mask.points_v.at(ipt).y);

      int r = (int)y;
      if ( r<0 || r>=(int)meta.rows() ) continue;        
      int c = (int)x;
      if ( c<0 || c>=(int)meta.cols() ) continue;
      pixrow.push_back(r);
      pixcol.push_back(c);
      img.set_pixel( r, c, 1 );
    }
  }

  void MaskRCNNreco::process( larcv::IOManager& iolcv,
                              larlite::storage_manager& ioll )
  {

    // get larmatch points
    larlite::event_larflow3dhit* ev_larmatch =
      (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit,"larmatch");
    LARCV_INFO() << "number of larmatch points: " << (int)ev_larmatch->size() << std::endl;
    
    // get mask-rcnn container
    larcv::EventClusterMask* ev_mrcnn =
      (larcv::EventClusterMask*)iolcv.get_data(larcv::kProductClusterMask,"mask_proposals_y");
    const std::vector<larcv::ClusterMask>& mask_v = ev_mrcnn->as_vector().front();
    LARCV_INFO() << "number of masks: " << mask_v.size() << std::endl;

    // get full wire images
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wire");
    const std::vector<larcv::Image2D>& adc_v = ev_adc->as_vector();

    // convert clusermask object into form better for this class
    _mask_v.clear();
    for ( auto& mask : mask_v ) {
      _mask_v.push_back( Mask_t( mask, adc_v[2].meta() ) );
    }

    // absorb smaller masks
    std::vector< Mask_t > after_absorb = merge_proposals( _mask_v );
    std::swap( after_absorb, _mask_v );
    after_absorb.clear();
    
    std::vector<larlite::larflowcluster> cluster_v = clusterbyproposals( *ev_larmatch,
                                                                         _mask_v,
                                                                         0.5 );

    larlite::event_larflowcluster* ev_out
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster,"mrcnn");
    
    larlite::event_larflowcluster* ev_out_unused
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster,"notmrcnn");
    
    for ( int ic=0; ic<(int)cluster_v.size()-1; ic++ ) {
      auto& cluster = cluster_v[ic];
      ev_out->emplace_back( std::move(cluster) );
    }
    ev_out_unused->emplace_back( std::move( cluster_v.back() ) );
  }

  /**
   * @brief absorb smaller masks into larger ones
   */
  std::vector<larflow::mrcnnreco::Mask_t> MaskRCNNreco::merge_proposals( std::vector<larflow::mrcnnreco::Mask_t>& mask_v )
  {
    // make a summary of each mask
    struct MaskSum_t {
      int xbounds[2];
      int ybounds[2];
      float area;
      int index;
      int used;
      std::vector<int> add;
      MaskSum_t( int xmin, int xmax, int ymin, int ymax, int idx ) {
        xbounds[0] = xmin;
        xbounds[1] = xmax;
        ybounds[0] = ymin;
        ybounds[1] = ymax;
        area = (float)(xmax-xmin)*(ymax-ymin);
        used = 0;
        index = idx;
        add.clear();
      };
      bool operator<( MaskSum_t& rhs ) {
        if ( area > rhs.area )
          return true;
        return false;
      };
    };

    std::vector<MaskSum_t> sum_v;
    sum_v.reserve( mask_v.size() );
    for ( size_t idx=0; idx<mask_v.size(); idx++ ) {
      auto const& mask = mask_v[idx];
      sum_v.push_back(  MaskSum_t( mask.col_range[0], mask.col_range[1], mask.row_range[0], mask.row_range[1], (int)idx ) );
    }

    // sort so that biggest by area is first
    std::sort( sum_v.begin(), sum_v.end() );
    
    for (int i=0; i<(int)sum_v.size(); i++ ) {
      auto& m1 = sum_v[i];
      if ( m1.used==1 ) continue; // don't consider if already absorbed
      auto& mask1 = mask_v[m1.index];
      
      for (int j=i+1; j<(int)sum_v.size(); j++) {
        auto& m2 = sum_v[j];
        if ( m2.used==1 ) continue;
        auto& mask2 = mask_v[m2.index];
        
        // test merge first with bounding bound overlap

        // check for null overlap
        if ( m2.ybounds[0] > m1.ybounds[1] || m2.ybounds[1] < m1.ybounds[0]
             || m2.xbounds[0] > m1.xbounds[1] || m2.xbounds[1] < m1.xbounds[0] )
          continue;

        // get overlap range
        int oymin = ( m2.ybounds[0]>m1.ybounds[0] ) ? m2.ybounds[0] : m1.ybounds[0];
        int oymax = ( m2.ybounds[1]<m1.ybounds[1] ) ? m2.ybounds[1] : m1.ybounds[1];
        int oxmin = ( m2.xbounds[0]>m1.xbounds[0] ) ? m2.xbounds[0] : m1.xbounds[0];
        int oxmax = ( m2.xbounds[1]<m1.xbounds[1] ) ? m2.xbounds[1] : m1.xbounds[1];
        float oarea = (float)(oymax-oymin)*(oxmax-oxmin);

        // check if we hit overlap threshold
        if ( oarea<0 || oarea/m2.area < 0.1 ) continue;

        // now we check mask pixel overlap
        int noverlap = 0;
        for (int ipix=0; ipix<(int)mask2.pixcol.size(); ipix++ ) {
          int c = mask2.pixcol[ipix];
          int r = mask2.pixrow[ipix];

          if ( mask1.img.pixel(r,c,__FILE__,__LINE__)>0 )
            noverlap++;
        }

        float frac_overlap = (float)noverlap/(float)mask2.pixcol.size();
        if ( frac_overlap>0.25 ) {
          // absorb
          m1.add.push_back( m2.index );
          m2.used = 1;
        }
      }
    }
    
    // apply merging
    std::vector<Mask_t> merged_v;
    for (int i=0; i<(int)sum_v.size(); i++ ) {
      auto& sum1  = sum_v[i];
      auto& mask1 = mask_v[ sum1.index ];
      if ( sum1.used==1 ) continue; // absorbed;
      if ( sum1.add.size()>0 ) {
        // absorb
        for ( auto& idx : sum1.add ) {
          auto& mask2 = mask_v[ idx ];
          for (size_t ipix=0; ipix<mask2.pixcol.size(); ipix++ ) {

            int r2 = mask2.pixrow[ipix];
            int c2 = mask2.pixcol[ipix];
            // update bounding box of mask1
            if ( r2 < mask1.row_range[0] ) mask1.row_range[0] = r2;
            if ( r2 > mask1.row_range[1] ) mask1.row_range[1] = r2;
            if ( c2 < mask1.col_range[0] ) mask1.col_range[0] = c2;
            if ( c2 > mask1.col_range[1] ) mask1.col_range[1] = c2;

            // check if pixel already in mask1
            if ( mask1.img.pixel(r2,c2, __FILE__, __LINE__ )==0 ) {
              mask1.pixcol.push_back( c2 );
              mask1.pixrow.push_back( r2 );
              mask1.img.set_pixel( r2, c2, 1 );
            }
          }
        }//end of indices to absorb        
      }//if there are images to absorb
      merged_v.emplace_back( std::move(mask1) );
    }
    
    return merged_v;
  }

  std::vector<larlite::larflowcluster>
  MaskRCNNreco::clusterbyproposals( const larlite::event_larflow3dhit& ev_larmatch,
                                    const std::vector<larflow::mrcnnreco::Mask_t>& mask_v,
                                    const float hit_threshold )
  {

    const int dpix = 0; // bleed mask pixel by +/- 1 pixel
    
    std::vector<larlite::larflowcluster> cluster_v;
    cluster_v.reserve( mask_v.size()+1 );
    std::vector<int> used_v( ev_larmatch.size(), 0 );

    for ( auto& mask : mask_v ) {

      auto const& meta = mask.img.meta();
      
      larlite::larflowcluster cluster;
      
      for ( size_t hitidx=0; hitidx<ev_larmatch.size(); hitidx++ ) {
        
        auto& hit = ev_larmatch[hitidx];
        
        if ( hit[9]<hit_threshold ) continue;
        
        float yplane_tick = hit.tick;
        float yplane_wire  = hit.targetwire[2];
        if ( yplane_tick < meta.min_y() || yplane_tick>meta.max_y() ) continue;
        if ( yplane_wire < meta.min_x() || yplane_wire>meta.max_x() ) continue;

        int yplane_row = meta.row( yplane_tick );
        int yplane_col = meta.col( yplane_wire );

        
        if ( yplane_row>=mask.row_range[0] && yplane_row<=mask.row_range[1]
             && yplane_col>=mask.col_range[0] && yplane_col<=mask.col_range[1] ) {
          // inside bounding box
          // now need to test against mask

          if ( mask.img.pixel( yplane_row, yplane_col )==1 ) {
            cluster.push_back( hit );
            used_v[hitidx] = 1;
          }
        }
      }
      
      LARCV_INFO() << "assigned " << cluster.size() << " hits to mask." << std::endl;
      
      cluster_v.push_back(cluster);
    }

    // make a not-used cluster
    larlite::larflowcluster notused;
    for ( size_t hitidx=0; hitidx<ev_larmatch.size(); hitidx++ ) {
      if ( used_v[hitidx]==0 )
        notused.push_back( ev_larmatch[hitidx] );
    }
    cluster_v.push_back( notused );
    
    return cluster_v;
  }

  
}
}
