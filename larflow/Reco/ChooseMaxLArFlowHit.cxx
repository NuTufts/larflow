#include "ChooseMaxLArFlowHit.h"

namespace larflow {
namespace reco {

  void ChooseMaxLArFlowHit::process( larcv::IOManager& iolcv,
                                     larlite::storage_manager& ioll )
  {

    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_adc->Image2DArray();

    larlite::event_larflow3dhit* ev_hit
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _input_larflow3dhit_treename );

    LARCV_INFO() << "Number of input hits: " << ev_hit->size() << std::endl;

    // output container
    larlite::event_larflow3dhit* evout_hit
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _output_larflow3dhit_treename );
    
    // track if used
    std::vector<int> used_v(ev_hit->size(),0);

    for ( size_t plane=0; plane<adc_v.size(); plane++ ) {
      
      _make_pixelmap( *ev_hit, adc_v, plane, used_v );
      LARCV_INFO() << "Source plane[" << plane << "] size of map: " << _srcpixel_to_spacepoint_m.size() << std::endl;
    
      // select hits
      for ( auto it=_srcpixel_to_spacepoint_m.begin(); it!=_srcpixel_to_spacepoint_m.end(); it++ ) {

        float maxscore = 0;
        int maxhit = -1;

        for ( size_t ii=0; ii<it->second.size(); ii++ ) {
          auto const& hit = ev_hit->at( it->second[ii] );
          if ( maxhit<0 || maxscore<hit.track_score ) {
            maxhit = it->second[ii];
            maxscore = hit.track_score;
          }
        }
        if ( maxhit>=0 && used_v[maxhit]==0 ) {
          evout_hit->push_back( ev_hit->at(maxhit) );
          used_v[maxhit] = 1;
        }
      }
      
    }//end of loop over plane
    
  }

  void ChooseMaxLArFlowHit::_make_pixelmap( const larlite::event_larflow3dhit& hit_v,
                                            const std::vector<larcv::Image2D>& img_v,
                                            const int source_plane,
                                            std::vector<int>& idx_used_v )
  {

    _srcpixel_to_spacepoint_m.clear();

    // collect pixel to larflow hits
    for ( size_t idx=0; idx<hit_v.size(); idx++ ) {

      if ( idx_used_v[idx]==1 ) continue;
      
      Pixel_t pix;
      pix.plane = source_plane;
      pix.row = img_v[pix.plane].meta().row( hit_v[idx].tick );
      pix.col = hit_v[idx].targetwire[pix.plane];

      auto it=_srcpixel_to_spacepoint_m.find( pix );
      if ( it==_srcpixel_to_spacepoint_m.end() ) {
        _srcpixel_to_spacepoint_m[ pix ] = std::vector<int>();
        it=_srcpixel_to_spacepoint_m.find( pix );
      }

      it->second.push_back( idx );

    }
    
  }

}
}
