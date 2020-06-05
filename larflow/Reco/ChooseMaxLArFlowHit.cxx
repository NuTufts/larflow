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

    _make_pixelmap( *ev_hit, adc_v );
    LARCV_INFO() << "Size of map: " << _srcpixel_to_spacepoint_m.size() << std::endl;
    
    // output container
    larlite::event_larflow3dhit* evout_hit
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _output_larflow3dhit_treename );
    
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
      evout_hit->push_back( ev_hit->at(maxhit) );
    }
    
  }

  void ChooseMaxLArFlowHit::_make_pixelmap( const larlite::event_larflow3dhit& hit_v,
                                            const std::vector<larcv::Image2D>& img_v )
  {

    _srcpixel_to_spacepoint_m.clear();

    // collect pixel to larflow hits
    for ( size_t idx=0; idx<hit_v.size(); idx++ ) {

      Pixel_t pix;
      pix.plane = 2;
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
