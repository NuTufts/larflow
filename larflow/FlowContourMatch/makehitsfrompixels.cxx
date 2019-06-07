#include "makehitsfrompixels.h"

namespace larflow {
  
  /**
   * create hits from pixels in the image.
   *
   * @param[in] arc_adc ADC image2d
   * @param[in] threshold ADC threshold to create hit
   *
   * @return vector of larlite::hit objects representing pixels
   */
  void FlowContourMatch::makeHitsFromWholeImagePixels( const larcv::Image2D& src_adc, const float threshold ) {

    /// instead of hits, which can be too sparsely defined,
    /// we can try to match pixels (or maybe eventually groups of pixels).
    /// to use same machinery, we turn pixels into hits

    larlite::event_hit evhit_v;
    evhit_v.clear();
    evhit_v.reserve(int(0.01*src_adc.meta().rows()*src_adc.meta().cols()));

    int maxcol = src_adc.meta().cols();
    int maxcol_plane = maxcol;
    if ( src_adc.meta().plane()<2 ) {
      maxcol_plane = src_adc.meta().col(2399)+1; // maxwire
    }
    else {
      maxcol_plane = src_adc.meta().col(3455)+1; // maxwire
    }
    maxcol = ( maxcol>maxcol_plane) ? maxcol_plane : maxcol;
    
    // we loop over all source pixels and make "hits" for all pixels above threshold
    int ihit = 0;
    for (int irow=0; irow<(int)src_adc.meta().rows(); irow++) {
      float hit_tick = src_adc.meta().pos_y( irow )-2400.0;
      
      for (int icol=0; icol<maxcol; icol++) {
	float pixval = src_adc.pixel( irow, icol );
	if (pixval<threshold )
	  continue;
	
	int wire = src_adc.meta().pos_x( icol );

	// make fake hit from pixel
	
	larlite::hit h;
	h.set_rms( 1.0 );
	h.set_time_range( hit_tick, hit_tick );
	h.set_time_peak( hit_tick, 1.0 );
	h.set_time_rms( 1.0 );
	h.set_amplitude( pixval, sqrt(pixval) );
	h.set_integral( pixval, sqrt(pixval) );
	h.set_sumq( pixval );
	h.set_multiplicity( 1 );
	h.set_local_index( ihit );
	h.set_goodness( 1.0 );
	h.set_ndf( 1 );

	larlite::geo::WireID wireid( 0, 0, src_adc.meta().plane(), wire );
	int ch = larutil::Geometry::GetME()->PlaneWireToChannel( wireid.Plane, wireid.Wire );
	h.set_channel( ch );
	h.set_view( (larlite::geo::View_t)wireid.Plane );
	h.set_wire( wireid );
	h.set_signal_type( larutil::Geometry::GetME()->SignalType( ch ) );
	evhit_v.emplace_back( std::move(h) );
	
	ihit++;
      }
    }
    
  }
  
  
}
