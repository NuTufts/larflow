#include "CompressRecoTrack.h"

#include "larflow/RecoUtils/geofuncs.h"
#include "TVector3.h"

namespace larflow {
namespace recoutils {

  larlite::track CompressRecoTrack::compress( const larlite::track& input_track,
					      float max_saggita_cm,
					      float max_step_size ) const
  {
    
    size_t norig_points = input_track.NumberTrajectoryPoints();
    if (norig_points<=2)
      return input_track;

    std::vector<size_t> newpt_index;
    newpt_index.reserve( norig_points );
    
    TVector3 current_pt = input_track.LocationAtPoint(0);
    std::vector<float> fcurrent = { (float)current_pt(0), (float)current_pt(1), (float)current_pt(2) };
    std::vector<float> flast(3,0);
    std::vector<float> ftest(3,0);    
    size_t current_idx = 0;
    size_t last_idx = 0;
    std::vector<float> fmax(3,0);
    float max_saggita = -1.0;

    // add first point
    newpt_index.push_back( 0 );

    for (size_t i=1; i<norig_points; i++) {

      const TVector3& testpt = input_track.LocationAtPoint(i);
      for (int v=0; v<3; v++)
	ftest[v] = testpt(v);
      
      float step_size = (float)(testpt-current_pt).Mag();

      if (max_saggita < 0.0 ) {
	// saggita unset
	fmax = ftest;
	max_saggita = 0.0;
      }
      else {
	float sag = larflow::recoutils::pointLineDistance3f( fcurrent, ftest, fmax );
	if ( sag > max_saggita ) {
	  max_saggita = sag;
	  fmax = ftest;
	}
      }

      if ( (i-current_idx)<=1 ) {
	flast = ftest;
	last_idx = i;
	continue;
      }
      
      // do we compress?
      if ( (step_size > max_step_size)
	   || ( max_saggita>max_saggita_cm ) ) {
	// register last point
	newpt_index.push_back( last_idx );
	// update and reset trackers
	fcurrent = flast;
	current_idx = last_idx;
	current_pt = testpt;
	fmax = std::vector<float>{0.0, 0.0, 0.0};
	max_saggita = -1.0;
      }

      flast = ftest;
      last_idx = i;
      
    }//end of loop over track

    if ( (int)newpt_index.back()!=(int)(norig_points-1) )
      newpt_index.push_back( norig_points-1 );

    // copy over locations of points: someone's elses problem to get dq/dx
    larlite::track compressed_track;
    for (auto const& idx : newpt_index ) {
      compressed_track.set_track_id( input_track.ID() );
      compressed_track.add_vertex( input_track.LocationAtPoint( idx ) );
      compressed_track.add_direction( input_track.DirectionAtPoint( idx ) );
    }

    return compressed_track;
  }

}
}
