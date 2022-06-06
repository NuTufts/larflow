#include "ShowerdQdx.h"

#include <stdio.h>
#include <string>
#include <sstream>
#include <omp.h>
#include "larlite/LArUtil/Geometry.h"
#include "larlite/LArUtil/DetectorProperties.h"
#include "ublarcvapp/MCTools/TruthShowerTrunkSCE.h"
#include "ublarcvapp/MCTools/TruthTrackSCE.h"
#include "ublarcvapp/RecoTools/DetUtils.h"
#include "geofuncs.h"
#include "ClusterImageMask.h"


namespace larflow {
namespace reco {

  int ShowerdQdx::ndebugcount = 0;
  larutil::SpaceChargeMicroBooNE* ShowerdQdx::_psce  = nullptr;

  /**
   * @begin default constructor
   */
  ShowerdQdx::ShowerdQdx()
    : larcv::larcv_base("ShowerdQdx")
  {
  }
  
  /** 
   * deconstructor
   */
  ShowerdQdx::~ShowerdQdx()
  {
  }

  /**
   * @brief reset result values
   */
  void ShowerdQdx::clear()
  {

    _pixsum_dqdx_v.clear();
    _best_pixsum_dqdx = 0;
    _best_pixsum_plane = -1;
    _best_pixsum_ortho = 0;
    
    _shower_dir.clear();
    _plane_dqdx_seg_v.clear();
    _plane_s_seg_v.clear();
    
    _plane_electron_srange_v.clear();
    _plane_electron_dqdx_v.clear();
    _plane_electron_dx_v.clear();
    _plane_electron_mean_v.clear();
    _plane_electron_rms_v.clear();
    _plane_electron_ngood_v.clear();    

    _plane_electron_best = -1;
    _plane_electron_best_mean = 0;
    _plane_electron_best_rms = 0;
    _plane_electron_best_ngood = 0;
    _plane_electron_best_start = -10;
    
    _plane_gamma_srange_v.clear();
    _plane_gamma_dqdx_v.clear();
    _plane_gamma_dx_v.clear();
    _plane_gamma_mean_v.clear();
    _plane_gamma_rms_v.clear();
    _plane_gamma_ngood_v.clear();

    _plane_gamma_best = -1;
    _plane_gamma_best_mean = 0;
    _plane_gamma_best_rms = 0;
    _plane_gamma_best_ngood = 0;    
    _plane_gamma_best_start = -10;    

    _true_min_feat_dist = -1.0;
    _true_vertex_err_dist = -100.0;
    _true_dir_cos = -2.0;
    _true_match_pdg = 0;
    _true_min_index = -1;
    _true_max_primary_cos = -2;
    
  }

  /**
   * @brief calculate dq/dx for a given shower
   *
   */
  void ShowerdQdx::processShower( const larlite::larflowcluster& shower,
                                  const larlite::track& trunk,
                                  const larlite::pcaxis& pca,
                                  const std::vector<larcv::Image2D>& adc_v,
                                  const larflow::reco::NuVertexCandidate& nuvtx )
  {
    
    // clear variables
    clear();
    trunk_tgraph_vv.clear();
    trunk_tgraph_vv.resize( adc_v.size() );

    const int tpcid  = shower[0].targetwire[4];
    const int cryoid = shower[0].targetwire[5];

    std::vector< const larcv::Image2D* > padc_v
      = ublarcvapp::recotools::DetUtils::getTPCImages( adc_v, tpcid, cryoid );    
    
    auto const& meta = padc_v.front()->meta();
    
    std::vector<float> start_pos = { (float)trunk.LocationAtPoint(0)[0],
                                     (float)trunk.LocationAtPoint(0)[1],
                                     (float)trunk.LocationAtPoint(0)[2] };
    std::vector<float> end_pos = { (float)trunk.LocationAtPoint(1)[0],
                                   (float)trunk.LocationAtPoint(1)[1],
                                   (float)trunk.LocationAtPoint(1)[2] };

    std::vector<float> fstart;    
    std::vector<float> fend;
    _shower_dir.resize(3,0);
    float dist = 0.;

    // make sure the shower start and end are inside the image
    bool validtrunk = checkShowerTrunk( start_pos, end_pos,
					fstart, fend,
					_shower_dir, dist,
					tpcid, cryoid, padc_v );

    // collect pixels that the trunk passes through in each wire plane image
    _createDistLabels( fstart, fend, padc_v, 10.0, tpcid, cryoid );

    // define line segment regions to measure dq/dx over the shower trunk
    _makeSegments( -3.0, 0.5 );

    // use track pixels in candidate neutrino vertex to mask pixels to zero
    std::vector<larcv::Image2D> track_masked_v = maskTrackPixels( padc_v, trunk, nuvtx, tpcid, cryoid );

    // calculate dq/dx for each segment along dq/dx, using masked image
    _sumChargeAlongSegments( fstart, fend, track_masked_v, 10.0, 1, 3, tpcid, cryoid );

    // find range along trunk that contain points with expected dq/dx for electron and photon
    // electron regions
    LARCV_DEBUG() << "/// FIND ELECTRON-LIKE RANGE ///" << std::endl;
    _findRangedQdx( fstart, fend, track_masked_v, 350.0, 100.0,
                    _plane_electron_dqdx_v,
                    _plane_electron_dx_v,
                    _plane_electron_srange_v,
                    _plane_electron_mean_v,
                    _plane_electron_rms_v,
                    _plane_electron_ngood_v,
                    _plane_electron_best,
                    _plane_electron_best_ngood,                    
                    _plane_electron_best_mean,
                    _plane_electron_best_rms,
                    _plane_electron_best_start );
    // photon regions (2MIP)
    LARCV_DEBUG() << "/// FIND GAMMA-LIKE RANGE ///" << std::endl;    
    _findRangedQdx( fstart, fend, track_masked_v, 1000.0, 400.0,
                    _plane_gamma_dqdx_v,
                    _plane_gamma_dx_v,
                    _plane_gamma_srange_v,
                    _plane_gamma_mean_v,
                    _plane_gamma_rms_v,
                    _plane_gamma_ngood_v,
                    _plane_gamma_best,
                    _plane_gamma_best_ngood,                    
                    _plane_gamma_best_mean,
                    _plane_gamma_best_rms,
                    _plane_gamma_best_start );

    // simple dq/dx measure using first X cm of trunk. use masked image.
    float pixsum_dist = 3.0; // default

    // use heuristics to check for early cascade start
    if ( _plane_electron_best>=0 && _plane_gamma_best>=0 ) {
      float erange_mid = 0.5*(_plane_electron_srange_v[ _plane_electron_best ][0]+_plane_electron_srange_v[ _plane_electron_best ][1]);
      float grange_mid = 0.5*(_plane_gamma_srange_v[ _plane_gamma_best ][0]+_plane_gamma_srange_v[ _plane_gamma_best ][1]);
      if ( _plane_electron_srange_v[ _plane_electron_best ][0]>-0.5
           && _plane_gamma_srange_v[ _plane_gamma_best ][0]>-0.5
           && erange_mid < grange_mid
           && _plane_electron_srange_v[ _plane_electron_best ][1]<3.0 ) {
        pixsum_dist = _plane_electron_srange_v[ _plane_electron_best ][1];
        // if (pixsum_dist>1.0 )
        //   pixsum_dist -= 0.50; // one segment step
      }
    }
    
    std::vector<float> end3cm(3,0);
    for (int i=0; i<3; i++) {
      end3cm[i] = fstart[i] + pixsum_dist*_shower_dir[i];
    }
    try {
      _pixsum_dqdx_v = sumChargeAlongTrunk( fstart, end3cm, track_masked_v, 10.0, 1, 3, tpcid, cryoid );
    }
    catch (std::exception& e) {
      LARCV_WARNING() << "Error calculating dq/dx: " << e.what() << std::endl;
      _pixsum_dqdx_v.resize( padc_v.size(), 0 );
      for (int p=0; p<(int)padc_v.size(); p++)
        _pixsum_dqdx_v[p] = 0.;
    }
    
    for (size_t p=0; p<padc_v.size(); p++) {
      LARCV_INFO() << "Plane[" << p << "] " << pixsum_dist << " cm trunk dq/dx: " << _pixsum_dqdx_v[p] << " pixsum/cm" << std::endl;
    }

    // heuristic to choosing best dq/dx
    // most good points?
    // start of electron shower?
    std::vector<int> ngood_v( padc_v.size(), 0);
    _best_pixsum_plane = -1;
    _best_pixsum_ortho = 0.;    
    _pixsum_dqdx_v.resize(padc_v.size(),0);
    for (size_t p=0; p<padc_v.size(); p++) {
      float ngood_max = 0;
      auto const& wirepitchdir = larlite::larutil::Geometry::GetME()->GetPlane( p, tpcid, cryoid ).fWirePitchDir;
      std::vector<double> orthovect= { 0.0, 0.0, 0.0 };

      float plane_ortho_cos = 0.;
      float ortholen = 0.;
      for (int i=0; i<3; i++) {
        plane_ortho_cos += wirepitchdir[i]*_shower_dir[i];
        ortholen += wirepitchdir[i]*wirepitchdir[i];
      }
      if (ortholen>0) {
        ortholen = sqrt(ortholen);
        plane_ortho_cos /= ortholen;
      }
      plane_ortho_cos = fabs(plane_ortho_cos);

      // check if electron or gamma had good region within the first 3 cm trunk
      float e_range = _plane_electron_srange_v[p][0];
      float e_len   = _plane_electron_dx_v[p];
      if ( e_range>-2.0 && e_range<3.0 && e_len>1.0 ) {
        if ( e_len>ngood_max )
          ngood_max = e_len;
      }
      float g_range = _plane_gamma_srange_v[p][0];
      float g_len   = _plane_gamma_dx_v[p];
      if ( g_range>-2.0 && g_range<3.0 && g_len>1.0 ) {
        if ( g_len>ngood_max )
          ngood_max = g_len;
      }
      
      if ( ngood_max > 0 && _best_pixsum_ortho < plane_ortho_cos && _pixsum_dqdx_v[p]>50.0 ) {
        _best_pixsum_ortho = plane_ortho_cos;        
        _best_pixsum_plane = (int)p;
      }

      LARCV_DEBUG() << "plane[" << p << "] plane_ortho_cos=" << plane_ortho_cos << "  ngood=" << ngood_max << std::endl;
    }
    if ( _best_pixsum_plane>=0 )
      _best_pixsum_dqdx = _pixsum_dqdx_v[ _best_pixsum_plane ];
    LARCV_INFO() << "chosen dq/dx from plane=" << _best_pixsum_plane << ": " << _best_pixsum_dqdx << std::endl;
    
    // make graphs for debug plotting
    std::vector<float> cgrad;
    std::vector<float> rgrad;
    for ( int p=0; p<(int)padc_v.size(); p++ ) {
      TGraph gpath(2);      
      float col = colcoordinate_and_grad( fstart, p, tpcid, cryoid, padc_v.size(), meta, cgrad );
      float row = rowcoordinate_and_grad( fstart, p, tpcid, cryoid, meta, rgrad );
      float ptx = col*meta.pixel_width()+meta.min_x();
      float pty = row*meta.pixel_height()+meta.min_y();
      gpath.SetPoint(0, ptx, pty );

      col = colcoordinate_and_grad( fend, p, tpcid, cryoid, padc_v.size(), meta, cgrad );
      row = rowcoordinate_and_grad( fend, p, tpcid, cryoid, meta, rgrad );
      ptx = col*meta.pixel_width()+meta.min_x();
      pty = row*meta.pixel_height()+meta.min_y();
      gpath.SetPoint(1, ptx, pty );
      trunk_tgraph_vv[p].emplace_back( std::move(gpath) );
    }
    
  }

  /** 
   * @brief make sure trunk end points are inside the image
   *
   */
  bool ShowerdQdx::checkShowerTrunk( const std::vector<float>& start_pos,
                                     const std::vector<float>& end_pos,
                                     std::vector<float>& modstart3d,
                                     std::vector<float>& modend3d,
                                     std::vector<float>& shower_dir,
                                     float& dist,
				     const int tpcid,
				     const int cryoid,
                                     const std::vector<const larcv::Image2D*>& padc_v )
  {

    auto const detp = larutil::DetectorProperties::GetME();
    
    // set output vector size
    modstart3d.resize(3,0);
    modend3d.resize(3,0);
   
    auto const& meta = padc_v.front()->meta();

    // initial length and direction
    dist = 0.;
    shower_dir.resize(3,0);        
    for (int i=0; i<3; i++) {
      shower_dir[i] = (end_pos[i]-start_pos[i]);
      dist += shower_dir[i]*shower_dir[i];
    }
    dist = sqrt(dist);
    if ( dist>0 ) {
      for (int i=0; i<3; i++)
        shower_dir[i] /= dist;
    }

    for (int i=0; i<3; i++) {
      modstart3d[i] = start_pos[i];
      modend3d[i]   = end_pos[i];
    }

    // check start and end point
    bool modified_trunk = false;
    
    // check the starting point
    float start_tick = detp->ConvertXToTicks( start_pos[0], 0, tpcid, cryoid );
    if ( start_tick<=meta.min_y() || start_tick>=meta.max_y() ) {

      // try to shorten trunk to stay in image
      bool fix = false;
      if ( start_tick<=meta.min_y() && shower_dir[0]!=0.0) {
	float mintick = detp->ConvertTicksToX( meta.pos_y( 1 ), 0, tpcid, cryoid );
	float s = (mintick-start_pos[0])/shower_dir[0];
	fix = true;
	for (int i=0; i<3; i++)
	  modstart3d[i] = start_pos[i] + s*shower_dir[i];
	start_tick = meta.pos_y(1);        
      }
      else if ( start_tick>=meta.max_y() && shower_dir[0]!=0.0) {
	float maxtick = detp->ConvertTicksToX(meta.pos_y( (int)meta.rows()-1 ), 0, tpcid, cryoid );
	float s = (maxtick-start_pos[0])/shower_dir[0];	
	fix = true;
	for (int i=0; i<3; i++)
	  modstart3d[i] = start_pos[i] + s*shower_dir[i];
	start_tick = meta.pos_y( (int)meta.rows()-1 );
      }

      if ( !fix ) {	      
	std::stringstream errmsg;
	errmsg << "out of bounds shower trunk starting point "
	       << " (" << start_pos[0] << "," << start_pos[1] << "," << start_pos[2] << ") "
	       << " start_tick=" << start_tick << std::endl;
	LARCV_CRITICAL() << errmsg.str() << std::endl;
	throw std::runtime_error(errmsg.str());
        return false;
      }
      else {
        modified_trunk = true;
      }
    }

    // final starting row
    float start_row = (float)meta.row(start_tick);
        
    float end_tick = detp->ConvertXToTicks( end_pos[0], 0, tpcid, cryoid );
    if ( end_tick<=meta.min_y() || end_tick>=meta.max_y() ) {

      // try to shorten trunk to stay in image
      bool fix = false;
      float s = 0.0;
      if ( end_tick<=meta.min_y() && shower_dir[0]!=0.0) {
	float mintick = detp->ConvertTicksToX( meta.pos_y( 1 ), 0, tpcid, cryoid );
	s = (mintick-start_pos[0])/shower_dir[0];
	fix = true;
	for (int i=0; i<3; i++)
	  modend3d[i] = start_pos[i] + s*shower_dir[i];
	end_tick = meta.pos_y(1);

      }
      else if ( end_tick>=meta.max_y() && shower_dir[0]!=0.0) {
	float maxtick = detp->ConvertTicksToX( meta.pos_y( (int)meta.rows()-1 ), 0, tpcid, cryoid );
	s = (maxtick-start_pos[0])/shower_dir[0];

	fix = true;
	for (int i=0; i<3; i++)
	  modend3d[i] = start_pos[i] + s*shower_dir[i];
	end_tick = meta.pos_y( (int)meta.rows()-1 );

      }

      if ( !fix ) {
	std::stringstream errmsg;
	errmsg << "out of bounds shower trunk ending point "
	       << " s=" << s
	       << " (" << end_pos[0] << "," << end_pos[1] << "," << end_pos[2] << ") "
	       << " end_tick=" << end_tick << std::endl;
	LARCV_CRITICAL() << errmsg.str() << std::endl;      
	throw std::runtime_error(errmsg.str());
        return false;
      } 
      else {
        modified_trunk = true;
      }
     
    }
    // final end row
    float end_row = (float)meta.row(end_tick);

    if ( modified_trunk ) {
      dist = 0;
      for (int i=0; i<3; i++) {
        shower_dir[i] = modend3d[i]-modstart3d[i];
        dist += shower_dir[i]*shower_dir[i];
      }
      dist = sqrt(dist);
    }

    return true;
  }
                                           

  /**
   * @brief get column coordinate in wire plane for given 3D point
   *
   */
  float ShowerdQdx::colcoordinate_and_grad( const std::vector<float>& pos,
                                            const int plane,
					    const int tpcid,
					    const int cryoid,
					    const int nplanes,
                                            const larcv::ImageMeta& meta,
                                            std::vector<float>& grad )
  {
    
    if ( plane<0 || plane>=nplanes )
      throw std::runtime_error("ProjectionTools::wirecoordinate_and_grad invalid plane number");
    
    // microboone only
    auto const& planegeo = larlite::larutil::Geometry::GetME()->GetPlane( plane, tpcid, cryoid );
    const TVector3& wirepitchdir = planegeo.fWirePitchDir;
    const TVector3& firstwirepos = planegeo.fWires_v[0].fWireStartVtx;
    const double wirepitch       = planegeo.fWirePitchLen;
    
    // from larlite Geometry::WireCoordinate(...)
    float wirecoord = 0.;
    for (int i=0; i<3; i++)
      wirecoord += wirepitchdir[i]*pos[i];

    float colcoord = (wirecoord-meta.min_x())/meta.pixel_width();
    
    grad.resize(3,0);
    for (int i=0; i<3; i++) {
      grad[i] = wirepitchdir[i]/meta.pixel_width();
    }

    return colcoord;
  }

  /**
   * @brief get column coordinate in wire plane for given 3D point
   *
   */
  float ShowerdQdx::rowcoordinate_and_grad( const std::vector<float>& pos,
					    const int plane,
					    const int tpcid,
					    const int cryoid,
					    const larcv::ImageMeta& meta,
					    std::vector<float>& grad )
  {
    // basing on larutil::DetectorProperties::ConvertTicksToX(...)
    
    float tick = larutil::DetectorProperties::GetME()->ConvertXToTicks( pos[0], plane, tpcid, cryoid );
    float row = (tick-meta.min_y())/meta.pixel_height();
    grad.resize(3,0);
    grad[0] = (pos[0]*larutil::DetectorProperties::GetME()->GetXTicksCoefficient())/meta.pixel_height();
    grad[1] = 0.0;
    grad[2]=  0.0;
    return row;
  }

  /**
   * @brief calculate dq/dx for given shower trunk defined by a 3d line segment
   */
  std::vector<float> ShowerdQdx::sumChargeAlongTrunk( const std::vector<float>& start3d,
                                                      const std::vector<float>& end3d,
                                                      const std::vector<larcv::Image2D>& img_v,
                                                      const float threshold,
                                                      const int dcol, const int drow,
						      const int tpcid, const int cryoid )
  {
    const int nplanes = img_v.size();

    float dist = 0.;
    std::vector<float> dir(3,0);
    for (int i=0; i<3; i++) {
      dir[i] = end3d[i]-start3d[i];
      dist += dir[i]*dir[i];
    }
    dist = sqrt(dist);
    for (int i=0; i<3; i++)
      dir[i] /= dist;

    float maxstepsize = 0.2; // cm
    int nsteps = dist/maxstepsize+1;
    float stepsize = dist/nsteps;

    std::vector<float> planesum_v(nplanes,0);

    if ( _plane_trunkpix_v.size()!=nplanes ) {
      std::vector< const larcv::Image2D* > pimg_v;
      for ( auto const& img : img_v )
	pimg_v.push_back( &img );
      _createDistLabels( start3d, end3d, pimg_v, threshold, tpcid, cryoid );
    }
    
    for ( int p=0; p<nplanes; p++ ) {
      auto const& img = img_v[p];
      auto& tplist = _plane_trunkpix_v[p];
      LARCV_DEBUG() << "Number in trunkpixel list, plane " << p << ": " << tplist.size() << std::endl;
      // we use the trunkpix list made in _createDistLabels
      // to degine min,max bounds for col and row
      int colbounds[2] = {-1};
      int rowbounds[2] = {-1};
      for (auto const& tp : tplist ) {
        if ( colbounds[0]>tp.col || colbounds[0]<0 )
          colbounds[0] = tp.col;
        if ( colbounds[1]<tp.col || colbounds[1]<0 )
          colbounds[1] = tp.col;
        if ( rowbounds[0]>tp.row || rowbounds[0]<0 )
          rowbounds[0] = tp.row;
        if ( rowbounds[1]<tp.row || rowbounds[1]<0 )
          rowbounds[1] = tp.row;
        // if ( false ) {
        //   LARCV_DEBUG() << "tp[" << tp.col << "," << tp.row << "] "
        //                 << " colbounds={" << colbounds[0] << "," << colbounds[1] << "}"
        //                 << " rowbounds={" << rowbounds[0] << "," << rowbounds[1] << "}"
        //                 << std::endl;
        // }
      }

      // crop width with padding added
      int colwidth = (colbounds[1]-colbounds[0]+1) + 2*dcol;
      int rowwidth = (rowbounds[1]-rowbounds[0]+1) + 2*drow;
      int rowlen   = (rowbounds[1]-rowbounds[0]+1);
      int col_origin = colbounds[0]-dcol;
      int row_origin = rowbounds[0]-drow;
      LARCV_DEBUG() << "colbounds: [" << colbounds[0] << "," << colbounds[1] << "]" << std::endl;
      LARCV_DEBUG() << "rowbounds: [" << rowbounds[0] << "," << rowbounds[1] << "]" << std::endl;      
      LARCV_DEBUG() << "Crop meta. Origin=(" << col_origin << "," << row_origin << ")"
                    << " ncol=" << colwidth << " nrow=" << rowwidth
                    << std::endl;

      if ( colwidth<=0 || rowwidth<=0 ) {
        LARCV_WARNING() << "[ShowerdQdx::sumChargeAlongTrunk] Plane[" << p << "] Bad cropped window size" << std::endl;
        planesum_v[p] = 0.0;        
        continue;
      }

      // make data array
      std::vector<float> crop(colwidth*rowwidth,0);
      std::vector<float> mask(colwidth*rowwidth,0);

      // copy the data into the cropped matrix
      // preserve the row-ordered data
      for (int c=0; c<(int)(colbounds[1]-colbounds[0]); c++) {
        // for the larcv image, the row-dimension is the contiguous element
        size_t origin_index = img.meta().index( rowbounds[0], colbounds[0]+c, __FILE__, __LINE__ );
        const float* source_ptr = img.as_vector().data() + origin_index;
        float* dest_ptr   = crop.data() + (c+dcol)*rowwidth + drow;
        memcpy( dest_ptr, source_ptr, sizeof(float)*rowlen );
      }

      // make array where we store the min dist along trunk for pixels
      // on trunk path
      std::vector<float> smin_img(colwidth*rowwidth,0);
      std::vector<float> smax_img(colwidth*rowwidth,0);      
      
      for (int itp=0; itp<(int)tplist.size(); itp++) {
        auto const& tp = tplist[itp];
        // change (col,row) from original image to mask image (with padding)
        // then add kernal shift
        int icol = (int)tp.col-col_origin;
        int irow = (int)tp.row-row_origin;
        smin_img[icol*rowwidth+irow] = tp.smin;
        smax_img[icol*rowwidth+irow] = tp.smax;          
      }      

      // kernal loop, making mask
      // trying to write in a way that is easy to accelerate
      int N = (2*dcol+1)*(2*drow+1); // number of elements in kernel

      // parallelize masking with block kernel
      float pixsum = 0.;
      float masksum = 0.;

      #pragma omp parallel for         
      for (int ikernel=0; ikernel<N; ikernel++) {

        int dr = ikernel%(2*drow+1) - drow;
        int dc = ikernel/(2*drow+1) - dcol;          

        for (size_t itp=0; itp<tplist.size(); itp++) {
          auto const& tp = tplist[itp];
          if ( tp.smin>0 && tp.smax<dist)  {
            // change (col,row) from original image to mask image (with padding)
            // then add kernal shift
            int icol = (int)tp.col-col_origin + dc;
            int irow = (int)tp.row-row_origin + dr;
            if ( icol>=colwidth || irow>=rowwidth || icol<0 || irow<0) {
              std::stringstream ss;
              ss << "OUT OF CROP BOUNDS (" << icol << "," << irow << ") TP[" << itp << "] orig=(" << tp.col << "," << tp.row << ")" << std::endl;
              throw std::runtime_error(ss.str());
            }

            int cropindex = icol*rowwidth+irow;
            float pixval = crop[ cropindex ];

            if ( pixval>threshold ) {
              #pragma omp atomic
              mask[ cropindex ] += 1.0;
            }
          }
        }
      }
      #pragma omp barrier
        
      // now sum charge
      for (size_t ii=0; ii<mask.size(); ii++ ) {
        if ( mask[ii]>0.5 ) {
          mask[ii] = 1.0;
          pixsum +=  crop[ii];          
          masksum += mask[ii];
        }
      }

      planesum_v[p] = pixsum;

      // now get the dx
      float seen_smin = 1e9;
      float seen_smax = -1e9;
      
      // find smin and smax along path
      for (size_t ii=0; ii<mask.size(); ii++ ) {
        if ( mask[ii]>0 ) {
          int ic = ii/rowwidth;
          int ir = ii%rowwidth;
          //_debug_crop_v.back().SetBinContent(ic+1,ir+1,1.0);            
          float pix_smin = smin_img[ic*rowwidth+ir];
          float pix_smax = smax_img[ic*rowwidth+ir];
          if ( pix_smin>0 && seen_smin>pix_smin ) {
            seen_smin = pix_smin;
          }
          if ( pix_smax>0 && seen_smax<pix_smax ) {
            seen_smax = pix_smax;
          }
        }
      }
      
      float ds = seen_smax - seen_smin;
      LARCV_DEBUG () << "plane[" << p << "] pixsum=" << planesum_v[p] << " ds=" << ds << " seen_smin=" << seen_smin << " seen_smax=" << seen_smax << std::endl;
      if ( ds>0 && ds<10.0 )
        planesum_v[p] /= ds;
      else
        planesum_v[p] = 0.;
      
    }//end of plane loop
    
    return planesum_v;
  }

  /**
   * @brief create collection of TrunkPix_t for each plane
   *
   */
  void ShowerdQdx::_createDistLabels( const std::vector<float>& start3d,
                                      const std::vector<float>& end3d,
                                      const std::vector<const larcv::Image2D*>& img_v,
                                      const float threshold,
				      const int tpcid,
				      const int cryoid )
  {
    
    _visited_v.clear();    
    _plane_trunkpix_v.clear();
    
    const int nplanes = img_v.size();
    _plane_trunkpix_v.reserve(nplanes);
    _visited_v.resize(nplanes);

    float dist = 0.;
    std::vector<float> dir(3,0);
    for (int i=0; i<3; i++) {
      dir[i] = end3d[i]-start3d[i];
      dist += dir[i]*dir[i];
    }
    dist = sqrt(dist);
    for (int i=0; i<3; i++)
      dir[i] /= dist;

    float maxstepsize = 0.2; // cm
    int nsteps = dist/maxstepsize+1;
    float stepsize = dist/nsteps;
    
    for ( int p=0; p<nplanes; p++ ) {
      TrunkPixMap_t& _visited_m = _visited_v[p];
      _visited_m.clear();
      
      auto const& img = *img_v[p];
      //larcv::Image2D blank(img.meta());
      //blank.paint(0.0);
      float pixsum = 0.;
      for (int istep=-nsteps; istep<=nsteps; istep++) {
        float s = istep*stepsize;
        std::vector<float> pos(3,0);
        for (int i=0; i<3; i++)
          pos[i] = start3d[i] + s*dir[i];

        // project into image
        std::vector<float> col_grad;
        std::vector<float> row_grad;        
        float col = colcoordinate_and_grad( pos, p, tpcid, cryoid, nplanes, img.meta(), col_grad );
        float row = rowcoordinate_and_grad( pos, p, tpcid, cryoid, img.meta(), row_grad );
        if ( col<0 || col>=(float)img.meta().cols()
             || row<0 || row>=(float)img.meta().rows() )
          continue;

        int icol = (int)col;
        int irow = (int)row;

        std::pair<int,int> pix(icol,irow);

        auto it = _visited_m.find(pix);
        if ( it==_visited_m.end() ) {
          TrunkPix_t tp(irow,icol,s,s);          
          _visited_m[pix] = tp;
        }
        else {
          // found, update min and max s
          auto& trunkpix = it->second;
          if ( trunkpix.smin > s )
            trunkpix.smin = s;
          if ( trunkpix.smax < s )
            trunkpix.smax = s;
        }
        
      }//end of step loop

      // now copy into forward list
      TrunkPixList_t flist; // sorted by smin
      flist.reserve( _visited_m.size() );
      for ( auto it=_visited_m.begin(); it!=_visited_m.end(); it++ ) {
        flist.push_back( it->second );
      }
      for ( auto& tp : flist )
        tp.s = tp.smin;      
      std::sort( flist.begin(), flist.end() );


      // dump
      if ( false ) {
        LARCV_DEBUG() << "[plane " << p << "] trunk pixel list ------------------" << std::endl;
        for ( auto& tp : flist ) {
          LARCV_DEBUG() << " [" << tp.col << "," << tp.row << "] smin=" << tp.smin << " smax=" << tp.smax << std::endl;
        }
      }

      _plane_trunkpix_v.emplace_back( std::move(flist) );      
    }//end of plane loop
    
  }

  /**
   * @brief define segments along the trunk
   */
  void ShowerdQdx::_makeSegments( const float starting_s, const float seg_size )
  {
    
    _plane_seg_dedx_v.clear();
    
    // we first define the zeroth segment
    // we will define segments going forward and backward from there
    for (int p=0; p<(int)_plane_trunkpix_v.size(); p++) {

      SegList_t seg_v;
      seg_v.reserve(20);
      
      auto& tplist = _plane_trunkpix_v[p];
      int isegz = -1;
      float smax = 0;
      for ( size_t i=0; i<tplist.size(); i++ ) {
        auto& tp = tplist[i];
        if ( tp.smin<starting_s && tp.smax>starting_s ) {
          isegz = (int)i;
        }
        else if ( tp.smin>starting_s && isegz<0 ) {
          isegz = (int)i;
        }        
        if ( 0.5*(tp.smin+tp.smax)>smax )
          smax = 0.5*(tp.smin+tp.smax);
      }
      if ( isegz<0 )
        continue; // should never happen
      auto& tpz = tplist[isegz];
      float smid = 0.5*(tpz.smin+tpz.smax);

      // ok, now we define segments along the trunk
      // we will calculate dqdx at these segments
      float s_start = smid;
      while ( s_start<smax ) {
        float s1 = s_start;
        float s2 = s1 + seg_size;
        // now we find the range over tplist
        float slimit1 = -2e3;
        float slimit2 = -2e3;
        int itp1 = -1;
        int itp2 = -1;
        for ( int itp=0; itp<(int)tplist.size(); itp++) {
          auto& tp = tplist[itp];
          if ( (tp.smin<=s1 && tp.smax>=s1) || (slimit1<-1e3 && tp.smin>=s1 ) ) {
            slimit1 = tp.smin;
            itp1 = itp;
          }
          if ( (tp.smin<=s2 && tp.smax>=s2) || (slimit2<-1e3 && tp.smin>=s2 ) ) {
            slimit2 = tp.smax;
            itp2 = itp;
          }
          if ( slimit1>-1e3 && slimit2>-1e3 )
            break;
        }
        if ( slimit1>=-1e3 && slimit2<-1e3 ) {
          slimit2 = tplist.back().smax;
          itp2 = (int)tplist.size()-1;
        }
        
        if ( slimit1>-1e3 && slimit2>-1e3 ) {
          Seg_t seg;
          seg.smin = slimit1;
          seg.smax = slimit2;
          seg.s = 0.5*(slimit1+slimit2);
          seg.plane = p;
          seg.pixsum = 0.;
          seg.dqdx = 0.;
          seg.ds = 0.;
          seg.itp1 = itp1;
          seg.itp2 = itp2;
          
          LARCV_DEBUG() << "plane[" << p << "]-seg[" << seg_v.size() << "] "
                        << " smin[" << seg.itp1 << "]=" << seg.smin
                        << " smax[" << seg.itp2 << "]=" << seg.smax
                        << std::endl;
        
          seg_v.push_back(seg);
        }
        
        // define the segment
        s_start += 0.5;
      }//end of seg loop
      _plane_seg_dedx_v.emplace_back( std::move(seg_v) );
    }//end of plane loop
  }

  /**
   * @brief calculate dq/dx for the collection of segment points for each plane
   *
   */
  void ShowerdQdx::_sumChargeAlongSegments( const std::vector<float>& start3d,
                                            const std::vector<float>& end3d,
                                            const std::vector<larcv::Image2D>& img_v,
                                            const float threshold,
                                            const int dcol, const int drow,
					    const int tpcid, const int cryoid )
  {

    _plane_dqdx_seg_v.clear();
    _plane_s_seg_v.clear();
    _debug_crop_v.clear();
    
    const int nplanes = img_v.size();
    
    _plane_dqdx_seg_v.resize(nplanes);
    _plane_s_seg_v.resize(nplanes);    

    float dist = 0.;
    std::vector<float> dir(3,0);
    for (int i=0; i<3; i++) {
      dir[i] = end3d[i]-start3d[i];
      dist += dir[i]*dir[i];
    }
    dist = sqrt(dist);
    for (int i=0; i<3; i++)
      dir[i] /= dist;

    for ( int p=0; p<nplanes; p++ ) {
      auto const& img = img_v[p];
      larcv::Image2D blank(img.meta());
      auto const& tplist = _plane_trunkpix_v[p];

      // get min,max bounds
      int colbounds[2] = {-1};
      int rowbounds[2] = {-1};
      for (auto const& tp : tplist ) {
        if ( colbounds[0]>tp.col || colbounds[0]<0 )
          colbounds[0] = tp.col;
        if ( colbounds[1]<tp.col || colbounds[1]<0 )
          colbounds[1] = tp.col;
        if ( rowbounds[0]>tp.row || rowbounds[0]<0 )
          rowbounds[0] = tp.row;
        if ( rowbounds[1]<tp.row || rowbounds[1]<0 )
          rowbounds[1] = tp.row;        
      }

      // crop width with padding added
      int colwidth = (colbounds[1]-colbounds[0]+1) + 2*dcol;
      int rowwidth = (rowbounds[1]-rowbounds[0]+1) + 2*drow;
      int rowlen   = (rowbounds[1]-rowbounds[0]+1);
      int col_origin = colbounds[0]-dcol;
      int row_origin = rowbounds[0]-drow;
      LARCV_DEBUG() << "colbounds: [" << colbounds[0] << "," << colbounds[1] << "]" << std::endl;
      LARCV_DEBUG() << "rowbounds: [" << rowbounds[0] << "," << rowbounds[1] << "]" << std::endl;      
      LARCV_DEBUG() << "Crop meta. Origin=(" << col_origin << "," << row_origin << ")"
                    << " ncol=" << colwidth << " nrow=" << rowwidth
                    << std::endl;

      if ( colwidth<=0 || rowwidth<=0 ) {
        char zdebughist[200];
        sprintf(zdebughist,"showerbilinear_crop_plane%d_ii%d",p,ndebugcount);
        ndebugcount++;
        TH2D hcrop(zdebughist,"",1,0,1,1,0,1);
        _debug_crop_v.emplace_back( std::move(hcrop) );
        LARCV_WARNING() << "[ShowerdQdx::_sumChargeAlongSegments] Plane[" << p << "] Bad cropped window size" << std::endl;
        continue;
      }

      // make data array
      std::vector<float> crop(colwidth*rowwidth,0);
      std::vector<float> mask(colwidth*rowwidth,0);
      std::vector<float> smin_img(colwidth*rowwidth,0);
      std::vector<float> smax_img(colwidth*rowwidth,0);

      // copy the data into the cropped matrix
      // preserve the row-ordered data
      for (int c=0; c<(int)(colbounds[1]-colbounds[0]); c++) {
        // for the larcv image, the row-dimension is the contiguous element
        size_t origin_index = img.meta().index( rowbounds[0], colbounds[0]+c, __FILE__, __LINE__ );
        const float* source_ptr = img.as_vector().data() + origin_index;
        float* dest_ptr   = crop.data() + (c+dcol)*rowwidth + drow;
        memcpy( dest_ptr, source_ptr, sizeof(float)*rowlen );
      }

      for (int itp=0; itp<(int)tplist.size(); itp++) {
        auto const& tp = tplist[itp];
        // change (col,row) from original image to mask image (with padding)
        // then add kernal shift
        int icol = (int)tp.col-col_origin;
        int irow = (int)tp.row-row_origin;
        smin_img[icol*rowwidth+irow] = tp.smin;
        smax_img[icol*rowwidth+irow] = tp.smax;          
      }

      char zdebughist[200];
      sprintf(zdebughist,"showerbilinear_crop_plane%d_ii%d",p,ndebugcount);
      ndebugcount++;
      TH2D hcrop(zdebughist,"",colwidth,0,colwidth,rowwidth,0,rowwidth);
      for (size_t ii=0; ii<crop.size(); ii++) {
        int ic = ii/rowwidth;
        int ir = ii%rowwidth;
        hcrop.SetBinContent(ic+1,ir+1,crop[ii]);
      }
      _debug_crop_v.emplace_back( std::move(hcrop) );
           
      // store dqdx for each segment in vector to be used for TTree
      std::vector<float>& out_dqdx_seg_v = _plane_dqdx_seg_v[p];
      std::vector<float>& out_s_seg_v    = _plane_s_seg_v[p];
      out_dqdx_seg_v.reserve( _plane_seg_dedx_v[p].size() );
      out_s_seg_v.reserve( _plane_seg_dedx_v[p].size() );

      int iseg = 0;
      for ( auto& seg : _plane_seg_dedx_v[p] ) {

        //blank.paint(0.0);
        float seen_smin = 1e9;
        float seen_smax = -1e9;
        float pixsum = 0.;
        float masksum = 0.;

        // blank out mask
        memset( mask.data(),    0, sizeof(float)*mask.size() );

        // kernal loop, making mask
        // trying to write in a way that is easy to accelerate
        int N = (2*dcol+1)*(2*drow+1);
        int itp1 = seg.itp1;
        int Nitp = (seg.itp2-seg.itp1)+1;

        // parallelize masking with block kernel
        #pragma omp parallel for         
        for (int ikernel=0; ikernel<N; ikernel++) {

          int dr = ikernel%(2*drow+1) - drow;
          int dc = ikernel/(2*drow+1) - dcol;          

          for (int itp=0; itp<Nitp; itp++) {
            int idx = itp1+itp;
            auto const& tp = tplist[idx];
            // change (col,row) from original image to mask image (with padding)
            // then add kernal shift
            int icol = (int)tp.col-col_origin + dc;
            int irow = (int)tp.row-row_origin + dr;
            if ( icol>=colwidth || irow>=rowwidth || icol<0 || irow<0) {
              std::stringstream ss;
              ss << "OUT OF CROP BOUNDS (" << icol << "," << irow << ") TP[" << idx << "] orig=(" << tp.col << "," << tp.row << ")" << std::endl;
              throw std::runtime_error(ss.str());
            }

            int cropindex = icol*rowwidth+irow;
            float pixval = crop[ cropindex ];
            if ( pixval>threshold ) {
              #pragma omp atomic
              mask[ cropindex ] += 1.0;
            }
          }
        }
        #pragma omp barrier
        
        // now sum charge
        for (size_t ii=0; ii<mask.size(); ii++ ) {
          if ( mask[ii]>0.5 ) {
            mask[ii] = 1.0;
            pixsum +=  crop[ii];          
            masksum += mask[ii];
          }
        }

        // find smin and smax along path
        for (size_t ii=0; ii<mask.size(); ii++ ) {
          if ( mask[ii]>0 ) {
            int ic = ii/rowwidth;
            int ir = ii%rowwidth;
            //_debug_crop_v.back().SetBinContent(ic+1,ir+1,1.0);            
            float pix_smin = smin_img[ic*rowwidth+ir];
            float pix_smax = smax_img[ic*rowwidth+ir];
            if ( pix_smin>0 && seen_smin>pix_smin ) {
              seen_smin = pix_smin;
            }
            if ( pix_smax>0 && seen_smax<pix_smax ) {
              seen_smax = pix_smax;
            }
          }
        }

        float ds = seen_smax - seen_smin;
        if ( ds>0 && ds<10.0 ) {
          // good
          seg.dqdx = pixsum/ds;        
          seg.pixsum = pixsum;
          seg.ds = ds;
          out_dqdx_seg_v.push_back( seg.dqdx );
          out_s_seg_v.push_back( seg.s );
        }
        else {
          // bad
          seg.dqdx = 0.;
          seg.pixsum = pixsum;
          seg.ds = 0.;
        }
        
        LARCV_DEBUG() << "plane[" << p << "]-seg[" << iseg << "] "
                      << " tp-index=[" << seg.itp1 << "-" << seg.itp2 << "]"
                      << " pixsum=" << seg.pixsum
                      << " masksum=" << masksum
                      << " dx=" << ds
                      << " dqdx=" << seg.dqdx << std::endl;
        
        iseg++;
      }//end of seg loop
    }//end of plane loop

  }

  /**
   * @brief sum charge along a single segment
   *
   */
  float ShowerdQdx::_sumChargeAlongOneSegment( ShowerdQdx::Seg_t& seg,
                                               const int plane,
                                               const std::vector<larcv::Image2D>& img_v,
                                               const float threshold,
                                               const int dcol, const int drow )
  {

    auto const& img = img_v[plane];
    auto const& tplist = _plane_trunkpix_v[plane];

    // get min,max bounds
    int colbounds[2] = {-1};
    int rowbounds[2] = {-1};
    for (auto const& tp : tplist ) {
      
      if ( colbounds[0]>tp.col || colbounds[0]<0 )
        colbounds[0] = tp.col;
      if ( colbounds[1]<tp.col || colbounds[1]<0 )
        colbounds[1] = tp.col;
      if ( rowbounds[0]>tp.row || rowbounds[0]<0 )
        rowbounds[0] = tp.row;
      if ( rowbounds[1]<tp.row || rowbounds[1]<0 )
        rowbounds[1] = tp.row;

      // LARCV_DEBUG() << "tp[" << tp.col << "," << tp.row << "] "
      //   	    << " colbounds={" << colbounds[0] << "," << colbounds[1] << "}"
      //   	    << " rowbounds={" << rowbounds[0] << "," << rowbounds[1] << "}"
      //   	    << std::endl;      
    }

    // crop width with padding added
    int colwidth = (colbounds[1]-colbounds[0]+1) + 2*dcol;
    int rowwidth = (rowbounds[1]-rowbounds[0]+1) + 2*drow;
    int rowlen   = (rowbounds[1]-rowbounds[0]+1);
    int col_origin = colbounds[0]-dcol;
    int row_origin = rowbounds[0]-drow;
    LARCV_DEBUG() << "seg itp: [" << seg.itp1 << "," << seg.itp2 << "]" << std::endl;
    LARCV_DEBUG() << "colbounds: [" << colbounds[0] << "," << colbounds[1] << "]" << std::endl;
    LARCV_DEBUG() << "rowbounds: [" << rowbounds[0] << "," << rowbounds[1] << "]" << std::endl;      
    LARCV_DEBUG() << "Crop meta. Origin=(" << col_origin << "," << row_origin << ")"
                  << " ncol=" << colwidth << " nrow=" << rowwidth
                  << std::endl;

    if ( colwidth<=0 || rowwidth<=0 ) {
      throw std::runtime_error("[ShowerdQdx::_sumChargeAlongOneSegment] Bad cropped window size");
    }

    // make data array
    std::vector<float> crop(colwidth*rowwidth,0);
    std::vector<float> mask(colwidth*rowwidth,0);
    std::vector<float> smin_img(colwidth*rowwidth,0);
    std::vector<float> smax_img(colwidth*rowwidth,0);

    // copy the data into the cropped matrix
    // preserve the row-ordered data
    for (int c=0; c<(int)(colbounds[1]-colbounds[0]); c++) {
      // for the larcv image, the row-dimension is the contiguous element
      size_t origin_index = img.meta().index( rowbounds[0], colbounds[0]+c, __FILE__, __LINE__ );
      const float* source_ptr = img.as_vector().data() + origin_index;
      float* dest_ptr   = crop.data() + (c+dcol)*rowwidth + drow;
      memcpy( dest_ptr, source_ptr, sizeof(float)*rowlen );
    }

    for (int itp=0; itp<(int)tplist.size(); itp++) {
      auto const& tp = tplist[itp];
      // change (col,row) from original image to mask image (with padding)
      // then add kernal shift
      int icol = (int)tp.col-col_origin;
      int irow = (int)tp.row-row_origin;
      smin_img[icol*rowwidth+irow] = tp.smin;
      smax_img[icol*rowwidth+irow] = tp.smax;          
    }
           
    float seen_smin = 1e9;
    float seen_smax = -1e9;
    float pixsum = 0.;
    float masksum = 0.;

    // blank out mask
    memset( mask.data(),    0, sizeof(float)*mask.size() );

    // kernal loop, making mask
    // trying to write in a way that is easy to accelerate
    int N = (2*dcol+1)*(2*drow+1);
    int itp1 = seg.itp1;
    int Nitp = (seg.itp2-seg.itp1)+1;

    // parallelize masking with block kernel
    #pragma omp parallel for         
    for (int ikernel=0; ikernel<N; ikernel++) {

      int dr = ikernel%(2*drow+1) - drow;
      int dc = ikernel/(2*drow+1) - dcol;          

      for (int itp=0; itp<Nitp; itp++) {
        int idx = itp1+itp;
        auto const& tp = tplist[idx];
        // change (col,row) from original image to mask image (with padding)
        // then add kernal shift
        int icol = (int)tp.col-col_origin + dc;
        int irow = (int)tp.row-row_origin + dr;
        if ( icol>=colwidth || irow>=rowwidth || icol<0 || irow<0) {
          std::stringstream ss;
          ss << "OUT OF CROP BOUNDS (" << icol << "," << irow << ") TP[" << idx << "] orig=(" << tp.col << "," << tp.row << ")" << std::endl;
          throw std::runtime_error(ss.str());
        }

        int cropindex = icol*rowwidth+irow;
        float pixval = crop[ cropindex ];
        if ( pixval>threshold ) {
          #pragma omp atomic
          mask[ cropindex ] += 1.0;
        }
      }
    }
    #pragma omp barrier
        
    // now sum charge
    for (size_t ii=0; ii<mask.size(); ii++ ) {
      if ( mask[ii]>0.5 ) {
        mask[ii] = 1.0;
        pixsum +=  crop[ii];          
        masksum += mask[ii];
      }
    }

    // find smin and smax along path
    for (size_t ii=0; ii<mask.size(); ii++ ) {
      if ( mask[ii]>0 ) {
        int ic = ii/rowwidth;
        int ir = ii%rowwidth;
        //_debug_crop_v.back().SetBinContent(ic+1,ir+1,1.0);            
        float pix_smin = smin_img[ic*rowwidth+ir];
        float pix_smax = smax_img[ic*rowwidth+ir];
        if ( pix_smin>0 && seen_smin>pix_smin ) {
          seen_smin = pix_smin;
        }
        if ( pix_smax>0 && seen_smax<pix_smax ) {
          seen_smax = pix_smax;
        }
      }
    }

    float ds = seen_smax - seen_smin;
    if ( ds>0 && ds<10.0 ) {
      return pixsum/ds;
    }
    else {
      // bad
      return 0;
    }
        
    return 0;
  }
  
  /**
   * @brief bind member variables to a given TTree
   */
  void ShowerdQdx::bindVariablesToTree( TTree* outtree )
  {
    outtree->Branch( "pixsum_dqdx_v",     &_pixsum_dqdx_v );
    outtree->Branch( "best_pixsum_dqdx",  &_best_pixsum_dqdx );
    outtree->Branch( "best_pixsum_plane", &_best_pixsum_plane );
    outtree->Branch( "best_pixsum_ortho", &_best_pixsum_ortho );
    outtree->Branch( "shower_dir",        &_shower_dir );
    outtree->Branch( "plane_dqdx_seg_vv", &_plane_dqdx_seg_v );
    outtree->Branch( "plane_s_seg_vv",    &_plane_s_seg_v );

    outtree->Branch( "plane_electron_srange_vv",  &_plane_electron_srange_v );
    outtree->Branch( "plane_electron_dqdx_v",     &_plane_electron_dqdx_v );
    outtree->Branch( "plane_electron_dx_v",       &_plane_electron_dx_v );
    outtree->Branch( "plane_electron_mean_v",     &_plane_electron_mean_v );
    outtree->Branch( "plane_electron_rms_v",      &_plane_electron_rms_v );
    outtree->Branch( "plane_electron_ngood_v",    &_plane_electron_ngood_v );
    outtree->Branch( "plane_electron_best",       &_plane_electron_best );
    outtree->Branch( "plane_electron_best_mean",  &_plane_electron_best_mean );
    outtree->Branch( "plane_electron_best_rms",   &_plane_electron_best_rms );
    outtree->Branch( "plane_electron_best_ngood", &_plane_electron_best_ngood );
    outtree->Branch( "plane_electron_best_start", &_plane_electron_best_start );        

    outtree->Branch( "plane_gamma_srange_vv",  &_plane_gamma_srange_v );
    outtree->Branch( "plane_gamma_dqdx_v",     &_plane_gamma_dqdx_v );
    outtree->Branch( "plane_gamma_dx_v",       &_plane_gamma_dx_v );
    outtree->Branch( "plane_gamma_mean_v",     &_plane_gamma_mean_v );
    outtree->Branch( "plane_gamma_rms_v",      &_plane_gamma_rms_v );
    outtree->Branch( "plane_gamma_ngood_v",    &_plane_gamma_ngood_v );        
    outtree->Branch( "plane_gamma_best",       &_plane_gamma_best );
    outtree->Branch( "plane_gamma_best_mean",  &_plane_gamma_best_mean );
    outtree->Branch( "plane_gamma_best_rms",   &_plane_gamma_best_rms );
    outtree->Branch( "plane_gamma_best_ngood", &_plane_gamma_best_ngood );
    outtree->Branch( "plane_gamma_best_start", &_plane_gamma_best_start );        

    outtree->Branch( "true_match_pdg",       &_true_match_pdg );
    outtree->Branch( "true_min_feat_dist",   &_true_min_feat_dist );
    outtree->Branch( "true_vertex_err_dist", &_true_vertex_err_dist );
    outtree->Branch( "true_dir_cos",         &_true_dir_cos );
    outtree->Branch( "true_max_primary_cos", &_true_max_primary_cos );    
  }

  /**
   * @brief mark pixels in TrunkPix_t collection for a given plane
   *
   * The pixels along the trunk are labeled the by the
   * shortest 3d distance from pixel to begining of shower trunk.
   *
   * @param[in] plane The plane whose trunk pixel collection we will mask with
   * @param[in] hist  The 2D image to mask
   */
  void ShowerdQdx::maskPixels( int plane, TH2D* hist )
  {
    // mostly for debugging
    auto const& tplist = _plane_trunkpix_v[plane];
    for (int itp=0; itp<(int)tplist.size(); itp++ ) {
      auto const& tp  = tplist[itp];
      hist->SetBinContent( tp.col+1, tp.row+1, tp.smin );
    }
  }

  /**
   * @brief plot the dq/dx vs. trunk position for a plane's segment collection
   *
   * @param[in] plane The index of the plane whose segment collection we will plot
   * @return the curve represented by a ROOT TGraph
   *
   */
  TGraph ShowerdQdx::makeSegdQdxGraphs(int plane)
  {

    if ( plane<0 || plane>=(int)_plane_seg_dedx_v.size() ) {
      TGraph g(1);
      return g;
    }
    
    auto const& seglist = _plane_seg_dedx_v.at(plane);
    int N = (int)seglist.size();
    TGraph g(N);
    for (int i=0; i<N; i++) {
      g.SetPoint(i,seglist[i].s,seglist[i].dqdx);
    }
    return g;
  }

  /**
   * @brief given a collection of segment data, find a consecutive set of dq/dx values within some range
   * 
   */
  void ShowerdQdx::_findRangedQdx( const std::vector<float>& start3d,
                                           const std::vector<float>& end3d,
                                           const std::vector<larcv::Image2D>& adc_v,
                                           const float dqdx_max,
                                           const float dqdx_threshold,
                                           std::vector<float>& plane_dqdx_v,
                                           std::vector<float>& plane_dx_v,
                                           std::vector< std::vector<float> >& plane_srange_v,
                                           std::vector<float>& plane_mean_v,
                                           std::vector<float>& plane_rms_v,
                                           std::vector<int>& plane_ngood_v,
                                           int& best_plane,
                                           int& plane_max_ngood,
                                           float& plane_best_dqdx,
                                           float& plane_best_rms,
                                           float& plane_best_start )
  {

    // step forward from 0
    // find first repeat segments in good range [dqdx_threshold,dqdx_max]
    // count number of good segments in range
    // calc rms of segment values in range

    const int nplanes = adc_v.size();
    plane_dqdx_v.clear();
    plane_srange_v.clear();
    plane_dx_v.clear();
    plane_mean_v.clear();
    plane_rms_v.clear();
    plane_ngood_v.clear();
    
    plane_srange_v.resize(nplanes);
    plane_dqdx_v.resize(nplanes,0);
    plane_dx_v.resize(nplanes,0);
    plane_mean_v.resize(nplanes,0);
    plane_rms_v.resize(nplanes,0);
    plane_ngood_v.resize(nplanes,0);

    best_plane = -1;
    plane_max_ngood = 0;
    plane_best_dqdx = 0;
    plane_best_rms = 0;
    plane_best_start = -10;
           
    // get distance between start and end points
    float dist = 0.;
    std::vector<float> dir(3,0);
    for (int i=0; i<3; i++) {
      dir[i] = end3d[i]-start3d[i];
      dist += (end3d[i]-start3d[i])*(end3d[i]-start3d[i]);
    }
    dist = sqrt(dist);
    if ( dist>0 ) {
      for (int i=0; i<3; i++)
        dir[i] /= dist;
    }    

    for (int p=0; p<nplanes; p++) {
      auto const& seg_v = _plane_seg_dedx_v[p]; // segment list
      auto& tplist = _plane_trunkpix_v[p]; // track point list
      int iseg_start = 0;
      int iseg_end = (int)seg_v.size()-1;
      // find first value below
      int num_inregion = 0;
      int num_outregion = 0;
      int tot_inregion = 0;
      bool good_region_started = false;

      float in_region_mean_x = 0.;
      float in_region_mean_x2 = 0.;
      
      for (int iseg=0; iseg<(int)seg_v.size(); iseg++) {
        auto const& seg = seg_v[iseg];
        if ( seg.s<0.0 )
          continue;
        if ( seg.dqdx>dqdx_threshold && seg.dqdx<dqdx_max ) {
          num_inregion++;                    
          if (!good_region_started && num_inregion>=2) {
            good_region_started = true;
            iseg_start = iseg-1;
            in_region_mean_x  += seg_v[iseg_start].dqdx;
            in_region_mean_x2 += seg_v[iseg_start].dqdx*seg_v[iseg_start].dqdx;
            tot_inregion = 1;
          }
          if (good_region_started) {
            in_region_mean_x += seg.dqdx;
            in_region_mean_x2 += seg.dqdx*seg.dqdx;
            tot_inregion++;
          }
          num_outregion = 0; //reset out of region counter
        }
        else {
          num_outregion++;
          if (good_region_started && num_outregion>=2) {
            iseg_end = iseg-2;
            break; // we're done
          }
          num_inregion = 0;
        }
        
      }

      // define range
      float s_min = seg_v[iseg_start].smin;
      float s_max = seg_v[iseg_end].smax;
      std::vector<float> srange = { s_min, s_max };
      if ( tot_inregion>0 ) {
        plane_mean_v[p]  = in_region_mean_x/float(tot_inregion);
        plane_rms_v[p]   = sqrt( in_region_mean_x2/float(tot_inregion) - plane_mean_v[p]*plane_mean_v[p] );
        plane_ngood_v[p] = tot_inregion;
      }
      
      plane_srange_v[p] = srange;

      std::vector<float> s3d(3,0);
      std::vector<float> e3d(3,0);
      for (int i=0; i<3; i++) {
        s3d[i] = start3d[i] + s_min*dir[i];
        e3d[i] = start3d[i] + s_max*dir[i];
      }

      // make a new segment to sum charge over
      Seg_t seg;
      seg.smin = s_min;
      seg.smax = s_max;
      seg.s = 0.5*(s_min+s_max);
      seg.itp1 = seg_v[iseg_start].itp1;
      seg.itp2 = seg_v[iseg_end].itp2;
      seg.plane = p;
      for  (int i=0; i<3; i++) {
        seg.endpt[0][i] = s3d[i]; 
        seg.endpt[1][i] = e3d[i];
      }
      seg.pixsum = 0;
      seg.dqdx = 0;
      seg.ds = s_max-s_min;

      LARCV_DEBUG() << "Plane[" << p << "] define range with iseg=[" << iseg_start << "," << iseg_end << "] itp=[" << seg.itp1 << "," << seg.itp2 << "]" << std::endl;
      if ( seg.itp2>(int)tplist.size() ) {
	LARCV_DEBUG() << "Correct bad segment trunkpix index. itp2=" << seg.itp2 << std::endl;
	seg.itp2 = (int)tplist.size()-1;
      }

      try {
        seg.dqdx = _sumChargeAlongOneSegment( seg, p, adc_v, 10.0, 1, 3 );
        LARCV_DEBUG() << "Plane[" << p << "] range dqdx s=(" << s_min << "," << s_max << ") dqdx=" << seg.dqdx << std::endl;        
      }
      catch (std::exception& e) {
        LARCV_WARNING() << "Plane[" << p << "] Error calculating dq/dx along segment: " << e.what() << std::endl;
        seg.dqdx = 0.0;
        seg.ds = 0.;
      }


      plane_dqdx_v[p] = seg.dqdx;
      plane_dx_v[p]   = seg.ds;
      
    }//end of plane loop

    // best plane determination
    for (int p=0; p<nplanes; p++) {
      if ( plane_ngood_v[p]>0 && plane_ngood_v[p]>plane_max_ngood ) {
        best_plane = p;        
        plane_max_ngood = plane_ngood_v[p];
        plane_best_dqdx = plane_mean_v[p];
        plane_best_rms  = plane_rms_v[p];
        plane_best_start = plane_srange_v[p][0];
      }
    }
    
  }

  /**
   * @brief For a given reco shower, find the truth mcshower that it most closely matches
   *
   */
  void ShowerdQdx::calcGoodShowerTaggingVariables( const larlite::larflowcluster& shower,
                                                   const larlite::track& trunk,
                                                   const larlite::pcaxis& pca,
                                                   const std::vector<const larcv::Image2D*>& padc_v,
                                                   const std::vector<larlite::mcshower>& mcshower_v,
						   const int tpcid, const int cryoid )
  {

    if ( ShowerdQdx::_psce==nullptr ) {
      ShowerdQdx::_psce = new larutil::SpaceChargeMicroBooNE;
    }
    
    auto const& meta = padc_v.front()->meta();
    auto const detp = larutil::DetectorProperties::GetME();


    std::vector<float> start_pos = { (float)trunk.LocationAtPoint(0)[0],
				     (float)trunk.LocationAtPoint(0)[1],
				     (float)trunk.LocationAtPoint(0)[2] };
    std::vector<float> end_pos = { (float)trunk.LocationAtPoint(1)[0],
				   (float)trunk.LocationAtPoint(1)[1],
				   (float)trunk.LocationAtPoint(1)[2] };

    float dist = 0.;
    std::vector<float> dir(3,0);
    for (int i=0; i<3; i++) {
      dir[i] = (end_pos[i]-start_pos[i]);
      dist += dir[i]*dir[i];
    }
    dist = sqrt(dist);
    if ( dist>0 ) {
      for (int i=0; i<3; i++)
        dir[i] /= dist;
    }
    
    //float start_tick = start_pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
    float start_tick = detp->ConvertXToTicks( start_pos[0], 0, tpcid, cryoid );
    if ( start_tick<meta.min_y() || start_tick>meta.max_y() ) {

      // try to shorten trunk to stay in image
      bool fix = false;
      if ( start_tick<=meta.min_y() && dir[0]!=0.0) {
	float mintick = detp->ConvertTicksToX(meta.pos_y( 1 ), 0, tpcid, cryoid);
	float s = (mintick-start_pos[0])/dir[0];
	
	fix = true;
	for (int i=0; i<3; i++)
	  start_pos[i] = start_pos[0] + s*dir[i];
	start_tick = meta.pos_y(1);
	
      }
      else if ( start_tick>=meta.max_y() && dir[0]!=0.0) {
	float maxtick = detp->ConvertTicksToX(meta.pos_y( (int)meta.rows()-1 ), 0, tpcid, cryoid );
	float s = (maxtick-start_pos[0])/dir[0];
	
	fix = true;
	for (int i=0; i<3; i++)
	  start_pos[i] = start_pos[0] + s*dir[i];
	start_tick = meta.pos_y( (int)meta.rows()-1 );
	
      }

      if ( !fix ) {	      
      
	std::stringstream errmsg;
	errmsg << "out of bounds shower trunk starting point "
	       << " (" << start_pos[0] << "," << start_pos[1] << "," << start_pos[2] << ") "
	       << " start_tick=" << start_tick << std::endl;
	LARCV_CRITICAL() << errmsg.str() << std::endl;
	throw std::runtime_error(errmsg.str());
      }
    }    
    float start_row = (float)meta.row(start_tick);
        
    float end_tick = detp->ConvertXToTicks( end_pos[0], 0, tpcid, cryoid );
    if ( end_tick<=meta.min_y() || end_tick>=meta.max_y() ) {
      
      // try to shorten trunk to stay in image
      bool fix = false;
      float s = 0.;
      if ( end_tick<=meta.min_y() && dir[0]!=0.0) {
	float mintick = detp->ConvertTicksToX( meta.pos_y( 1 ), 0, tpcid, cryoid );
	s = (mintick-start_pos[0])/dir[0];

	fix = true;
	for (int i=0; i<3; i++)
	  end_pos[i] = start_pos[0] + s*dir[i];
	end_tick = meta.pos_y(1);

      }
      else if ( end_tick>=meta.max_y() && dir[0]!=0.0) {
	float maxtick = detp->ConvertTicksToX( meta.pos_y( (int)meta.rows()-1 ), 0, tpcid, cryoid );
	s = (maxtick-start_pos[0])/dir[0];

	fix = true;
	for (int i=0; i<3; i++)
	  end_pos[i] = start_pos[0] + s*dir[i];
	end_tick = meta.pos_y( (int)meta.rows()-1 );

      }

      if ( !fix ) {
	std::stringstream errmsg;
	errmsg << "out of bounds shower trunk ending point "
	       << " s=" << s
	       << " (" << end_pos[0] << "," << end_pos[1] << "," << end_pos[2] << ") "
	       << " end_tick=" << end_tick << std::endl;
	LARCV_CRITICAL() << errmsg.str() << std::endl;      
	throw std::runtime_error(errmsg.str());
      }
    }
    float end_row = (float)meta.row(end_tick);
  
    dist = 0.;
    for (int i=0; i<3; i++) {
      dir[i] = end_pos[i]-start_pos[i];
      dist += dir[i]*dir[i];
    }
    dist = sqrt(dist);
    for (int i=0; i<3; i++)
      dir[i] /= dist;

    int min_index = -1;
    float min_feat_dist = 1e9;
    float vertex_err_dist = 0;
    int   match_pdg = 0;
    float dir_cos = 0.;

    for ( int ishower=0; ishower<(int)mcshower_v.size(); ishower++ ) {
      
      auto const& mcshower = mcshower_v[ishower];

      if ( mcshower.Origin()!=1 )
        continue; // not neutrino origin

      auto const& profile = mcshower.DetProfile();

      // default shower_dir: from the truth
      
      TVector3 shower_dir = mcshower.Start().Momentum().Vect();
      float pmom = shower_dir.Mag();
      TVector3 mcstart = mcshower.Start().Position().Vect();
      TVector3 pstart  = profile.Position().Vect(); // start of shower profile

      if ( mcshower.PdgCode()==22 ) {
        // if gamma, we want to use the dir and start from the profile
        shower_dir = profile.Momentum().Vect();
        pmom = shower_dir.Mag();
        mcstart = pstart;
      }

      // copy TVector3 to vector<float>, so we can use geofunc
      std::vector<float> mcdir(3,0);
      std::vector<float> fmcstart(3,0);
      std::vector<float> fmcend(3,0);
      for (int i=0; i<3; i++) {
        shower_dir[i] /= pmom;
        mcdir[i] = (float)shower_dir[i];
        fmcstart[i] = mcstart[i];
        fmcend[i] = fmcstart[i] + 10.0*mcdir[i];
      }

      if ( mcshower.PdgCode()==22 ) {           
        // space charge correction
        std::vector<double> s_offset = _psce->GetPosOffsets(mcstart[0],mcstart[1],mcstart[2]);
        fmcstart[0] = fmcstart[0] - s_offset[0] + 0.7;
        fmcstart[1] = fmcstart[1] + s_offset[1];
        fmcstart[2] = fmcstart[2] + s_offset[2];
        
        std::vector<double> e_offset = _psce->GetPosOffsets(fmcend[0],fmcend[1],fmcend[2]);
        fmcend[0] = fmcend[0] - e_offset[0] + 0.7;
        fmcend[1] = fmcend[1] + e_offset[1];
        fmcend[2] = fmcend[2] + e_offset[2];
      }

      std::vector<float> fsce_dir(3,0);
      float sce_dir_len = 0.;
      for (int i=0; i<3; i++) {
        fsce_dir[i] = fmcend[i] - fmcstart[i];
        sce_dir_len += (fsce_dir[i]*fsce_dir[i]);
      }
      sce_dir_len = sqrt( sce_dir_len );
      if ( sce_dir_len>0 ) {
        for (int i=0; i<3; i++) {
          fsce_dir[i] /= sce_dir_len;
        }
      }      

      // finally!
      float dvertex = larflow::reco::pointRayProjection3f( start_pos, dir, fmcstart );
      float fcos = 0.;
      for (int i=0; i<3; i++) {
        fcos += fsce_dir[i]*dir[i];
      }

      float goodmetric = (1.0-fcos)*(1.0-fcos) + (dvertex*dvertex/9.0); // dvertex has a sigma of 3 cm
      if ( min_feat_dist>goodmetric ) {
        dir_cos = fcos;
        vertex_err_dist = dvertex;
        min_feat_dist = goodmetric;
        match_pdg = mcshower.PdgCode();
        min_index =  ishower;
      }
    }
    
    // store in member variables
    _true_min_feat_dist   = min_feat_dist;
    _true_vertex_err_dist = vertex_err_dist;
    _true_dir_cos = dir_cos;
    _true_match_pdg = match_pdg;
    _true_min_index = min_index;

    LARCV_INFO() << "Best true shower match: " << std::endl;
    LARCV_INFO() << " - feat_dist=" << _true_min_feat_dist << std::endl;
    LARCV_INFO() << " - vertex_dist="<< _true_vertex_err_dist << std::endl;
    LARCV_INFO() << " - true-dir-cos=" << _true_dir_cos << std::endl;
    LARCV_INFO() << " - match PDG code=" << _true_match_pdg << std::endl;
    LARCV_INFO() << " - true min index=" << _true_min_index << std::endl;
    
  }

  /**
   * @brief Make images where the reco tracks from a neutrino candidate are masked to zero
   *
   */
  std::vector<larcv::Image2D>
  ShowerdQdx::maskTrackPixels( const std::vector<const larcv::Image2D*>& padc_v,
                               const larlite::track& shower_trunk,
                               const larflow::reco::NuVertexCandidate& nuvtx,
			       const int tpcid, const int cryoid )
  {

    ClusterImageMask masker; // class contains functions to mask pixels along track object

    // create blank images. will add track mask to this
    std::vector<larcv::Image2D> mask_v;
    for ( auto const& adc : padc_v ) {
      larcv::Image2D mask( adc->meta() );
      mask.paint(0.0);
      mask_v.emplace_back( std::move(mask) );
    }

    // could do this two ways: use track cluster or fitted track ...
    for ( auto const& track : nuvtx.track_v ) {

      // don't mask colinear tracks
      float dist1 = 0;
      float dist2 = 0;
      auto const& tstart = track.LocationAtPoint(0);
      auto const& tend   = track.LocationAtPoint( (int)track.NumberTrajectoryPoints()-1 );
      auto const& sstrt  = shower_trunk.LocationAtPoint(0);
      auto const& send   = shower_trunk.LocationAtPoint(1);
      std::vector<float> tdir(3,0);
      std::vector<float> sdir(3,0);
      float tlen = 0.;
      float slen = 0.;
      for (int i=0; i<3; i++) {
        dist1 += ( tstart[i]-sstrt[i] )*( tstart[i]-sstrt[i] );
        dist2 += ( tend[i]-sstrt[i] )*( tend[i]-sstrt[i] );        
        tdir[i] = tend[i]-tstart[i];
        tlen += tdir[i]*tdir[i];
        sdir[i] = send[i]-sstrt[i];
        slen += sdir[i]*sdir[i];
      }
      if ( tlen>0 ) {
        tlen = sqrt(tlen);
        for (int i=0; i<3; i++)
          tdir[i] /= tlen;
      }
      if ( slen>0 ) {
        slen = sqrt(slen);
        for (int i=0; i<3; i++) {
          sdir[i] /= slen;
        }
      }

      float dist = (dist1<dist2) ? dist1 : dist2;
      
      float cosdir = 0.;
      for (int i=0; i<3; i++)
        cosdir += tdir[i]*sdir[i];

      LARCV_DEBUG() << "track: cosdir=" << cosdir << " dist1=" << dist1 << " dist2=" << dist2 << " dist=" << dist << std::endl;

      if ( fabs(cosdir)<0.95 || dist>10 ) {
        masker.maskTrack( track, padc_v, mask_v, tpcid, cryoid, 1.0, 2, 4, 0.1, 0.3 );        
      }
    }

    // create masked images. will add track mask to this
    std::vector<larcv::Image2D> track_removed_v;
    for ( size_t p=0; p<padc_v.size(); p++) {
      auto const& adc  = *padc_v[p];
      auto const& mask = mask_v[p];
      larcv::Image2D removed( adc );
      auto const& data_v = adc.as_vector();
      auto const& m_v    = mask.as_vector();
      auto& removed_v    = removed.as_mod_vector();
      for (size_t i=0; i<data_v.size(); i++) {
        if ( m_v[i]==0 )
          removed_v[i] = data_v[i];
        else
          removed_v[i] = 0.0;
      }
      track_removed_v.emplace_back( std::move(removed) );
    }
    
    return track_removed_v;
  }

  /**
   * @brief Run the processShower algorithms for a truth mcshower
   *
   */
  void ShowerdQdx::processMCShower( const larlite::mcshower& shower,
                                    const std::vector<larcv::Image2D>& adc_v,
                                    const larflow::reco::NuVertexCandidate& nuvtx )
  {
    
    // clear variables
    clear();

    // we make shower trunk
    ublarcvapp::mctools::TruthShowerTrunkSCE trunk_maker( _psce );

    larlite::track trunk = trunk_maker.applySCE(shower);
    larlite::larflowcluster dummy_cluster;
    larlite::pcaxis dummy_pca;

    throw std::runtime_error("FIX ME: ShowerdQdx::processMCShower");
    //processShower( dummy_cluster, trunk, dummy_pca, adc_v, nuvtx );
  }

  /**
   * @brief find the maximum overlap due to other true particles in the interaction.
   *
   */
  void ShowerdQdx::checkForOverlappingPrimary( const larlite::event_mctrack& ev_mctrack,
                                               const larlite::event_mcshower& ev_mcshower )
  {
    _true_max_primary_cos = -2;

    if ( _true_min_index < 0 ) {
      LARCV_DEBUG() << "No matched mcshower to reco shower (index=" << _true_min_index << "). return." << std::endl;
      // no match
      return;
    }
    LARCV_DEBUG() << "Matching shower index: " << _true_min_index << " (of " << ev_mcshower.size() << " mcshowers)" << std::endl;    

    auto const& shower = ev_mcshower.at( _true_min_index );

    ublarcvapp::mctools::TruthTrackSCE       track_maker( _psce );
    ublarcvapp::mctools::TruthShowerTrunkSCE trunk_maker( _psce );
    larlite::track trunk = trunk_maker.applySCE(shower);
    
    const TVector3& start = trunk.LocationAtPoint(0);
    const TVector3& shower_dir = trunk.DirectionAtPoint(0);

    int noverlapping = 0;
    int closest_track_idx = -1;
    float closest_track_cos = -2;
    
    // check track
    LARCV_DEBUG() << "search mctracks (N=" << ev_mctrack.size() << ")" << std::endl;
    for ( int itrack=0; itrack<(int)ev_mctrack.size(); itrack++ ) {

      const larlite::mctrack& track = ev_mctrack.at(itrack);
      if (track.Origin()!=1)
        continue; // consider neutrino interaction tracks only

      larlite::track track_sce = track_maker.applySCE( track );

      if ( track_sce.NumberTrajectoryPoints()==0 )
        continue;           

      LARCV_DEBUG() << "mctrack[" << itrack << "] n=" << track_sce.NumberTrajectoryPoints() << std::endl;

      const TVector3& track_cand_start = track_sce.LocationAtPoint(0);
      const TVector3& track_cand_dir   = track_sce.DirectionAtPoint(0);

      float vtx_dist = 0.;
      float cosdir = 0.;
      float showerlen = 0.;
      float tracklen = 0.;
      for (int i=0; i<3; i++) {
        vtx_dist  += ( track_cand_start[i] - start[i] )*( track_cand_start[i] - start[i] );
        cosdir    += track_cand_dir[i]*shower_dir[i];
        showerlen += shower_dir[i]*shower_dir[i];
        tracklen  += track_cand_dir[i]*track_cand_dir[i];
      }
      if (showerlen*tracklen>0 )
        cosdir /= sqrt(showerlen*tracklen);

      if ( vtx_dist<1.0 && cosdir>closest_track_cos ) {
        closest_track_cos = cosdir;
        closest_track_idx = itrack;
        noverlapping++;
      }
    }

    int closest_shower_idx = -1;
    float closest_shower_cos = -2;
    
    // check shower
    LARCV_DEBUG() << "search mcshowers" << std::endl;    
    for ( int ishower=0; ishower<(int)ev_mcshower.size(); ishower++ ) {

      if (ishower==_true_min_index)
        continue; // don't compare against self
      
      auto const& xshower = ev_mcshower.at(ishower);
      if (xshower.Origin()!=1)
        continue; // consider neutrino interaction showers only

      larlite::track xshower_sce = trunk_maker.applySCE( xshower );

      const TVector3& xshower_cand_start = xshower_sce.LocationAtPoint(0);
      const TVector3& xshower_cand_dir   = xshower_sce.DirectionAtPoint(0);

      float vtx_dist = 0.;
      float cosdir = 0.;
      float showerlen = 0.;
      float xshowerlen = 0.;
      for (int i=0; i<3; i++) {
        vtx_dist  += ( xshower_cand_start[i] - start[i] )*( xshower_cand_start[i] - start[i] );
        cosdir    += xshower_cand_dir[i]*shower_dir[i];
        showerlen += shower_dir[i]*shower_dir[i];
        xshowerlen  += xshower_cand_dir[i]*xshower_cand_dir[i];
      }
      if (xshowerlen*showerlen>0 )
        cosdir /= sqrt(xshowerlen*showerlen);

      if ( vtx_dist<9.0 && cosdir>closest_shower_cos ) {
        closest_shower_cos = cosdir;
        closest_shower_idx = ishower;
        noverlapping++;
      }
    }

    _true_max_primary_cos = ( closest_shower_cos>closest_track_cos ) ? closest_shower_cos: closest_track_cos;
    
    LARCV_INFO() << "Num overlapping: " << noverlapping << std::endl;
    LARCV_INFO() << "closest_track_idx="  << closest_track_idx  << " closest_track_cos=" << closest_track_cos << std::endl;
    LARCV_INFO() << "closest_shower_idx=" << closest_shower_idx << " closest_shower_idx=" << closest_shower_idx << std::endl;
      
  }

  /**
   * @brief find a matching true shower to given reco shower, and then if good match, calc dq/dx for the true shower
   *
   */
  void ShowerdQdx::matchMCShowerAndProcess( const larlite::larflowcluster& reco_shower,
                                            const larlite::track& reco_shower_trunk,
                                            const larlite::pcaxis& reco_pca,
                                            const std::vector<larcv::Image2D>& adc_v,
                                            const larflow::reco::NuVertexCandidate& nuvtx,                                                    
                                            const std::vector<larlite::mcshower>& mcshower_v )
  {

    clear();

    const int tpcid  = nuvtx.tpcid;
    const int cryoid = nuvtx.cryoid;

    std::vector< const larcv::Image2D* > padc_v
      = ublarcvapp::recotools::DetUtils::getTPCImages( adc_v, tpcid, cryoid );        
    
    calcGoodShowerTaggingVariables( reco_shower, reco_shower_trunk, reco_pca,
                                    padc_v, mcshower_v, tpcid, cryoid );

    if ( _true_dir_cos>0.9 && fabs(_true_vertex_err_dist)<3.0 && _true_min_index>=0 ) {
      
      auto const& match_shower = mcshower_v[_true_min_index];
      processMCShower( match_shower, adc_v, nuvtx );
    }
  }
  
  larlite::track ShowerdQdx::makeLarliteTrackdqdx(int plane)
  {
    auto const& seg_v  = _plane_seg_dedx_v.at(plane);
    TVector3 vdir;
    for (int i=0; i<3; i++)
      vdir[i] = _shower_dir[i];
    
    larlite::track ll;
    ll.reserve(seg_v.size());
    for (int ipt=0; ipt<(int)seg_v.size(); ipt++) {
      auto const& seg = seg_v[ipt];
      TVector3 pos3d;
      for (int i=0; i<3; i++)
        pos3d[i] = 0.5*( seg.endpt[0][i]+seg.endpt[1][i] );
      std::vector<double> dqdx_v(4, seg.dqdx );
      ll.add_vertex( pos3d );
      ll.add_direction(vdir);
      ll.add_dqdx( dqdx_v );
    }
    return ll;
  }
  
}
}
