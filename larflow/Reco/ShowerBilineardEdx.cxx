#include "ShowerBilineardEdx.h"

#include <stdio.h>
#include <string>
#include <sstream>
#include <omp.h>
#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"
#include "geofuncs.h"
#include "ClusterImageMask.h"

namespace larflow {
namespace reco {

  int ShowerBilineardEdx::ndebugcount = 0;
  larutil::SpaceChargeMicroBooNE* ShowerBilineardEdx::_psce  = nullptr;
  
  ShowerBilineardEdx::ShowerBilineardEdx()
    : larcv::larcv_base("ShowerBilineardEdx")
  {
  }

  ShowerBilineardEdx::~ShowerBilineardEdx()
  {
  }

  void ShowerBilineardEdx::clear()
  {

    bilinear_path_vv.clear();    

    _pixsum_dqdx_v.clear();
    _best_pixsum_dqdx = 0;
    _best_pixsum_plane = -1;
    _best_pixsum_ngood = 0;
    _best_pixsum_ortho = 0;
    
    _bilin_dqdx_v.clear();    
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
    
  }
  
  void ShowerBilineardEdx::processShower( larlite::larflowcluster& shower,
                                          larlite::track& trunk,
                                          larlite::pcaxis& pca,
                                          const std::vector<larcv::Image2D>& adc_v,
                                          const larflow::reco::NuVertexCandidate& nuvtx )
  {
    
    // clear variables
    clear();
    bilinear_path_vv.clear();
    bilinear_path_vv.resize( adc_v.size() );
    
    auto const& meta = adc_v.front().meta();
    
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
    bool validtrunk = checkShowerTrunk( start_pos, end_pos, fstart, fend, _shower_dir, dist, adc_v );

    // collect pixels that the trunk passes through in each wire plane image
    _createDistLabels( fstart, fend, adc_v, 10.0 );

    // define line segment regions to measure dq/dx over the shower trunk
    _makeSegments( -3.0 );

    // use track pixels in candidate neutrino vertex to mask pixels to zero
    std::vector<larcv::Image2D> track_masked_v = maskTrackPixels( adc_v, trunk, nuvtx );

    // calculate dq/dx for each segment along dq/dx, using masked image
    _sumChargeAlongSegments( fstart, fend, track_masked_v, 10.0, 1, 3 );

    // find range along trunk that contain points with expected dq/dx for electron and photon
    // electron regions
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

    // simple dq/dx measure using first 3 cm of trunk. use masked image.    
    std::vector<float> end3cm(3,0);
    for (int i=0; i<3; i++) {
      end3cm[i] = fstart[i] + 3.0*_shower_dir[i];
    }
    _pixsum_dqdx_v = sumChargeAlongTrunk( fstart, end3cm, track_masked_v, 10.0, 1, 3 );
    for (size_t p=0; p<adc_v.size(); p++) {
      LARCV_DEBUG() << "//////////////// PLANE " << p << " ///////////////////" << std::endl;      
      LARCV_DEBUG() << "3 cm trunk dq/dx: " << _pixsum_dqdx_v[p] << " pixsum/cm" << std::endl;
    }

    // heuristic to choosing best dq/dx
    // most good points?
    // start of electron shower?
    std::vector<int> ngood_v( adc_v.size(), 0);
    _best_pixsum_ngood = 0;
    _best_pixsum_plane = -1;
    _best_pixsum_ortho = 0.;    
    _pixsum_dqdx_v.resize(adc_v.size(),0);
    for (size_t p=0; p<adc_v.size(); p++) {
      float ngood_max = 0;

      std::vector<double> orthovect= { 0.0,
                                       larutil::Geometry::GetME()->GetOrthVectorsY().at(p),
                                       larutil::Geometry::GetME()->GetOrthVectorsZ().at(p) };

      float plane_ortho_cos = 0.;
      float ortholen = 0.;
      for (int i=0; i<3; i++) {
        plane_ortho_cos += orthovect[i]*_shower_dir[i];
        ortholen += orthovect[i]*orthovect[i];
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
    
    // make graphs for debug plotting
    std::vector<float> cgrad;
    std::vector<float> rgrad;
    for ( int p=0; p<(int)adc_v.size(); p++ ) {
      TGraph gpath(2);      
      float col = colcoordinate_and_grad( fstart, p, meta, cgrad );
      float row = rowcoordinate_and_grad( fstart, meta, rgrad );
      float ptx = col*meta.pixel_width()+meta.min_x();
      float pty = row*meta.pixel_height()+meta.min_y();
      gpath.SetPoint(0, ptx, pty );

      col = colcoordinate_and_grad( fend, p, meta, cgrad );
      row = rowcoordinate_and_grad( fend, meta, rgrad );
      ptx = col*meta.pixel_width()+meta.min_x();
      pty = row*meta.pixel_height()+meta.min_y();
      gpath.SetPoint(1, ptx, pty );
      bilinear_path_vv[p].emplace_back( std::move(gpath) );
    }

    
    // use use the trunk to define a start and end point
    // bilinear calculations not used right now    
    // _bilin_dqdx_v.resize(3,0);
    
    // for (size_t p=0; p<adc_v.size(); p++) {
    //   _pixsum_dqdx_v[p] = pixsum_v[p]/dist;
      
    //   // projection
    //   //LARCV_DEBUG() << "//////////////// PLANE " << p << " ///////////////////" << std::endl;
    //   auto const& img = adc_v.at(p);
    //   std::vector<float> grad(6,0);
    //   //LARCV_DEBUG() << "Get Bilinear Charge w/ grad, plane " << p << std::endl;
    //   float avedQdx = aveBilinearCharge_with_grad( img, fstart, fend, 20, 75.0, grad );
    //   _bilin_dqdx_v[p] = avedQdx;
      
    //   //LARCV_DEBUG() << "ave dQ/dx: "  << avedQdx << std::endl;
    //   LARCV_DEBUG() << "pixel sum: " << pixsum_v[p] << " dpixsum/dist=" << pixsum_v[p]/dist << std::endl;
    // }

    
  }

  /** 
   * @brief make sure trunk end points are inside the image
   *
   */
  bool ShowerBilineardEdx::checkShowerTrunk( const std::vector<float>& start_pos,
                                             const std::vector<float>& end_pos,
                                             std::vector<float>& modstart3d,
                                             std::vector<float>& modend3d,
                                             std::vector<float>& shower_dir,
                                             float& dist,
                                             const std::vector<larcv::Image2D>& adc_v )
  {

    // set output vector size
    modstart3d.resize(3,0);
    modend3d.resize(3,0);
   
    auto const& meta = adc_v.front().meta();

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
    float start_tick = start_pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
    if ( start_tick<=meta.min_y() || start_tick>=meta.max_y() ) {

      // try to shorten trunk to stay in image
      bool fix = false;
      if ( start_tick<=meta.min_y() && shower_dir[0]!=0.0) {
	float mintick = (meta.pos_y( 1 )-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
	float s = (mintick-start_pos[0])/shower_dir[0];
	fix = true;
	for (int i=0; i<3; i++)
	  modstart3d[i] = start_pos[i] + s*shower_dir[i];
	start_tick = meta.pos_y(1);        
      }
      else if ( start_tick>=meta.max_y() && shower_dir[0]!=0.0) {
	float maxtick = (meta.pos_y( (int)meta.rows()-1 )-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
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
        
    float end_tick = end_pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
    if ( end_tick<=meta.min_y() || end_tick>=meta.max_y() ) {

      // try to shorten trunk to stay in image
      bool fix = false;
      float s = 0.0;
      if ( end_tick<=meta.min_y() && shower_dir[0]!=0.0) {
	float mintick = (meta.pos_y( 1 )-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
	s = (mintick-start_pos[0])/shower_dir[0];
	fix = true;
	for (int i=0; i<3; i++)
	  modend3d[i] = start_pos[i] + s*shower_dir[i];
	end_tick = meta.pos_y(1);

      }
      else if ( end_tick>=meta.max_y() && shower_dir[0]!=0.0) {
	float maxtick = (meta.pos_y( (int)meta.rows()-1 )-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
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

  float ShowerBilineardEdx::aveBilinearCharge_with_grad( const larcv::Image2D& img,
                                                         std::vector<float>& start3d,
                                                         std::vector<float>& end3d,
                                                         int npts,
                                                         float avedQdx,
                                                         std::vector<float>& grad ) {

    const int plane = (int)img.meta().plane();
    auto const& meta = img.meta();
    
    grad.resize(6,0);
    for (int i=0; i<6; i++)
      grad[i] = 0.;
    
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

    // calculate the dx for each point (we assume trunk going in same direction along trunk)
    float len_cell_x_cm = img.meta().pixel_height()*0.5*larutil::LArProperties::GetME()->DriftVelocity();
    float len_cell_pitch = 0.3; // cm
    std::vector<double> orthovect= { 0.0,
                                     larutil::Geometry::GetME()->GetOrthVectorsY().at(plane),
                                     larutil::Geometry::GetME()->GetOrthVectorsZ().at(plane) };
    float orthlen = 0.;    
    for (int i=0; i<3; i++) {
      orthlen += orthovect[i]*orthovect[i];
    }
    orthlen = sqrt(orthlen);
    
    float orthproj = 0.0;
    for (int i=0; i<3; i++) {
      orthproj += orthovect[i]*dir[i]/orthlen;
    }
    orthproj = fabs(orthproj);

    
    float wiredir_len = dist; // distance assuming ray cross wire plane cell walls
    if ( orthproj>0 ) {
      if ( len_cell_pitch/orthproj<dist )
        wiredir_len = len_cell_pitch/orthproj;
    }
    float tickdir_len = dist;
    if ( fabs(dir[0])>0 ) {
      float l = len_cell_x_cm/fabs(dir[0]);
      if ( l<tickdir_len )
        tickdir_len = l;
    }
    float dx = (tickdir_len<wiredir_len) ? tickdir_len : wiredir_len;

    float stepsize = dist/float(npts);

    float aveQ = 0.;
    std::vector<float> aveQ_grad(6,0);
    TGraph gpath(npts+1);
    
    for (int istep=0; istep<=npts; istep++) {
      std::vector<float> pt(3,0);
      float f = istep*stepsize/dist;

      for (int i=0; i<3; i++)
        pt[i] = start3d[i]*(1-f) + f*end3d[i];


      std::vector<float> cgrad;
      std::vector<float> rgrad;    
      float col = colcoordinate_and_grad( pt, plane, meta, cgrad );
      float row = rowcoordinate_and_grad( pt, meta, rgrad );
      float ptx = col*meta.pixel_width()+meta.min_x();
      float pty = row*meta.pixel_height()+meta.min_y();
      gpath.SetPoint(istep, ptx, pty );
      
      std::vector<float> pix_grad(3,0);
      float pix = bilinearPixelValue_and_grad( pt, plane, img, pix_grad );
      //LARCV_DEBUG() << "  bilinear-pixval[" << istep << "] " << pix << " (col,tick)=(" << ptx << "," << pty << ")" << std::endl;

      // dpix/dx_s = dpix/dx*dx/dx_s
      std::vector<float> pt_grad(6,0);
      for (int i=0; i<3; i++) {
        pt_grad[i]   = pix_grad[i]*(1-f);
        pt_grad[3+i] = pix_grad[i]*f;
      }
      for (int i=0; i<6; i++) {
        aveQ_grad[i] += pt_grad[i];
      }
      
      aveQ += pix;      
    }
    bilinear_path_vv[plane].emplace_back( std::move(gpath) );

    aveQ /= float(npts);
    for (int i=0; i<6; i++)
      aveQ_grad[i] /= float(npts);
    
    // aveQ/dist
    float dqdx = aveQ/dx;

    LARCV_DEBUG() << "------------------------------------" << std::endl;
    LARCV_DEBUG() << "start: (" << start3d[0] << "," << start3d[1] << "," << start3d[2] << ")" << std::endl;
    LARCV_DEBUG() << "end: (" << end3d[0] << "," << end3d[1] << "," << end3d[2] << ")" << std::endl;
    LARCV_DEBUG() << "dist: " << dist << " cm" << std::endl;
    LARCV_DEBUG() << "dir: (" << dir[0] << "," << dir[1] << "," << dir[2] << ")" << std::endl;
    LARCV_DEBUG() << "cell dims: wire=" << len_cell_pitch << " cm  tick=" << len_cell_x_cm << " cm" << std::endl;
    LARCV_DEBUG() << "plane-orthovect: (" << orthovect[0] << "," << orthovect[1] << "," << orthovect[2] << ")" << std::endl;
    LARCV_DEBUG() << "projection on orthovect: " << orthproj << std::endl;
    LARCV_DEBUG() << "aveQ: " << aveQ << std::endl;
    LARCV_DEBUG() << "dx: " << dx << std::endl;
    LARCV_DEBUG() << "wire-pitch distance: " << wiredir_len << " cm" << std::endl;
    LARCV_DEBUG() << "tick-pitch distance: " << tickdir_len << " cm" << std::endl;
    LARCV_DEBUG() << "grad: (" << aveQ_grad[0] << "," << aveQ_grad[1] << "," << aveQ_grad[2] << ") "
                  << "(" << aveQ_grad[3] << "," << aveQ_grad[4] << "," << aveQ_grad[5] << ") "
                  << std::endl;
    
    return dqdx;
  }
                                           

  float ShowerBilineardEdx::colcoordinate_and_grad( const std::vector<float>& pos,
                                                    const int plane,
                                                    const larcv::ImageMeta& meta,
                                                    std::vector<float>& grad )
  {
    
    if ( plane<0 || plane>=(int)larutil::Geometry::GetME()->Nplanes() )
      throw std::runtime_error("ProjectionTools::wirecoordinate_and_grad invalid plane number");
    
    // microboone only
    const std::vector<Double_t>& firstwireproj = larutil::Geometry::GetME()->GetFirstWireProj(); 
    std::vector<double> orthovect = { 0,
                                      larutil::Geometry::GetME()->GetOrthVectorsY().at(plane),
                                      larutil::Geometry::GetME()->GetOrthVectorsZ().at(plane) };

    // from larlite Geometry::WireCoordinate(...)
    float wirecoord = pos[1]*orthovect[1] + pos[2]*orthovect[2] - firstwireproj.at(plane);

    float colcoord = (wirecoord-meta.min_x())/meta.pixel_width();
    
    grad.resize(3,0);
    grad[0] = 0.0;
    grad[1] = orthovect[1]/meta.pixel_width();
    grad[2] = orthovect[2]/meta.pixel_width();
    return colcoord;
  }
  
  float ShowerBilineardEdx::rowcoordinate_and_grad( const std::vector<float>& pos,
                                                    const larcv::ImageMeta& meta,
                                                    std::vector<float>& grad )
  {
    float tick = pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
    float row = (tick-meta.min_y())/meta.pixel_height();
    grad.resize(3,0);
    grad[0] = 1.0/larutil::LArProperties::GetME()->DriftVelocity()/0.5/meta.pixel_height();
    grad[1] = 0.0;
    grad[2]=  0.0;
    return row;
  }

  float ShowerBilineardEdx::bilinearPixelValue_and_grad( std::vector<float>& pos3d,
                                                         const int plane,
                                                         const larcv::Image2D& img,
                                                         std::vector<float>& grad )
  {

    auto const& meta = img.meta();
    std::vector<float> col_grad;
    std::vector<float> row_grad;    
    float col = colcoordinate_and_grad( pos3d, plane, meta, col_grad );
    float row = rowcoordinate_and_grad( pos3d, meta, row_grad );
    float dc = col - std::floor(col);
    float dr = row - std::floor(row);
    int pixco[2][2][2] = { 0 };

    int ll_col = (dc>=0.5) ? (int)col : (int)col-1;
    int ll_row = (dr>=0.5) ? (int)row : (int)row-1;

    float pixvals[2][2] = { 0 };
    for (int ic=0; ic<2; ic++) {
      int c = ll_col+ic;
      if ( c<0 || c>=(int)meta.cols() )
        continue;
      for (int ir=0; ir<2; ir++) {
        int r = ll_row+ir;
        if ( r<0 || r>=(int)meta.rows() )
          continue;
        pixvals[ic][ir] = img.pixel(r,c);
      }
    }

    float x1 = ((float)ll_col)+0.5;    
    float x2 = ((float)ll_col)+1.5;
    float y1 = ((float)ll_row)+0.5;
    float y2 = ((float)ll_row)+1.5;

    // bilinear interpolation of pixel value
    float Pix = pixvals[0][0]*(x2-col)*(y2-row)
      + pixvals[1][0]*(col-x1)*(y2-row)
      + pixvals[0][1]*(x2-col)*(row-y1)
      + pixvals[1][1]*(col-x1)*(row-y1);

    // gradient w.r.t the 3D position
    // dPix/dy = dPix/dcol*dcol/dy
    // dPix/dz = dPix/dcol*dcol/dz
    // dPix/dx = dPix/drow*drow/dx

    grad.resize(3,0);
    grad[0] = (-pixvals[0][0]*(x2-col) - pixvals[1][0]*(col-x1) + pixvals[0][1]*(x2-col) + pixvals[1][1]*(col-x1))*row_grad[0];
    grad[1] = (-pixvals[0][0]*(y2-row) + pixvals[1][0]*(y2-row) - pixvals[0][1]*(row-y1) + pixvals[1][1]*(row-y1))*row_grad[1];
    grad[2] = (-pixvals[0][0]*(y2-row) + pixvals[1][0]*(y2-row) - pixvals[0][1]*(row-y1) + pixvals[1][1]*(row-y1))*row_grad[2];
    
    
    return Pix;
  }

  std::vector<float> ShowerBilineardEdx::sumChargeAlongTrunk( const std::vector<float>& start3d,
                                                              const std::vector<float>& end3d,
                                                              const std::vector<larcv::Image2D>& img_v,
                                                              const float threshold,
                                                              const int dcol, const int drow )
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
      _createDistLabels( start3d, end3d, img_v, threshold );
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
        throw std::runtime_error("Bad cropped window size");
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
          if ( tp.smin>0 || tp.smax>0 ) {          
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
      if ( ds>0 && ds<10.0 )
        planesum_v[p] /= ds;
      
    }//end of plane loop
    
    return planesum_v;
  }

  void ShowerBilineardEdx::_createDistLabels( const std::vector<float>& start3d,
                                              const std::vector<float>& end3d,
                                              const std::vector<larcv::Image2D>& img_v,
                                              const float threshold )
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
      
      auto const& img = img_v[p];
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
        float col = colcoordinate_and_grad( pos, p, img.meta(), col_grad );
        float row = rowcoordinate_and_grad( pos, img.meta(), row_grad );
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

  void ShowerBilineardEdx::_makeSegments( float starting_s )
  {

    const float seg_size = 0.5;
    
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

  void ShowerBilineardEdx::_sumChargeAlongSegments( const std::vector<float>& start3d,
                                                    const std::vector<float>& end3d,
                                                    const std::vector<larcv::Image2D>& img_v,
                                                    const float threshold,
                                                    const int dcol, const int drow )
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
        throw std::runtime_error("Bad cropped window size");
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
          //<< " npix=" << npix << " nskip=" << nskip << " nvisited=" << nvisited
                      << " dx=" << ds
                      << " dqdx=" << seg.dqdx << std::endl;
        
        iseg++;
      }//end of seg loop
    }//end of plane loop

  }
  
  float ShowerBilineardEdx::_sumChargeAlongOneSegment( ShowerBilineardEdx::Seg_t& seg,
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
      throw std::runtime_error("Bad cropped window size");
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
  
  
  void ShowerBilineardEdx::bindVariablesToTree( TTree* outtree )
  {
    outtree->Branch( "pixsum_dqdx_v", &_pixsum_dqdx_v );
    outtree->Branch( "best_pixsum_dqdx", &_best_pixsum_dqdx );
    outtree->Branch( "best_pixsum_plane", &_best_pixsum_plane );
    outtree->Branch( "best_pixsum_ngood", &_best_pixsum_ngood );
    outtree->Branch( "best_pixsum_ortho", &_best_pixsum_ortho );
    outtree->Branch( "bilin_dqdx_v",  &_bilin_dqdx_v );
    outtree->Branch( "shower_dir", &_shower_dir );
    outtree->Branch( "plane_dqdx_seg_vv", &_plane_dqdx_seg_v );
    outtree->Branch( "plane_s_seg_vv",    &_plane_s_seg_v );

    outtree->Branch( "plane_electron_srange_vv", &_plane_electron_srange_v );
    outtree->Branch( "plane_electron_dqdx_v", &_plane_electron_dqdx_v );
    outtree->Branch( "plane_electron_dx_v", &_plane_electron_dx_v );
    outtree->Branch( "plane_electron_mean_v", &_plane_electron_mean_v );
    outtree->Branch( "plane_electron_rms_v", &_plane_electron_rms_v );
    outtree->Branch( "plane_electron_ngood_v", &_plane_electron_ngood_v );
    outtree->Branch( "plane_electron_best", &_plane_electron_best );
    outtree->Branch( "plane_electron_best_mean", &_plane_electron_best_mean );
    outtree->Branch( "plane_electron_best_rms", &_plane_electron_best_rms );
    outtree->Branch( "plane_electron_best_ngood", &_plane_electron_best_ngood );
    outtree->Branch( "plane_electron_best_start", &_plane_electron_best_start );        

    outtree->Branch( "plane_gamma_srange_vv", &_plane_gamma_srange_v );
    outtree->Branch( "plane_gamma_dqdx_v", &_plane_gamma_dqdx_v );
    outtree->Branch( "plane_gamma_dx_v", &_plane_gamma_dx_v );
    outtree->Branch( "plane_gamma_mean_v", &_plane_gamma_mean_v );
    outtree->Branch( "plane_gamma_rms_v", &_plane_gamma_rms_v );
    outtree->Branch( "plane_gamma_ngood_v", &_plane_gamma_ngood_v );        
    outtree->Branch( "plane_gamma_best", &_plane_gamma_best );
    outtree->Branch( "plane_gamma_best_mean", &_plane_gamma_best_mean );
    outtree->Branch( "plane_gamma_best_rms", &_plane_gamma_best_rms );
    outtree->Branch( "plane_gamma_best_ngood", &_plane_gamma_best_ngood );
    outtree->Branch( "plane_gamma_best_start", &_plane_gamma_best_start );        

    outtree->Branch( "true_match_pdg", &_true_match_pdg );
    outtree->Branch( "true_min_feat_dist", &_true_min_feat_dist );
    outtree->Branch( "true_vertex_err_dist", &_true_vertex_err_dist );
    outtree->Branch( "true_dir_cos", &_true_dir_cos );
  }

  void ShowerBilineardEdx::maskPixels( int plane, TH2D* hist )
  {
    // mostly for debugging
    auto const& tplist = _plane_trunkpix_v[plane];
    for (int itp=0; itp<(int)tplist.size(); itp++ ) {
      auto const& tp  = tplist[itp];
      hist->SetBinContent( tp.col+1, tp.row+1, tp.smin );
    }
  }

  TGraph ShowerBilineardEdx::makeSegdQdxGraphs(int plane)
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

  void ShowerBilineardEdx::_findRangedQdx( const std::vector<float>& start3d,
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
      
      seg.dqdx = _sumChargeAlongOneSegment( seg, p, adc_v, 10.0, 1, 3 );
      LARCV_DEBUG() << "Plane[" << p << "] range dqdx s=(" << s_min << "," << s_max << ") dqdx=" << seg.dqdx << std::endl;

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

  void ShowerBilineardEdx::calcGoodShowerTaggingVariables( const larlite::larflowcluster& shower,
                                                           const larlite::track& trunk,
                                                           const larlite::pcaxis& pca,
                                                           const std::vector<larcv::Image2D>& adc_v,
                                                           const std::vector<larlite::mcshower>& mcshower_v )
  {

    if ( ShowerBilineardEdx::_psce==nullptr ) {
      ShowerBilineardEdx::_psce = new larutil::SpaceChargeMicroBooNE;
    }
    
    auto const& meta = adc_v.front().meta();


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
    
    float start_tick = start_pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
    if ( start_tick<meta.min_y() || start_tick>meta.max_y() ) {

      // try to shorten trunk to stay in image
      bool fix = false;
      if ( start_tick<=meta.min_y() && dir[0]!=0.0) {
	float mintick = (meta.pos_y( 1 )-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
	float s = (mintick-start_pos[0])/dir[0];
	
	fix = true;
	for (int i=0; i<3; i++)
	  start_pos[i] = start_pos[0] + s*dir[i];
	start_tick = meta.pos_y(1);
	
      }
      else if ( start_tick>=meta.max_y() && dir[0]!=0.0) {
	float maxtick = (meta.pos_y( (int)meta.rows()-1 )-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
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
        
    float end_tick = end_pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
    if ( end_tick<=meta.min_y() || end_tick>=meta.max_y() ) {
      
      // try to shorten trunk to stay in image
      bool fix = false;
      float s = 0.;
      if ( end_tick<=meta.min_y() && dir[0]!=0.0) {
	float mintick = (meta.pos_y( 1 )-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
	s = (mintick-start_pos[0])/dir[0];

	fix = true;
	for (int i=0; i<3; i++)
	  end_pos[i] = start_pos[0] + s*dir[i];
	end_tick = meta.pos_y(1);

      }
      else if ( end_tick>=meta.max_y() && dir[0]!=0.0) {
	float maxtick = (meta.pos_y( (int)meta.rows()-1 )-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
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

    float min_feat_dist = 1e9;
    float vertex_err_dist = 0;
    int   match_pdg = 0;
    float dir_cos = 0.;

    for ( auto const& mcshower : mcshower_v ) {

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
      }
    }
    
    // store in member variables
    _true_min_feat_dist   = min_feat_dist;
    _true_vertex_err_dist = vertex_err_dist;
    _true_dir_cos = dir_cos;
    _true_match_pdg = match_pdg;

    LARCV_DEBUG() << "Best true shower match: " << std::endl;
    LARCV_DEBUG() << " - feat_dist=" << _true_min_feat_dist << std::endl;
    LARCV_DEBUG() << " - vertex_dist="<< _true_vertex_err_dist << std::endl;
    LARCV_DEBUG() << " - true-dir-cos=" << _true_dir_cos << std::endl;
    LARCV_DEBUG() << " - match PDG code=" << match_pdg << std::endl;
    
  }

  std::vector<larcv::Image2D>
  ShowerBilineardEdx::maskTrackPixels( const std::vector<larcv::Image2D>& adc_v,
                                       const larlite::track& shower_trunk,
                                       const larflow::reco::NuVertexCandidate& nuvtx )
  {

    ClusterImageMask masker; // class contains functions to mask pixels along track object

    // create blank images. will add track mask to this
    std::vector<larcv::Image2D> mask_v;
    for ( auto const& adc : adc_v ) {
      larcv::Image2D mask( adc.meta() );
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
        masker.maskTrack( track, adc_v, mask_v, 1.0, 2, 4, 0.1, 0.3 );        
      }
    }

    // create masked images. will add track mask to this
    std::vector<larcv::Image2D> track_removed_v;
    for ( size_t p=0; p<adc_v.size(); p++) {
      auto const& adc  = adc_v[p];
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
  
}
}
