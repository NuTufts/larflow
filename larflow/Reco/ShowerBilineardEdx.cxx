#include "ShowerBilineardEdx.h"

#include <stdio.h>
#include <string>
#include <omp.h>
#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"
#include "geofuncs.h"

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
    _shower_dir.clear();
    _pixsum_dqdx_v.clear();
    _bilin_dqdx_v.clear();
    _plane_dqdx_seg_v.clear();
    _plane_s_seg_v.clear();    
  }
  
  void ShowerBilineardEdx::processShower( larlite::larflowcluster& shower,
                                          larlite::track& trunk,
                                          larlite::pcaxis& pca,
                                          const std::vector<larcv::Image2D>& adc_v )
  {
    
    // clear variables
    bilinear_path_vv.clear();
    bilinear_path_vv.resize( adc_v.size() );
    _pixsum_dqdx_v.clear();
    _bilin_dqdx_v.clear();
    _shower_dir.clear();
    _plane_dqdx_seg_v.clear();
    
    auto const& meta = adc_v.front().meta();
    
    std::vector<double> start_pos = { trunk.LocationAtPoint(0)[0],
                                      trunk.LocationAtPoint(0)[1],
                                      trunk.LocationAtPoint(0)[2] };
    float start_tick = start_pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
    if ( start_tick<meta.min_y() || start_tick>meta.max_y() ) {
      throw std::runtime_error("out of bounds shower trunk starting point");
    }    
    float start_row = (float)meta.row(start_tick);
    
    std::vector<double> end_pos = { trunk.LocationAtPoint(1)[0],
                                    trunk.LocationAtPoint(1)[1],
                                    trunk.LocationAtPoint(1)[2] };
    float end_tick = start_pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
    if ( end_tick<=meta.min_y() || end_tick>=meta.max_y() ) {
      throw std::runtime_error("out of bounds shower trunk starting point");
    }
    float end_row = (float)meta.row(end_tick);

    std::vector<float> fstart = { (float)start_pos[0], (float)start_pos[1], (float)start_pos[2] };
    std::vector<float> fend   = { (float)end_pos[0],   (float)end_pos[1],   (float)end_pos[2] };          
    _createDistLabels( fstart, fend, adc_v, 10.0 );
    _makeSegments( -3.0 );
    _sumChargeAlongSegments( fstart, fend, adc_v, 10.0, 1, 3 );
    
    std::vector<float> pixsum_v = sumChargeAlongTrunk( fstart, fend, adc_v, 10.0, 1, 3 );    

    float dist = 0.;
    _shower_dir.resize(3,0);
    for (int i=0; i<3; i++) {
      _shower_dir[i] = (fend[i]-fstart[i]);
      dist += (fend[i]-fstart[i])*(fend[i]-fstart[i]);
    }
    dist = sqrt(dist);
    if ( dist>0 ) {
      for (int i=0; i<3; i++)
        _shower_dir[i] /= dist;
    }
    
    // use use the trunk to define a start and end point
    _pixsum_dqdx_v.resize(3,0);
    _bilin_dqdx_v.resize(3,0);
    
    for (size_t p=0; p<adc_v.size(); p++) {
      _pixsum_dqdx_v[p] = pixsum_v[p]/dist;
      
      // projection
      //LARCV_DEBUG() << "//////////////// PLANE " << p << " ///////////////////" << std::endl;
      auto const& img = adc_v.at(p);
      std::vector<float> grad(6,0);
      //LARCV_DEBUG() << "Get Bilinear Charge w/ grad, plane " << p << std::endl;
      float avedQdx = aveBilinearCharge_with_grad( img, fstart, fend, 20, 75.0, grad );
      _bilin_dqdx_v[p] = avedQdx;
      
      //LARCV_DEBUG() << "ave dQ/dx: "  << avedQdx << std::endl;
      LARCV_DEBUG() << "pixel sum: " << pixsum_v[p] << " dpixsum/dist=" << pixsum_v[p]/dist << std::endl;
    }
    _findRangedQdx( fstart, fend, adc_v, 350.0, 150.0 );

    
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
      if ( true ) {
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
        float s2 = s1 + 2.0;
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
    outtree->Branch( "bilin_dqdx_v",  &_bilin_dqdx_v );
    outtree->Branch( "shower_dir", &_shower_dir );
    outtree->Branch( "plane_dqdx_seg_vv", &_plane_dqdx_seg_v );
    outtree->Branch( "plane_s_seg_vv",    &_plane_s_seg_v );
    outtree->Branch( "plane_electron_srange_vv", &_plane_electron_srange_v );
    outtree->Branch( "plane_electron_dqdx_v", &_plane_electron_dqdx_v );
    outtree->Branch( "plane_electron_dx_v", &_plane_electron_dx_v );
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
                                           const float dqdx_threshold )
  {
    _plane_electron_srange_v.clear();
    _plane_electron_dqdx_v.clear();
    _plane_electron_dx_v.clear();
    
    _plane_gamma_srange_v.clear();

    const int nplanes = adc_v.size();
    _plane_electron_srange_v.resize( nplanes );
    _plane_electron_dqdx_v.resize( nplanes );
    _plane_electron_dx_v.resize( nplanes );    
    
    _plane_gamma_srange_v.resize( nplanes );

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
      auto const& seg_v = _plane_seg_dedx_v[p];
      int iseg_start = 0;
      int iseg_end = (int)seg_v.size()-1;
      // find first value below 
      for (int iseg=0; iseg<(int)seg_v.size(); iseg++) {
        auto const& seg = seg_v[iseg];
        if ( seg.dqdx>dqdx_threshold && seg.dqdx<dqdx_max ) {
          iseg_start = iseg;
          break;
        }
      }

      for (int iseg=(int)seg_v.size()-1; iseg>=0; iseg-- ) {
        auto const& seg = seg_v[iseg];
        if ( seg.dqdx>dqdx_threshold && seg.dqdx<dqdx_max ) {
          iseg_end = iseg;
          break;
        }
      }

      float s_min = seg_v[iseg_start].smin;
      float s_max = seg_v[iseg_end].smax;

      std::vector<float> srange = { s_min, s_max };

      _plane_electron_srange_v[p] = srange;

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
      seg.itp2 = seg_v[iseg_end].itp1;
      seg.plane = p;
      for  (int i=0; i<3; i++) {
        seg.endpt[0][i] = s3d[i]; 
        seg.endpt[1][i] = e3d[i];
      }
      seg.pixsum = 0;
      seg.dqdx = 0;
      seg.ds = s_max-s_min;           
      seg.dqdx = _sumChargeAlongOneSegment( seg, p, adc_v, 10.0, 1, 3 );
      LARCV_DEBUG() << "Plane[" << p << "] range dqdx s=(" << s_min << "," << s_max << ") dqdx=" << seg.dqdx << std::endl;

      _plane_electron_dqdx_v[p] = seg.dqdx;
      _plane_electron_dx_v[p] = seg.ds;
      
    }//end of plane loop
    
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

    TVector3 vstart = trunk.LocationAtPoint(0);
    std::vector<float> fstart = { (float)trunk.LocationAtPoint(0)[0],
                                  (float)trunk.LocationAtPoint(0)[1],
                                  (float)trunk.LocationAtPoint(0)[2] };
    float start_tick = fstart[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
    if ( start_tick<meta.min_y() || start_tick>meta.max_y() ) {
      throw std::runtime_error("out of bounds shower trunk starting point");
    }    
    float start_row = (float)meta.row(start_tick);

    TVector3 vend = trunk.LocationAtPoint(1);
    std::vector<float> fend = { (float)trunk.LocationAtPoint(1)[0],
                                (float)trunk.LocationAtPoint(1)[1],
                                (float)trunk.LocationAtPoint(1)[2] };
    float end_tick = fend[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
    if ( end_tick<=meta.min_y() || end_tick>=meta.max_y() ) {
      throw std::runtime_error("out of bounds shower trunk starting point");
    }
    float end_row = (float)meta.row(end_tick);

    float dist = 0.;
    std::vector<float> dir(3,0);
    for (int i=0; i<3; i++) {
      dir[i] = fend[i]-fstart[i];
      dist += dir[i]*dir[i];
    }
    dist = sqrt(dist);
    for (int i=0; i<3; i++)
      dir[i] /= dist;

    float min_feat_dist = 1e9;
    float vertex_err_dist = 0;
    float dir_cos = 0.;

    for ( auto const& mcshower : mcshower_v ) {

      if ( mcshower.Origin()!=1 )
        continue; // not neutrino origin

      TVector3 shower_dir = mcshower.Start().Momentum().Vect();
      float pmom = shower_dir.Mag();
      TVector3 mcstart = mcshower.Start().Position().Vect();

      std::vector<float> mcdir(3,0);
      std::vector<float> fmcstart(3,0);
      std::vector<float> fmcend(3,0);
      TVector3 mcend;
      for (int i=0; i<3; i++) {
        shower_dir[i] /= pmom;
        mcdir[i] = (float)shower_dir[i];
        fmcstart[i] = mcstart[i];
        fmcend[i] = fmcstart[i] + 10.0*mcdir[i];
        mcend[i] = fmcend[i];
      }

            // space charge correction
      std::vector<double> s_offset = _psce->GetPosOffsets(mcstart[0],mcstart[1],mcstart[2]);
      fmcstart[0] = fmcstart[0] - s_offset[0] + 0.7;
      fmcstart[1] = fmcstart[1] + s_offset[1];
      fmcstart[2] = fmcstart[2] + s_offset[2];

      std::vector<double> e_offset = _psce->GetPosOffsets(mcend[0],mcend[1],mcend[2]);
      fmcend[0] = fmcend[0] - e_offset[0] + 0.7;
      fmcend[1] = fmcend[1] + e_offset[1];
      fmcend[2] = fmcend[2] + e_offset[2];

      TVector3 sce_dir = mcend-mcstart;
      std::vector<float> fsce_dir(3,0);
      float sce_dir_len = sce_dir.Mag();
      if ( sce_dir_len>0 ) {
        for (int i=0; i<3; i++) {
          sce_dir[i] /= sce_dir_len;
          fsce_dir[i] = sce_dir[i];
        }
      }      


      // finally!
      float dvertex = larflow::reco::pointRayProjection3f( fmcstart, fsce_dir, fstart );
      float fcos = 0.;
      for (int i=0; i<3; i++) {
        fcos += fsce_dir[i]*dir[i];
      }

      float goodmetric = (1.0-fcos)*(1.0-fcos) + (dvertex*dvertex/9.0); // dvertex has a sigma of 3 cm
      if ( min_feat_dist>goodmetric ) {
        dir_cos = fcos;
        vertex_err_dist = dvertex;
        min_feat_dist = goodmetric;
      }
    }
    
    // store in member variables
    _true_min_feat_dist   = min_feat_dist;
    _true_vertex_err_dist = vertex_err_dist;
    _true_dir_cos = dir_cos;

    LARCV_DEBUG() << "Best true shower match: " << std::endl;
    LARCV_DEBUG() << " - feat_dist=" << _true_min_feat_dist << std::endl;
    LARCV_DEBUG() << " - vertex_dist="<< _true_vertex_err_dist << std::endl;
    LARCV_DEBUG() << " - true-dir-cos=" << _true_dir_cos << std::endl;
    
  }

  
}
}
