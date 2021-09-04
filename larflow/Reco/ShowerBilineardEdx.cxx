#include "ShowerBilineardEdx.h"

#include <stdio.h>
#include <string>
#include <sstream>
#include <omp.h>
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/Geometry.h"
#include "ublarcvapp/MCTools/TruthShowerTrunkSCE.h"
#include "ublarcvapp/MCTools/TruthTrackSCE.h"
#include "geofuncs.h"
#include "ClusterImageMask.h"


namespace larflow {
namespace reco {
  
  ShowerBilineardEdx::ShowerBilineardEdx()
    : larcv::larcv_base("ShowerBilineardEdx")
  {
  }

  ShowerBilineardEdx::~ShowerBilineardEdx()
  {
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
  
}
}
