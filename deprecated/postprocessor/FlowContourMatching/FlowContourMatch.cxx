#include "FlowContourMatch.h"

#include <exception>
#include <cmath>

// larlite
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/DetectorProperties.h"

// larcv
#include "larcv/core/DataFormat/ImageMeta.h"
#include "larcv/core/ROOTUtil/ROOTUtils.h"

// ROOT
#include "TStyle.h"
#include "TCanvas.h"
#include "TH2D.h"

namespace larflow {
  
  
  FlowMatchData_t::FlowMatchData_t( int srcid, int tarid )
    : src_ctr_id(srcid), tar_ctr_id(tarid)
  {}
  FlowMatchData_t::FlowMatchData_t( const FlowMatchData_t& x ) {
    src_ctr_id = x.src_ctr_id;
    tar_ctr_id = x.tar_ctr_id;
    score      = x.score;
    matchingflow_v = x.matchingflow_v;
  }

  // ==========================================
  // FlowContourMatch Algo
  // ---------------------

  const int FlowContourMatch::kSourcePlane[2] = { 2, 2 };
  const int FlowContourMatch::kTargetPlane[2] = { 0, 1 };
  
  FlowContourMatch::FlowContourMatch() {
    for (int i=0; i<2; i++) {
      m_score_matrix[i] = NULL;
      m_plot_scorematrix[i] = NULL;
      m_tar_img2ctrindex[i] = NULL;
    }
    m_src_img2ctrindex = NULL;
      
    // parameters: see header for descriptions
    kTargetChargeRadius = 2;

    // larutil 
    m_psce = new ::larutil::SpaceChargeMicroBooNE;
    m_ptsv = ::larutil::TimeService::GetME();
    
  }

  FlowContourMatch::~FlowContourMatch() {
    clear();
  }

  void FlowContourMatch::clear( bool clear2d, bool clear3d, int flowdir ) {
    if ( clear2d ) {
      // needs to be cleared for each subimage (entry)
      for (int i=0; i<2; i++) {
	if ( flowdir>=0 && i!=flowdir ) continue;
	
	delete m_score_matrix[i];
	delete m_plot_scorematrix[i];
	m_score_matrix[i]     = nullptr;
	m_plot_scorematrix[i] = nullptr;
	m_flowdata[i].clear();
	m_src_targets[i].clear();
	delete m_tar_img2ctrindex[i];
	m_tar_img2ctrindex[i] = nullptr;
      }
      delete m_src_img2ctrindex;
      m_src_img2ctrindex = nullptr;            
    }//end of clear 2nd
    
    if (clear3d) {
      // needs to be cleared after every event
      m_plhit2flowdata.clear();
    }
  }

  // ==================================================================================
  // Primary Interfaces
  // -----------------------
  void FlowContourMatch::fillPlaneHitFlow( const larlitecv::ContourCluster& contour_data,
					   const larcv::Image2D& src_adc,
					   const std::vector<larcv::Image2D>& tar_adc,
					   const std::vector<larcv::Image2D>& flow_img,
					   const larlite::event_hit& hit_v,
					   const float threshold,
					   bool runY2U,
					   bool runY2V) {
    // Primary call.
    // We expect this to be called many time per event, for each subimage
    // goal is to fill m_plhit2flowdata

    // we clear 2d info only (first). 3d info gets accumulated over many subimages
    // this is wonky and needs to get fixed with eventual interface where whole-image gets provided
    // (or vector of subimages? -- lots of mem to hold that simultaneously?)
    clear( true, false ); 

    if(runY2U && runY2V && tar_adc.size()<2){
      throw std::runtime_error("FlowContourMatch::fillPlaneHitFlow: requested both planes but single target");  
    }
    if(runY2U && runY2V && flow_img.size()<2){
      throw std::runtime_error("FlowContourMatch::fillPlaneHitFlow: requested both planes but single flow image");  
    }
    
    if(runY2U){
      //std::cout << "Run Y2U Match" << std::endl;
      _match( FlowContourMatch::kY2U,
	     contour_data,
	     src_adc,
	     tar_adc[0],
	     flow_img[0],
	     hit_v,
	     threshold );
      
      m_plhit2flowdata.ranY2U = true;
    }
    if(runY2V){
      //std::cout << "Run Y2V Match" << std::endl;
      _match( FlowContourMatch::kY2V,
	      contour_data,
	      src_adc,
	      tar_adc[1],
	      flow_img[1],
	      hit_v,
	      threshold );

      m_plhit2flowdata.ranY2V = true;
    }
    //check for size mismatch: should not happen if both run
    if(runY2U && runY2V){
      if ( m_plhit2flowdata.Y2U.size()!=m_plhit2flowdata.Y2V.size() )
	throw std::runtime_error("FlowContourMatch::fillPlaneHitFlowData -- Y2U and Y2V do not match");
    }
    _fill_consistency3d(m_plhit2flowdata.Y2U, m_plhit2flowdata.Y2V, m_plhit2flowdata.consistency3d, m_plhit2flowdata.dy, m_plhit2flowdata.dz);
    
  }


  // =====================================================================
  // SSNet+Endpoint Integration
  // ---------------------------

  void FlowContourMatch::integrateSSNetEndpointOutput( const std::vector<larcv::Image2D>& track_scoreimgs,
						       const std::vector<larcv::Image2D>& shower_scoreimgs,
						       const std::vector<larcv::Image2D>& endpt_scoreimgs )
  {

    // store ssnet+endpoint information into hit2flow data
    // inputs
    // ------
    // track_scoreimgs: assuming vector is (u,v,y)
    // shower_scoreimgs: assuming vector is (u,v,y)
    // endpt_scoreimgs: assuming vector is (u,v,y)
    // ** we assume the above cover the same subimage region
    //
    // output
    // -------
    // updates m_plhit2flowdata[Y2U and Y2V] (if exists)
    
    // can be whole view or subimage
    // we loop over hits, and check if image set is applicable

    bool filled[2] = { m_plhit2flowdata.ranY2U, m_plhit2flowdata.ranY2V };
    std::vector<HitFlowData_t>* phitdata_v[2] = { &(m_plhit2flowdata.Y2U), &(m_plhit2flowdata.Y2V) };

    for ( int iflow=kY2U; iflow<(int)kNumFlowDirs; iflow++ ) {
      if ( !filled[iflow] ) continue;

      std::vector<HitFlowData_t>&  hitdata_v = *(phitdata_v[iflow]);
	      
      // we check if hit is contained
      const larcv::Image2D& track_img  = track_scoreimgs[  kSourcePlane[iflow] ];
      const larcv::Image2D& shower_img = shower_scoreimgs[ kSourcePlane[iflow] ];
      const larcv::Image2D& endpt_img  = endpt_scoreimgs[  kSourcePlane[iflow] ];
      const larcv::ImageMeta& track_meta = track_img.meta();
      
      for ( auto& hitdata : hitdata_v ) {
	if ( track_meta.min_x()<=hitdata.srcwire && hitdata.srcwire<track_meta.max_x()
	     && track_meta.min_y()<=hitdata.pixtick && hitdata.pixtick<track_meta.max_y() ) {
	  // inside image

	  int col = track_meta.col( hitdata.srcwire );
	  int row = track_meta.row( hitdata.pixtick );
	  hitdata.track_score  = track_img.pixel(row,col);
	  hitdata.shower_score = shower_img.pixel(row,col);
	  hitdata.endpt_score  = endpt_img.pixel(row,col);

	  // renormed scores: remove endpt and recalc shower/track
	  float renorm = 1.0 - hitdata.endpt_score; // bg+track+shower
	  hitdata.renormed_track_score  = hitdata.track_score/renorm;
	  hitdata.renormed_shower_score = hitdata.shower_score/renorm;

	}
      }
    }
  }

  // =====================================================================
  // INFILL INTEGRATION
  // --------------------
  void FlowContourMatch::stitchInfill( const larcv::Image2D& infill_crop,
				       larcv::Image2D& trusted,
				       larcv::Image2D& infill_whole,
				       const larcv::EventChStatus& ev_chstatus){

    // fills an infill subimage in a whole image.
    // Note: this is set to prefer prediction in center of image.

    const larcv::ImageMeta& out_meta = infill_whole.meta();
    int src_min_c = out_meta.col( infill_crop.meta().min_x() );
    int src_min_r = out_meta.row( infill_crop.meta().min_y() );

    for ( int r=0; r<(int)infill_crop.meta().rows(); r++ ) {
      for ( int c=0; c<(int)infill_crop.meta().cols(); c++ ) {

	int src_c = src_min_c+c;
	int src_r = src_min_r+r;

	if ( trusted.pixel(src_r,src_c)>0 )
	  continue; // already filled by trusted region

	infill_whole.set_pixel( src_r, src_c, infill_crop.pixel(r,c) );
	if ( c>=261 || c<=571 )
	  trusted.set_pixel( src_r, src_c, 1.0 );
      }
    }

  }

  void FlowContourMatch::maskInfill( const larcv::Image2D& infill,
				     const larcv::EventChStatus& ev_chstatus,
				     const float score_thresh,
				     larcv::Image2D& masked_infill){
    // masks infill prediction outside of dead regions.
    // creates activation image (pix = 0 or 1) w.r.t. infill score threshold
    // inputs
    //------
    // original infill image
    //
    // outputs
    //-------
    // masked & thresholded infill activation
    //
    const larcv::ChStatus& status = ev_chstatus.Status(infill.meta().plane());
    const std::vector<short> st_v = status.as_vector();
    for(int col=0; col<infill.meta().cols(); col++){
      if ( (col+(int)infill.meta().min_x())>=(int)st_v.size() ) continue; // bounds check
      if(st_v[col+infill.meta().min_x()]==4) continue; //if good ch skip
      for(int row=0; row<infill.meta().rows(); row++){
	if(infill.pixel(row,col)<score_thresh) continue; //infill score threshold
	masked_infill.set_pixel(row,col,1);
      }//end of row
    }//end of col
  }

  void FlowContourMatch::addInfill( const larcv::Image2D& masked_infill,
				    const larcv::EventChStatus& ev_chstatus,
				    const float threshold,
				    larcv::Image2D& img_fill_v){
    // adds infill activation to existing adc image
    // adc is set w.r.t. some charge threshold
    // inputs
    //-------
    // infill activation
    // original adc image
    //
    // outputs
    //---------
    // edited adc image
    //

    // int maxcol = img_fill_v.meta().cols();
    // int maxcol_plane = maxcol;
    // if ( img_fill_v.meta().plane()<2 ) {
    //   maxcol_plane = img_fill_v.meta().col(2399)+1; // maxwire
    // }
    // else {
    //   maxcol_plane = img_fill_v.meta().col(3455)+1; // maxwire
    // }
    // maxcol = ( maxcol>maxcol_plane) ? maxcol_plane : maxcol;
    
    const larcv::ChStatus& status = ev_chstatus.Status(masked_infill.meta().plane());
    const std::vector<short> st_v = status.as_vector();
    for(int col=0; col<masked_infill.meta().cols(); col++){
      if ( (col+(int)masked_infill.meta().min_x())>=(int)st_v.size() ) continue; // bounds check
      if(st_v[col+masked_infill.meta().min_x()]==4) continue; //if good ch skip
      for(int row=0; row<masked_infill.meta().rows(); row++){
	if(masked_infill.pixel(row,col)<1) continue; //masked infill
	img_fill_v.set_pixel(row,col,threshold+5.); //set adc to above threshold
      }//end of row
    }//end of col
  }

  void FlowContourMatch::labelInfillHits( const std::vector<larcv::Image2D>& masked_infill) {
    // store Infill above thresh information into hit2flow data
    // inputs
    // ------
    // masked_infill: assuming vector is (u,v,y)
    // ** we assume the above cover the same subimage region
    //
    // output
    // -------
    // updates m_plhit2flowdata[Y2U and Y2V] (if exists)

    // can be whole view or subimage
    // we loop over hits, and check if image set is applicable

    bool filled[2] = { m_plhit2flowdata.ranY2U, m_plhit2flowdata.ranY2V };
    std::vector<HitFlowData_t>* phitdata_v[2] = { &(m_plhit2flowdata.Y2U), &(m_plhit2flowdata.Y2V) };

    for ( int iflow=kY2U; iflow<(int)kNumFlowDirs; iflow++ ) {
      if ( !filled[iflow] ) continue;

      std::vector<HitFlowData_t>&  hitdata_v = *(phitdata_v[iflow]);
      // we check if hit is contained
      const larcv::Image2D& infill_src = masked_infill[ kSourcePlane[iflow] ];
      const larcv::Image2D& infill_tar = masked_infill[ kTargetPlane[iflow] ];
      const larcv::ImageMeta& src_meta = infill_src.meta();
      const larcv::ImageMeta& tar_meta = infill_tar.meta();

      for ( auto& hitdata : hitdata_v ) {
	if ( src_meta.min_x()<=hitdata.srcwire && hitdata.srcwire<src_meta.max_x()
	     && src_meta.min_y()<=hitdata.pixtick && hitdata.pixtick<src_meta.max_y() ) {
	  // inside image

	  int col = src_meta.col( hitdata.srcwire );
	  int row = src_meta.row( hitdata.pixtick );
	  hitdata.src_infill = (infill_src.pixel(row,col)>0 ) ? true : false;
	  // check if target pix is inside image
	  if ( tar_meta.min_x()<=hitdata.targetwire && hitdata.targetwire<tar_meta.max_x()
	       && tar_meta.min_y()<=hitdata.pixtick && hitdata.pixtick<tar_meta.max_y() ) {
	    int colt= tar_meta.col( hitdata.targetwire);
	    hitdata.tar_infill = (infill_tar.pixel(row,colt)>0) ? true : false;
	  }
	  else{
	    //not in crop
	    hitdata.tar_infill = false;
	  }
	}
      }
    }
  }
  // =====================================================================
  // MCTRACK MATCH
  // --------------------
  void FlowContourMatch::mctrack_match( const larlite::event_mctrack& evtrack,
					const std::vector<larcv::Image2D>& img_v){

    return mctrack_match( m_plhit2flowdata, evtrack, img_v, m_psce, m_ptsv);
  }

  void FlowContourMatch::mctrack_match(PlaneHitFlowData_t& plhit2flowdata,
				       const larlite::event_mctrack& evtrack,
				       const std::vector<larcv::Image2D>& img_v,
				       ::larutil::SpaceChargeMicroBooNE* psce,
				       const ::larutil::TimeService* ptsv){
    
    // This function updates internal std::vector<HitFlowData_t> to add mctruth.
    // It is intended to be run once per event,
    // when we have collected all hits in the whole image. 
    // First all mctracks in the event are projected onto the whole Y image.
    // Then we match via hit source pixel and tick 
    
    //space charge and time service; ideally initialized in algo constructor
    ::larutil::SpaceChargeMicroBooNE* sce = psce;
    if ( psce==NULL ){
      sce = new ::larutil::SpaceChargeMicroBooNE;
    }
    // note on TimeService: if not initialized from root file via GetME(true)
    // needs trig_offset hack in tufts_larflow branch head
    const ::larutil::TimeService* tsv = ptsv;
    if( ptsv==NULL){
      tsv = ::larutil::TimeService::GetME();
    }

    // blank track images: trackid, x, y, z, E, flag, dWall with Y meta
    std::vector<larcv::Image2D> trackimg_v;
    for(int i=0; i<7; i++){
      larcv::Image2D trackimg(img_v[2].meta());
      trackimg.paint(-1.0);
      trackimg_v.emplace_back(std::move(trackimg));
    }
    for(const auto& truthtrack : evtrack){
      //initialize internal vectors
      int nstep = truthtrack.size();
      std::vector<unsigned int> trackid;//(nstep,0);
      std::vector<double> E;//(nstep,0);
      std::vector<std::vector<float>> tyz;//(nstep,std::vector<double>(3,0));
      std::vector<float> dWall;
      
      _mctrack_to_tyz(truthtrack,tyz,trackid,E,dWall,sce,tsv);
      _tyz_to_pixels(tyz,trackid,E,dWall,img_v[2],trackimg_v);
    }
    std::vector<HitFlowData_t>* hit2flowdata = &plhit2flowdata.Y2U; // assign Y2U first
    std::vector<HitFlowData_t>* hit2flowdata2 = NULL; 
    if(plhit2flowdata.ranY2U && plhit2flowdata.ranY2V){
      hit2flowdata2 = &plhit2flowdata.Y2V; // also assign Y2V
    }
    else if(!plhit2flowdata.ranY2U && plhit2flowdata.ranY2V){
      hit2flowdata = &plhit2flowdata.Y2V; //assign Y2V only
    }
    else if(!plhit2flowdata.ranY2U && !plhit2flowdata.ranY2V){
      throw std::runtime_error("FlowContourMatch::mctrack_match -- no hits filled");
    }
    else{
      //do nothing
    }
    // now we loop over HitFlowData_t
    // note: this copies the same mctruth to Y2U and Y2V, b/c we don't know
    // at this point which one will be selected
    //
    // to check: are srcwire, pixtick in global (whole img) scope??
    // if not, I need the crop image meta
    
    for(auto& hit : *(hit2flowdata)){
      hit.X_truth.resize(3,-1.);
      if(hit.pixtick >= trackimg_v[0].meta().min_y() && hit.pixtick < trackimg_v[0].meta().max_y()
	 && hit.srcwire >= trackimg_v[0].meta().min_x() && hit.srcwire < trackimg_v[0].meta().max_x() ){
	int col = trackimg_v[0].meta().col( hit.srcwire );
	int row = trackimg_v[0].meta().row( hit.pixtick );
	
	hit.trackid = (int)trackimg_v[0].pixel(row,col);
	hit.truthflag = (int)trackimg_v[5].pixel(row,col)+1; //nomatch is -1 in blank image
	hit.X_truth[0] = trackimg_v[1].pixel(row,col);
	hit.X_truth[1] = trackimg_v[2].pixel(row,col);
	hit.X_truth[2] = trackimg_v[3].pixel(row,col);
	hit.dWall      = trackimg_v[6].pixel(row,col);
	//std::cout <<"dWall= "<< hit.dWall << std::endl;
      }
    }
    
    if(hit2flowdata2 != NULL){
      for(auto& hit : *(hit2flowdata2)){
	hit.X_truth.resize(3,-1.);
	if(hit.pixtick >= trackimg_v[0].meta().min_y() && hit.pixtick < trackimg_v[0].meta().max_y()
	   && hit.srcwire >= trackimg_v[0].meta().min_x() && hit.srcwire < trackimg_v[0].meta().max_x() ){
	  int col = trackimg_v[0].meta().col( hit.srcwire );
	  int row = trackimg_v[0].meta().row( hit.pixtick );

	  hit.trackid = (int)trackimg_v[0].pixel(row,col);
	  hit.truthflag = (int)trackimg_v[5].pixel(row,col)+1;
	  hit.X_truth[0] = trackimg_v[1].pixel(row,col);
	  hit.X_truth[1] = trackimg_v[2].pixel(row,col);
	  hit.X_truth[2] = trackimg_v[3].pixel(row,col);
	  hit.dWall      = trackimg_v[6].pixel(row,col);
	}
      }
    }
    
  }

  void FlowContourMatch::_mctrack_to_tyz(const larlite::mctrack& truthtrack,
					 std::vector<std::vector<float>>& tyz,
					 std::vector<unsigned int>& trackid,
					 std::vector<double>& E,
					 std::vector<float>& dWall,
					 ::larutil::SpaceChargeMicroBooNE* sce,
					 const ::larutil::TimeService* tsv){

    const float cm_per_tick = ::larutil::LArProperties::GetME()->DriftVelocity()*(::larutil::DetectorProperties::GetME()->SamplingRate()*1.0e-3); 
    // detector boundaries
    const ::larutil::Geometry* geo = ::larutil::Geometry::GetME();
    const float xmin = 0.;
    const float xmax = geo->DetHalfWidth()*2.;
    const float ymin = -1.*geo->DetHalfHeight();
    const float ymax = geo->DetHalfHeight();
    const float zmin = 0.;
    const float zmax = geo->DetLength();

    for(int step=1; step<truthtrack.size(); step++){
      std::vector<float> pos(3); //x,y,z
      std::vector<float> dr(4); // x,y,z,t
      dr[0] = -truthtrack[step-1].X()+truthtrack[step].X();
      dr[1] = -truthtrack[step-1].Y()+truthtrack[step].Y();
      dr[2] = -truthtrack[step-1].Z()+truthtrack[step].Z();
      dr[3] = -truthtrack[step-1].T()+truthtrack[step].T();
      float dR = sqrt(pow(dr[0],2)+pow(dr[1],2)+pow(dr[2],2));
      for (int i=0; i<3; i++)
	dr[i] /= dR;

      int N = (dR>0.15) ? dR/0.15+1 : 1;
      float dstep = dR/float(N);
      for(int i=0; i<N; i++){
	float mindist = 1.0e4;
	pos[0] = truthtrack[step-1].X()+dstep*i*dr[0];
	pos[1] = truthtrack[step-1].Y()+dstep*i*dr[1];
	pos[2] = truthtrack[step-1].Z()+dstep*i*dr[2];
	// for time use linear approximation
	float t = truthtrack[step-1].T()+i*(dstep*::larutil::LArProperties::GetME()->DriftVelocity()*1.0e3); // cm * cm/usec * usec/ns

	std::vector<double> pos_offset = sce->GetPosOffsets( pos[0], pos[1], pos[2] );
	pos[0] = pos[0]-pos_offset[0]+0.7;
	pos[1] += pos_offset[1];
	pos[2] += pos_offset[2];
	//time tick
	float tick = tsv->TPCG4Time2Tick(t) + pos[0]/cm_per_tick;
	pos[0] = (tick + tsv->TriggerOffsetTPC()/(::larutil::DetectorProperties::GetME()->SamplingRate()*1.0e-3))*cm_per_tick; // x in cm
	// SCE shifted min dist to wall
	//if(std::abs(pos[0] - xmin)< mindist) mindist = std::abs(pos[0] - xmin);
	//if(std::abs(pos[0] - xmax)< mindist) mindist = std::abs(pos[0] - xmax);
	if(std::abs(pos[1] - ymin)< mindist) mindist = std::abs(pos[1] - ymin);
	if(std::abs(pos[1] - ymax)< mindist) mindist = std::abs(pos[1] - ymax);
	if(std::abs(pos[2] - zmin)< mindist) mindist = std::abs(pos[2] - zmin);
	if(std::abs(pos[2] - zmax)< mindist) mindist = std::abs(pos[2] - zmax);

	tyz.push_back(pos);
	dWall.push_back(mindist);
	trackid.push_back(truthtrack.TrackID());//this is the same for all steps in a track
	E.push_back(truthtrack[step-1].E());
      }
    }
  }

  void FlowContourMatch::_tyz_to_pixels(const std::vector<std::vector<float>>& tyz,
					const std::vector<unsigned int>& trackid,
					const std::vector<double>& E,
					const std::vector<float>& dWall,
					const larcv::Image2D& adc,
					std::vector<larcv::Image2D>& trackimg_v){

    // this function updates the mctrack images for each mctrack
    // if two tracks are crossing, the corresponding pixel will be filled by the track with higher E at that point
    std::vector<std::vector<int>> imgpath;
    imgpath.reserve(tyz.size());
    for (auto const& pos : tyz ) {
      std::vector<int> crossing_imgcoords = getProjectedPixel( pos, adc.meta(), 3 );
      imgpath.push_back( crossing_imgcoords );
    }
    
    int istep = 0;
    float thresh = 10.0; // adc threshold
    // note: pix are image row, col numbers
    for(auto const& pix : imgpath ){
      if(pix[0]==-1 || pix[3]==-1){istep++; continue;}
      for(int b=pix[3]-2; b<pix[3]+3; b++){
	if( b<0 || b>= trackimg_v[0].meta().cols()) continue; //border case
	double prevE = trackimg_v[4].pixel(pix[0],b);
	if( prevE>0 && prevE > E.at(istep) ) continue; //fill only highest E deposit
	if( adc.pixel(pix[0],b)<thresh) continue;
	trackimg_v[0].set_pixel(pix[0],b,(float)trackid.at(istep));// trackid
	trackimg_v[1].set_pixel(pix[0],b,(float)tyz.at(istep)[0]); // x
	trackimg_v[2].set_pixel(pix[0],b,(float)tyz.at(istep)[1]); // y
	trackimg_v[3].set_pixel(pix[0],b,(float)tyz.at(istep)[2]); // z
	trackimg_v[4].set_pixel(pix[0],b,(float)E.at(istep)); // E
	int flag = (b==pix[3]) ? 0 : 1; // 1: on track, 2: on smear //+1 when filling hit
	trackimg_v[5].set_pixel(pix[0],b,(float)flag); // flag
	trackimg_v[6].set_pixel(pix[0],b,(float)dWall.at(istep)); //dWall
      }
      istep++;
    }
    
  }
  
  // copy of UBWireTool::getProjectedImagePixel
  std::vector<int> FlowContourMatch::getProjectedPixel( const std::vector<float>& pos3d,
							const larcv::ImageMeta& meta,
							const int nplanes,
							const float fracpixborder ) {
    std::vector<int> img_coords( nplanes+1, -1 );
    float row_border = fabs(fracpixborder)*meta.pixel_height();
    float col_border = fabs(fracpixborder)*meta.pixel_width();

    // tick/row
    float tick = pos3d[0]/(::larutil::LArProperties::GetME()->DriftVelocity()*::larutil::DetectorProperties::GetME()->SamplingRate()*1.0e-3) + 3200.0;
    if ( tick<meta.min_y() ) {
      if ( tick>meta.min_y()-row_border )
	// below min_y-border, out of image
	img_coords[0] = meta.rows()-1; // note that tick axis and row indicies are in inverse order (same order in larcv2)
      else
	// outside of image and border
	img_coords[0] = -1;
    }
    else if ( tick>meta.max_y() ) {
      if ( tick<meta.max_y()+row_border )
	// within upper border
	img_coords[0] = 0;
      else
	// outside of image and border
	img_coords[0] = -1;
    }
    else {
      // within the image
      img_coords[0] = meta.row( tick );
    }

    // Columns
    Double_t xyz[3] = { pos3d[0], pos3d[1], pos3d[2] };
    // there is a corner where the V plane wire number causes an error
    if ( (pos3d[1]>-117.0 && pos3d[1]<-116.0) && pos3d[2]<2.0 ) {
      //std::cout << __PRETTY_FUNCTION__ << ": v-plane corner hack (" << xyz[0] << "," << xyz[1] << "," << xyz[2] << ")" << std::endl;
      xyz[1] = -116.0;
    }
    for (int p=0; p<nplanes; p++) {
      float wire = larutil::Geometry::GetME()->WireCoordinate( xyz, p );
      // round wire
      //wire = std::roundf(wire);

      // get image coordinates
      if ( wire<meta.min_x() ) {
	if ( wire>meta.min_x()-col_border ) {
	  //std::cout << __PRETTY_FUNCTION__ << " plane=" << p << " wire=" << wire << "<" << meta.min_x()-col_border << std::endl;
	  // within lower border
	  img_coords[p+1] = 0;
	}
	else
	  img_coords[p+1] = -1;
      }
      else if ( wire>=meta.max_x() ) {
	if ( wire<meta.max_x()+col_border ) {
	  //std::cout << __PRETTY_FUNCTION__ << " plane=" << p << " wire=" << wire << ">" << meta.max_x()+col_border << std::endl;
	  // within border
	  img_coords[p+1] = meta.cols()-1;
	}
	else
	  // outside border
	  img_coords[p+1] = -1;
      }
      else
	// inside image
	img_coords[p+1] = meta.col( wire );
    }//end of plane loop

    // there is a corner where the V plane wire number causes an error
    if ( pos3d[1]<-116.3 && pos3d[2]<2.0 && img_coords[1+1]==-1 ) {
      img_coords[1+1] = 0;
    }
    return img_coords;
  }

  // ----------------------------------
  // TRUTH MATCH WITH ANCESTOR IMAGES
  // ----------------------------------

  void FlowContourMatch::label_mcid_w_ancestor_img( const std::vector<larcv::Image2D>& ancestor_v,
						    const std::vector<larcv::Image2D>& adcimg_v) {

    std::vector<HitFlowData_t>* pflowdata[2] = { nullptr, nullptr };
    
    if ( m_plhit2flowdata.ranY2U )
      pflowdata[0] = &(m_plhit2flowdata.Y2U);
    if ( m_plhit2flowdata.ranY2V )
      pflowdata[1] = &(m_plhit2flowdata.Y2V);
    
    if ( !pflowdata[0] && !pflowdata[1] ) {
      throw std::runtime_error("[FlowContourMatch::mctrack_match_w_ancestor_img] No hit information filled yet");
    }

    int nlabeled[2] = {0,0};
    for ( int iflow=0; iflow<2; iflow++) {
      if ( !pflowdata[iflow] ) continue;
      std::vector<HitFlowData_t>& flowdata = *(pflowdata[iflow]);

      // ------------------------------------------------
      // debug
      // char hname[100];
      // sprintf( hname, "ancestorlabel_flow%d", iflow );
      // TH2D hancestor = larcv::as_th2d( adcimg_v[2], hname );
      // hancestor.Reset();
      // for ( size_t r=0; r<ancestor_v[2].meta().rows(); r++ ) {
      // 	for ( size_t c=0; c<ancestor_v[2].meta().cols(); c++ ) {
      // 	  if ( ancestor_v[2].pixel(r,c)>=0 )
      // 	    hancestor.SetBinContent( c+1, r+1, 5 );
      // 	}
      // }
      // ------------------------------------------------      

      int maxcol = ancestor_v[2].meta().cols();
      int maxrow = ancestor_v[2].meta().rows();
      
      for ( auto& hit : flowdata ) {
	if ( hit.srcwire<0 || hit.pixtick<0 ) continue;
	int col = ancestor_v[2].meta().col( hit.srcwire );
	int row = ancestor_v[2].meta().row( hit.pixtick );

	// look in a neighborhood
	float maxadc = -1;
	int mcid_at_max = -1;
	for ( int dc=-3; dc<=3; dc++ ) {
	  int c=col+dc;
	  if ( c<0 || c>=maxcol ) continue;
	  for ( int dr=-10; dr<=10; dr++ ) {
	    int r=row+dr;
	    if ( r<0 || r>=maxrow ) continue;
	    float adc = adcimg_v[2].pixel(r,c);
	    int mcid  = ancestor_v[2].pixel(r,c);
	    if ( adc>maxadc && mcid>=0 ) {
	      maxadc = adc;
	      mcid_at_max = mcid;
	    }
	  }
	}
	
	hit.trackid = mcid_at_max;
	if ( mcid_at_max>=0 ) {
	  //hancestor.SetBinContent( col+1, row+1, 10 );
	  nlabeled[iflow]++;
	}
	else {
	  //hancestor.SetBinContent( col+1, row+1, -5 );	  
	}
      }
      std::cout << "[FlowContourMatch::label_mcid_w_ancestor_img] "
		<< "Flow path " << iflow << " labeled hits=" << nlabeled[iflow]
		<< " of " << flowdata.size() << " all hits"
		<< std::endl;

      // ---------------------------------
      // debug
      // gStyle->SetOptStat(0);
      // TCanvas c("c","c",800,600);
      // hancestor.Draw("colz");
      // std::string canvname = std::string(hname)+".png";
      // c.SaveAs( canvname.c_str() );
      // std::cout << "[FlowContourMatch::label_mcid_w_ancestor_img] [DEBUG] save " << hname << std::endl;      
      // ---------------------------------      

    }
    
  }
  
  // =====================================================================
  // INTERNAL FUNCTIONS
  // --------------------

  void FlowContourMatch::_fill_consistency3d(std::vector<HitFlowData_t>& Y2U,
					     std::vector<HitFlowData_t>& Y2V,
					     std::vector<int>& consistency3d,
					     std::vector<float>& dy,
					     std::vector<float>& dz) {    
    float ddy =0;
    float ddz =0;
    if(Y2U.size()<=0 && Y2V.size()<=0) return; //nothing was filled

    // if here, at least flow was filled. we allocate consistency vectors
    if ( (Y2U.size()>0 && consistency3d.size()!=Y2U.size())
	 || (Y2V.size()>0 && consistency3d.size()!=Y2V.size()) ) {
      consistency3d.assign( Y2U.size(), -1);
      dy.assign( Y2U.size(), -1 );
      dz.assign( Y2U.size(), -1 );
    }
    
    //first fill 3D coord
    if(Y2U.size()>0){
      for(int i=0; i<Y2U.size(); i++){
	Y2U[i].X.assign( 3, -1);
	_calc_coord3d(Y2U[i],Y2U[i].X,FlowContourMatch::kY2U);
      }
    }
    if(Y2V.size()>0){
      for(int i=0; i<Y2V.size(); i++){
	Y2V[i].X.assign( 3, -1);
	_calc_coord3d(Y2V[i],Y2V[i].X,FlowContourMatch::kY2V);
      }
    }
    
    //if both flow information is provided, we can run consistency measures
    if(Y2U.size()>0 && Y2V.size()>0){
      //both are same size (by construction)
      for(int i=0; i<Y2U.size(); i++){
	_calc_dist3d(Y2U[i].X,Y2V[i].X,ddy,ddz);
	dy[i] = ddy;
	dz[i] = ddz;
	consistency3d[i] = _calc_consistency3d(ddy,ddz);
      }
    } //end of both
  }
    

  void FlowContourMatch::_calc_coord3d(HitFlowData_t& hit_y2u,
				       std::vector<float>& X,
				       FlowDirection_t flowdir) {

    if ( hit_y2u.srcwire<0 || hit_y2u.srcwire>=3456) {
      // this hit has no flow-match data
      X[0] = -1;
      X[1] = -1;
      X[2] = -1;
      return;
    }
    // for debug
    //std::cout << __PRETTY_FUNCTION__ << "src wire=" << hit_y2u.srcwire << " tar(u)=" << hit_y2u.targetwire << std::endl;
    if ( hit_y2u.targetwire<0 || hit_y2u.targetwire>=2400 ) {
      // flow is out of the plane (track how this happend).
      // for now, we do not have a vaule
      X[0] = -1;
      X[1] = -1;
      X[2] = -1;
      return;
    }
    
    // larlite geometry tool
    const ::larutil::Geometry* geo = ::larutil::Geometry::GetME();
    const float cm_per_tick      = ::larutil::LArProperties::GetME()->DriftVelocity()*0.5; // cm/usec * usec/tick
    X[0] = (hit_y2u.pixtick-3200.0)*cm_per_tick;
    double y=-1.;
    double z=-1.;
    //Get intersection
    switch ( flowdir ) {
    case kY2U:
      geo->IntersectionPoint( hit_y2u.srcwire, hit_y2u.targetwire, 2, 0, y, z );
      break;
    case kY2V:
      geo->IntersectionPoint( hit_y2u.srcwire, hit_y2u.targetwire, 2, 1, y, z );
      break;
    default:
      throw std::runtime_error("FlowContourMatch::calc_coord3d: invalid FlowDirection_t option"); // shouldnt be possible
      break;
    }
    X[1] = y;
    X[2] = z;
    
  }

  void FlowContourMatch::_calc_dist3d(std::vector<float>& X0,
				      std::vector<float>& X1,
				      float& dy,
				      float& dz) {

    if ( X0.size()<=0 ||  X1.size()<=0 || X0.size()!=X1.size() ) {
      // one of the 3D positions was not properly initialized: should not happen
      dy = -1;
      dz = -1;
      return;
    }
    if ( std::any_of(X0.cbegin(), X0.cend(), [](int i){ return i == -1; })
	 || std::any_of(X1.cbegin(), X1.cend(), [](int i){ return i == -1; })) {
      // one of the flows is out of the plane (track how this happend).
      // for now, we do not have a vaule
      dy = -1;
      dy = -1;
      return;

    }
        
    dy = sqrt(pow(X1[1]-X0[1],2));
    dz = sqrt(pow(X1[2]-X0[2],2));
    
  }

  int FlowContourMatch::_calc_consistency3d(float& dy,
					    float& dz) {

    if ( dy<0 || dz<0 )
	   return larlite::larflow3dhit::kNoValue; // no

    float dR = sqrt(dy*dy + dz*dz);
    //dR should be in cm?
    if(dR<0.5) return larlite::larflow3dhit::kIn5mm;  // kIn5mm
    if(dR<1.0) return larlite::larflow3dhit::kIn10mm; // kIn10mm
    if(dR<5.0) return larlite::larflow3dhit::kIn50mm; // kIn50mm
    return larlite::larflow3dhit::kOut50mm;           // kOut50mm

  }
 
  void FlowContourMatch::_match( FlowDirection_t flowdir,
				 const larlitecv::ContourCluster& contour_data,
				 const larcv::Image2D& src_adc,
				 const larcv::Image2D& tar_adc,
				 const larcv::Image2D& flow_img,
				 const larlite::event_hit& hit_v,
				 const float threshold ) {
    
    // produces 3D hits from from one flow image
    int src_planeid = -1;
    int tar_planeid = -1;

    switch ( flowdir ) {
    case kY2U:
      src_planeid = 2;
      tar_planeid = 0;
      break;
    case kY2V:
      src_planeid = 2;
      tar_planeid = 1;
      break;
    default:
      throw std::runtime_error("FlowContourMatch::match: invalid FlowDirection_t option"); // shouldnt be possible
      break;
    }

    // we clear the 2d data, but keep the hit data (which we will update with _make3Dhits)
    clear( true, false, (int)flowdir );
    
    // first we create match data within the image
    _createMatchData( contour_data, flow_img, src_adc, tar_adc, flowdir );

    // use the match data to score contour-contour matching
    _scoreMatches( contour_data, src_planeid, tar_planeid, flowdir );

    // use score matrix to define matches
    _greedyMatch(flowdir);

    // make 3D hits and update hit2flowdata vector
    if ( flowdir==kY2U )
      _make3Dhits( hit_v, src_adc, tar_adc, src_planeid, tar_planeid, threshold, m_plhit2flowdata.Y2U, flowdir );
    else if ( flowdir==kY2V )
      _make3Dhits( hit_v, src_adc, tar_adc, src_planeid, tar_planeid, threshold, m_plhit2flowdata.Y2V, flowdir );
    else
      throw std::runtime_error("should never get here");

  }


  void FlowContourMatch::makeHitsFromWholeImagePixels( const larcv::Image2D& src_adc, larlite::event_hit& evhit_v, const float threshold ) {

    // instead of hits, which can be too sparsely defined,
    // we can try to match pixels (or maybe eventually groups of pixels).
    // to use same machinery, we turn pixels into hits
    //
    // inputs
    // ------
    // src_adc: source ADC image
    // threshold: ADC threshold
    //
    // outputs
    // -------
    // evhit_v (by address): vector of hits created from above threshold pixels

    evhit_v.clear();
    evhit_v.reserve(10000);

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
    
  // ==================================================================================
  // Algorithm (internal) Methods
  // -----------------------------
    
  void FlowContourMatch::_createMatchData( const larlitecv::ContourCluster& contour_data,
					   const larcv::Image2D& flow_img,
					   const larcv::Image2D& src_adc,
					   const larcv::Image2D& tar_adc,
					   const FlowDirection_t kflowdir ) {

    // we compile the relationships between pixels and the different contour-clusters
    // the goal is to start to see what contours on source and target imager are paired together
    //   through flow predictions
    // things we fill
    // --------------
    // m_src_img2ctrindex: 2D array. position of array corresponds to position of src_adc img.
    //                     value is the index to a contour.
    // m_tar_img2ctrindex: same as above, but for target image
    // m_srcimg_meta: pointer to source image meta
    // m_tarimg_meta: pointer to target image meta
    // m_src_targets: map between contour index and a ContourTargets_t object.
    //                contourtargets object is a container storing TargetPix_t.
    //                TargetPix_t stores info about source -> target pixel pair from the flow predictions
    // m_flowdata: map between src-target contour pair to FlowMatchData_t which contains score
    

    int src_planeid = src_adc.meta().plane();
    int tar_planeid = tar_adc.meta().plane();
    float threshold = 10;
    
    const larcv::ImageMeta& srcmeta = src_adc.meta();
    const larcv::ImageMeta& tarmeta = tar_adc.meta();
    m_srcimg_meta = &srcmeta;
    m_tarimg_meta[kflowdir] = &tarmeta;
    
    // allocate arrays for image pixel to contour index lookup
    m_src_img2ctrindex                       = new int[m_srcimg_meta->cols()*m_srcimg_meta->rows()];
    m_tar_img2ctrindex[kflowdir]             = new int[m_tarimg_meta[kflowdir]->cols()*m_tarimg_meta[kflowdir]->rows()];
    memset( m_src_img2ctrindex, 0,           sizeof(int)*m_srcimg_meta->cols()*m_srcimg_meta->rows() );
    memset( m_tar_img2ctrindex[kflowdir], 0, sizeof(int)*m_tarimg_meta[kflowdir]->cols()*m_tarimg_meta[kflowdir]->rows() );    
    
    for ( int r=0; r<(int)srcmeta.rows(); r++) {
      
      // for each row, we find the available contours on the images.
      // saves us search each time

      std::set< int > tar_ctr_ids;
      std::vector<int> src_cols_in_ctr;
      src_cols_in_ctr.reserve(20);
      std::map<int,int> src_cols2ctrid;

      // std::cout << "------------------------------------------" << std::endl;
      // std::cout << "Find row=" << r << " contours" << std::endl;

      // Find contours on source image in this row
      // std::cout << "source: ";      
      for ( int c=0; c<(int)srcmeta.cols(); c++) {
	if ( src_adc.pixel(r,c)<threshold )
	  continue;

	cv::Point pt( c,r );
	int ictr = 0;
	for ( auto const& ctr : contour_data.m_plane_atomics_v[src_planeid] ) {
	  double result =  cv::pointPolygonTest( ctr, pt, false );
	  if ( result>=0 ) {
	    src_cols_in_ctr.push_back( c );
	    src_cols2ctrid[c] = ictr;	    
	    //std::cout << " " << ictr;
	    m_src_img2ctrindex[ r*m_srcimg_meta->cols() + c ] = ictr;
	    break;
	  }
	  ictr++;
	}
      }
      //std::cout << std::endl;

      // Find Contours on the target image in this row
      //std::cout << "target: ";      
      for ( int c=0; c<(int)tarmeta.cols(); c++) {
	if ( tar_adc.pixel(r,c)<threshold )
	  continue;

	cv::Point pt( c,r );	
	int ictr = 0;	
	for ( auto const& ctr : contour_data.m_plane_atomics_v[tar_planeid] ) {
	  double result =  cv::pointPolygonTest( ctr, pt, false );
	  if ( result>=0 ) {
	    tar_ctr_ids.insert( ictr );
	    m_tar_img2ctrindex[kflowdir][ r*m_tarimg_meta[kflowdir]->cols() + c ] = ictr;
	    //std::cout << ictr << " ";
	    break;
	  }
	  ictr++;
	}
      }//end of col loop
      //std::cout << std::endl;      

      // Nothing in this row, move on to the next row
      if ( src_cols_in_ctr.size()==0 || tar_ctr_ids.size()==0 ) {
	//std::cout << "nothing to match" << std::endl;
	continue;
      }

      // now loop over source columns in contours and make matches to target contours
      for ( auto const& source_col : src_cols_in_ctr ) {

	float flow = flow_img.pixel(r,source_col);
	int target_col = source_col+flow;
	cv::Point src_pt( source_col, r );
	cv::Point tar_pt( target_col, r );	
	
	// remember the contour we're in
	int src_ctr_id = src_cols2ctrid[source_col];
	const larlitecv::Contour_t& src_ctr = contour_data.m_plane_atomics_v[src_planeid][src_ctr_id];

	// store the target point for this contour
	auto it_srcctr_targets = m_src_targets[kflowdir].find( src_ctr_id );
	if ( it_srcctr_targets==m_src_targets[kflowdir].end() ) {
	  // create a container
	  m_src_targets[kflowdir].insert( std::pair<int,ContourTargets_t>(src_ctr_id,ContourTargets_t()) );
	  it_srcctr_targets = m_src_targets[kflowdir].find( src_ctr_id );
	}
	TargetPix_t tpix;
	tpix.row = r;
	tpix.col = target_col;
	tpix.srccol = source_col;
	it_srcctr_targets->second.push_back( tpix );
	
	// now, find the distance to the contours on the target row
	for ( auto const& ctrid : tar_ctr_ids ) {
	  float dist = cv::pointPolygonTest( contour_data.m_plane_atomics_v[tar_planeid][ctrid], tar_pt, true );
	  if ( dist>-1.0 )
	    dist = -1.0;
	  
	  dist = fabs(dist);

	  // apply some matching threshold [WARNING HIDDEN PARAMETER]
	  if ( dist>30.0 ) {
	    continue;
	  }

	  // // store the match data
	  SrcTarPair_t idpair = { src_ctr_id, ctrid };
	  auto it_flowdata = m_flowdata[kflowdir].find( idpair );
	  if ( it_flowdata==m_flowdata[kflowdir].end() ) {
	  //   // if the map doesn't have the pair we're looking for, we create the data
	    FlowMatchData_t x( src_ctr_id,  ctrid);
	    m_flowdata[kflowdir].insert( std::pair<SrcTarPair_t,FlowMatchData_t>(idpair,x));
	    it_flowdata = m_flowdata[kflowdir].find(idpair);
	  }

	  FlowMatchData_t& flowdata = it_flowdata->second;
	  FlowMatchData_t::FlowPixel_t flowpix;
	  flowpix.src_wire = source_col;
	  flowpix.tar_wire = target_col;
	  flowpix.row = r;
	  flowpix.pred_miss = std::fabs(dist);
	  flowdata.matchingflow_v.push_back( flowpix );
	  
	}
      }
    }
  }

  void FlowContourMatch::_scoreMatches( const larlitecv::ContourCluster& contour_data, int src_planeid, int tar_planeid, const FlowDirection_t kflowdir ) {
    // takes src-target contour pairs and starts to calculate scores
    // scores are based on what fraction of pixels get matched from source to the target contour
    //
    // results use m_flowdata information
    //
    // things we fill
    // --------------
    // m_score_matrix: (src,target) contour index are the pos. in the array/matrix. value is score
    //
    
    m_src_ncontours = contour_data.m_plane_atomics_v[src_planeid].size();
    m_tar_ncontours[(int)kflowdir] = contour_data.m_plane_atomics_v[tar_planeid].size();
    // std::cout << __PRETTY_FUNCTION__ << std::endl;
    // std::cout << "scr ncontours: " << m_src_ncontours << std::endl;
    // std::cout << "tar ncontours: " << m_tar_ncontours << std::endl;

    if ( m_score_matrix[kflowdir]!=NULL )
      delete m_score_matrix[kflowdir];
    
    m_score_matrix[kflowdir] = new double[m_src_ncontours*m_tar_ncontours[kflowdir]]; // should probably its own class
    memset(m_score_matrix[kflowdir], 0, sizeof(double)*m_src_ncontours*m_tar_ncontours[kflowdir] );
    
    for ( auto it : m_flowdata[kflowdir] ) {
      FlowMatchData_t& flowdata = it.second;
      float score = _scoreMatch( flowdata );
      flowdata.score = score;
      m_score_matrix[kflowdir][ flowdata.src_ctr_id*m_tar_ncontours[kflowdir] + flowdata.tar_ctr_id ] = score;
    }

    // normalize it
    for (int is=0; is<m_src_ncontours; is++) {
      float norm_s = 0;
      for (int it=0; it<m_tar_ncontours[kflowdir]; it++) {
	norm_s += m_score_matrix[kflowdir][ is*m_tar_ncontours[kflowdir] + it ];
      }
      if (norm_s>0 ) {
	for (int it=0; it<m_tar_ncontours[kflowdir]; it++) {
	  m_score_matrix[kflowdir][ is*m_tar_ncontours[kflowdir] + it ] /= norm_s;
	}
      }
    }
    
  }
  
float FlowContourMatch::_scoreMatch( const FlowMatchData_t& matchdata ) {
    float score = 0.0;
    int nscores = 0;
    for ( auto const& flow : matchdata.matchingflow_v ) {
      score += 1.0/flow.pred_miss;
      nscores++;
    }
    
    return score;
  }

  void FlowContourMatch::_greedyMatch(const FlowDirection_t kflowdir) {
    // goal is to assign a cluster on the
    // source plane purely to one on the target
    //
    // this function modifies m_score_matrix[kflowdir]
    //
    
    for (int is=0; is<m_src_ncontours; is++) {
      float max_s = -1.0;
      int   idx   = 0;
      for (int it=0; it<m_tar_ncontours[kflowdir]; it++) {
	float score = m_score_matrix[kflowdir][ is*m_tar_ncontours[kflowdir] + it ];
	if ( score>max_s ) {
	  max_s = 0;
	  idx = it;
	}
      }
      if (max_s>0 ) {
	for (int it=0; it<m_tar_ncontours[kflowdir]; it++) {
	  if ( it!=idx )
	    m_score_matrix[kflowdir][ is*m_tar_ncontours[kflowdir] + it ] = 0;
	  else
	    m_score_matrix[kflowdir][ is*m_tar_ncontours[kflowdir] + it ] = 1.0;
	}
      }
    }
    
  }

  void FlowContourMatch::dumpMatchData() {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    for (int i=0; i<2; i++) {
      std::cout << "===Match Data. Direction " << i << " =======" << std::endl;
      for ( auto it : m_flowdata[i] ) {
	std::cout << "[CONTOURS: src(" << it.first[0] << ") -> tar(" << it.first[1] << ")]" << std::endl;
	const FlowMatchData_t& flowdata = it.second;
	std::cout << "  Flow entries " << flowdata.matchingflow_v.size() << std::endl;
	for ( auto const& flow : flowdata.matchingflow_v ) {
	  std::cout << "    " << flow.row << ": " << flow.src_wire
		    << " -> " << flow.tar_wire << "  err=" << flow.pred_miss << std::endl;
	}
      }
    }
  }

  TH2D& FlowContourMatch::plotScoreMatrix(const FlowDirection_t kflowdir) {
    if ( m_plot_scorematrix!=NULL ) {
      delete m_plot_scorematrix[kflowdir];
    }
    m_plot_scorematrix[kflowdir] = new TH2D( "h2d_flowmatch_scorematrix", ";Source Contour;Target Contour",
					     m_src_ncontours, 0, m_src_ncontours,
					     m_tar_ncontours[kflowdir], 0, m_tar_ncontours[kflowdir] );
    for (int is=0; is<m_src_ncontours; is++) {
      for (int it=0; it<m_tar_ncontours[kflowdir]; it++) {
	m_plot_scorematrix[kflowdir]->SetBinContent( is+1, it+1, m_score_matrix[kflowdir][ is*m_tar_ncontours[kflowdir] + it ] );
      }
    }

    m_plot_scorematrix[kflowdir]->SetMaximum(1.0);
    m_plot_scorematrix[kflowdir]->SetMinimum(0.0);	
    
    return *(m_plot_scorematrix[kflowdir]);
  }

  void FlowContourMatch::_make3Dhits( const larlite::event_hit& hit_v,
				      const larcv::Image2D& srcimg_adc,
				      const larcv::Image2D& tar_adc,
				      const int src_plane,
				      const int tar_plane,
				      const float threshold,
				      std::vector<HitFlowData_t>& hit2flowdata,
				      const FlowDirection_t kflowdir ) {

    // make3Dhits
    // turn flow predictions and contour matches into 3D hits
    //
    // inputs
    // ------
    // hit_v: vector of hits found on the different planes
    // srcimg_adc: source ADC image
    // tar_adc: target ADC image
    // src_plane: plane ID of source image
    // tar_plane: plane ID of target image
    // threshold: ADC threshold used to determine if landed on pixel w/ charge
    //
    // implicit input from class members
    // ----------------------------------
    // m_score_matrix: tells us which target contours most strongly associated to cluster (from scoreMatches)
    // m_flowdata: data of what clusters associated to which (createMatchData)
    // m_src_img2ctrindex: array mapping (row,col) to contour index for source image (createMatchData)
    // m_tar_img2ctrindex: array mapping (row,col) to contour index for target image (createMatchData)
    //
    // outputs
    // -------
    // hit2flowdata: this vector is constant for a whole event image, and gets updated for each sub-image.
    //   it stores where we think we should put the target hit -- note this might be a modification of the
    //   the initial flow prediction
    //
    // description of method
    // ---------------------
    // 1) loop over hits
    // 2) for each hit, determine if it is within a source contour using m_src_img2ctrindex
    // 3) using flow image, determine target pixel
    // 4) determine match quality
    // 5) depending on match quality, determine matching target column
    // 6) convert source and target columns into wires, row into ticks
    // 7) turn that information, make 3D hit position (X-axis relative to trigger time)

    int _verbosity = 1;
    
    if ( hit2flowdata.size()!=hit_v.size() ) {
      if (_verbosity>0 )
	std::cout << "hit2flowdata vector is reset" << std::endl;
      hit2flowdata.clear();
      hit2flowdata.resize( hit_v.size() );
    }
    
    // for (int hitidx=0; hitidx<(int)hit_v.size(); hitidx++) {
    //   // get hit for this index
    //   const larlite::hit& ahit = hit_v[hitidx];
    //   std::cout << "hit[" << hitidx << "]: " << ahit.StartTick() << std::endl;
    // }
    // std::cin.get();

    int numscoredhits = 0;
    int not_on_src_plane = 0;
    int not_in_bounds = 0;
    int src_col_nomatch = 0;
    int no_contour = 0;
    
    for (int hitidx=0; hitidx<(int)hit_v.size(); hitidx++) {
      // get hit for this index
      const larlite::hit& ahit = hit_v[hitidx];

      // is this on the source plane? if not, skip
      if ( src_plane!=(int)ahit.WireID().planeID().Plane ) {
	not_on_src_plane++;
	continue;
      }

      // is it in the image (crop)

      // time limits
      int hit_tstart = 2400+(int)ahit.StartTick();
      int hit_tend   = 2400+(int)ahit.EndTick();
      if ( hit_tend < m_srcimg_meta->min_y() || hit_tstart >= m_srcimg_meta->max_y() ) {
	// if not within the tick bounds, skip
	not_in_bounds++;
	continue;
      }
      if ( hit_tend >= m_srcimg_meta->max_y() )
	hit_tend = m_srcimg_meta->pos_y( m_srcimg_meta->rows()-1 );
      if ( hit_tstart < m_srcimg_meta->min_y() )
	hit_tstart = m_srcimg_meta->min_y();
      
      // wire bounds
      int wire = (int)ahit.WireID().Wire;
      if ( wire < m_srcimg_meta->min_x() || wire >= m_srcimg_meta->max_x() ) {
	// if not within wire bounds, skip
	not_in_bounds++;
	continue;
      }

      
      // translate wire and ticks into col and row
      int wirecol  = m_srcimg_meta->col( wire );
      int rowstart = m_srcimg_meta->row( hit_tstart );
      int rowend   = m_srcimg_meta->row( hit_tend );

      if ( _verbosity>1 )
	std::cout << "---------------------------------------" << std::endl;
      if ( _verbosity>1 )
	std::cout << "valid hit: " << hitidx << " wire=" << wire << " wirecol=" << wirecol 
		  << " tickrange=[" << hit_tstart << "," << hit_tend << "]"
		  << " rowrange=["  << rowstart << "," << rowend << "]"
		  << std::endl;

      // ok our hit is in the image. match the hit to a contour
      // we loop through source image contours and check if any of their pixels are inside the hit tick range.
      bool foundcontour = false;
      int  sourcecontourindex = -1;
      for ( auto const& it_ctr : m_src_targets[kflowdir] ) {
	int src_ctridx = it_ctr.first; // source image contour index
	const ContourTargets_t& ctrtargets = it_ctr.second; // list of src and target pixels

	// loop over pixels within this source contour, find those that fit within given hit above
	for ( auto const& pixinfo : ctrtargets ) {
	  //std::cout << "  srcctr[" << src_ctridx << "] (" << pixinfo.row << "," << pixinfo.srccol << ")" << std::endl;
	  
	  if ( wirecol!=pixinfo.srccol ) {
	    //std::cout << "source contour pixel does not match, skip. wirecol=" << wirecol << " pixinfo.srccol=" << pixinfo.srccol << std::endl;
	    src_col_nomatch++;
	    continue;
	  }
	  if ( rowstart <= pixinfo.row && pixinfo.row <= rowend ) {
	    // found the overlap
	    sourcecontourindex = src_ctridx;
	    foundcontour = true;

	    if ( _verbosity>1 )
	      std::cout << "source contour pixel within hit. source contour=" << src_ctridx
			<< " source(r,c)=(" << pixinfo.row << "," << pixinfo.srccol << ")"
			<< " w/ flow to target(r,c)=(" << pixinfo.row << "," << pixinfo.col << ")"
			<< std::endl;
	    
	    // update the src/target pix from larflow based on
	    // 1) adc value of source pixel
	    // 2) match quality
	    float src_adc    = srcimg_adc.pixel( pixinfo.row, pixinfo.srccol );

	    // match quality
	    // 1-best) target pixel inside (primary?) contour and on charge
	    // 2) target pixel inside contour -- find closest charge inside contour
	    // 3) target pixel outside contour -- find closest charge inside best-score contour
	    // x) can provide a null result too. when truth is in dead region, the resulting hit can often be poor.
	    //    eventually we would use matchability here. alternatively, during tracking pass, we can infer if
	    //    next step is inside dead region. if flow prediction at this point is wildly off, then we throw
	    //    out hit. but there is nothing one can do for that approach here.
	    
	    // get the data for this hit
	    HitFlowData_t& hitdata = hit2flowdata[hitidx];

	    // so we first calculate the hit quality, using the flow, src image, target image,
	    //  and src-target contour matches (via score matrix)
	    int matchquality = -1; // << starting value
	    int pastquality  = hitdata.matchquality;
	    int dist2center  = std::abs( pixinfo.srccol - m_srcimg_meta->cols()/2 );
	    int dist2charge  = -1;
	    if ( _verbosity>1 )
	      std::cout << "  -- past hitquality=" << pastquality << std::endl;
	    
	    // does target point to charge? look within a range of wires
	    int tarcolmin = pixinfo.col-kTargetChargeRadius;
	    int tarcolmax = pixinfo.col+kTargetChargeRadius;
	    if ( tarcolmin<0 )
	      tarcolmin = 0;
	    if ( tarcolmax>=(int)m_tarimg_meta[kflowdir]->cols() )
	      tarcolmax = (int)m_tarimg_meta[kflowdir]->cols() - 1;

	    // we look for the peak adc value between tarcolmin and tarcolmax
	    float target_adc = 0; // <<
	    int   target_col = pixinfo.col; // << default, set to flow prediction
	    bool oncharge = false;
	    for (int tarcol=tarcolmin; tarcol<=tarcolmax; tarcol++) {
	      float tadc = tar_adc.pixel( pixinfo.row, tarcol );
	      if ( target_adc<tadc ) {
		target_adc = tadc;
		target_col = tarcol;
	      }
	    }
	    if ( target_adc>threshold ) {
	      oncharge = true;
	    }

	    // is this pixel in a (matched) contour
	    bool incontour = false;
	    int target_contour = m_tar_img2ctrindex[kflowdir][ int(pixinfo.row*m_tarimg_meta[kflowdir]->cols() + target_col) ];
	    if ( target_contour>0 ) {
	      incontour = true;
	    }
	    
	    // CONDITIONS FOR QUALITY LEVEL
	    if ( incontour && oncharge ) {
	      // quality level 1: oncharge and incontour	      
	      matchquality = 1;
	      dist2charge = 0;
	      if ( _verbosity>1 )	      
		std::cout << "  -- hit quality=1 " << " past(" << pastquality << ") " << std::endl;
	    }//end of quality 1 condition
	    else if ( incontour && !oncharge && (pastquality<0 || pastquality>=2) ) {
	      // quality level 2: incontour but not on charge
	      
	      // we calculate the matched pixel for this case if in the past we didnt get a better match
	      matchquality = 2;
	      if ( _verbosity>1 )	      
		std::cout << "  -- hit quality=2 " << " past(" << pastquality << ") " << std::endl;
		      
	      // we look for the closest pixel inside the contour that has charge, and that is what we match to
	      int possearch_col = target_col+1;
	      if ( possearch_col<0 )
		possearch_col = 0;
	      if ( possearch_col>=m_tarimg_meta[kflowdir]->cols() )
		possearch_col = m_tarimg_meta[kflowdir]->cols()-1;	      
	      
	      while ( possearch_col<(int)m_tarimg_meta[kflowdir]->cols() && possearch_col-target_col<30 && possearch_col>target_col) {
		if ( m_tar_img2ctrindex[kflowdir][ int(pixinfo.row*m_tarimg_meta[kflowdir]->cols() + possearch_col) ]==target_contour ) {
		  // column in contour
		  float tadc = tar_adc.pixel( pixinfo.row, possearch_col );
		  if ( tadc>threshold ) {
		    break;
		  }
		}
		possearch_col++;
	      }
	      
	      int negsearch_col = target_col-1;
	      if ( negsearch_col<0 )
		negsearch_col = 0;
	      if ( negsearch_col>=m_tarimg_meta[kflowdir]->cols() )
		negsearch_col = m_tarimg_meta[kflowdir]->cols()-1;	      

	      while ( negsearch_col>=0 && target_col-negsearch_col<30 && negsearch_col < target_col  ) {	      
		if ( m_tar_img2ctrindex[kflowdir][ int(pixinfo.row*m_tarimg_meta[kflowdir]->cols() + negsearch_col) ]==target_contour ) {
		  // column in contour
		  float tadc = tar_adc.pixel( pixinfo.row, negsearch_col );
		  if ( tadc>threshold ) {
		    break;
		  }
		}
		negsearch_col--;
	      }

	      // bound results
	      if ( negsearch_col<0 )
		negsearch_col = 0;
	      if ( possearch_col>=m_tarimg_meta[kflowdir]->cols() )
		possearch_col = m_tarimg_meta[kflowdir]->cols()-1;
	      
	      int negdist = abs(negsearch_col-target_col);
	      int posdist = abs(possearch_col-target_col);
	      
	      if (  negdist < posdist ) {
		target_col = negsearch_col;
		dist2charge = negdist;
	      }
	      else {
		target_col  = possearch_col;
		dist2charge = posdist;
	      }
	      target_adc = tar_adc.pixel( pixinfo.row, target_col );
	      
	    }// end of quality 2
	    else if ( !incontour && (pastquality<0 || pastquality>=3) ) {
	      // quality 3: out of contour and out of charge
	      // we calculate the matched pixel for this case if in the past we didnt get a better match
	      // doing the best we can
	      matchquality = 3;
	      if ( _verbosity>1 )
		std::cout << "  -- hit quality=3 " << " past(" << pastquality << ") starting search from target_col=" << target_col << std::endl;
	      
	      // check the best matching contour first for charge to match
	      // we search in a neighborhood around the predicted point
	      // we track all the contours we cross into
	      // we use the contour with the best value from m_score_matrix
	      // take the pixel using the best match
	      std::vector<ClosestContourPix_t> matched_contour_list;
	      std::set<int> used_contours;

	      bool found_candidate_contour = false;
	      int possearch_col = target_col+1;
	      if ( possearch_col<0 )
		possearch_col = 0;
	      if ( possearch_col>=m_tarimg_meta[kflowdir]->cols() )
		possearch_col = m_tarimg_meta[kflowdir]->cols()-1;	      
	      while ( possearch_col<(int)m_tarimg_meta[kflowdir]->cols() && possearch_col>target_col && possearch_col-target_col<50 ) {
		float tadc = tar_adc.pixel( pixinfo.row, possearch_col );		
		if ( tadc > threshold )  {
		  int target_contour_idx = m_tar_img2ctrindex[kflowdir][ int(pixinfo.row*m_tarimg_meta[kflowdir]->cols() + possearch_col) ];
		  if ( used_contours.find( target_contour_idx )==used_contours.end() ) {
		    // have not search this contour, provide a match candidate
		    ClosestContourPix_t close_ctr_info;
		    close_ctr_info.ctridx = target_contour_idx;
		    close_ctr_info.dist = abs(possearch_col - target_col);
		    close_ctr_info.col	= possearch_col;
		    close_ctr_info.adc  = tadc;		    
		    close_ctr_info.scorematch = m_score_matrix[kflowdir][ int(src_ctridx*m_tar_ncontours[kflowdir] + target_contour_idx) ];
		    matched_contour_list.push_back( close_ctr_info );
		    used_contours.insert( target_contour_idx );
		    found_candidate_contour = true;
		  }
		}
		possearch_col++;
	      }//end of pos loop
	      int negsearch_col = target_col-1;
	      if ( negsearch_col<0 )
		negsearch_col = 0;
	      if ( negsearch_col>=m_tarimg_meta[kflowdir]->cols() )
		negsearch_col = m_tarimg_meta[kflowdir]->cols()-1;	      
	      while ( negsearch_col>=0 && target_col-negsearch_col<50 && negsearch_col < target_col) {
		float tadc = tar_adc.pixel( pixinfo.row, negsearch_col );
		if (  tadc > threshold )  {
		  int target_contour_idx = m_tar_img2ctrindex[kflowdir][ int(pixinfo.row*m_tarimg_meta[kflowdir]->cols() + negsearch_col) ];
		  if ( used_contours.find( target_contour_idx )==used_contours.end() ) {
		    // have not search this contour, provide a match candidate
		    ClosestContourPix_t close_ctr_info;
		    close_ctr_info.ctridx = target_contour_idx;
		    close_ctr_info.dist = abs(negsearch_col - target_col);
		    close_ctr_info.col	= negsearch_col;
		    close_ctr_info.adc  = tadc;		    
		    close_ctr_info.scorematch = m_score_matrix[kflowdir][ src_ctridx*m_tar_ncontours[kflowdir] + target_contour_idx ];
		    matched_contour_list.push_back( close_ctr_info );
		    used_contours.insert( target_contour_idx );
		    found_candidate_contour = true;
		  }
		}
		negsearch_col--;
	      }//end of neg loop
	      
	      // ok, now we pick the best one! if we found a candidate that is
	      float best_score = 0.0;
	      float best_adc = 0.0;
	      int best_col  = -1;
	      int best_dist = -1;
	      if ( found_candidate_contour ) {
		for ( auto& match : matched_contour_list ) {
		  if ( best_score < match.scorematch ) {
		    best_col   = match.col;
		    best_score = match.scorematch;
		    best_dist  = match.dist;
		    best_adc   = match.adc;
		  }
		}

		target_col  = best_col;
		dist2charge = best_dist;
		target_adc  = best_adc;
	      }
	      else {
		// didn't find a nearby contour 
		target_col = -1; // indicates that no good target found
		matchquality = -1;
	      }
	    }

	    // did we do better?
	    bool update_hitdata = false;
	    if ( matchquality>0 && target_col>=0 ) {
	      // target_col<0 indicates no good target found (happens when true target pixel in dead region)
	      //we found a case where we did better or the same

	      if ( matchquality<pastquality || pastquality<0 )  {
		// we did better. replace the hit
		update_hitdata = true;
	      }
	      else {
		if ( matchquality==1 && hitdata.dist2center>dist2center) {
		  // we decide on this by using the src pixel closest to the center of the y-image
		  update_hitdata = true;
		}
		else if ( matchquality==2 && hitdata.dist2charge>dist2charge ) {
		  // we decide on this by using which flow prediction was closest to the eventual charge match
		  update_hitdata = true;
		}
		else if ( matchquality==3 && hitdata.dist2charge>dist2charge ) {
		  // same criterion as 2
		  update_hitdata = true;
		}
	      }
	      
	    }

	    if ( matchquality<0 && pastquality<0 ) {
	      // matchquality is currenly not set still (didn't find a good match)
	      // we want to set the default to for the hitdata
	      hitdata.maxamp       = 0;
	      hitdata.hitidx       = hitidx;
	      hitdata.srcwire      = m_srcimg_meta->pos_x( pixinfo.srccol );
	      hitdata.targetwire   = m_tarimg_meta[kflowdir]->pos_x( pixinfo.col );
	      hitdata.pixtick      = m_srcimg_meta->pos_y( pixinfo.row );
	      hitdata.matchquality = -1;
	      hitdata.dist2center  = 10000;  // large sentinal value
	      hitdata.dist2charge  = 10000;  // large sentinal value
	      hitdata.src_ctr_idx  = sourcecontourindex;
	      hitdata.tar_ctr_idx  = -1;
	      if ( _verbosity>1 )	      
		std::cout << "  -- set default for unmatched hit [srccol=" << pixinfo.srccol << "] [targetcol=" << pixinfo.col << "]" << std::endl;
	    }
	    
	    
	    if ( update_hitdata ) {
	      if ( _verbosity>1 )	      
		std::cout << "  -- update hit flow data" << std::endl;
	      hitdata.maxamp       = target_adc;
	      hitdata.hitidx       = hitidx;
	      hitdata.srcwire      = m_srcimg_meta->pos_x( pixinfo.srccol );
	      hitdata.targetwire   = m_tarimg_meta[kflowdir]->pos_x( target_col ); //< we used the column we found
	      hitdata.pixtick      = m_srcimg_meta->pos_y( pixinfo.row );
	      hitdata.matchquality = matchquality;
	      hitdata.dist2center  = dist2center;
	      hitdata.dist2charge  = dist2charge;
	      hitdata.src_ctr_idx  = sourcecontourindex;
	      hitdata.tar_ctr_idx  = target_contour;
	    }
	    
	  }//if row within hit row range
	}//end of ctrtargets pixel loop
	
      }//end of loop over list of src-target pairs

      // did it find a source contour?
      if ( !foundcontour ) {
	if ( _verbosity>1 )	
	  std::cout << "pixel src(r,c)=(" << (rowstart+rowend)/2 << "," << wirecol << ") does not have matching source contour." << std::endl;
	no_contour++;
      }
      else {
	numscoredhits++;
      }
      
    }//end of hit index loop
    std::cout << "number of scored hits: " << numscoredhits
	      << " (no_contour=" << no_contour << ","
	      << " src_col_nomatch=" << src_col_nomatch << ","
	      << " not_in_bounds=" << not_in_bounds << ","
	      << " not_on_srcplane=" << not_on_src_plane << ")"
	      << std::endl;

    return;
  }

  std::vector<larlite::larflow3dhit> FlowContourMatch::get3Dhits_1pl( FlowDirection_t flowdir, bool makehits_for_nonmatches ) {
    // we convert the information we've compiled in m_hit2flowdata, which provides our best guess
    //  as the correct source-column + target-column pair.
    //  the objects of this container also contain info about matchquality
    if ( flowdir==kY2U )
      return get3Dhits_1pl( m_plhit2flowdata.Y2U, makehits_for_nonmatches );
    else if ( flowdir==kY2V )
      return get3Dhits_1pl( m_plhit2flowdata.Y2U, makehits_for_nonmatches );
    else
      throw std::runtime_error("should not get here");
  }
  
  std::vector<larlite::larflow3dhit> FlowContourMatch::get3Dhits_1pl( const std::vector<HitFlowData_t>& hit2flowdata, bool makehits_for_nonmatches ) {

    // now we have, in principle, the best/modified flow prediction for hits that land on flow predictions
    // we can make 3D hits!
    std::vector<larlite::larflow3dhit> output_hit3d_v;
    for (int hitidx=0; hitidx<(int)hit2flowdata.size(); hitidx++) {
      const HitFlowData_t& hitdata = hit2flowdata[ hitidx ];
      if (  hitdata.matchquality<=0 && !makehits_for_nonmatches ) {
	//std::cout << "no good match for hitidx=" << hitidx << ", skip this hit if we haven't set the makehits_for_nonmatches flag" << std::endl;
    	continue;
      }
      // otherwise make a hit
      larlite::larflow3dhit flowhit;
      flowhit.idxhit     = hitidx;
      flowhit.tick       = hitdata.pixtick;
      flowhit.srcwire    = hitdata.srcwire;
      flowhit.targetwire[0] = hitdata.targetwire;
      flowhit[0]          = hitdata.X[0];
      flowhit[1]          = hitdata.X[1];
      flowhit[2]          = hitdata.X[2];
      flowhit.dy            = -1.;
      flowhit.dz            = -1.;
      flowhit.consistency3d=larlite::larflow3dhit::kNoValue;
      flowhit.X_truth[0]    = hitdata.X_truth[0];
      flowhit.X_truth[1]    = hitdata.X_truth[1];
      flowhit.X_truth[2]    = hitdata.X_truth[2];
      flowhit.trackid       = hitdata.trackid;
      flowhit.dWall         = hitdata.dWall;
      switch ( hitdata.matchquality ) {
      case 1:
	flowhit.matchquality=larlite::larflow3dhit::kQandCmatch;
	break;
      case 2:
	flowhit.matchquality=larlite::larflow3dhit::kCmatch;
	break;
      case 3:
	flowhit.matchquality=larlite::larflow3dhit::kClosestC;
	break;
      default:
	flowhit.matchquality=larlite::larflow3dhit::kNoMatch;
	break;
      }
      switch ( hitdata.truthflag ) {
      case 0:
	flowhit.truthflag=larlite::larflow3dhit::kNoTruthMatch;
	break;
      case 1:
	flowhit.truthflag=larlite::larflow3dhit::kOnTrack;
	break;
      case 2:
	flowhit.truthflag=larlite::larflow3dhit::kOnSmear;
	break;
      default:
	flowhit.truthflag=larlite::larflow3dhit::kUnknown;
	break;
      }
      output_hit3d_v.emplace_back( flowhit );
    }

    return output_hit3d_v;    
  }

  std::vector<larlite::larflow3dhit> FlowContourMatch::get3Dhits_2pl( bool makehits_for_nonmatches, bool require_3Dconsistency ) {
    // we convert the information we've compiled in m_hit2flowdata, which provides our best guess
    //  as the correct source-column + target-column pair.
    //  the objects of this container also contain info about matchquality
      return get3Dhits_2pl( m_plhit2flowdata, makehits_for_nonmatches, require_3Dconsistency );
  }
  
  std::vector<larlite::larflow3dhit> FlowContourMatch::get3Dhits_2pl( const PlaneHitFlowData_t& plhit2flowdata, bool makehits_for_nonmatches, bool require_3Dconsistency ) {

    // now we have, in principle, the best/modified flow prediction for hits that land on flow predictions
    // here we choose which of the two predictions to keep (for each hit)
    std::vector<larlite::larflow3dhit> output_hit3d_v;

    // case 1, we only ran one flow direction
    if ( !plhit2flowdata.ranY2U || !plhit2flowdata.ranY2V ) {

      // determine which one we ran
      FlowDirection_t flowdir = ( plhit2flowdata.ranY2U ) ? kY2U : kY2V;
      const std::vector<HitFlowData_t>& hit2flowdata = ( plhit2flowdata.ranY2U ) ? plhit2flowdata.Y2U : plhit2flowdata.Y2V;

      for (int hitidx=0; hitidx<(int)hit2flowdata.size(); hitidx++) {
	const HitFlowData_t& hitdata = hit2flowdata[ hitidx ];
	if (  hitdata.matchquality<=0 && !makehits_for_nonmatches ) {
	  //std::cout << "no good match for hitidx=" << hitidx << ", skip this hit if we haven't set the makehits_for_nonmatches flag" << std::endl;
	  continue;
	}
	// otherwise make a hit
	larlite::larflow3dhit flowhit;
	flowhit.resize(3,-1);
	flowhit.idxhit        = hitidx;
	flowhit.tick          = hitdata.pixtick;
	flowhit.srcwire       = hitdata.srcwire;
	flowhit.targetwire[0] = hitdata.targetwire;
	flowhit[0]            = hitdata.X[0];
	flowhit[1]            = hitdata.X[1];
	flowhit[2]            = hitdata.X[2];
	flowhit.dy            = -1; // no 3d consistency calc
	flowhit.dz            = -1; // no 3d consistency calc
	flowhit.track_score   = hitdata.track_score;
	flowhit.shower_score  = hitdata.shower_score;
	flowhit.endpt_score   = hitdata.endpt_score;
	flowhit.renormed_track_score   = hitdata.renormed_track_score;
	flowhit.renormed_shower_score  = hitdata.renormed_shower_score;
	flowhit.src_infill    = (unsigned short)(hitdata.src_infill);
	flowhit.tar_infill[0] = (unsigned short)(hitdata.tar_infill);
	flowhit.X_truth[0]    = hitdata.X_truth[0];
	flowhit.X_truth[1]    = hitdata.X_truth[1];
	flowhit.X_truth[2]    = hitdata.X_truth[2];
	flowhit.trackid       = hitdata.trackid;
	flowhit.dWall         = hitdata.dWall;
	switch ( hitdata.matchquality ) {
	case 1:
	  flowhit.matchquality=larlite::larflow3dhit::kQandCmatch;
	  break;
	case 2:
	  flowhit.matchquality=larlite::larflow3dhit::kCmatch;
	  break;
	case 3:
	  flowhit.matchquality=larlite::larflow3dhit::kClosestC;
	  break;
	default:
	  flowhit.matchquality=larlite::larflow3dhit::kNoMatch;
	  break;
	}
	flowhit.consistency3d=larlite::larflow3dhit::kNoValue;
	switch ( hitdata.truthflag ) {
	case 0:
	  flowhit.truthflag=larlite::larflow3dhit::kNoTruthMatch;
	  break;
	case 1:
	  flowhit.truthflag=larlite::larflow3dhit::kOnTrack;
	  break;
	case 2:
	  flowhit.truthflag=larlite::larflow3dhit::kOnSmear;
	  break;
	default:
	  flowhit.truthflag=larlite::larflow3dhit::kUnknown;
	  break;
	}
	output_hit3d_v.emplace_back( flowhit );
      }
    }// end of only 1 ran
    //case 2: we ran Y2U and Y2V
    else if(plhit2flowdata.ranY2U && plhit2flowdata.ranY2V){
      std::cout << "Picking hits using 2-flow information" << std::endl;
      const std::vector<HitFlowData_t>& hit2flowdata_y2u = plhit2flowdata.Y2U;
      const std::vector<HitFlowData_t>& hit2flowdata_y2v = plhit2flowdata.Y2V;
      //we only loop over length of one, they should be same anyway
      for (int hitidx=0; hitidx<(int)hit2flowdata_y2u.size(); hitidx++) {
	const HitFlowData_t& hitdata0 = hit2flowdata_y2u[ hitidx ];
	const HitFlowData_t& hitdata1 = hit2flowdata_y2v[ hitidx ];
	HitFlowData_t hitdata;
	//select here
	larlite::larflow3dhit::FlowDirection_t used_dir = larlite::larflow3dhit::kY2U;
	if(require_3Dconsistency && plhit2flowdata.consistency3d[ hitidx ]>3) continue; //if require_3Dconsistency, skip if no consistency
	if(hitdata0.matchquality<0 && hitdata1.matchquality<0 && !makehits_for_nonmatches ) {
	  //std::cout << "no good match for hitidx=" << hitidx << ", skip this hit if we haven't set the makehits_for_nonmatches flag" << std::endl;
	  continue;
	}
	else if(hitdata0.matchquality<0 && hitdata1.matchquality<0 && makehits_for_nonmatches) {
	  //neither has a match. pick the one that does not land on infill
	  if(hitdata0.tar_infill && !hitdata1.tar_infill){
	    hitdata = hitdata1;
	    used_dir = larlite::larflow3dhit::kY2V;
	  }
	  else if(!hitdata0.tar_infill && hitdata1.tar_infill){
	    hitdata = hitdata0;
	  }
	  else{
	    hitdata = hitdata0; //neither has a match, same infill, pick y2u
	  }
	}
	else if(hitdata0.matchquality>=0 && hitdata1.matchquality>=0 && hitdata0.matchquality==hitdata1.matchquality) {
	  //same quality, pick the one that does not land on infill
	  if(hitdata0.tar_infill && !hitdata1.tar_infill){
	    hitdata = hitdata1;
	    used_dir = larlite::larflow3dhit::kY2V;
	  }
	  else if(hitdata0.tar_infill && !hitdata1.tar_infill){
	    hitdata = hitdata0;
	  }
	  else{
	    hitdata = hitdata0; //same infill, pick y2u
	  }
	}
	else if((hitdata0.matchquality<0 && hitdata1.matchquality>=0)
		|| (hitdata0.matchquality>=0 && hitdata1.matchquality>=0 && hitdata0.matchquality < hitdata1.matchquality))  {
	  hitdata = hitdata1; //y2v better matchqual
	  used_dir = larlite::larflow3dhit::kY2V;
	}
	else if((hitdata1.matchquality<0 && hitdata0.matchquality>=0) 
		|| (hitdata1.matchquality>=0 && hitdata0.matchquality>=0 && hitdata1.matchquality < hitdata0.matchquality))  {
	  hitdata = hitdata0; //y2u better matchqual
	}
	else{
	  hitdata = hitdata0; 
	}
	//make a hit
	larlite::larflow3dhit flowhit;
	flowhit.resize(3,-1);
	flowhit.idxhit     = hitidx;
	flowhit.tick       = hitdata.pixtick;
	flowhit.srcwire    = hitdata.srcwire;
	flowhit.flowdir    = used_dir;
	flowhit.targetwire[0] = hitdata0.targetwire;
	flowhit.targetwire[1] = hitdata1.targetwire;
	flowhit[0]            = hitdata.X[0];
	flowhit[1]            = hitdata.X[1];
	flowhit[2]            = hitdata.X[2];	
	flowhit.dy            = plhit2flowdata.dy[ hitidx ];
	flowhit.dz            = plhit2flowdata.dz[ hitidx ];
	flowhit.track_score   = hitdata.track_score;
	flowhit.shower_score  = hitdata.shower_score;
	flowhit.endpt_score   = hitdata.endpt_score;
	flowhit.renormed_track_score   = hitdata.renormed_track_score;
	flowhit.renormed_shower_score  = hitdata.renormed_shower_score;
	flowhit.src_infill    = (unsigned short)(hitdata.src_infill);
	flowhit.tar_infill[0] = (unsigned short)(hitdata0.tar_infill);
	flowhit.tar_infill[1] = (unsigned short)(hitdata1.tar_infill);
	flowhit.X_truth[0]    = hitdata.X_truth[0];
	flowhit.X_truth[1]    = hitdata.X_truth[1];
	flowhit.X_truth[2]    = hitdata.X_truth[2];
	flowhit.trackid       = hitdata.trackid;
	flowhit.dWall         = hitdata.dWall;
	//std::cout <<"hitdata dWall "<< hitdata.dWall <<" flowhit dWall " << flowhit.dWall << std::endl;
	switch ( hitdata.matchquality ) {
	case 1:
	  flowhit.matchquality=larlite::larflow3dhit::kQandCmatch;
	  break;
	case 2:
	  flowhit.matchquality=larlite::larflow3dhit::kCmatch;
	  break;
	case 3:
	  flowhit.matchquality=larlite::larflow3dhit::kClosestC;
	  break;
	default:
	  flowhit.matchquality=larlite::larflow3dhit::kNoMatch;
	  break;
	}
	switch ( plhit2flowdata.consistency3d[ hitidx ] ) {
	case 0:
	  flowhit.consistency3d=larlite::larflow3dhit::kIn5mm;
	  break;
	case 1:
	  flowhit.consistency3d=larlite::larflow3dhit::kIn10mm;
	  break;
	case 2:
	  flowhit.consistency3d=larlite::larflow3dhit::kIn50mm;
	  break;
	case 3:
	  flowhit.consistency3d=larlite::larflow3dhit::kOut50mm;
	  break;
	default:
	  flowhit.consistency3d=larlite::larflow3dhit::kNoValue;
	  break;
	}
	switch ( hitdata.truthflag ) {
	case 0:
	  flowhit.truthflag=larlite::larflow3dhit::kNoTruthMatch;
	  break;
	case 1:
	  flowhit.truthflag=larlite::larflow3dhit::kOnTrack;
	  break;
	case 2:
	  flowhit.truthflag=larlite::larflow3dhit::kOnSmear;
	  break;
	default:
	  flowhit.truthflag=larlite::larflow3dhit::kUnknown;
	  break;
	}
	output_hit3d_v.emplace_back( flowhit );
      }
    }// end of both ran

    return output_hit3d_v;    
  }

  std::vector<larcv::Image2D> FlowContourMatch::makeStitchedFlowImages( const std::vector<larcv::Image2D>& img_v ) {
    std::vector< larlite::larflow3dhit > hit_v = get3Dhits_2pl( false, false );
    std::vector<larcv::Image2D> outimg_v;

    larcv::Image2D y2uimg( img_v[0].meta() );
    larcv::Image2D y2vimg( img_v[1].meta() );
    larcv::Image2D srcimg( img_v[2].meta() );
    y2uimg.paint(0.0);
    y2vimg.paint(0.0);
    srcimg.paint(0.0);

    std::cout << "[FlowContourMatch::makeStichedFlowImages] start" << std::endl;
    for (int i=0; i<3; i++)
      std::cout << "  " <<img_v[i].meta().dump() << std::endl;
	
    for ( auto& hit : hit_v ) {
      if ( hit[0]==hit[2] && hit[0]==hit[1] && hit[0]==-1 ) continue; // bad hit
      if ( hit.tick >=img_v[2].meta().min_y() && hit.tick<img_v[2].meta().max_y()
	   && hit.srcwire>=img_v[2].meta().min_x() && hit.srcwire<img_v[2].meta().max_x() ) {
	
	int row = img_v[2].meta().row( hit.tick );
	int col = img_v[2].meta().col( hit.srcwire );
	//std::cout << "hit: src(" << row << "," << col << ")";
	
	if ( hit.flowdir==larlite::larflow3dhit::kY2U )  {
	  srcimg.set_pixel( row, col, hit.targetwire[0]-hit.srcwire );
	  if ( img_v[0].meta().min_x()<=hit.targetwire[kY2U] && img_v[0].meta().max_x()>hit.targetwire[kY2U] ) {
	    y2uimg.set_pixel( row, img_v[0].meta().col(hit.targetwire[0]), img_v[2].pixel(row,col) );
	    //std::cout << " target(" << row << "," << img_v[0].meta().col(hit.targetwire[0]) << ")";
	  }	  
	}
	else {
	  srcimg.set_pixel( row, col, hit.targetwire[1]-hit.srcwire );
	  if ( img_v[1].meta().min_x()<=hit.targetwire[kY2V] && img_v[1].meta().max_x()>hit.targetwire[kY2V] ) {
	    y2vimg.set_pixel( row, img_v[1].meta().col(hit.targetwire[kY2V]), img_v[2].pixel(row,col) );
	    //std::cout << " target(" << row << "," << img_v[1].meta().col(hit.targetwire[1]) << ")";
	  }	  
	}
	//std::cout << std::endl;
	
      }// if inside image
    }// hit loop

    outimg_v.emplace_back( std::move(y2uimg) );	
    outimg_v.emplace_back( std::move(y2vimg) );	
    outimg_v.emplace_back( std::move(srcimg) );
    
    return outimg_v;
  }

}
