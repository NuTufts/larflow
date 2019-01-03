R__ADD_LIBRARY_PATH($LARLITE_LIBDIR)
R__LOAD_LIBRARY(libLArLite_DataFormat.so)
R__ADD_INCLUDE_PATH($LARLITE_COREDIR)

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"

void draw_geometry(std::string inputfile, std::string tree,int event, bool userootfile, std::string color="default", bool drawpca=false)
{
  //-----------------------//
  // color:
  // default: all white
  // ssnet: color by ssnet label
  // cluster: color by cluster
  //-----------------------//
  // if color=="cluster" && drawpca==true:
  // load & draw main pca axis
  //------------------------//
  
  TEveManager::Create();

   TRandom r(0);

   TFile::SetCacheFileDir(".");
   if(userootfile){
     gGeoManager = gEve->GetGeometry("uboone_simplifiedCryo.root");
   }
   else{
     gGeoManager = gEve->GetGeometry("simplified_uboone.gdml");
   }
   gGeoManager->DefaultColors();
   gGeoManager->GetVolume("volSteelVessel")->InvisibleAll();
   gGeoManager->GetVolume("volTPCActive")->SetTransparency(50);
   gGeoManager->GetVolume("volCathodePlate")->InvisibleAll();
   
   TEveGeoTopNode* box = new TEveGeoTopNode(gGeoManager,gGeoManager->GetTopNode());
   box->SetVisLevel(6);
   gEve->AddGlobalElement(box);
   
   gEve->FullRedraw3D(kTRUE);

   //load data
   larlite::storage_manager io( larlite::storage_manager::kREAD );
   io.add_in_filename( inputfile );
   io.open();   
   io.go_to( event );
   larlite::event_larflow3dhit* ev_hits;
   larlite::event_larflowcluster* ev_clusters;
   larlite::event_pcaxis* ev_pca;
   if(color=="default" || color=="ssnet")
     ev_hits = (larlite::event_larflow3dhit*)io.get_data( larlite::data::kLArFlow3DHit, tree );
   //load larflowcluster
   if(color=="cluster"){
     ev_clusters = (larlite::event_larflowcluster*)io.get_data( larlite::data::kLArFlowCluster, tree );
     if(drawpca){
       //load pcaxis
       ev_pca = (larlite::event_pcaxis*)io.get_data( larlite::data::kPCAxis, tree);
     }
   }

   //default
   if(color=="default"){
     Int_t npoints = ev_hits->size();
     TEvePointSet* ps = new TEvePointSet();
     ps->SetOwnIds(kTRUE);
     
     for(Int_t i = 0; i<npoints; i++){
       ps->SetNextPoint(ev_hits->at(i).at(0)-130.0, ev_hits->at(i).at(1), ev_hits->at(i).at(2)-524.0);
       ps->SetPointId(new TNamed(Form("Point %d", i), ""));
     }
     
     ps->SetMarkerColor(kWhite);
     ps->SetMarkerSize(0.3);
     ps->SetMarkerStyle(20);
     
     gEve->AddElement(ps);
     gEve->Redraw3D();
   }

   //if color by ssnet
   if(color=="ssnet"){
     Int_t npoints = ev_hits->size();
     TEvePointSet* ps[4]; //trk, shwr, endpt, other
     for(int i=0; i<4; i++){
       ps[i]= new TEvePointSet();
       ps[i]->SetOwnIds(kTRUE);
       ps[i]->SetMarkerSize(0.3);
       ps[i]->SetMarkerStyle(20);

     }
     for(Int_t i = 0; i<npoints; i++){
       if(ev_hits->at(i).endpt_score>0.8){
	 ps[2]->SetNextPoint(ev_hits->at(i).at(0)-130.0, ev_hits->at(i).at(1), ev_hits->at(i).at(2)-524.0);
	 ps[2]->SetPointId(new TNamed(Form("Point %d", i), ""));
       }
       else if(ev_hits->at(i).renormed_track_score>0.5){
	 ps[0]->SetNextPoint(ev_hits->at(i).at(0)-130.0, ev_hits->at(i).at(1), ev_hits->at(i).at(2)-524.0);
	 ps[0]->SetPointId(new TNamed(Form("Point %d", i), ""));
       }
       else if(ev_hits->at(i).renormed_shower_score>0.5){
	 ps[1]->SetNextPoint(ev_hits->at(i).at(0)-130.0, ev_hits->at(i).at(1), ev_hits->at(i).at(2)-524.0);
	 ps[1]->SetPointId(new TNamed(Form("Point %d", i), ""));
       }
       else{
	 ps[3]->SetNextPoint(ev_hits->at(i).at(0)-130.0, ev_hits->at(i).at(1), ev_hits->at(i).at(2)-524.0);
	 ps[3]->SetPointId(new TNamed(Form("Point %d", i), ""));
       }

     }
     
     ps[0]->SetMarkerColor(kWhite);
     ps[1]->SetMarkerColor(kGreen);
     ps[2]->SetMarkerColor(kRed);
     ps[3]->SetMarkerColor(kBlue);
     
     gEve->AddElement(ps[0]);
     gEve->AddElement(ps[1]);
     gEve->AddElement(ps[2]);
     gEve->AddElement(ps[3]);
     gEve->Redraw3D();
   }

   //if color by cluster 
   if(color=="cluster"){
     Int_t nclust = ev_clusters->size();

     TEvePointSet* ps[nclust];
     TEveStraightLineSet* pca = new TEveStraightLineSet();

     for(int k=0; k<nclust; k++){
       ps[k]= new TEvePointSet();
       ps[k]->SetOwnIds(kTRUE);

       Int_t npoints = ev_clusters->at(k).size();
       for(Int_t i = 0; i<npoints; i++){
	 ps[k]->SetNextPoint(ev_clusters->at(k).at(i).at(0)-130.0, ev_clusters->at(k).at(i).at(1), ev_clusters->at(k).at(i).at(2)-524.0);
	 ps[k]->SetPointId(new TNamed(Form("Point %d", i), ""));
       }
       
       ps[k]->SetMarkerColor(TMath::Nint(r.Uniform(2, nclust)));
       //ps[k]->SetMarkerColor(nclust+1);
       ps[k]->SetMarkerSize(0.3);
       ps[k]->SetMarkerStyle(20);
       
       gEve->AddElement(ps[k]);
       gEve->Redraw3D();

       // pca of all clusters, except last one, which is unassigned hits
       if(drawpca && k<nclust-1){
	 larlite::pcaxis pcaxis = ev_pca->at(k);
	 const double* eigval = pcaxis.getEigenValues();
	 const double*  meanpos = pcaxis.getAvePosition();
	 larlite::pcaxis::EigenVectors eigvec = pcaxis.getEigenVectors();
	 double eiglen = sqrt(eigval[0]);
	 //start,  end of primary axis
	 float x1=meanpos[0]-eigvec[0][0]*2*eiglen-130.0;
	 float x2=meanpos[0]+eigvec[0][0]*2*eiglen-130.0;
	 float y1=meanpos[1]-eigvec[1][0]*2*eiglen;
	 float y2=meanpos[1]+eigvec[1][0]*2*eiglen; 
	 float z1=meanpos[2]-eigvec[2][0]*2*eiglen-524.0;
	 float z2=meanpos[2]+eigvec[2][0]*2*eiglen-524.0;
	 pca->AddLine(x1,y1,z1,x2,y2,z2);

	 pca->SetLineColor(kBlack);
	 pca->SetLineWidth(3);
       }
     }
     
     gEve->AddElement(pca);
     gEve->Redraw3D();

   }
   TGLViewer *v = gEve->GetDefaultGLViewer();
   //v->GetClipSet()->SetClipType(TGLClip::EType(1));
   v->ColorSet().Background().SetColor(kMagenta+4);
   v->SetGuideState(TGLUtil::kAxesEdge, kTRUE, kFALSE, 0);
   v->RefreshPadEditor(v);
   v->CurrentCamera().RotateRad(-0.6, -2.5);
   v->DoDraw();
  
}
