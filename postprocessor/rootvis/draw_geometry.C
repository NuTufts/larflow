R__ADD_LIBRARY_PATH($LARLITE_LIBDIR)
R__LOAD_LIBRARY(libLArLite_DataFormat.so)
R__ADD_INCLUDE_PATH($LARLITE_COREDIR)

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"

void draw_geometry(std::string inputfile, std::string tree,int event, bool userootfile)
{
  
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

   //load in larflowhits
   larlite::storage_manager io( larlite::storage_manager::kREAD );
   io.add_in_filename( inputfile );
   io.open();   
   io.go_to( event );
   larlite::event_larflow3dhit& ev_hits = *((larlite::event_larflow3dhit*)io.get_data( larlite::data::kLArFlow3DHit, tree ));
   
   Int_t npoints = ev_hits.size();
   TEveElement* parent= 0;
   TEvePointSet* ps = new TEvePointSet();
   ps->SetOwnIds(kTRUE);

   for(Int_t i = 0; i<npoints; i++)
     {
       ps->SetNextPoint(ev_hits.at(i).at(0)-130.0, ev_hits.at(i).at(1), ev_hits.at(i).at(2)-524.0);
       ps->SetPointId(new TNamed(Form("Point %d", i), ""));
     }

   ps->SetMarkerColor(TMath::Nint(r.Uniform(2, 9)));
   ps->SetMarkerSize(0.3);
   ps->SetMarkerStyle(20);

   if (parent)
     {
       parent->AddElement(ps);
     }
   else
     {
       gEve->AddElement(ps);
       gEve->Redraw3D();
     }

   TGLViewer *v = gEve->GetDefaultGLViewer();
   //v->GetClipSet()->SetClipType(TGLClip::EType(1));
   v->ColorSet().Background().SetColor(kMagenta+4);
   v->SetGuideState(TGLUtil::kAxesEdge, kTRUE, kFALSE, 0);
   v->RefreshPadEditor(v);
   //v->CurrentCamera().RotateRad(-1.2, 0.5);
   v->DoDraw();
  
}
