
void make_simplified_uboone(bool use_gdml=false) {
  gSystem->Load("libGeom");
  if(use_gdml) gSystem->Load("libGdml");
  
  new TGeoManager("cryo", "cryostat");

  //TGeoElementTable *table = gGeoManager->GetElementTable();
  /**** Define materials and mixtures ***/
  TGeoMaterial *vac = new TGeoMaterial("Vacuum", 1.0079, 0, 1.e-25 );
  TGeoMixture *AL = new TGeoMixture("ALUMINIUM_Al",1, 2.699 );
  AL->DefineElement(0, 26.98,13.0, 1.000);
  TGeoMixture *SI = new TGeoMixture("SILICON_Si",1, 2.33 );
  SI->DefineElement(0, 28.09,14.0, 1.000);
  TGeoMixture *SiO2 = new TGeoMixture("SiO2",2, 2.2 );
  SiO2->DefineElement(0, 8, 2);
  SiO2->DefineElement(1, 14,1);
  SiO2->SetRadLen(12.2627, 44.6977);
  TGeoMixture *fib = new TGeoMixture("fibrous_glass",8, 2.74351);
  fib->DefineElement(0,26.98, 13.0, 0.062 );
  fib->DefineElement(1,16.0, 8.0, 0.461 );
  fib->DefineElement(2,40.08, 20.0, 0.160 );
  fib->DefineElement(3,55.84, 26.0, 0.001 );
  fib->DefineElement(4,24.30, 12.0, 0.021 );
  fib->DefineElement(5,22.99, 11.0, 0.007 );
  fib->DefineElement(6,28.09, 14.0, 0.280 );
  fib->DefineElement(7,47.87, 22.0, 0.008 );
  fib->SetRadLen(8.99351, 36.8643);
  TGeoMixture *ST = new TGeoMixture("STEEL_STAINLESS_Fe7Cr2Ni",4,7.93);
  ST->DefineElement(0,12.01, 6.0, 0.001);
  ST->DefineElement(1,52.00, 24.00, 0.179);
  ST->DefineElement(2,55.84, 26.00, 0.730);
  ST->DefineElement(3,58.69, 28.0, 0.090);
  ST->SetRadLen(1.75424, 16.9879);
  TGeoMixture *LAr = new TGeoMixture("LAr",1,1.4);
  LAr->DefineElement(0,39.95, 18.0, 1.);
  LAr->SetRadLen(13.9472, 86.841);
  LAr->SetTransparency( 50 );  // set LAr half transparent
  TGeoMixture *Air = new TGeoMixture("Air",3,0.001205);
  Air->DefineElement(0, 39.95, 18.0, 0.009);
  Air->DefineElement(1, 14.01, 7.0, 0.781);
  Air->DefineElement(2, 16.00, 8.0, 0.209);
  Air->SetRadLen(30435.7, 71490.1);
  TGeoMixture *H2O = new TGeoMixture("Water",2,1.0);
  H2O->DefineElement(0,1.01, 1.0, 0.112);
  H2O->DefineElement(1,16.0, 8.0, 0.888);
  H2O->SetRadLen(35.7577, 78.0127);
  TGeoMixture *Ti = new TGeoMixture("Titanium",1,4.506);
  Ti->DefineElement(0,47.87, 22.0, 1.0);
  Ti->SetRadLen(3.58434, 28.5483);
  TGeoMixture *TPB = new TGeoMixture("TPB",1,1.4);
  TPB->DefineElement(0,39.95, 18.0, 1.0);
  TPB->SetRadLen(13.9472, 86.841);
  TGeoMixture *Glass = new TGeoMixture("Glass",8,2.74351);
  Glass->DefineElement(0,26.98, 13.0, 0.062 );
  Glass->DefineElement(1,16.0, 8.0, 0.461 );
  Glass->DefineElement(2,40.08, 20.0, 0.160 );
  Glass->DefineElement(3,55.84, 26.0, 0.001 );
  Glass->DefineElement(4,24.30, 12.0, 0.021 );
  Glass->DefineElement(5,22.99, 11.0, 0.007 );
  Glass->DefineElement(6,28.09, 14.0, 0.280 );
  Glass->DefineElement(7,47.87, 22.0, 0.008 );
  Glass->SetRadLen(8.99351, 36.8643);
  TGeoMixture *Acr = new TGeoMixture("Acrylic",3,1.19);
  Acr->DefineElement(0,12.01, 6.0, 0.600);
  Acr->DefineElement(1,1.01, 1.0, 0.80);
  Acr->DefineElement(2,16.0, 8.0, 0.320);
  TGeoMixture *Poly = new TGeoMixture("Polystyrene",2, 1.06);
  Poly->DefineElement(0,12.01,6.0,0.923);
  Poly->DefineElement(1,1.01,1.0,0.077);
  TGeoMixture* G10 = new TGeoMixture("G10",4,1.7);
  G10->DefineElement(0,12.01, 6.0, 0.299);
  G10->DefineElement(1,1.01, 1.0, 0.025);
  G10->DefineElement(2,16.0, 8.0, 0.395);
  G10->DefineElement(3,28.09, 14.0, 0.281);
  G10->SetRadLen(18.4195, 51.7558);

  /*** Translations and rotations ***/
  TGeoRotation* r1  = new TGeoRotation("rPlus30AboutX",30,0,0);
  TGeoRotation* r2  = new TGeoRotation("rPlus60AboutX",60,0,0);
  TGeoRotation* r3  = new TGeoRotation("rPlus90AboutX",90,0,0);
  TGeoRotation* r4  = new TGeoRotation("rMinus90AboutX",-90,0,0);
  TGeoRotation* r5  = new TGeoRotation("rPlusUVAngleAboutX",150,0,0);
  TGeoRotation* r6  = new TGeoRotation("rPlus150AboutX",150,0,0);
  TGeoRotation* r7  = new TGeoRotation("rPlus180AboutX",180,0,0);
  TGeoRotation* r8  = new TGeoRotation("rMinusUVAngleAboutX",-30,0,0);
  TGeoRotation* r9  = new TGeoRotation("rPlus30AboutY",0,30,0);
  TGeoRotation* r10 = new TGeoRotation("rPlus60AboutY",0,60,0);
  TGeoRotation* r11 = new TGeoRotation("rPlus90AboutY",0,90,0);
  TGeoRotation* r12 = new TGeoRotation("rPlus180AboutY",0,180,0);
  TGeoRotation* r13 = new TGeoRotation("rMinus90AboutY",0,-90,0);
  TGeoRotation* r14 = new TGeoRotation("rPlus90AboutZ",0,0,90);
  TGeoRotation* r15 = new TGeoRotation("rMinus90AboutZ",0,0,-90);
  TGeoRotation* r16 = new TGeoRotation("rPlus180AboutZ",0,0,180);
  TGeoRotation* r17 = new TGeoRotation("rMinus180AboutZ",0,0,-180);
  TGeoRotation* r18 = new TGeoRotation("rMinus90AboutYPlus180AboutZ",0,-90,180);
  TGeoRotation* r19 = new TGeoRotation("rMinus90AboutYMinus90AboutZ",0,-90,-90);
  TGeoRotation* r20 = new TGeoRotation("rPlus90AboutYPlus180AboutZ",0,90,180);
  TGeoRotation* r21 = new TGeoRotation("rMinus90AboutYPlus90AboutZ",0,-90,90);
  TGeoRotation* r22 = new TGeoRotation("rPlus90AboutYMinus90AboutZ",0,90,-90);
  TGeoRotation* r23 = new TGeoRotation("rPlus90AboutXPlus90AboutZ",90,0,90);
  TGeoRotation* r24 = new TGeoRotation("rPlus90AboutXPlus180AboutZ",90,0,180);
  TGeoRotation* r25 = new TGeoRotation("rPlus90AboutXPMinus90AboutY",90,-90,0);
  TGeoRotation* r26 = new TGeoRotation("rPlus90AboutXPlus90AboutY",90,90,0);
  TGeoRotation* r27 = new TGeoRotation("rPlus90AboutXMinus90AboutZ",90,0,-90);
  TGeoRotation* r28 = new TGeoRotation("rPMTRotation1",90,90,0);
  TGeoRotation* r29 = new TGeoRotation("r39degFix",0,39,0);
  TGeoTranslation* posCenter = new TGeoTranslation("posCenter",192.,0,525);
  TGeoTranslation* posEndCap1 = new TGeoTranslation("posEndCap1",0.,0.,305.6243);
  TGeoTranslation* posEndCap2 = new TGeoTranslation("posEndCap2",0.,0.,-305.6243);

  TGeoTranslation* posFrameSubt = new TGeoTranslation("posFrameSubt",0,0,0);
  TGeoTranslation* posaCrossBeamUnion2 = new TGeoTranslation("posaCrossBeamUnion2",0,0,111.64);
  TGeoTranslation* posaCrossBeamUnion3 = new TGeoTranslation("posaCrossBeamUnion3",0,0,72.);
  TGeoTranslation* posaSideCrossBeamUnion2 = new TGeoTranslation("posaSideCrossBeamUnion2",0,0,91.26);
  TGeoTranslation* posVertBarUnion1 = new TGeoTranslation("posVertBarUnion1",-5.5,0,-301.05);
  TGeoTranslation* posVertBarUnion2 = new TGeoTranslation("posVertBarUnion2",-5.5,0,-100.35);
  TGeoTranslation* posVertBarUnion3 = new TGeoTranslation("posVertBarUnion3",-5.5,0,100.35);
  TGeoTranslation* posVertBarUnion4 = new TGeoTranslation("posVertBarUnion4",-5.5,0,301.05);
  TGeoRotation* CrossBeam93rot = new TGeoRotation("CrossBeam93rot",93,0,0);
  TGeoRotation* rMinus60AboutY0 = new TGeoRotation("rMinus60AboutY0",0,-30,0);
  TGeoRotation* rMinus60AboutY1 = new TGeoRotation("rMinus60AboutY1",0,-30,0);
  TGeoRotation* rPlus76AboutY = new TGeoRotation("rPlus76AboutY",0,76,0);
  TGeoRotation* rMinus17AboutY = new TGeoRotation("rMinus17AboutY",0,-33,0);
  TGeoRotation* rotTopCross1 = new TGeoRotation("rotTopCross1",0,-38,0);
  TGeoRotation* rotSideCross33AboutY = new TGeoRotation("rotSideCross33AboutY",0,66,0);

  TGeoCombiTrans* rotEndCap2 = new TGeoCombiTrans("rotEndCap2",0.,0.,-305.6243,r12);
  TGeoCombiTrans* rotCrossBeamUnion= new TGeoCombiTrans("rotCrossBeamUnion",0,0,0,CrossBeam93rot);
  TGeoCombiTrans* rotaTopCrossUnion= new TGeoCombiTrans("rotaTopCrossUnion",0,0,0,r10);
  TGeoCombiTrans* rotaCrossBeamUnion0= new TGeoCombiTrans("rotaCrossBeamUnion0",64,0,55.82,rMinus60AboutY0);
  TGeoCombiTrans* rotaCrossBeamUnion1= new TGeoCombiTrans("rotaCrossBeamUnion1",-64,0,55.82,rMinus60AboutY1);
  TGeoCombiTrans* rotaTopCrossOutUnion= new TGeoCombiTrans("rotaTopCrossOutUnion",0,0,0,rPlus76AboutY);
  TGeoCombiTrans* rotCrossOutUnion0= new TGeoCombiTrans("rotCrossOutUnion0",64,0,36,rotTopCross1);
  TGeoCombiTrans* rotCrossOutUnion1= new TGeoCombiTrans("rotCrossOutUnion1",-64,0,36,rotTopCross1);
  TGeoCombiTrans* rotSideCross1= new TGeoCombiTrans("rotSideCross1",0,0,0,rotSideCross33AboutY);
  TGeoCombiTrans* rotSideCross2= new TGeoCombiTrans("rotSideCross2",64,0,45.63,rMinus17AboutY);
  TGeoCombiTrans* rotSideCross3= new TGeoCombiTrans("rotSideCross3",-64,0,45.63,rMinus17AboutY);
  
  posFrameSubt->RegisterYourself();
  posaCrossBeamUnion2->RegisterYourself();
  posaCrossBeamUnion3->RegisterYourself();
  posaSideCrossBeamUnion2->RegisterYourself();
  posVertBarUnion1->RegisterYourself();
  posVertBarUnion2->RegisterYourself();
  posVertBarUnion3->RegisterYourself();
  posVertBarUnion4->RegisterYourself();
  CrossBeam93rot->RegisterYourself();
  rMinus60AboutY0->RegisterYourself();
  rMinus60AboutY1->RegisterYourself();
  rPlus76AboutY->RegisterYourself();
  rMinus17AboutY->RegisterYourself();
  rotTopCross1->RegisterYourself();
  rotSideCross33AboutY->RegisterYourself();
  posCenter->RegisterYourself();
  posEndCap1->RegisterYourself();
  posEndCap2->RegisterYourself();
  rotEndCap2->RegisterYourself();
  rotCrossBeamUnion->RegisterYourself();
  rotaTopCrossUnion->RegisterYourself();
  rotaCrossBeamUnion0->RegisterYourself();
  rotaCrossBeamUnion1->RegisterYourself();
  rotaTopCrossOutUnion->RegisterYourself();
  rotCrossOutUnion0->RegisterYourself();
  rotCrossOutUnion1->RegisterYourself();
  rotSideCross1->RegisterYourself();
  rotSideCross2->RegisterYourself();
  rotSideCross3->RegisterYourself();
  r1->RegisterYourself();
  r2->RegisterYourself();
  r3->RegisterYourself();
  r4->RegisterYourself();
  r5->RegisterYourself();
  r6->RegisterYourself();
  r7->RegisterYourself();
  r8->RegisterYourself();
  r9->RegisterYourself();
  r10->RegisterYourself();
  r11->RegisterYourself();
  r12->RegisterYourself();
  r13->RegisterYourself();
  r14->RegisterYourself();
  r15->RegisterYourself();
  r16->RegisterYourself();
  r17->RegisterYourself();
  r18->RegisterYourself();
  r19->RegisterYourself();
  r20->RegisterYourself();
  r21->RegisterYourself();
  r22->RegisterYourself();
  r23->RegisterYourself();
  r24->RegisterYourself();
  r25->RegisterYourself();
  r26->RegisterYourself();
  r27->RegisterYourself();
  r28->RegisterYourself();
  r29->RegisterYourself();

  /*** Shapes and composite shapes ***/
  TGeoBBox* b1 = new TGeoBBox("TPCPlane",0.15/2., 233.0/2., 1037.35/2.);
  TGeoBBox* b2 = new TGeoBBox("TPCPlaneVert",0.15/2., 233.0/2., 1037.0/2.);
  TGeoBBox* b3 = new TGeoBBox("GroundPlate",0.1/2., 240.0/2., 1100./2.);
  TGeoBBox* b4 = new TGeoBBox("CathodePlate",0.1/2., 240.0/2., 1042./2.);
  TGeoBBox* b5 = new TGeoBBox("TPC",260./2., 256.0/2., 1045./2.);
  TGeoBBox* b6 = new TGeoBBox("TPCActive",256.35/2., 233.0/2., 1036.8/2.);
  TGeoBBox* b7 = new TGeoBBox("Paddle_PMT",0.3175/2., 50.8/2., 18.626667/2.);
  TGeoTube* t1 = new TGeoTube("TPCWireVert",0.,0.0075, 233.0/2.);
  TGeoTube* t2 = new TGeoTube("FieldCageTubeZ",0.635,1.27, 1026.84/2.);
  TGeoTube* t3 = new TGeoTube("FieldCageTubeY",0.635,1.27, 222.84/2.);
  TGeoTube* t4 = new TGeoTube("PMTVolume",0.,15.494, 28.199/2.);
  TGeoTube* t5 = new TGeoTube("PMT_AcrylicPlate",0.,15.24, 0.2/2.);
  TGeoTube* t6 = new TGeoTube("PMT_Stalk",0.,3.175, 7.62/2.);
  TGeoTube* t7 = new TGeoTube("PMT_SteelBase",0.,15.24, 3.81/2.);
  TGeoTube* t8 = new TGeoTube("PMT_Underside",0.,10.16, 6.35/2.);
  TGeoTube* t9 = new TGeoTube("PMT_Lens",0.,10.16, 6.35/2.);
  TGeoTube* t10 = new TGeoTube("CryostatTube",0.,191.61, 1086.49/2.);
  TGeoSphere* s1 = new TGeoSphere("CryostatEnd",0.0,305.25069,0.,38.88,0.,360.);
  TGeoTube* SteelTube = new TGeoTube("SteelTube",190.5,191.6099,1086.49/2.);
  TGeoSphere* EndCap = new TGeoSphere("EndCap",304.1407,305.2507,0.,38.88,0.,360.);

  //TPC frame
  TGeoBBox* b8 = new TGeoBBox("aSideBeam",256./2., 2./2., 5./2.);
  TGeoBBox* b9 = new TGeoBBox("aTopBeam",256./2., 10.16/2., 2.54/2.);
  TGeoBBox* b10 = new TGeoBBox("aSideCross0",10./2., 1.905/2., 109.22/2.);
  TGeoBBox* b11 = new TGeoBBox("aTopCrossBeamA",10./2., 1.905/2., 126.746/2.);
  TGeoBBox* b12 = new TGeoBBox("aTopCrossBeamB",10./2., 1.905/2., 96.52/2.);
  TGeoBBox* b13 = new TGeoBBox("FrameA",11./2., 254./2., 1040./2.);
  TGeoBBox* b14 = new TGeoBBox("FrameB",11.1/2., 218.44/2., 1003.52/2.);
  TGeoBBox* b15 = new TGeoBBox("VertBar",6.35/2., 218.44/2., 6.35/2.);
  TGeoBBox* b16 = new TGeoBBox("CrossBeamA",9.0/2., 301.12/2., 7./2.);

  TGeoCompositeShape *CryostatUnion1 = new TGeoCompositeShape("CryostatUnion1","CryostatTube + (CryostatEnd:posEndCap1)");
  TGeoCompositeShape *Cryostat       = new TGeoCompositeShape("Cryostat","(CryostatUnion1) + (CryostatEnd:rotEndCap2)");
  TGeoCompositeShape *SteelVesselUnion1 = new TGeoCompositeShape("SteelVesselUnion1","SteelTube + (EndCap:posEndCap1)");
  TGeoCompositeShape *SteelVessel       = new TGeoCompositeShape("SteelVessel","(SteelVesselUnion1) + (EndCap:rotEndCap2)");
  
  TGeoCompositeShape *Frame0         = new TGeoCompositeShape("Frame0","FrameA - (FrameB:posFrameSubt)");
  TGeoCompositeShape *Frame1         = new TGeoCompositeShape("Frame1","Frame0 + (VertBar:posVertBarUnion1)");
  TGeoCompositeShape *Frame2         = new TGeoCompositeShape("Frame2","Frame1 + (VertBar:posVertBarUnion2)");
  TGeoCompositeShape *Frame3         = new TGeoCompositeShape("Frame3","Frame2 + (VertBar:posVertBarUnion3)");
  TGeoCompositeShape *Frame4         = new TGeoCompositeShape("Frame4","Frame3 + (VertBar:posVertBarUnion4)");
  
  //TGeoCompositeShape *CrossBeam      = new TGeoCompositeShape("CrossBeam","CrossBeamA + (CrossBeamA:rotCrossBeamUnion)"); 
  TGeoCompositeShape *aTopCross0     = new TGeoCompositeShape("aTopCross0","aTopCrossBeamA + (aTopCrossBeamA:rotaTopCrossUnion)"); 
  TGeoCompositeShape *aTopCross1     = new TGeoCompositeShape("aTopCross1","aTopBeam  + (aTopCross0:rotaCrossBeamUnion0)"); 
  TGeoCompositeShape *aTopCross2     = new TGeoCompositeShape("aTopCross2","aTopCross1 + (aTopCross0:rotaCrossBeamUnion1)"); 
  TGeoCompositeShape *aTopCross      = new TGeoCompositeShape("aTopCross","aTopCross2 + (aTopBeam:posaCrossBeamUnion2)"); 
  /*
  TGeoCompositeShape *aTopCrossOuter0 = new TGeoCompositeShape("aTopCrossOuter0","aTopCrossBeamB + (aTopCrossBeamB:rotaTopCrossOutUnion)"); 
  TGeoCompositeShape *aTopCrossOuter1 = new TGeoCompositeShape("aTopCrossOuter1","aTopBeam + (aTopCrossOuter0:rotCrossOutUnion0)"); 
  TGeoCompositeShape *aTopCrossOuter2 = new TGeoCompositeShape("aTopCrossOuter2","aTopCrossOuter1 + (aTopCrossOuter0:rotCrossOutUnion1)"); 
  TGeoCompositeShape *aTopCrossOuter = new TGeoCompositeShape("aTopCrossOuter","aTopCrossOuter + (aTopBeam:posaCrossBeamUnion3)"); 
  */
  TGeoCompositeShape *aSideCross1     = new TGeoCompositeShape("aSideCross1","aSideCross0  + (aSideCross0:rotSideCross1)"); 
  TGeoCompositeShape *aSideCross2     = new TGeoCompositeShape("aSideCross2","aTopBeam  + (aSideCross1:rotSideCross2)"); 
  TGeoCompositeShape *aSideCross3     = new TGeoCompositeShape("aSideCross3","aSideCross2  + (aSideCross1:rotSideCross3)"); 
  TGeoCompositeShape *aSideCross      = new TGeoCompositeShape("aSideCross","aSideCross3  + (aTopBeam:posaSideCrossBeamUnion2)"); 

  /*
  <union name="aSideCross1">
      <first ref="aSideCross0"/>
      <second ref="aSideCross0"/>
      <position name="posaSideCrossUnion" unit="cm" x="0" y="0" z="0"/>
      <rotation name="rotSideCross33AboutY" unit="deg" x="0" y="66" z="0"/>
    </union>
    <union name="aSideCross2">
      <first ref="aTopBeam"/>
      <second ref="aSideCross1"/>
      <position name="posaSideCrossBeamUnion0" unit="cm" x="64" y="0" z="45.63"/>
      <rotation name="rMinus17AboutY" unit="deg" x="0" y="-33" z="0"/>
    </union>
    <union name="aSideCross3">
      <first ref="aSideCross2"/>
      <second ref="aSideCross1"/>
      <position name="posaSideCrossBeamUnion1" unit="cm" x="-64" y="0" z="45.63"/>
      <rotation name="rMinus60AboutY" unit="deg" x="0" y="-33" z="0"/>
    </union>
    <union name="aSideCross">
      <first ref="aSideCross2"/>
      <second ref="aTopBeam"/>
      <position name="posaSideCrossBeamUnion2" unit="cm" x="0" y="0" z="91.26"/>
    </union>
   */
  /*** Define media ***/
  TGeoMedium *mvac = new TGeoMedium("Vacuum", 1, vac );
  TGeoMedium *mAL = new TGeoMedium("ALUMINIUM_Al",2, AL );
  TGeoMedium *mSI = new TGeoMedium("SILICON_Si",3, SI );
  TGeoMedium *mSiO2 = new TGeoMedium("SiO2",4, SiO2 );
  TGeoMedium *mfib = new TGeoMedium("fibrous_glass",5,fib);
  TGeoMedium *mST = new TGeoMedium("STEEL_STAINLESS_Fe7Cr2Ni",6,ST);
  TGeoMedium *mLAr = new TGeoMedium("LAr",7,LAr);
  TGeoMedium *mAir = new TGeoMedium("Air",8,Air);
  TGeoMedium *mH2O = new TGeoMedium("Water",9,H2O);
  TGeoMedium *mTi = new TGeoMedium("Titanium",10,Ti);
  TGeoMedium *mTPB = new TGeoMedium("TPB",11,TPB);
  TGeoMedium *mGlass = new TGeoMedium("Glass",12,Glass);
  TGeoMedium *mAcr = new TGeoMedium("Acrylic",13,Acr);
  TGeoMedium *mPoly = new TGeoMedium("Polystyrene",14,Poly);
  TGeoMedium *mG10 = new TGeoMedium("G10",15,G10);

  /*** Define volumes ***/
  // cryostat is top volume
  TGeoVolume *top = new TGeoVolume("volCryostat", Cryostat, mLAr );
  gGeoManager->SetTopVolume( top );
  TGeoVolume *volSteelVessel = new TGeoVolume("volSteelVessel",SteelVessel,mST);
  TGeoVolume *volTPC = new TGeoVolume("volTPC",b5,mLAr );
  TGeoVolume *volGroundPlate = new TGeoVolume("volGroundPlate",b3,mST);
  TGeoVolume *volCathodePlate = new TGeoVolume("volCathodePlate",b4,mST);
  TGeoVolume *volTPCActive = new TGeoVolume("volTPCActive",b6,mLAr);
  volTPC->AddNode(volCathodePlate,1,  new TGeoTranslation(127.45, 0, 0));
  volTPC->AddNode(volTPCActive,2,new TGeoTranslation(-1.55, 0, 0));
  TGeoVolume *volPMT = new TGeoVolume("volPMT",t4,mLAr);
  TGeoVolume *volOpDetSensitive = new TGeoVolume("volOpDetSensitive",t5,mLAr);
  TGeoVolume *vol_PMT_AcrylicPlate = new TGeoVolume("vol_PMT_AcrylicPlate",t5,mLAr);
  TGeoVolume *vol_PMT_Stalk = new TGeoVolume("vol_PMT_Stalk",t6,mGlass);
  TGeoVolume *vol_PMT_SteelBase = new TGeoVolume("vol_PMT_SteelBase",t7,mST);
  TGeoVolume *vol_PMT_Underside = new TGeoVolume("vol_PMT_Underside",t8,mGlass);
  TGeoVolume *vol_PMT_Lens = new TGeoVolume("vol_PMT_Lens",t9,mLAr);
  volPMT->AddNode(volOpDetSensitive,1,new TGeoTranslation(0,0,13.87));
  volPMT->AddNode(vol_PMT_AcrylicPlate,2,new TGeoTranslation(0,0,13.67));
  volPMT->AddNode(vol_PMT_Stalk,3,new TGeoTranslation(0,0,-6.35));
  volPMT->AddNode(vol_PMT_SteelBase,4,new TGeoTranslation(0,0,-12.065));
  volPMT->AddNode(vol_PMT_Lens,5,new TGeoTranslation(0,0,3.81));
  volPMT->AddNodeOverlap(vol_PMT_Underside,6,new TGeoTranslation(0,0,3.81));
  TGeoVolume *volPaddle_PMT = new TGeoVolume("volPaddle_PMT",b7,mAcr);
  TGeoVolume *volFrame = new TGeoVolume("volFrame",Frame4,mST);
  //TGeoVolume *volCrossBeam = new TGeoVolume("volCrossBeam",CrossBeam,mST);
  //TGeoVolume *volaTopBeam = new TGeoVolume("volaTopBeam",b9,mG10);
  //TGeoVolume *volaTopCross = new TGeoVolume("volaTopCross",aTopCross,mG10);
  //TGeoVolume *volaTopCrossOuter = new TGeoVolume("volaTopCrossOuter",aTopCrossOuter,mG10);
  //TGeoVolume *volaSideCross = new TGeoVolume("volaSideCross",aSideCross,mG10);

  /*** Add nodes ***/
  top->AddNode(volSteelVessel,0);
  //TPC
  top->AddNode(volTPC,1, new TGeoTranslation(0,0.97,0));
  //PMT
  TGeoCombiTrans* posPMT[32];
  posPMT[0] = new TGeoCombiTrans(-155.5865,55.249,-430.7935,r28);
  posPMT[1] = new TGeoCombiTrans(-155.4415,55.249,-390.1455,r28);
  posPMT[2] = new TGeoCombiTrans(-155.4795,27.431,-467.3985,r28);
  posPMT[3] = new TGeoCombiTrans(-155.4865,-0.303,-344.7565,r28);
  posPMT[4] = new TGeoCombiTrans(-155.1975,-28.576,-468.0255,r28);
  posPMT[5] = new TGeoCombiTrans(-155.2825,-56.615,-430.6305,r28);
  posPMT[6] = new TGeoCombiTrans(-155.3234,-56.203,-390.3205,r28);
  posPMT[7] = new TGeoCombiTrans(-155.4625,54.646,-230.5235,r28);
  posPMT[8] = new TGeoCombiTrans(-155.2315,54.693,-190.2875,r28);
  posPMT[9] = new TGeoCombiTrans(-155.1955,-0.829,-276.4855,r28);
  posPMT[10] = new TGeoCombiTrans(-155.1305,-0.706,-144.6615,r28);
  posPMT[11] = new TGeoCombiTrans(-155.0525,-56.261,-230.8605,r28);
  posPMT[12] = new TGeoCombiTrans(-154.6935,-57.022,-190.1595,r28);
  posPMT[13] = new TGeoCombiTrans(-155.0285,55.771,-18.3665,r28);
  posPMT[14] = new TGeoCombiTrans(-154.9185,55.822,22.4295,r28);
  posPMT[15] = new TGeoCombiTrans(-154.6635,-0.875,-65.4045,r28);
  posPMT[16] = new TGeoCombiTrans(-154.6965,-0.549,66.7845,r28);
  posPMT[17] = new TGeoCombiTrans(-154.6395,-56.323,-18.2795,r28);
  posPMT[18] = new TGeoCombiTrans(-154.6655,-56.205,22.1165,r28);
  posPMT[19] = new TGeoCombiTrans(-154.7065,55.8,192.5735,r28);
  posPMT[20] = new TGeoCombiTrans(-154.5855,55.625,233.3835,r28);
  posPMT[21] = new TGeoCombiTrans(-154.6695,-0.051,145.7035,r28);
  posPMT[22] = new TGeoCombiTrans(-154.3495,-0.502,277.7085,r28);
  posPMT[23] = new TGeoCombiTrans(-154.6575,-56.408,192.7745,r28);
  posPMT[24] = new TGeoCombiTrans(-154.6495,-56.284,233.4055,r28);
  posPMT[25] = new TGeoCombiTrans(-153.8795,55.822,392.5655,r28);
  posPMT[26] = new TGeoCombiTrans(-153.6865,55.313,433.3615,r28);
  posPMT[27] = new TGeoCombiTrans(-153.4625,27.607,471.2115,r28);
  posPMT[28] = new TGeoCombiTrans(-154.2215,-0.722,347.0985,r28);
  posPMT[29] = new TGeoCombiTrans(-153.4995,-28.625,471.8555,r28);
  posPMT[30] = new TGeoCombiTrans(-154.1035,-56.309,393.4395,r28);
  posPMT[31] = new TGeoCombiTrans(-153.8205,-56.514,433.3645,r28);
  for(int i=0; i<32; i++){
    top->AddNodeOverlap(volPMT,i+2,posPMT[i]);
  }
  top->AddNodeOverlap(volPaddle_PMT,34, new TGeoTranslation(-161.341,-2.801,-252.2715));
  top->AddNodeOverlap(volPaddle_PMT,35, new TGeoTranslation(-160.858,-2.594,-40.9315));
  top->AddNodeOverlap(volPaddle_PMT,36, new TGeoTranslation(-160.882,-2.7, 43.9005));
  top->AddNodeOverlap(volPaddle_PMT,37, new TGeoTranslation(-160.654,-2.355,255.1425));

  top->AddNode(volFrame,38,new TGeoTranslation(-136.,0.,0.));
  //top->AddNode(volaSideCross,39,new TGeoTranslation(0.,-45.63,530.095));
  //top->AddNode(volaSideCross,40,new TGeoTranslation(0.,-45.63,-530.095));
  //top->AddNode(volaTopCross,39,new TGeoTranslation(0.,134.35,-279.1));
  //top->AddNode(volaTopCross,40,new TGeoTranslation(0.,-134.35,-279.1));
  //glose the geometry and draw top
  gGeoManager->CloseGeometry();
  gGeoManager->SetTopVisible();
  top->Draw("ogle");

  TFile *tf = new TFile("uboone_simplifiedCryo.root", "RECREATE"); 
  gGeoManager->Write();
  tf->Close();

  if(use_gdml) gGeoManager->Export("simplified_uboone.gdml");
}
       /*     
        <volumeref ref="volaTopCross"/>
        <position name="posaTopCross0" unit="cm" x="0,134.35,-279.1"/>
        <volumeref ref="volaTopCross"/>
        <position name="posaBottomCross0" unit="cm" x="0,-134.35,-279.1"/>
        <volumeref ref="volaTopCross"/>
        <position name="posaTopCross1" unit="cm" x="0,134.35,-55.82"/>
        <volumeref ref="volaTopCross"/>
        <position name="posaBottomCross1" unit="cm" x="0,-134.35,-55.82"/>
        <volumeref ref="volaTopCross"/>
        <position name="posaTopCross2" unit="cm" x="0,134.35,167.46"/>
        <volumeref ref="volaTopCross"/>
        <position name="posaBottomCross2" unit="cm" x="0,-134.35,167.46"/>
        <volumeref ref="volaTopCrossOuter"/>
        <position name="posaTopCrossOuter0" unit="cm" x="0,134.35,-446.56"/>
        <volumeref ref="volaTopCrossOuter"/>
        <position name="posaBottomCrossOuter0" unit="cm" x="0,-134.35,-446.56"/>
        <volumeref ref="volaTopCrossOuter"/>
        <position name="posaTopCrossOuter1" unit="cm" x="0,134.35,390.74"/>
        <volumeref ref="volaTopCrossOuter"/>
        <position name="posaBottomCrossOuter1" unit="cm" x="0,-134.35,390.74"/>
       */
