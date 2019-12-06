//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//#pragma link C++ namespace larflow;
#pragma link C++ namespace larflow::reco;
#pragma link C++ class larflow::reco::cluster_t+;
#pragma link C++ function larflow::reco::cluster_larflow3dhits+;
#pragma link C++ function larflow::reco::cluster_dump2jsonfile+;
#pragma link C++ function larflow::reco::cluster_splitbytrackshower+;
#pragma link C++ function larflow::reco::cluster_imageprojection+;
#pragma link C++ function larflow::reco::cluster_getcontours+;
#pragma link C++ class larflow::PCACluster+;
//ADD_NEW_CLASS ... do not change this line
#endif
