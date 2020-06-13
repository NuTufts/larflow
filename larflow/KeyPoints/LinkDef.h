//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace larflow;
#pragma link C++ namespace larflow::keypoints;
#pragma link C++ class larflow::keypoints::KPdata+;
#pragma link C++ class larflow::keypoints::PrepKeypointData+;
#pragma link C++ class larflow::keypoints::LoaderKeypointData+;

//ADD_NEW_CLASS ... do not change this line
#endif
