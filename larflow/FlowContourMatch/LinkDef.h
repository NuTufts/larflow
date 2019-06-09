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
#pragma link C++ class larflow::ContourFlowMatchDict+;
#pragma link C++ class std::pair<int,int> SrcTarPair_t+;
#pragma link C++ class std::map<SrcTarPair_t,ContourFlowMatch_t>+;

#pragma link C++ function larflow::makeHitsFromPixels+;
#pragma link C++ function larflow::makeSimpleFlowHits+;

#pragma link C++ function larflow::makeFlowHitsFromSparseCrops+;

//ADD_NEW_CLASS ... do not change this line
#endif
