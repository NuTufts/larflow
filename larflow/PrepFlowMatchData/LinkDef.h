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

#pragma link C++ class larflow::FlowMatchMap+;
#pragma link C++ class larflow::PrepFlowMatchData+;
#pragma link C++ function larflow::sample_pair_array( const int&, const FlowMatchMap&, int& )+;

//ADD_NEW_CLASS ... do not change this line
#endif
