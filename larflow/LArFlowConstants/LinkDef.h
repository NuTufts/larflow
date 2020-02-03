//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link off all namespaces;

//#pragma link C++ namespace larflow;
#pragma link C++ namespace larflow;
#pragma link C++ enum  larflow::FlowDir_t+;
#pragma link C++ class larflow::LArFlowConstants+;
//ADD_NEW_CLASS ... do not change this line
#endif
