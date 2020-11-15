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
#pragma link C++ namespace larflow::spatialembed;
#pragma link C++ struct larflow::spatialembed::AncestorIDPix_t+;
#pragma link C++ class larflow::spatialembed::PrepMatchEmbed+;
#pragma link C++ class larflow::spatialembed::Prep3DSpatialEmbed+;
#pragma link C++ struct larflow::spatialembed::Prep3DSpatialEmbed::VoxelData_t+;
#pragma link C++ class std::vector<larflow::spatialembed::Prep3DSpatialEmbed::VoxelData_t>+;
#pragma link C++ class larflow::spatialembed::SpatialEmbed3DNetProducts+;

//ADD_NEW_CLASS ... do not change this line
#endif
