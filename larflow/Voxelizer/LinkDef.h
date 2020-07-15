//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace std;
#pragma link C++ namespace larflow;
#pragma link C++ namespace larflow::voxelizer;
#pragma link C++ class larflow::voxelizer::VoxelizeTriplets+;
#pragma link C++ class std::vector<larflow::voxelizer::VoxelizeTriplets>+;

//ADD_NEW_CLASS ... do not change this line
#endif
