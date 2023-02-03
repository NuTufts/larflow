/** \defgroup Voxelizer Voxelizer
 *
 * \brief Tools to match CRT hit and track objects to TPC tracks
 *
 *
 * cint script to generate libraries and python bindings.
 * Declare namespace & classes you defined
 * pragma statement: order matters! Google it ;)
 *
 */

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace larflow;
#pragma link C++ namespace larflow::voxelizer;
#pragma link C++ class larflow::voxelizer::VoxelizeTriplets+;
#pragma link C++ class std::vector<larflow::voxelizer::VoxelizeTriplets>+;
#pragma link C++ class larflow::voxelizer::LArVoxelHitMaker+;
#pragma link C++ class std::vector<larflow::voxelizer::LArVoxelHitMaker>+;
#pragma link C++ class larflow::voxelizer::TPCVoxelData+;
#pragma link C++ class std::vector<larflow::voxelizer::TPCVoxelData>+;

//ADD_NEW_CLASS ... do not change this line
#endif
