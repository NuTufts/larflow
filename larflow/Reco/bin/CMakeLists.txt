add_executable(run_kpsrecomanager run_kpsrecomanager.cxx)
message( STATUS "larlite libs: ${LARLITE_LIBS}" )
target_compile_features( run_kpsrecomanager
  PRIVATE
  cxx_std_17 )
set_target_properties( run_kpsrecomanager PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
target_include_directories( run_kpsrecomanager
  PRIVATE
  $ENV{UBDL_BASEDIR}/argparse/include )
target_link_libraries( run_kpsrecomanager
  PUBLIC
  LArFlow_Reco
  LArFlow_SCBoundary
  LArCVApp_ubdllee
  LArCVApp_Reco3D
  LArCVApp_MCTools
  LArCVApp_ContourTools
  LArCVApp_LArliteHandler
  LArCVApp_UBWireTool  
  larlite_Base
  larlite_DataFormat
  larlite_LArUtil
  ROOT::Core )
