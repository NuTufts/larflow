add_executable (unittest unittest.cxx)

# larlite libs
file( GLOB LARLITE_LIBS "$ENV{LARLITE_LIBDIR}/lib*.so" )
message("larlite libs: ${LARLITE_LIBS}")

# larcv libs used
set(LARCV_LIBS_USED LArCVCoreBase LArCVCoreDataFormat LArCVCoreProcessor)

# ublarcvapp libs used
set(UBLARCVAPP_LIBS_USED LArCVApp_MCTools LArCVApp_UBImageMod LArCVApp_UBWireTool LArCVApp_ubdllee LArCVApp_UBWireTool )

# larflow libs
set(LARFLOW_LIBS_USED LArFlow_KeyPoints LArFlow_PrepFlowMatchData )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR} ${LARCV_INCLUDE_DIR} ${LARLITE_INC_DIRS} ${UBLARCVAPP_INCLUDE_DIR} ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDES} )
target_link_libraries(unittest ${LARFLOW_LIBS_USED} ${LARCV_LIBS_USED} ${UBLARCVAPP_LIBS_USED} ${UBLARCVAPP_LIBS_USED} ${LARLITE_LIBS} ${LARLITE_LIBS} ${OpenCV_LIBS} ${PYTHON_LIBRARIES} )
