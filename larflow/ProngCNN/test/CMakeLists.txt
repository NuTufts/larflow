# Minimal test executable
add_executable(test_load_model test_load_model.cxx)
target_link_libraries(test_load_model ${ROOT_LIBRARIES})
install(TARGETS test_load_model RUNTIME DESTINATION bin)

# Simple test executable (has linking issues with ROOT dictionaries)
# add_executable(test_prongcnn_simple test_prongcnn_simple.cxx)

# Full executable (commented out for now due to header conflicts)
# add_executable(run_prongcnn run_prongcnn.cxx)
# target_compile_definitions(run_prongcnn PRIVATE ${ROOT_DEFINITIONS})

# Include directories for simple test
# target_include_directories(test_prongcnn_simple
#   PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}
#   ${PROJECT_SOURCE_DIR}
#   ${LARCV_INCLUDE_DIR}
#   ${LARLITE_INC_DIRS}
#   ${TORCH_INCLUDE_DIRS}
#   ${ROOT_INCLUDE_DIRS}
# )
# 
# # Add larpid include directory explicitly
# if(DEFINED ENV{LARPID_DIR})
#   target_include_directories(test_prongcnn_simple PRIVATE "$ENV{LARPID_DIR}/include")
# endif()
# 
# # Link libraries for simple test
# target_link_libraries(test_prongcnn_simple
#   LArFlow_ProngCNN
#   LArCVCoreBase
#   LArCVCoreDataFormat
#   larlite_DataFormat
#   larlite_Base
#   LArPID_Interface
#   ${TORCH_LIBRARIES}
#   ${ROOT_LIBRARIES}
#   dl  # For dynamic loading
# )
# 
# # Install the executable
# install(TARGETS test_prongcnn_simple
#   RUNTIME DESTINATION bin)