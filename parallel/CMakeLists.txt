cmake_minimum_required(VERSION 3.10)  

project(parallel_brain_model)  

# Set MPI paths explicitly  
set(MPI_C_INCLUDE_DIR "E:/MS_MPI_SDK/Include")  # Adjust these paths as necessary  
set(MPI_C_LIBRARIES "E:/MS_MPI_SDK/Lib/x64/msmpi.lib")  
set(MPI_CXX_INCLUDE_DIR "E:/MS_MPI_SDK/Include")  
set(MPI_CXX_LIBRARIES "E:/MS_MPI_SDK/Lib/x64/msmpi.lib")  

# Include directories  
include_directories(  
    ${CMAKE_CURRENT_SOURCE_DIR}/header  
    ${MPI_C_INCLUDE_DIR} 
)  

# Source files  
aux_source_directory(./src SRC)  

# Set output path for the executable  
set(EXECUTABLE_OUTPUT_PATH ../)  

# Add the executable  
add_executable(main ${SRC})  

# Link MPI libraries  
target_link_libraries(main ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})  
