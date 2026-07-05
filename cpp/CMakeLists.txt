cmake_minimum_required(VERSION 3.20)
project(go2_wbc_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GO2WBC_USE_SYSTEM_EIGEN "Use a system-provided Eigen3 package" OFF)
option(GO2WBC_FETCH_EIGEN "Download Eigen if it is not installed" ON)
option(GO2WBC_FETCH_OSQP "Download OSQP if it is not installed" ON)

if(GO2WBC_USE_SYSTEM_EIGEN)
    find_package(Eigen3 QUIET)
endif()
if(NOT Eigen3_FOUND)
    if(GO2WBC_FETCH_EIGEN)
        include(FetchContent)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
        set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            eigen
            URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_GetProperties(eigen)
        if(NOT eigen_POPULATED)
            FetchContent_Populate(eigen)
        endif()
        if(NOT TARGET Eigen3::Eigen)
            add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
            target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})
        endif()
    else()
        message(FATAL_ERROR "Eigen3 was not found. Install libeigen3-dev or enable GO2WBC_FETCH_EIGEN.")
    endif()
endif()

find_package(osqp QUIET)
if(NOT TARGET osqp::osqp AND NOT TARGET osqpstatic AND NOT TARGET osqp)
    if(GO2WBC_FETCH_OSQP)
        include(FetchContent)
        set(OSQP_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
        set(OSQP_BUILD_DEMO_EXECUTABLE OFF CACHE BOOL "" FORCE)
        set(OSQP_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
        set(OSQP_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
        set(OSQP_ENABLE_PRINTING OFF CACHE BOOL "" FORCE)
        set(OSQP_ENABLE_PROFILING OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            osqp
            URL https://github.com/osqp/osqp/releases/download/v1.0.0/osqp-v1.0.0-src.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(osqp)
    else()
        message(FATAL_ERROR "OSQP was not found. Install OSQP or enable GO2WBC_FETCH_OSQP.")
    endif()
endif()

if(TARGET osqp::osqp)
    set(GO2WBC_OSQP_TARGET osqp::osqp)
elseif(TARGET osqpstatic)
    set(GO2WBC_OSQP_TARGET osqpstatic)
elseif(TARGET osqp)
    set(GO2WBC_OSQP_TARGET osqp)
else()
    message(FATAL_ERROR "Could not identify an OSQP CMake target.")
endif()

set(MUJOCO_ROOT "" CACHE PATH "Path to a MuJoCo SDK root containing include/ and lib/")
set(MUJOCO_PYTHON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/Lib/site-packages/mujoco" CACHE PATH
    "Optional path to a MuJoCo Python package directory")

find_path(MUJOCO_INCLUDE_DIR
    NAMES mujoco/mujoco.h
    PATHS
        "${MUJOCO_ROOT}/include"
        "${MUJOCO_ROOT}/include/mujoco"
        "${MUJOCO_PYTHON_ROOT}/include"
        "${MUJOCO_PYTHON_ROOT}/include/mujoco"
)

find_library(MUJOCO_LIBRARY
    NAMES mujoco libmujoco
    PATHS
        "${MUJOCO_ROOT}/lib"
        "${MUJOCO_ROOT}/bin"
        "${MUJOCO_PYTHON_ROOT}"
)

set(MUJOCO_DLL "${MUJOCO_PYTHON_ROOT}/mujoco.dll")
if(NOT MUJOCO_LIBRARY AND MINGW AND EXISTS "${MUJOCO_DLL}")
    find_program(GENDEF_EXECUTABLE gendef)
    find_program(DLLTOOL_EXECUTABLE dlltool)
    if(GENDEF_EXECUTABLE AND DLLTOOL_EXECUTABLE)
        set(MUJOCO_IMPORT_DIR "${CMAKE_BINARY_DIR}/mujoco_import")
        set(MUJOCO_DEF "${MUJOCO_IMPORT_DIR}/mujoco.def")
        set(MUJOCO_IMPORT_LIB "${MUJOCO_IMPORT_DIR}/libmujoco.dll.a")
        file(MAKE_DIRECTORY "${MUJOCO_IMPORT_DIR}")
        execute_process(
            COMMAND "${GENDEF_EXECUTABLE}" "${MUJOCO_DLL}"
            WORKING_DIRECTORY "${MUJOCO_IMPORT_DIR}"
            RESULT_VARIABLE GENDEF_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(GENDEF_RESULT EQUAL 0 AND EXISTS "${MUJOCO_DEF}")
            execute_process(
                COMMAND "${DLLTOOL_EXECUTABLE}" -d "${MUJOCO_DEF}" -l "${MUJOCO_IMPORT_LIB}" -D mujoco.dll
                RESULT_VARIABLE DLLTOOL_RESULT
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(DLLTOOL_RESULT EQUAL 0 AND EXISTS "${MUJOCO_IMPORT_LIB}")
                set(MUJOCO_LIBRARY "${MUJOCO_IMPORT_LIB}" CACHE FILEPATH "MuJoCo MinGW import library" FORCE)
            endif()
        endif()
    endif()
endif()

if(NOT MUJOCO_INCLUDE_DIR)
    message(FATAL_ERROR "MuJoCo headers were not found. Set MUJOCO_ROOT or MUJOCO_PYTHON_ROOT.")
endif()

if(NOT MUJOCO_LIBRARY)
    message(FATAL_ERROR "MuJoCo library was not found. Set MUJOCO_ROOT to an installed MuJoCo SDK.")
endif()

add_library(go2wbc_core
    src/CentroidalMpc.cpp
    src/GeneralContactWbc.cpp
    src/MujocoModelInterface.cpp
    src/OsqpSolver.cpp
)

target_include_directories(go2wbc_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${MUJOCO_INCLUDE_DIR}
)

target_link_libraries(go2wbc_core
    PUBLIC
        Eigen3::Eigen
        ${MUJOCO_LIBRARY}
        ${GO2WBC_OSQP_TARGET}
)

add_executable(inspect_dynamics
    apps/inspect_dynamics.cpp
)

target_link_libraries(inspect_dynamics PRIVATE go2wbc_core)

add_executable(solve_wbc_once
    apps/solve_wbc_once.cpp
)

target_link_libraries(solve_wbc_once PRIVATE go2wbc_core)

add_executable(solve_mpc_once
    apps/solve_mpc_once.cpp
)

target_link_libraries(solve_mpc_once PRIVATE go2wbc_core)

add_executable(run_trot_rollout
    apps/run_trot_rollout.cpp
)

target_link_libraries(run_trot_rollout PRIVATE go2wbc_core)

if(WIN32 AND EXISTS "${MUJOCO_DLL}")
    add_custom_command(TARGET inspect_dynamics POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MUJOCO_DLL}"
            "$<TARGET_FILE_DIR:inspect_dynamics>/mujoco.dll"
    )
    add_custom_command(TARGET solve_wbc_once POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MUJOCO_DLL}"
            "$<TARGET_FILE_DIR:solve_wbc_once>/mujoco.dll"
    )
    add_custom_command(TARGET solve_mpc_once POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MUJOCO_DLL}"
            "$<TARGET_FILE_DIR:solve_mpc_once>/mujoco.dll"
    )
    add_custom_command(TARGET run_trot_rollout POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MUJOCO_DLL}"
            "$<TARGET_FILE_DIR:run_trot_rollout>/mujoco.dll"
    )
endif()
