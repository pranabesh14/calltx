cmake_minimum_required(VERSION 3.16)
project(calltx LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# ---- whisper.cpp (speech recognition engine, pure C/C++) ----
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS      OFF CACHE BOOL "" FORCE)
# GPU: configure with -DGGML_CUDA=ON (NVIDIA), -DGGML_VULKAN=ON, or Metal is on by default on macOS.

FetchContent_Declare(
  whisper
  GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
  GIT_TAG        v1.9.5
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(whisper)

# ---- miniaudio (single header: WAV/MP3/FLAC decoding + resampling) ----
set(THIRD_PARTY_DIR "${CMAKE_BINARY_DIR}/third_party")
set(MINIAUDIO_H "${THIRD_PARTY_DIR}/miniaudio.h")

file(MAKE_DIRECTORY "${THIRD_PARTY_DIR}")

if(NOT EXISTS "${MINIAUDIO_H}")
  message(FATAL_ERROR
    "miniaudio.h is missing at ${MINIAUDIO_H}. "
    "Download miniaudio.h manually and place it at this path."
  )
endif()

file(SIZE "${MINIAUDIO_H}" MINIAUDIO_SIZE)
if(MINIAUDIO_SIZE LESS 100000)
  message(FATAL_ERROR
    "miniaudio.h appears incomplete or invalid: ${MINIAUDIO_H}"
  )
endif()

add_executable(calltx src/main.cpp src/audio.cpp src/deadair.cpp)
target_include_directories(calltx PRIVATE ${THIRD_PARTY_DIR})
target_link_libraries(calltx PRIVATE whisper)
if(UNIX AND NOT APPLE)
  target_link_libraries(calltx PRIVATE pthread m dl)
endif()

# ---- tests (optional): cmake -B build -DCALLTX_BUILD_TESTS=ON && ctest --test-dir build ----
option(CALLTX_BUILD_TESTS "Build calltx unit tests" OFF)
if(CALLTX_BUILD_TESTS)
  enable_testing()
  add_executable(test_deadair tests/test_deadair.cpp src/deadair.cpp)
  target_include_directories(test_deadair PRIVATE src)
  add_test(NAME deadair COMMAND test_deadair)
endif()
