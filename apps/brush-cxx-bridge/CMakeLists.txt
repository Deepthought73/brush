if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_CMD cargo build -p brush-cxx-bridge --verbose)
    set(TARGET_DIR "debug")
else ()
    set(CARGO_CMD cargo build -p brush-cxx-bridge --release --verbose)
    set(TARGET_DIR "release")
endif ()

# If the user is cross compiling to a different target, update the location of the target directory
# *Note*: Make sure to install the target with rustup, as well as setting the proper environment variables, such as CC, HOST_CC, CARGO_TARGET_<triple>_LINKER, ...
# See https://doc.rust-lang.org/cargo/reference/environment-variables.html#configuration-environment-variables
if (DEFINED ENV{CARGO_BUILD_TARGET})
    set(TARGET_DIR "$ENV{CARGO_BUILD_TARGET}/${TARGET_DIR}")
endif()

if(ENABLE_LTO)
    set(RUST_FLAGS "-Clinker-plugin-lto -Clinker=${CMAKE_C_COMPILER} -Clink-arg=-fuse-ld=lld")
endif()

set(BRUSH_CXX_BRIDGE_LIB "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/libbrush_cxx_bridge.a")
set(BRUSH_CXX_BRIDGE_CXX "${CMAKE_CURRENT_BINARY_DIR}/brush_cxx_bridge.cpp")
set(BRUSH_CXX_BRIDGE_H   "${CMAKE_CURRENT_BINARY_DIR}/brush_cxx_bridge.h")

add_custom_target(brush_cxx_bridge_rust
    BYPRODUCTS ${BRUSH_CXX_BRIDGE_LIB} ${BRUSH_CXX_BRIDGE_CXX} ${BRUSH_CXX_BRIDGE_H}
    COMMAND ${CMAKE_COMMAND} -E env
            CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}
            RUSTFLAGS=${RUST_FLAGS}
            ${CARGO_CMD}
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/brush-cxx-bridge/src/lib.rs.cc
            ${BRUSH_CXX_BRIDGE_CXX}
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/brush-cxx-bridge/src/lib.rs.h
            ${BRUSH_CXX_BRIDGE_H}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    USES_TERMINAL
    VERBATIM
)
set_source_files_properties(${BRUSH_CXX_BRIDGE_CXX} PROPERTIES GENERATED TRUE)

add_library(brush_cxx_bridge STATIC ${BRUSH_CXX_BRIDGE_CXX})
add_dependencies(brush_cxx_bridge brush_cxx_bridge_rust)
target_link_libraries(brush_cxx_bridge PUBLIC pthread dl ${BRUSH_CXX_BRIDGE_LIB})
target_include_directories(brush_cxx_bridge PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
