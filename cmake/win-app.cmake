# windows application manifest for llama.cpp executables
#
# embedded as RT_MANIFEST ID 1 in every executable owned by this project. it
# activates the UTF-8 process code page (argv, env and narrow file APIs see
# UTF-8) and long-path awareness. keep it PRIVATE to executables: the libraries
# must keep working in host applications without this manifest.
#
# requires Windows 10 1903 (build 18362) or newer for activeCodePage=UTF-8.
# unprefixed long paths additionally require LongPathsEnabled=1 on the machine.

set(LLAMA_WIN_APP_MANIFEST "${CMAKE_CURRENT_LIST_DIR}/../common/win-app.manifest")

if (WIN32)
    # MSVC is set for the cl command line only; clang with the MSVC ABI simulates it without setting MSVC
    if (MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
        set(LLAMA_WIN_APP_MSVC_ABI TRUE)
    else()
        set(LLAMA_WIN_APP_MSVC_ABI FALSE)
    endif()

    # generated resource script for the GNU-ABI path. it references the shared
    # manifest by absolute path, so no include path resolution is needed
    set(LLAMA_WIN_APP_MANIFEST_RC "${CMAKE_CURRENT_BINARY_DIR}/win-app-manifest.rc")
    configure_file("${CMAKE_CURRENT_LIST_DIR}/win-app.rc.in" "${LLAMA_WIN_APP_MANIFEST_RC}" @ONLY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${LLAMA_WIN_APP_MANIFEST}")

    if (NOT LLAMA_WIN_APP_MSVC_ABI)
        enable_language(RC)
        if (NOT CMAKE_RC_COMPILER MATCHES "windres")
            # CMake's Windows-GNU link rules route objects through objects.a; GNU links dispatch members by content and reject a .res container there
            # resinator can emit a COFF object instead. pin its machine to the target arch (its default X64 would break ARM64)
            string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" LLAMA_WIN_APP_RC_ARCH)
            if (LLAMA_WIN_APP_RC_ARCH MATCHES "^(arm64|aarch64)$")
                set(LLAMA_WIN_APP_RC_ARCH aarch64)
            elseif (LLAMA_WIN_APP_RC_ARCH MATCHES "^(x86_64|x64|amd64)$")
                set(LLAMA_WIN_APP_RC_ARCH x86_64)
            elseif (LLAMA_WIN_APP_RC_ARCH MATCHES "^(x86|i[3-6]86)$")
                set(LLAMA_WIN_APP_RC_ARCH x86)
            endif()
            set(CMAKE_RC_COMPILE_OBJECT "<CMAKE_RC_COMPILER> <DEFINES> <INCLUDES> <FLAGS> /:output-format coff /:target ${LLAMA_WIN_APP_RC_ARCH} /fo <OBJECT> <SOURCE>")
        endif()
    endif()
endif()

function(llama_add_app_manifest target)
    if (NOT WIN32)
        return()
    endif()
    get_target_property(_aliased ${target} ALIASED_TARGET)
    if (_aliased)
        return()
    endif()
    get_target_property(_type ${target} TYPE)
    if (NOT _type STREQUAL "EXECUTABLE")
        return()
    endif()
    if (LLAMA_WIN_APP_MSVC_ABI)
        # the linker merges the manifest source into its generated manifest (execution level etc.)
        target_sources(${target} PRIVATE ${LLAMA_WIN_APP_MANIFEST})
    else()
        # MinGW / MSYS2 / Zig: a .manifest source is not embedded by the GNU ABI, compile a resource instead
        if (NOT TARGET llama-win-app-manifest-res)
            add_library(llama-win-app-manifest-res OBJECT ${LLAMA_WIN_APP_MANIFEST_RC})
            set_source_files_properties(${LLAMA_WIN_APP_MANIFEST_RC} PROPERTIES OBJECT_DEPENDS "${LLAMA_WIN_APP_MANIFEST}")
        endif()
        target_link_libraries(${target} PRIVATE llama-win-app-manifest-res)
    endif()
endfunction()

function(llama_apply_app_manifests_dir dir)
    get_property(_targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target IN LISTS _targets)
        llama_add_app_manifest(${_target})
    endforeach()
    get_property(_subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
    foreach(_subdir IN LISTS _subdirs)
        llama_apply_app_manifests_dir("${_subdir}")
    endforeach()
endfunction()

# apply the manifest to all executable targets created in the project-owned
# directories. independent builds (e.g. vulkan-shaders-gen) are not reached
# here; they embed the shared manifest directly, see ggml/src/ggml-vulkan
function(llama_apply_app_manifests)
    if (NOT WIN32)
        return()
    endif()
    get_property(_subdirs DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY SUBDIRECTORIES)
    foreach(_dir IN LISTS _subdirs)
        get_filename_component(_name "${_dir}" NAME)
        if (_name STREQUAL "app" OR _name STREQUAL "examples" OR _name STREQUAL "pocs" OR _name STREQUAL "tests" OR _name STREQUAL "tools")
            llama_apply_app_manifests_dir("${_dir}")
        endif()
    endforeach()
endfunction()
