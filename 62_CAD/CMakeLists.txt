include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(EXAMPLE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/Curves.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/Curves.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/Hatch.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/Hatch.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/Polyline.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/DrawResourcesFiller.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/DrawResourcesFiller.h"
  "../../src/nbl/ext/TextRendering/TextRendering.cpp" # TODO: this one will be a part of dedicated Nabla ext called "TextRendering" later on which uses MSDF + Freetype
)
set(EXAMPLE_INCLUDES
  "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/boost/superproject/libs/math/include")
nbl_create_executable_project("${EXAMPLE_SOURCES}" "" "${EXAMPLE_INCLUDES}" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")

option(NBL_CAD_EX_LIST_BUILTIN_RESOURCES "[CAD example]: enable if you want to list dynamic builtin resources as spirv shaders in the example archive - otherwise it will be empty" OFF)

if(NBL_EMBED_BUILTIN_RESOURCES)
	set(_BR_TARGET_ ${EXECUTABLE_NAME}_builtin_spirv)

	# shaders IO directories
	set(NBL_CAD_INPUT_SHADERS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
	get_filename_component(_CAD_SPIRV_BR_BUNDLE_SEARCH_DIRECTORY_ "${CMAKE_CURRENT_BINARY_DIR}/shaders/include" ABSOLUTE)
	get_filename_component(_CAD_SPIRV_BR_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/builtin/include" ABSOLUTE)
	get_filename_component(_CAD_SPIRV_BR_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/builtin/src" ABSOLUTE)

	# valid key for a resource is ${NBL_ARCHIVE_ABSOLUTE_PATH}/<entry path>, I put all <entry path>s as ${NBL_ARCHIVE_COMMON_ENTRY} for simplicity
	set(NBL_BUILTIN_RESOURCES_NAMESPACE "nbl::this_example::builtin") # do not change, will break compilation because of hardcoded MonoAssetManagerAndBuiltinResourceApplication inheritance & usage
	set(NBL_ARCHIVE_ABSOLUTE_PATH "nbl/cad")
	set(NBL_ARCHIVE_COMMON_ENTRY "shaders/spirv")

	set(NBL_CAD_OUTPUT_SPIRV_DIRECTORY "${_CAD_SPIRV_BR_BUNDLE_SEARCH_DIRECTORY_}/${NBL_ARCHIVE_ABSOLUTE_PATH}/${NBL_ARCHIVE_COMMON_ENTRY}")

	# lets make use of NSC if enabled, otherwise no changes & at and we have empty archive - see comments regarding ComputerAidedDesign class definition in main.cpp!
	if(NBL_CAD_EX_LIST_BUILTIN_RESOURCES)
		message(FATAL_ERROR "CAD example uses device capabilities & NSC doesn't support it yet! Remove once fixed.")

		set(NBL_ABS_INPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/app_resources")

		# only headers which are included to compiled shaders
		set(NBL_CAD_INPUT_COMMONS
			"${NBL_ABS_INPUT_PATH}/common.hlsl"
		)

		macro(NBL_LIST_DYNAMIC_BUILTIN_RESOURCE _NBL_INPUT_SHADER_ _NBL_STAGE_)
			set(NBL_INPUT_SHADER "${_NBL_INPUT_SHADER_}")
			set(NBL_STAGE "${_NBL_STAGE_}")

			cmake_path(GET NBL_INPUT_SHADER STEM NBL_SHADER_STEM)
			set(NBL_OUTPUT_SPIRV_FILENAME "${NBL_SHADER_STEM}.spv")
			set(NBL_OUTPUT_SPIRV_PATH "${NBL_CAD_OUTPUT_SPIRV_DIRECTORY}/${NBL_OUTPUT_SPIRV_FILENAME}")

			if(NBL_STAGE STREQUAL vertex)
				set(NBL_NSC_COMPILE_OPTIONS -T vs_6_7 -E main)
			elseif(NBL_STAGE STREQUAL fragment)
				set(NBL_NSC_COMPILE_OPTIONS -T ps_6_7 -E main)
			else()
				message(FATAL_ERROR "Internal error, invalid NBL_STAGE = \"${NBL_STAGE}\"")
			endif()
	
			set(NBL_NSC_COMPILE_COMMAND
				"$<TARGET_FILE:nsc>"
				-Fc "${NBL_OUTPUT_SPIRV_PATH}"
				${NBL_NSC_COMPILE_OPTIONS}
				"${NBL_INPUT_SHADER}"
			)
	
			set(NBL_DEPENDS
				"${NBL_INPUT_SHADER}"
				${NBL_CAD_INPUT_COMMONS}
			)
		
			add_custom_command(OUTPUT "${NBL_OUTPUT_SPIRV_PATH}"
			   COMMAND ${NBL_NSC_COMPILE_COMMAND}
			   DEPENDS ${NBL_DEPENDS}
			   COMMENT "Generating \"${NBL_OUTPUT_SPIRV_PATH}\""
			   VERBATIM
			   COMMAND_EXPAND_LISTS
			)
	
			list(APPEND NBL_CAD_OUTPUT_SPIRV_BUILTINS "${NBL_OUTPUT_SPIRV_PATH}")
			LIST_BUILTIN_RESOURCE(CAD_SPIRV_RESOURCES_TO_EMBED "${NBL_ARCHIVE_COMMON_ENTRY}/${NBL_OUTPUT_SPIRV_FILENAME}")
		endmacro()

		# only shaders which are being compiled, no extra headers there
		NBL_LIST_DYNAMIC_BUILTIN_RESOURCE("${NBL_ABS_INPUT_PATH}/fragment_shader.hlsl" fragment)
		NBL_LIST_DYNAMIC_BUILTIN_RESOURCE("${NBL_ABS_INPUT_PATH}/fragment_shader_debug.hlsl" fragment)
		NBL_LIST_DYNAMIC_BUILTIN_RESOURCE("${NBL_ABS_INPUT_PATH}/resolve_alphas.hlsl" fragment)
		NBL_LIST_DYNAMIC_BUILTIN_RESOURCE("${NBL_ABS_INPUT_PATH}/vertex_shader.hlsl" vertex)

		target_compile_definitions(${EXECUTABLE_NAME} PUBLIC NBL_CAD_EX_LIST_BUILTIN_RESOURCES) # proxy pass to the example as #define
	endif()

	ADD_CUSTOM_BUILTIN_RESOURCES(${_BR_TARGET_} CAD_SPIRV_RESOURCES_TO_EMBED "${_CAD_SPIRV_BR_BUNDLE_SEARCH_DIRECTORY_}" "${NBL_ARCHIVE_ABSOLUTE_PATH}" "${NBL_BUILTIN_RESOURCES_NAMESPACE}" "${_CAD_SPIRV_BR_OUTPUT_DIRECTORY_HEADER_}" "${_CAD_SPIRV_BR_OUTPUT_DIRECTORY_SOURCE_}" "STATIC" "INTERNAL")
	LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} ${_BR_TARGET_})
endif()

# if enabled then try use Nabla "Text Rendering" extension 
# with an implemented interface using the 3rdparty deps 

set(NBL_CAD_EX_USE_TEXT_RENDERING_EXT OFF) # do not enable, for future usage when the extension is written

if(NBL_BUILD_TEXT_RENDERING AND NBL_CAD_EX_USE_TEXT_RENDERING_EXT)
	add_dependencies(${EXECUTABLE_NAME} ${NBL_EXT_TEXT_RENDERING_TARGET})
	target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${NBL_EXT_TEXT_RENDERING_TARGET})
	target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:${NBL_EXT_TEXT_RENDERING_TARGET},INCLUDE_DIRECTORIES>)
else()
	# Freetype
	add_dependencies(${EXECUTABLE_NAME} freetype)
	target_link_libraries(${EXECUTABLE_NAME} PRIVATE freetype)
	target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:freetype,INCLUDE_DIRECTORIES>)

	# msdfgen
	add_dependencies(${EXECUTABLE_NAME} ${NBL_MSDFGEN_TARGETS})
	target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${NBL_MSDFGEN_TARGETS})
	foreach(NBL_TARGET IN LISTS NBL_MSDFGEN_TARGETS)
		target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:${NBL_TARGET},INCLUDE_DIRECTORIES>)
	endforeach()
endif()
