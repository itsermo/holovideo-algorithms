cmake_minimum_REQUIRED(VERSION 2.8)

project(ripgen-fbo)

#file(COPY ripstart DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

file(GLOB ScriptFiles "*.sh")

add_custom_target(copy-ripgen-scripts)
get_target_property(MyTargetLocation ripgen-fbo LOCATION)
get_filename_component(MyTargetDir ${MyTargetLocation} PATH)
foreach(ScriptFile ${ScriptFiles})
  add_custom_command(TARGET copy-ripgen-scripts PRE_BUILD
                     COMMAND ${CMAKE_COMMAND} -E
                         copy_if_different ${ScriptFile} ${MyTargetDir})
endforeach()
add_dependencies(ripgen-fbo copy-ripgen-scripts)
