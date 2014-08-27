cmake_minimum_REQUIRED(VERSION 2.8)

project(holodepth)

file(GLOB ScriptFiles "*.sh")

add_custom_target(copy-holodepth-scripts)
get_target_property(MyTargetLocation holodepth LOCATION)
get_filename_component(MyTargetDir ${MyTargetLocation} PATH)
foreach(ScriptFile ${ScriptFiles})
  add_custom_command(TARGET copy-holodepth-scripts PRE_BUILD
                     COMMAND ${CMAKE_COMMAND} -E
                         copy_if_different ${ScriptFile} ${MyTargetDir})
endforeach()
add_dependencies(holodepth copy-holodepth-scripts)
