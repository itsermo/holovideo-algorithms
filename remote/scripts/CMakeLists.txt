cmake_minimum_required(VERSION 2.8)

project (holovideo-scripts)

file(GLOB ScriptFiles "*.sh") 

add_custom_target(copy-remote-scripts)
get_target_property(MyTargetLocation RemoteQT LOCATION)
get_filename_component(MyTargetDir ${MyTargetLocation} PATH)
foreach(ScriptFile ${ScriptFiles})
  add_custom_command(TARGET copy-remote-scripts PRE_BUILD
                     COMMAND ${CMAKE_COMMAND} -E
                         copy ${ScriptFile} ${MyTargetDir})
endforeach()

add_dependencies(RemoteQT copy-remote-scripts)
#file(COPY nvidia-framelock-enable DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
#file(COPY nvidia-framelock-disable DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
#file(COPY holovideo-enable DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
#file(COPY holovideo-disable DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
