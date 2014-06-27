#!/bin/bash
NUM_GPUS=3
CONTROL_DISPLAY=:0.0
NVIDIA_SETTINGS_PATH="/usr/bin/nvidia-settings"
NVIDIA_SETTINGS_COMMAND="${NVIDIA_SETTINGS_PATH} -c ${CONTROL_DISPLAY}"

RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
NORMAL=$(tput sgr0)

check_return()
{
  if [ $? -eq 0 ]
    then
      echo "$GREEN" '[OK]' "$NORMAL"
    else
      echo "$RED" '[FAILED]' "$NORMAL"
      exit 1
  fi
}

printf "Killing any nvidia-settings instance... "
eval killall nvidia-settings

printf "Disabling current framelock... "
eval ${NVIDIA_SETTINGS_COMMAND} -a [gpu]/FrameLockEnable=0 -a [framelock:0]/FrameLockUseHouseSync=0 >> /dev/null
check_return

printf "Disabling house sync signal on master device... "
eval ${NVIDIA_SETTINGS_COMMAND} -a [framelock:0]/FrameLockUseHouseSync=0 >> /dev/null
check_return
