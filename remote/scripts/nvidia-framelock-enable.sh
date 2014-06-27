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
      eval ./nvidia-framelock-disable.sh
      exit 1
  fi
}

eval ./nvidia-framelock-disable.sh

printf "Running nvidia-settings... "
$(${NVIDIA_SETTINGS_COMMAND} -p "Frame Lock") &
check_return

#printf "Querying all GPUs with framelock capability... "
#eval ${NVIDIA_SETTINGS_COMMAND} -V all -q gpus >> /dev/null
#check_return

#printf "Querying GPU refresh rates... "
#eval ${NVIDIA_SETTINGS_COMMAND} -q [dpy]/RefreshRate >> /dev/null
#check_return

#printf "Querying the valid framelock configurations for the display devices... "
#eval ${NVIDIA_SETTINGS_COMMAND} -q [dpy]/FrameLockDisplayConfig >> /dev/null
#check_return

printf "Setting the master display device... "
eval ${NVIDIA_SETTINGS_COMMAND} -a [gpu:0.DVI-I-0]/FrameLockDisplayConfig=2 >> /dev/null
check_return

printf "  Setting slave display device 0... "
eval ${NVIDIA_SETTINGS_COMMAND} -a [gpu:0.DVI-I-1]/FrameLockDisplayConfig=1 >> /dev/null
check_return

for ((i=1; i < ${NUM_GPUS}; i++))
do
	printf "  Setting slave display device $i..."
	eval ${NVIDIA_SETTINGS_COMMAND} -a [gpu:$i.dpy]/FrameLockDisplayConfig=1 >> /dev/null
	check_return
done

printf "Enabling frame lock on GPUs... "
eval ${NVIDIA_SETTINGS_COMMAND} -a [gpu]/FrameLockEnable=1 >> /dev/null 
check_return

#for ((i=0; i < ${NUM_GPUS}; i++))
#do
#	printf "Enabling frame lock on GPU $i... "
#	eval ${NVIDIA_SETTINGS_COMMAND} -a [gpu:$i]/FrameLockEnable=1 >> /dev/null
#	check_return
#done

printf "Performing test signal on GPU 0... "
eval ${NVIDIA_SETTINGS_COMMAND} -a [gpu:0]/FrameLockTestSignal=1
check_return

printf "Disabling test signal on GPU 0... "
eval ${NVIDIA_SETTINGS_COMMAND} -a [gpu:0]/FrameLockTestSignal=0
check_return
check_return
