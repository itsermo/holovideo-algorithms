#!/bin/sh

if [ $1 ]; then MODEL=$1
	echo Using model $MODEL 
else
	MODEL='../models/renderer.xml-ripcube'
	echo Defaulting to model $MODEL. Specify an argument for other models.
fi

echo using model $MODEL. 

#DISPLAY=:0.0 ./ripgen 4 $MODEL $2 > /dev/null &
#DISPLAY=:0.1 ./ripgen 2 $MODEL $2 > /dev/null &
#DISPLAY=:0.2 ./ripgen 0 $MODEL $2 > /dev/null &

DISPLAY=:0.0 './ripgen-fbo' 0 $MODEL $2  &
DISPLAY=:0.1 './ripgen-fbo' 2 $MODEL $2  &
#sleep may help decide who gets mouse focus at launch. Used for SID'10 experiment 
#sleep 2
DISPLAY=:0.2 './ripgen-fbo' 4 $MODEL $2  &
