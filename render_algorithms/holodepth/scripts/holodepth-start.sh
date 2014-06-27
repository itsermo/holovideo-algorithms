#!/bin/bash
DISPLAY=localhost:0.0 ./holodepth 2 &#>/dev/null &
DISPLAY=localhost:0.1 ./holodepth 1 >/dev/null &
DISPLAY=localhost:0.2 ./holodepth 0 >/dev/null &
