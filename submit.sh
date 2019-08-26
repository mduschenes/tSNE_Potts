#!/usr/bin/env bash


# Input arguments

DIRECTORY=${1:-"$PWD"}
CONFIGS=${2:-"template.config"}
OPTIONS=${3:-"template.sh"}
TASK=${4:-"main.py"}


# Generate jobs, and array = {DIRECTORY,COMMAND,SOURCE,TASK}
array=($(python submit.py -dir "$DIRECTORY" -configs $CONFIGS -options $OPTIONS -task $TASK ))

# Get ENV Variables from Array
DIRECTORY=${array[0]}
COMMAND=${array[1]}
SOURCE=${array[2]}
TASK=${array[3]}
echo array ${array[@]}

# Execute Jobs
${COMMAND} ${TASK}




