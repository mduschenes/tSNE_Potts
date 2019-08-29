#!/usr/bin/env bash


# Options
TEST=false
while getopts "T" o; do
	case "$o" in
		T)	TEST=true;;
  #       shift # The double dash which separates options from parameters
  #       break
  #       ;; # Exit the loop using break command
	esac
done
shift $((OPTIND - 1))

# Input arguments

DIRECTORY=${1:-""}
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
if $TEST; then
	echo "Command: ${COMMAND} ${TASK} in $DIRECTORY"
else
	${COMMAND} ${TASK}
fi




