#!/bin/bash

read -p 'Number of engines to spawn: ' num
for (( i=0; i<num; ++i )); do
	ipengine --profile=ssh &
done
