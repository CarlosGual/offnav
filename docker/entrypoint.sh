#!/bin/bash

. activate habitat
cd ~/code/ && pip install -e .
bash offnav/scripts/launch_training.sh