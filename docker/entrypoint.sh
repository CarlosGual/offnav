#!/bin/bash

. activate habitat
cd ~/code/ && pip install -e .
bash scripts/dgx/launch_eval.sh