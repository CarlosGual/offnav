#!/bin/bash

. activate habitat
cd ~/code/ && pip install -e .
bash scripts/1-objectnav-il.sh