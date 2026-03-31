#!/bin/bash
pkill -f tensorboard
tensorboard --logdir=saved/ --samples_per_plugin images=300 --port=6007
