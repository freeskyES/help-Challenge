#! /bin/bash

dir=$(dirname "$0")

docker build -t eunsong-tensorflow-1.14.0-gpu-py3 -f $dir/$dirDockerfile-1.14.0-gpu-py3