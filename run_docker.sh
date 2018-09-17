#nvidia-docker run -it --rm -v $PWD:/home nvidia/cuda:9.0-cudnn7-devel bash	
nvidia-docker run -it --rm -v $PWD:/root tensorflow/tensorflow:latest-devel-gpu-py3 bash
