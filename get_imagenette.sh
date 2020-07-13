#!/bin/bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
mkdir data
tar xzvf imagenette2.tgz -C data 
rm imagenette2.tgz