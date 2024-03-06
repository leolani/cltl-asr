#!/bin/bash

### Download whisper.cpp ###
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp


### Build whisper.cpp and download the model ###
# For Intel hardware run only `make` without any parameters
WHISPER_COREML=1 make -j
./models/download-ggml-model.sh medium


### Generate the model for the CoreML framework ###
# Python 3.10 is recommended
python -m venv venv
source venv/bin/activate

pip install ane_transformers
pip install openai-whisper
# Ensure XCode is installed from the app store!
pip install coremltools

./models/generate-coreml-model.sh medium

deactivte


### Run the serveer ###
./server --port 8989 -m models/ggml-medium.bin