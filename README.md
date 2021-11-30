# Object_detection

Step 1:
Download this repository

# Installation
pip install -U --pre tensorflow=="2.*"

pip install pycocotools

Step 2:

Extract the downloaded zip folder

Step 3:

cd models/research

protoc object_detection/protos/*.proto --python_out=.

Step 4:

pip install object_detection

Step 5:

Run the main_code.py

For reference: https://medium.com/@techmayank2000/object-detection-using-ssd-mobilenetv2-using-tensorflow-api-can-detect-any-single-class-from-31a31bbd0691
