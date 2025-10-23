# RaceComp
Comparison of CNN and ViT for image upscaling and facial recognition tasks trained on racially biased datasets

## CNNs

### ```cnn-unet.py```
This is a full-U UNet super resolution model with ResNet50 as the encoder backbone. It expects an input tensor of 3x112x12, and upscales to a tensor 3x224x224. Between layers, the decoder upscaling may be center-cropped to match the encoder size, as implemented in the original UNet paper (but here decoder outputs are larger than encoders therefore it is the decoder that is cropped).

### ```cnn_abridged-unet.py```
This is a half-U UNet super resolution model with ResNet50 as the encoder backbone. The top half of the U is not bridged to avoid cropping (zero-pad is used in the lower levels). 

## Data structure
The FairFace dataset should be implemented in the structure shown in ```training_data_structure.jpg```. Reference ```load.py``` to see how the data is loaded and what the resulting data structure may be like. ```data_testbed.ipynb``` allows for interactive viewing and testing of loading data.