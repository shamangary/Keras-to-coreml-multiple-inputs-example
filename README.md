# Keras-to-coreml-Using-ssrnet-as-a-multiple-inputs-custom-lambda-layer-example
keras to coreml example


## Introduction
Considering Keras is a convenience framework for building deep learning structure, we usually use it to develop our own network. However, network with complex custom layer is not directly supported by the "coremltools", and cannot be convert to coreml model directly (iOS friendly framework).

+ In this project, I share the way to rewrite the multiple inputs custom lambda layer in Keras into swift function to support coreml model.
+ I use SSR-Net as our example. For more information, please go to https://github.com/shamangary/SSR-Net

## How to run?
```
python keras2coreml.py
```
This command convert the keras model and weight file into coreml model.

### Multiple inputs Lambda layer bug?
Before I write this github repositori, coremltools does not support multiple inputs Lambda layer.
https://github.com/apple/coremltools/issues/188
In the above issue, they fixed it by adding some lines into their code. Plz check it out.

## Custom swift class
After the conversion between keras and coreml, the model is not directly usable for iOS app.
You need to define your own class for the custom layer for the model.

The guide of http://machinethink.net/blog/coreml-custom-layers/ is very useful.
However, we need complex custom Lambda layer other than 1 input 1 output.

Considering our Soft Stagewise Regression (SSR_module) contains 9 inputs and 1 output, 
I rewrite the swift class "SSR_module.swift".


## Reference
+ http://machinethink.net/blog/coreml-custom-layers/
+ https://github.com/hollance/CoreML-Custom-Layers
+ https://github.com/apple/coremltools
