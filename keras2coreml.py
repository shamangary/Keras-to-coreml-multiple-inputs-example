from SSRNET_model import SSR_net
import coremltools
from coremltools.proto import NeuralNetwork_pb2

import sys, os
import numpy as np

import keras
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import load_img, img_to_array


# The conversion function for Lambda layers.
def convert_lambda(layer):
    # Only convert this Lambda layer if it is for our swish function.
    
    if layer.name == 'pred_a':
        params = NeuralNetwork_pb2.CustomLayerParams()

        # The name of the Swift or Obj-C class that implements this layer.
        params.className = "SSR_module"

        # The desciption is shown in Xcode's mlmodel viewer.
        params.description = "Soft Stagewise Regression"
        params.parameters["s1"].doubleValue = layer.arguments['s1']
        params.parameters["s2"].doubleValue = layer.arguments['s2']
        params.parameters["s3"].doubleValue = layer.arguments['s3']
        params.parameters["lambda_d"].doubleValue = layer.arguments['lambda_d']
        params.parameters["lambda_local"].doubleValue = layer.arguments['lambda_local']

        return params
    else:
        return None

def main():
    
    weight_file = "./ssrnet_3_3_3_64_1.0_1.0.h5"

    # load model and weights
    img_size = 64
    stage_num = [3,3,3]
    lambda_local = 1
    lambda_d = 1
    model = SSR_net(img_size,stage_num, lambda_local, lambda_d)()
    model.load_weights(weight_file)
    model.save('ssrnet.h5')
    model.summary()


    coreml_model = coremltools.converters.keras.convert(
        model,
        input_names="image",
        image_input_names="image",
        output_names="output",
        add_custom_layers=True,
        custom_conversion_functions={ "Lambda": convert_lambda })


    # Look at the layers in the converted Core ML model.
    print("\nLayers in the converted model:")
    for i, layer in enumerate(coreml_model._spec.neuralNetwork.layers):
        if layer.HasField("custom"):
            print("Layer %d = %s --> custom layer = %s" % (i, layer.name, layer.custom.className))
        else:
            print("Layer %d = %s" % (i, layer.name))
    coreml_model.save('ssrnet.mlmodel')


if __name__ == '__main__':
    main()