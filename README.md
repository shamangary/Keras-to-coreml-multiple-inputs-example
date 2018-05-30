# Keras-to-coreml-Using-ssrnet-as-a-multiple-inputs-custom-lambda-layer-example
keras to coreml example


## Introduction
Considering Keras is a convenient framework for building deep learning structure, we usually use it to develop our own network. 

However, network with complex custom layer is not directly supported by the "coremltools", and cannot be easily convert to coreml model (iOS friendly framework).

+ In this project, I share the way to rewrite the ***multiple inputs custom Lambda layer in Keras*** into swift function to support coreml model.
+ I use SSR-Net as our example. For more information, please go to https://github.com/shamangary/SSR-Net

## How to run?
```
python keras2coreml.py
```
This command convert the keras model and weight file into coreml model.

### Multiple inputs Lambda layer bug?
Before I write this github repository, coremltools does not support multiple inputs Lambda layer.

https://github.com/apple/coremltools/issues/188

In the above issue, they fixed it by adding some lines into their code. Plz check it out.

## Custom swift class
After the conversion between keras and coreml, the model is not directly usable for iOS app.
You need to define your own class for the custom layer for the model.

The guide of http://machinethink.net/blog/coreml-custom-layers/ is very useful.
However, we need a more complex custom Lambda layer other than 1 input 1 output.

Considering our Soft Stagewise Regression (SSR_module) contains 9 inputs and 1 output, 
I rewrite the swift class "SSR_module.swift".

```
# Target custom layer in Keras SSR-Net model

        def merge_age(x,s1,s2,s3,lambda_local,lambda_d):
            a = x[0][:,0]*0
            b = x[0][:,0]*0
            c = x[0][:,0]*0
            V = 101

            for i in range(0,s1):
                a = a+(i+lambda_local*x[6][:,i])*x[0][:,i]
            a = K.expand_dims(a,-1)
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j+lambda_local*x[7][:,j])*x[1][:,j]
            b = K.expand_dims(b,-1)
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k+lambda_local*x[8][:,k])*x[2][:,k]
            c = K.expand_dims(c,-1)
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))


            age = (a+b+c)*V
            return age
        
        pred_a = Lambda(merge_age,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_local':self.lambda_local,'lambda_d':self.lambda_d},output_shape=(1,),name='pred_a')([pred_a_s1,pred_a_s2,pred_a_s3,delta_s1,delta_s2,delta_s3, local_s1, local_s2, local_s3])


## Swift class version evaluation function

    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        print(#function, inputs.count, outputs.count)
        var a: Double = 0
        for i in 0...2 {
            let index_a: [NSNumber] = [0,0,i as NSNumber,0,0]
            a = a + (Double(i)+inputs[6][index_a].doubleValue)*inputs[0][index_a].doubleValue
        }
        let index_0: [NSNumber] = [0,0,0,0,0]
        a = a/(3.0*(1.0+inputs[3][index_0].doubleValue))
        
        var b: Double = 0
        for j in 0...2 {
            let index_b: [NSNumber] = [0,0,j as NSNumber,0,0]
            b = b + (Double(j)+inputs[7][index_b].doubleValue)*inputs[1][index_b].doubleValue
        }
        b = b/(3.0*(1.0+inputs[3][index_0].doubleValue))/(3.0*(1.0+inputs[4][index_0].doubleValue))
        
        var c: Double = 0
        for k in 0...2 {
            let index_c: [NSNumber] = [0,0,k as NSNumber,0,0]
            c = c + (Double(k)+inputs[8][index_c].doubleValue)*inputs[2][index_c].doubleValue
        }
        c = c/(3.0*(1.0+inputs[3][index_0].doubleValue))/(3.0*(1.0+inputs[4][index_0].doubleValue))/(3.0*(1.0+inputs[5][index_0].doubleValue))
        
        let age: Double = (a+b+c)*101.0
        print(age) // for xcode console debug
        outputs[0][index_0] = NSNumber(value: age)
        
    }
```


## Reference
+ http://machinethink.net/blog/coreml-custom-layers/
+ https://github.com/hollance/CoreML-Custom-Layers
+ https://github.com/apple/coremltools
