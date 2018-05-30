import Foundation
import CoreML
import Accelerate

@objc(SSR_module) class SSR_module: NSObject, MLCustomLayer {
    
    required init(parameters: [String : Any]) throws {
        print(#function, parameters)
        super.init()
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        let out_shape: [NSNumber] = [1,1,1,1,1]
        return [out_shape]
        // Somehow the output shape has to be 5-dim.
        // Even the real prediction is only 1-dim.
        // Just check the input shape after you use keras coreml converter and follow the pattern of it.
    }
    
    func setWeightData(_ weights: [Data]) throws {
        print(#function, weights)
    }
    
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
}
