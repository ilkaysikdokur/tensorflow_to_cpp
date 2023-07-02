class Layer:
    layerType = ''
    isTrainable = False
    batchSize = 0
    useBias = False
    applyTraining = False
        
        
        
class Input(Layer):
    imageX = 0
    imageY = 0
    imageChannel = 0
    epoch = 1
    classSize = 0
    nTrain = 0
    nTest = 0
    trainLocation= ''
    testLocation= ''
    
    
    def __init__(self, imageX, imageY, imageChannel, epoch, batchSize, classSize, nTrain, nTest):
        Layer.layerType = 'Input'
        Layer.isTrainable = False
        self.imageX = imageX
        self.imageY = imageY
        self.imageChannel = imageChannel
        self.epoch = epoch
        Layer.batchSize = batchSize
        self.classSize = classSize
        self.nTrain = nTrain
        self.nTest = nTest
        
        
class Conv2D(Layer):
    kernelX = 0
    kernelY = 0
    filterAmount = 0
    padding = 0
    stride = 1
    activation = ''
    def __init__(self, kernelX, kernelY, filterAmount, padding=0, stride=1, applyTraining=True, useBias=True, activation=None):
        Layer.layerType = 'Conv2D'
        Layer.isTrainable = True
        self.kernelX = kernelX
        self.kernelY = kernelY
        self.filterAmount = filterAmount
        self.padding = padding
        self.stride = stride
        Layer.applyTraining = applyTraining
        Layer.useBias = useBias
        self.activation = activation
    
        
class MaxPool2D(Layer):
    poolX = 2
    poolY = 2
    def __init__(self, poolX=2, poolY=2):
        Layer.layerType = 'MaxPool2D'
        Layer.isTrainable = False
        self.poolX = poolX
        self.poolY = poolY
        
        
class AvgPool2D(Layer):
    poolX = 2
    poolY = 2
    def __init__(self, poolX=2, poolY=2):
        Layer.layerType = 'AvgPool2D'
        Layer.isTrainable = False
        self.poolX = poolX
        self.poolY = poolY
        
        
class Flatten(Layer):
    length = 0
    def __init__(self):
        Layer.layerType = 'Flatten'
        Layer.isTrainable = False
        
        
class Dense(Layer):
    length = 0
    activation = ''
    def __init__(self, length, applyTraining=True, useBias=True, activation=None):
        Layer.layerType = 'Dense'
        Layer.isTrainable = True
        self.length = length
        Layer.applyTraining = applyTraining
        Layer.useBias = useBias
        self.activation = activation

class Sequential:
    def __init__(self):
        self.modelDict = []
        
    def add(self, layer):
        layerTemp = {}
        layerTemp['layerType'] = layer.layerType
        layerTemp['isTrainable'] = layer.isTrainable
        
        if layer.isTrainable == True:
            layerTemp['applyTraining'] = layer.applyTraining
            layerTemp['useBias'] = layer.useBias
        
        if layer.layerType == 'Input':
            layerTemp['imageX'] = layer.imageX
            layerTemp['imageY'] = layer.imageY
            layerTemp['imageChannel'] = layer.imageChannel
            layerTemp['epoch'] = layer.epoch
            layerTemp['batchSize'] = layer.batchSize
            layerTemp['classSize'] = layer.classSize
            layerTemp['nTrain'] = layer.nTrain
            layerTemp['nTest'] = layer.nTest
        
        if layer.layerType == 'Conv2D':
            layerTemp['kernelX'] = layer.kernelX
            layerTemp['kernelY'] = layer.kernelY
            layerTemp['filterAmount'] = layer.filterAmount
            layerTemp['padding'] = layer.padding
            layerTemp['stride'] = layer.stride
            layerTemp['activation'] = layer.activation
            layerTemp['isTrainable'] = layer.isTrainable
            layerTemp['useBias'] = layer.useBias

            
            for i in range(len(self.modelDict)-1, -1, -1):
                if i != 0:
                    if self.modelDict[i]['isTrainable'] == True:
                        layerTemp['filterAmountPrev']=self.modelDict[i]['filterAmount']
                        break
                else:
                    layerTemp['filterAmountPrev']=self.modelDict[i]['imageChannel']
                    
            for i in range(len(self.modelDict)-1, -1, -1):
                if i != 0:
                    layerTemp['productX']=(self.modelDict[i]['productX'] - layer.kernelX + 2*layer.padding)//layer.stride + 1
                    layerTemp['productY']=(self.modelDict[i]['productY'] - layer.kernelY + 2*layer.padding)//layer.stride + 1
                    layerTemp['productZ']=layer.filterAmount
                    break
                else:
                    layerTemp['productX']=(self.modelDict[i]['imageX'] - layer.kernelX + 2*layer.padding)//layer.stride + 1
                    layerTemp['productY']=(self.modelDict[i]['imageY'] - layer.kernelY + 2*layer.padding)//layer.stride + 1
                    layerTemp['productZ']=layer.filterAmount
            
        if layer.layerType == 'MaxPool2D':
            layerTemp['poolX'] = layer.poolX
            layerTemp['poolY'] = layer.poolY
            layerTemp['applyTraining'] = layer.applyTraining
            
            for i in range(len(self.modelDict)-1, -1, -1):
                if i != 0:
                    layerTemp['productX']=self.modelDict[i]['productX']//layer.poolX
                    layerTemp['productY']=self.modelDict[i]['productY']//layer.poolY
                    layerTemp['productZ']=self.modelDict[i]['filterAmount']
                    break
                else:
                    layerTemp['productX']=self.modelDict[i]['imageX']//layer.poolX
                    layerTemp['productY']=self.modelDict[i]['imageY']//layer.poolY
                    layerTemp['productZ']=self.modelDict[i]['imageChannel']
                    
        if layer.layerType == 'AvgPool2D':
            layerTemp['poolX'] = layer.poolX
            layerTemp['poolY'] = layer.poolY
            layerTemp['applyTraining'] = layer.applyTraining
            
            for i in range(len(self.modelDict)-1, -1, -1):
                if i != 0:
                    layerTemp['productX']=self.modelDict[i]['productX']//layer.poolX
                    layerTemp['productY']=self.modelDict[i]['productY']//layer.poolY
                    layerTemp['productZ']=self.modelDict[i]['filterAmount']
                    break
                else:
                    layerTemp['productX']=self.modelDict[i]['imageX']//layer.poolX
                    layerTemp['productY']=self.modelDict[i]['imageY']//layer.poolY
                    layerTemp['productZ']=self.modelDict[i]['imageChannel']
                    
        if layer.layerType == 'Flatten':
            for i in range(len(self.modelDict)-1, -1, -1):
                if i != 0:
                    layerTemp['length'] = self.modelDict[i]['productX']*self.modelDict[i]['productY']*self.modelDict[i]['productZ']
                    break
                else:
                    layerTemp['length'] = self.modelDict[i]['imageX']*self.modelDict[i]['imageY']*self.modelDict[i]['imageChannel']
        
        if layer.layerType == 'Dense':
            layerTemp['activation'] = layer.activation
            layerTemp['isTrainable'] = layer.isTrainable
            layerTemp['useBias'] = layer.useBias
            for i in range(len(self.modelDict)-1, -1, -1):
                if i != 0:
                    layerTemp['length']=layer.length
                    layerTemp['lengthPrev']=self.modelDict[i]['length']
                    break

        self.modelDict.append(layerTemp)
        
    
    def optimizer(self, optimizerType, learningRate=0.01, useShuffle=True, beta_coeffs=[0.9, 0.999], epsilon=0.0000001):
        self.modelDict[len(self.modelDict)-1]['optimizerType'] = optimizerType
        self.modelDict[len(self.modelDict)-1]['learningRate'] = learningRate
        self.modelDict[len(self.modelDict)-1]['useShuffle'] = useShuffle
        self.modelDict[len(self.modelDict)-1]['beta_coeffs'] = beta_coeffs
        self.modelDict[len(self.modelDict)-1]['epsilon'] = epsilon
        
    def setDataset(self, trainLocation='', testLocation=''):
        self.modelDict[0]['trainLocation'] = trainLocation
        self.modelDict[0]['testLocation'] = testLocation
        
def exportHLS(modelDict, mode, inferenceIndex=0, parallelFactor=0):
    
    #design.hpp building
    designHpp = ''
    
    designHpp += '#include <iostream>\n'
    designHpp += '#include <cstdlib>\n'
    designHpp += '#include <ctime>\n'
    designHpp += '#include <cstring>\n'
    designHpp += '#include <random>\n'
    designHpp += '#include <vector>\n'
    designHpp += '#include <cmath>\n\n'
    designHpp += 'using namespace std;\n\n'
    
    
    designHpp += '//Layer 0: Input\n'
    designHpp += '#define IMAGEX '+str(modelDict[0]['imageX'])+'\n'
    designHpp += '#define IMAGEY '+str(modelDict[0]['imageY'])+'\n'
    designHpp += '#define IMAGECHANNEL '+str(modelDict[0]['imageChannel'])+'\n'
    designHpp += '#define EPOCH '+str(modelDict[0]['epoch'])+'\n'
    designHpp += '#define BATCHSIZE '+str(modelDict[0]['batchSize'])+'\n'
    designHpp += '#define CLASSSIZE '+str(modelDict[0]['classSize'])+'\n'
    designHpp += '#define NTRAIN '+str(modelDict[0]['nTrain'])+'\n'
    designHpp += '#define NTEST '+str(modelDict[0]['nTest'])+'\n\n'
    designHpp += '#define PRODUCTX0 '+str(modelDict[0]['imageX'])+'\n'
    designHpp += '#define PRODUCTY0 '+str(modelDict[0]['imageY'])+'\n'
    designHpp += '#define PRODUCTZ0 '+str(modelDict[0]['imageChannel'])+'\n'
    
    for i in range(1, len(modelDict)):
        layerDict = modelDict[i]
        
        designHpp += '//Layer '+str(i)+': '+layerDict['layerType']+'\n'
        
        if layerDict['layerType'] == 'Conv2D':
            designHpp += '#define KERNELX'+str(i)+' '+str(layerDict['kernelX'])+'\n'
            designHpp += '#define KERNELY'+str(i)+' '+str(layerDict['kernelY'])+'\n'
            designHpp += '#define FILTERAMOUNT'+str(i)+' '+str(layerDict['filterAmount'])+'\n'
            designHpp += '#define PADDING'+str(i)+' '+str(layerDict['padding'])+'\n'
            designHpp += '#define STRIDE'+str(i)+' '+str(layerDict['stride'])+'\n'
            designHpp += '#define FILTERAMOUNTPREV'+str(i)+' '+str(layerDict['filterAmountPrev'])+'\n'
            designHpp += '#define PRODUCTX'+str(i)+' '+str(layerDict['productX'])+'\n'
            designHpp += '#define PRODUCTY'+str(i)+' '+str(layerDict['productY'])+'\n'
            designHpp += '#define PRODUCTZ'+str(i)+' '+str(layerDict['productZ'])+'\n\n'
        
        
        if layerDict['layerType'] == 'MaxPool2D':
            designHpp += '#define POOLX'+str(i)+' '+str(layerDict['poolX'])+'\n'
            designHpp += '#define POOLY'+str(i)+' '+str(layerDict['poolY'])+'\n'
            designHpp += '#define PRODUCTX'+str(i)+' '+str(layerDict['productX'])+'\n'
            designHpp += '#define PRODUCTY'+str(i)+' '+str(layerDict['productY'])+'\n'
            designHpp += '#define PRODUCTZ'+str(i)+' '+str(layerDict['productZ'])+'\n\n'
        
        if layerDict['layerType'] == 'AvgPool2D':
            designHpp += '#define POOLX'+str(i)+' '+str(layerDict['poolX'])+'\n'
            designHpp += '#define POOLY'+str(i)+' '+str(layerDict['poolY'])+'\n'
            designHpp += '#define PRODUCTX'+str(i)+' '+str(layerDict['productX'])+'\n'
            designHpp += '#define PRODUCTY'+str(i)+' '+str(layerDict['productY'])+'\n'
            designHpp += '#define PRODUCTZ'+str(i)+' '+str(layerDict['productZ'])+'\n\n'
            
            
        if layerDict['layerType'] == 'Flatten':
            designHpp += '#define LENGTH'+str(i)+' '+str(layerDict['length'])+'\n\n'

            
            
        if layerDict['layerType'] == 'Dense':
            designHpp += '#define LENGTH'+str(i)+' '+str(layerDict['length'])+'\n'
            designHpp += '#define LENGTHPREV'+str(i)+' '+str(layerDict['lengthPrev'])+'\n\n'
    
    
        
    designHpp += '\n'
    designHpp += '//Optimizer Constants\n'
    designHpp += '#define ETA '+str(modelDict[len(modelDict)-1]['learningRate'])+'\n'
    if (modelDict[len(modelDict)-1]['optimizerType'] == 'ADAM'):
        designHpp += '#define BETA1 '+str(modelDict[len(modelDict)-1]['beta_coeffs'][0])+'\n'
        designHpp += '#define BETA2 '+str(modelDict[len(modelDict)-1]['beta_coeffs'][1])+'\n'
        designHpp += '#define EPSILON '+str(modelDict[len(modelDict)-1]['epsilon'])+'\n\n'
    
    
    designHpp += '\n'
    designHpp += '//Data Type Definitions\n'
    designHpp += 'typedef double type_weight;\n'
    designHpp += 'typedef int type_index;\n\n'
    
    designHpp += '\n'
    designHpp += '//Weight and Bias Arrays\n'
    for i in range(1, len(modelDict)):
        layerDict = modelDict[i]
        
        
        if (layerDict['layerType'] == 'Conv2D'):
            designHpp += 'extern type_weight F'+str(i)+'[KERNELX'+str(i)+'][KERNELY'+str(i)+'][FILTERAMOUNTPREV'+str(i)+'][FILTERAMOUNT'+str(i)+'];\n'
            if (layerDict['useBias'] == True):
               designHpp += 'extern type_weight BF'+str(i)+'[FILTERAMOUNT'+str(i)+'];\n'
             
        
        if (layerDict['layerType'] == 'Dense'):
            designHpp += 'extern type_weight W'+str(i)+'[LENGTHPREV'+str(i)+'][LENGTH'+str(i)+'];\n'
            if (layerDict['useBias'] == True):
               designHpp += 'extern type_weight BW'+str(i)+'[BATCHSIZE][LENGTH'+str(i)+'];\n'
            
    
    designHpp += '\n\n'
    designHpp += '//Other Function Definition\n'
    
    designHpp += '\n\n'
    designHpp += '//Training Function Definition\n'
    designHpp += 'void train(type_weight I[BATCHSIZE][1+IMAGEX*IMAGEY*IMAGECHANNEL], type_weight POUT[BATCHSIZE][CLASSSIZE], bool isInference);'
    
    
    
    #design.cpp building
    designCpp = ''
    
    designCpp += '#include "design.hpp"\n\n'
    
    
    
    designCpp += '\n'
    designCpp += '//Weight and Bias Arrays\n'
    for i in range(1, len(modelDict)):
        layerDict = modelDict[i]
        
        
        if (layerDict['layerType'] == 'Conv2D'):
            designCpp += 'type_weight F'+str(i)+'[KERNELX'+str(i)+'][KERNELY'+str(i)+'][FILTERAMOUNTPREV'+str(i)+'][FILTERAMOUNT'+str(i)+'] = {0};\n'
            if (layerDict['useBias'] == True):
               designCpp += 'type_weight BF'+str(i)+'[FILTERAMOUNT'+str(i)+'] = {0};\n'
             
        
        if (layerDict['layerType'] == 'Dense'):
            designCpp += 'type_weight W'+str(i)+'[LENGTHPREV'+str(i)+'][LENGTH'+str(i)+'] = {0};\n'
            if (layerDict['useBias'] == True):
               designCpp += 'type_weight BW'+str(i)+'[BATCHSIZE][LENGTH'+str(i)+'] = {0};\n'
    
    
    designCpp += '\n\n'
    designCpp += '//Training Function\n'
    designCpp += 'void train(type_weight I[BATCHSIZE][1+IMAGEX*IMAGEY*IMAGECHANNEL], type_weight POUT[BATCHSIZE][CLASSSIZE], bool isInference){\n'
    if parallelFactor != 0:
        designCpp += '#pragma HLS INTERFACE axis register both port=I\n'
        designCpp += '#pragma HLS INTERFACE axis register both port=POUT\n\n'
        
        for i in range(1, len(modelDict)):
            if (modelDict[i]['layerType'] == 'Conv2D'):
                designCpp += '#pragma HLS RESOURCE variable=F'+str(i)+' core=XPM_MEMORY\n'
                if modelDict[i]['useBias'] == True:
                    designCpp += '#pragma HLS RESOURCE variable=BF'+str(i)+' core=XPM_MEMORY\n'
            if (modelDict[i]['layerType'] == 'Dense'):
                designCpp += '#pragma HLS RESOURCE variable=W'+str(i)+' core=XPM_MEMORY\n'
                if modelDict[i]['useBias'] == True:
                    designCpp += '#pragma HLS RESOURCE variable=BW'+str(i)+' core=XPM_MEMORY\n\n'
        
            
        for i in range(1, len(modelDict)):
            if (modelDict[i]['layerType'] == 'Conv2D'):
                designCpp += '#pragma HLS ARRAY_PARTITION variable=F'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
                if modelDict[i]['useBias'] == True:
                    designCpp += '#pragma HLS ARRAY_PARTITION variable=BF'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
                
            if (modelDict[i]['layerType'] == 'Dense'):
                designCpp += '#pragma HLS ARRAY_PARTITION variable=W'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
                
                if modelDict[i]['useBias'] == True:
                    designCpp += '#pragma HLS ARRAY_PARTITION variable=BW'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n\n'
                
        
    designCpp += '\n'
    designCpp += '\t//Indices\n'
    designCpp += '\t'
    designCpp += 'type_index i, j, k, l, m, n, p;\n\n'
    if (modelDict[len(modelDict)-1]['optimizerType'] == 'ADAM'):
        designCpp += '\t//ADAM Momentums\n'
        for i in range(1, len(modelDict)):
            if modelDict[i]['isTrainable'] == True and modelDict[i]['applyTraining'] == True:
                
                if (modelDict[i]['layerType'] == 'Conv2D'):
                    designCpp += '\t'
                    designCpp += 'static type_weight mF'+str(i)+'[KERNELX'+str(i)+'][KERNELY'+str(i)+'][FILTERAMOUNTPREV'+str(i)+'][FILTERAMOUNT'+str(i)+'] = {0};\n'
                    designCpp += '\t'
                    designCpp += 'static type_weight vF'+str(i)+'[KERNELX'+str(i)+'][KERNELY'+str(i)+'][FILTERAMOUNTPREV'+str(i)+'][FILTERAMOUNT'+str(i)+'] = {0};\n'
                    if (modelDict[i]['useBias'] == True):
                        designCpp += '\t'
                        designCpp += 'static type_weight mBF'+str(i)+'[FILTERAMOUNT'+str(i)+'] = {0};\n'
                        designCpp += '\t'
                        designCpp += 'static type_weight vBF'+str(i)+'[FILTERAMOUNT'+str(i)+'] = {0};\n'
                        
                if (modelDict[i]['layerType'] == 'Dense'):
                    designCpp += '\t'
                    designCpp += 'static type_weight mW'+str(i)+'[LENGTHPREV'+str(i)+'][LENGTH'+str(i)+'] = {0};\n'
                    designCpp += '\t'
                    designCpp += 'static type_weight vW'+str(i)+'[LENGTHPREV'+str(i)+'][LENGTH'+str(i)+'] = {0};\n'
                    if (modelDict[i]['useBias'] == True):
                        designCpp += '\t'
                        designCpp += 'static type_weight mBW'+str(i)+'[BATCHSIZE][LENGTH'+str(i)+'] = {0};\n'
                        designCpp += '\t'
                        designCpp += 'static type_weight vBW'+str(i)+'[BATCHSIZE][LENGTH'+str(i)+'] = {0};\n'
                
    
        if parallelFactor != 0:
            designCpp += '\n'
            
            for i in range(1, len(modelDict)):
                if (modelDict[i]['layerType'] == 'Conv2D'):
                    designCpp += '#pragma HLS ARRAY_PARTITION variable=mF'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
                    designCpp += '#pragma HLS ARRAY_PARTITION variable=vF'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
                    
                    if modelDict[i]['useBias'] == True:
                        designCpp += '#pragma HLS ARRAY_PARTITION variable=mBF'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
                        designCpp += '#pragma HLS ARRAY_PARTITION variable=vBF'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
                if (modelDict[i]['layerType'] == 'Dense'):
                    designCpp += '#pragma HLS ARRAY_PARTITION variable=mW'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
                    designCpp += '#pragma HLS ARRAY_PARTITION variable=vW'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
                    
                    if modelDict[i]['useBias'] == True:
                        designCpp += '#pragma HLS ARRAY_PARTITION variable=mBW'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
                        designCpp += '#pragma HLS ARRAY_PARTITION variable=vBW'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
                
    
    
    designCpp += '\n'
    designCpp += '\t//Products\n'                  
    for i in range(1, len(modelDict)):
        if (modelDict[i]['layerType'] in ['Conv2D', 'MaxPool2D', 'AvgPool2D']):
            designCpp += '\ttype_weight P'+str(i)+'[BATCHSIZE][PRODUCTX'+str(i)+'][PRODUCTY'+str(i)+'][PRODUCTZ'+str(i)+'] = {0};\n'
            if (modelDict[i]['layerType'] in ['MaxPool2D']):
                designCpp += '\ttype_index IndexP'+str(i)+'[BATCHSIZE][PRODUCTX'+str(i)+'][PRODUCTY'+str(i)+'][PRODUCTZ'+str(i)+'][2] = {0};\n'
                                              
        else:
            designCpp += '\ttype_weight P'+str(i)+'[BATCHSIZE][LENGTH'+str(i)+'] = {0};\n'
    
    
    if parallelFactor != 0:
        designCpp += '\n'
        
        for i in range(1, len(modelDict)):
            designCpp += '#pragma HLS ARRAY_PARTITION variable=P'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
         
            
         
    designCpp += '\ttype_weight IArr[BATCHSIZE][1+PRODUCTX0*PRODUCTY0*PRODUCTZ0] = {0};\n\n'
    if parallelFactor != 0:
        designCpp += '\n'
        designCpp += '#pragma HLS ARRAY_PARTITION variable=IArr cyclic factor='+str(parallelFactor)+' dim=0\n'
    
    designCpp += '\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
    if parallelFactor != 0:
        designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
    designCpp += '\t\tfor (j=0 ; j<1+PRODUCTX0*PRODUCTY0*PRODUCTZ0 ; ++j){\n'
    if parallelFactor != 0:
        designCpp += '#pragma HLS PIPELINE\n'
    designCpp += '\t\t\tIArr[i][j] = I[i][j];\n'
    designCpp += '\t\t}\n'
    designCpp += '\t}\n'
    
    designCpp += '\n\n'
    designCpp += '\t//Actual Class Array\n'
    designCpp += '\ttype_index outActual[BATCHSIZE][CLASSSIZE] = {0};\n'
    if parallelFactor != 0:
        designCpp += '#pragma HLS ARRAY_PARTITION variable=outActual cyclic factor='+str(parallelFactor)+' dim=0\n'
        
    designCpp += '\tfor(i=0 ; i<BATCHSIZE ; ++i){\n'
    if parallelFactor != 0:
        designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
    designCpp += '\t\tfor(j=0 ; j<CLASSSIZE ; ++j){\n'
    if parallelFactor != 0:
        designCpp += '#pragma HLS PIPELINE\n'
    designCpp += '\t\t\tif (IArr[i][0] == j) outActual[i][j] = 1;\n'
    designCpp += '\t\t}\n'
    designCpp += '\t}\n\n'

                 
    designCpp += '\n\n'
    designCpp += '\t//Forward Pass\n\n'
    for i in range(1, len(modelDict)):
        designCpp += '\t//Layer '+str(i)+': '+modelDict[i]['layerType']+'\n'
        
        if modelDict[i]['layerType'] == 'Dense' and modelDict[i]['activation'] == 'Softmax':
             designCpp += '\ttype_weight softmaxA'+str(i)+'[BATCHSIZE] = {0};\n\n'
             if parallelFactor != 0:
                 designCpp += '#pragma HLS ARRAY_PARTITION variable=softmaxA'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
        
        
        if modelDict[i]['layerType'] == 'Conv2D':
            
            designCpp += '\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\tfor (j=0 ; j<FILTERAMOUNTPREV'+str(i)+' ; ++j){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\tfor (k=0 ; k<PRODUCTX'+str(i)+' ; ++k){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\tfor (l=0 ; l<PRODUCTY'+str(i)+' ; ++l){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\t\tfor (m=0 ; m<PRODUCTZ'+str(i)+' ; ++m){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\t\t\tfor (n=0 ; n<KERNELX'+str(i)+' ; ++n){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\t\t\t\tfor (p=0 ; p<KERNELY'+str(i)+' ; ++p){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS PIPELINE\n'
            designCpp += '\t\t\t\t\t\t\t\tif ((isInference == true && i==0)||(isInference == false)){\n'
            if i == 1:
                designCpp += '\t\t\t\t\t\t\t\t\tP'+str(i)+'[i][k][l][m] += IArr[i][1 + j*IMAGEX*IMAGEY + (k-PADDING'+str(i)+'+n)*IMAGEX + (l-PADDING'+str(i)+'+p)] * F'+str(i)+'[n][p][j][m];\n'
                if modelDict[i]['useBias'] == True:
                    designCpp += '\t\t\t\t\t\t\t\t\tif (j==FILTERAMOUNTPREV'+str(i)+'-1 &&  n==KERNELX'+str(i)+'-1 && p==KERNELY'+str(i)+'-1)\n'
                    designCpp += '\t\t\t\t\t\t\t\t\t\tP'+str(i)+'[i][k][l][m] += BF'+str(i)+'[m];\n'
            else:
                designCpp += '\t\t\t\t\t\t\t\t\tP'+str(i)+'[i][k][l][m] += P'+str(i-1)+'[i][k-PADDING'+str(i)+'+n][l-PADDING'+str(i)+'+p][m] * F'+str(i)+'[n][p][j][m];\n'
                if modelDict[i]['useBias'] == True:
                    designCpp += '\t\t\t\t\t\t\t\t\tif (j==FILTERAMOUNTPREV'+str(i)+'-1 && n==KERNELX'+str(i)+'-1 && p==KERNELY'+str(i)+'-1)\n'
                    designCpp += '\t\t\t\t\t\t\t\t\t\tP'+str(i)+'[i][k][l][m] += BF'+str(i)+'[m];\n'
                
            if modelDict[i]['activation'] == 'RELU':
                designCpp += '\t\t\t\t\t\t\t\t\tif (j==FILTERAMOUNTPREV'+str(i)+'-1 && n==KERNELX'+str(i)+'-1 && p==KERNELY'+str(i)+'-1 && P'+str(i)+'[i][k][l][m]<0)\n'
                designCpp += '\t\t\t\t\t\t\t\t\t\tP'+str(i)+'[i][k][l][m] = 0;\n'
                
            if modelDict[i]['activation'] == 'Sigmoid':
                designCpp += '\t\t\t\t\t\t\t\t\tif (j==FILTERAMOUNTPREV'+str(i)+'-1 && n==KERNELX'+str(i)+'-1 && p==KERNELY'+str(i)+'-1)\n'
                designCpp += '\t\t\t\t\t\t\t\t\t\tP'+str(i)+'[i][k][l][m] = 1 / (1+exp(-1*P'+str(i)+'[i][k][l][m]));\n'
            
            if modelDict[i]['activation'] == 'Tanh':
                designCpp += '\t\t\t\t\t\t\t\t\tif (j==FILTERAMOUNTPREV'+str(i)+'-1 && n==KERNELX'+str(i)+'-1 && p==KERNELY'+str(i)+'-1)\n'
                designCpp += '\t\t\t\t\t\t\t\t\t\tP'+str(i)+'[i][k][l][m] = tanh(P'+str(i)+'[i][k][l][m]);\n'
            
            
            designCpp += '\t\t\t\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t}\n'   
            designCpp += '\t\t\t}\n'
            designCpp += '\t\t}\n'
            designCpp += '\t}\n'
        
        if modelDict[i]['layerType'] == 'MaxPool2D':
            designCpp += '\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\tfor (j=0 ; j<PRODUCTX'+str(i)+' ; ++j){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\tfor (k=0 ; k<PRODUCTY'+str(i)+' ; ++k){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\tfor (l=0 ; l<PRODUCTZ'+str(i)+' ; ++l){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\t\tfor (n=0 ; n<POOLX'+str(i)+' ; ++n){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\t\t\tfor (p=0 ; p<POOLY'+str(i)+' ; ++p){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS PIPELINE\n'
            designCpp += '\t\t\t\t\t\t\tif ((isInference == true && i==0)||(isInference == false)){\n'
            if i != 1:
                designCpp += '\t\t\t\t\t\t\t\tif (P'+str(i)+'[i][j][k][l] < P'+str(i-1)+'[i][j*POOLX'+str(i)+'+n][k*POOLY'+str(i)+'+p][l]){\n'
                designCpp += '\t\t\t\t\t\t\t\t\tP'+str(i)+'[i][j][k][l] = P'+str(i-1)+'[i][j*POOLX'+str(i)+'+n][k*POOLY'+str(i)+'+p][l];\n'
                designCpp += '\t\t\t\t\t\t\t\t\tIndexP'+str(i)+'[i][j][k][l][0] = j*POOLX'+str(i)+'+n;\n'
                designCpp += '\t\t\t\t\t\t\t\t\tIndexP'+str(i)+'[i][j][k][l][1] = k*POOLY'+str(i)+'+p;\n'
                designCpp += '\t\t\t\t\t\t\t\t}\n'
            else:
                designCpp += '\t\t\t\t\t\t\t\tif (P'+str(i)+'[i][j][k][l] < IArr[i][1 + l*PRODUCTX'+str(i)+'*POOLX'+str(i)+'+PRODUCTY'+str(i)+'*POOLY'+str(i)+' + (j*POOLX+n)*PRODUCTX'+str(i)+'*POOLX'+str(i)+' + k*POOLY'+str(i)+'+p]){\n'
                designCpp += '\t\t\t\t\t\t\t\t\tP'+str(i)+'[i][j][k][l] = IArr[i][1 + l*PRODUCTX'+str(i)+'*POOLX'+str(i)+'+PRODUCTY'+str(i)+'*POOLY'+str(i)+' + (j*POOLX+n)*PRODUCTX'+str(i)+'*POOLX'+str(i)+' + k*POOLY'+str(i)+'+p];\n'
                designCpp += '\t\t\t\t\t\t\t\t\tIndexP'+str(i)+'[i][j][k][l][0] = j*POOLX'+str(i)+'+n;\n'
                designCpp += '\t\t\t\t\t\t\t\t\tIndexP'+str(i)+'[i][j][k][l][1] = k*POOLY'+str(i)+'+p;\n'
                designCpp += '\t\t\t\t\t\t\t\t}\n'
              
            designCpp += '\t\t\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t}\n'   
            designCpp += '\t\t\t}\n'
            designCpp += '\t\t}\n'
            designCpp += '\t}\n'
            
        
        if modelDict[i]['layerType'] == 'AvgPool2D':
            designCpp += '\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\tfor (j=0 ; j<PRODUCTX'+str(i)+' ; ++j){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\tfor (k=0 ; k<PRODUCTY'+str(i)+' ; ++k){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\tfor (l=0 ; l<PRODUCTZ'+str(i)+' ; ++l){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\t\tfor (n=0 ; n<POOLX'+str(i)+' ; ++n){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\t\t\tfor (p=0 ; p<POOLY'+str(i)+' ; ++p){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS PIPELINE\n'
            designCpp += '\t\t\t\t\t\t\tif ((isInference == true && i==0)||(isInference == false)){\n'
            if i != 1:
                designCpp += '\t\t\t\t\t\t\t\tP'+str(i)+'[i][j][k][l] += P'+str(i-1)+'[i][j*POOLX'+str(i)+'+n][k*POOLY'+str(i)+'+p][l]/(POOLX'+str(i)+'*POOLY'+str(i)+');\n'
            else:
                designCpp += '\t\t\t\t\t\t\t\tP'+str(i)+'[i][j][k][l] += IArr[i][1 + l*PRODUCTX'+str(i)+'*POOLX'+str(i)+'+PRODUCTY'+str(i)+'*POOLY'+str(i)+' + (j*POOLX+n)*PRODUCTX'+str(i)+'*POOLX'+str(i)+' + k*POOLY'+str(i)+'+p]/(POOLX'+str(i)+'*POOLY'+str(i)+');\n'
            
            designCpp += '\t\t\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t}\n'   
            designCpp += '\t\t\t}\n'
            designCpp += '\t\t}\n'
            designCpp += '\t}\n'
            
            
        if modelDict[i]['layerType'] == 'Flatten':
            designCpp += '\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\tfor (j=0 ; j<PRODUCTX'+str(i-1)+' ; ++j){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\tfor (k=0 ; k<PRODUCTY'+str(i-1)+' ; ++k){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\tfor (l=0 ; l<PRODUCTZ'+str(i-1)+' ; ++l){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS PIPELINE\n'
            designCpp += '\t\t\t\tif ((isInference == true && i==0)||(isInference == false)){\n'
            if i != 1:
                designCpp += '\t\t\t\t\t\tP'+str(i)+'[i][l*PRODUCTY'+str(i-1)+'*PRODUCTX'+str(i-1)+' + j*PRODUCTX'+str(i-1)+' + k] = P'+str(i-1)+'[i][j][k][l];\n'
            
            else:
                designCpp += '\t\t\t\t\t\tP'+str(i)+'[i][l*PRODUCTY'+str(i-1)+'*PRODUCTX'+str(i-1)+' + j*PRODUCTX'+str(i-1)+' + k] = IArr[i][1 + l*PRODUCTX'+str(i-1)+'*PRODUCTY'+str(i-1)+' + k*PRODUCTX'+str(i-1)+' + j];\n'
            
            designCpp += '\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t}\n'   
            designCpp += '\t\t\t}\n'
            designCpp += '\t\t}\n'
            designCpp += '\t}\n'
            
        if modelDict[i]['layerType'] == 'Dense':
            designCpp += '\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\tfor (j=0 ; j<LENGTH'+str(i)+' ; ++j){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\tfor (k=0 ; k<LENGTHPREV'+str(i)+' ; ++k){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS PIPELINE\n'
            designCpp += '\t\t\t\tif ((isInference == true && i==0)||(isInference == false)){\n'
            
            designCpp += '\t\t\t\t\tP'+str(i)+'[i][j] += P'+str(i-1)+'[i][k] * W'+str(i)+'[k][j];\n'
            
            if modelDict[i]['useBias'] == True:
                designCpp += '\t\t\t\t\tif (k==LENGTHPREV'+str(i)+'-1)\n'
                designCpp += '\t\t\t\t\t\tP'+str(i)+'[i][j] += BW'+str(i)+'[i][j];\n'
                
            if modelDict[i]['activation'] == 'RELU':
                designCpp += '\t\t\t\t\tif (k==LENGTHPREV'+str(i)+'-1 && P'+str(i)+'[i][j]<0)\n'
                designCpp += '\t\t\t\t\t\tP'+str(i)+'[i][j] = 0;\n'
                
            if modelDict[i]['activation'] == 'Sigmoid':
                designCpp += '\t\t\t\t\tif (k==LENGTHPREV'+str(i)+'-1)\n'
                designCpp += '\t\t\t\t\t\tP'+str(i)+'[i][j] = 1 / (1+exp(-1*P'+str(i)+'[i][j]));\n'
            
            if modelDict[i]['activation'] == 'Tanh':
                designCpp += '\t\t\t\t\tif (k==LENGTHPREV'+str(i)+'-1)\n'
                designCpp += '\t\t\t\t\t\tP'+str(i)+'[i][j] = tanh(P'+str(i)+'[i][j]);\n'
            
                
            if i == len(modelDict) - 1 and modelDict[i]['activation'] != 'Softmax':
                designCpp += '\t\t\t\t\tif (k==LENGTHPREV'+str(i)+'-1)\n'
                designCpp += '\t\t\t\t\t\tPOUT[i][j] = P'+str(i)+'[i][j];\n'
                
            if modelDict[i]['activation'] == 'Softmax':
                designCpp += '\t\t\t\t\tif (k==LENGTHPREV'+str(i)+'-1){\n'
                designCpp += '\t\t\t\t\t\tP'+str(i)+'[i][j] = exp(P'+str(i)+'[i][j]);\n'
                designCpp += '\t\t\t\t\t\tsoftmaxA'+str(i)+'[i] += P'+str(i)+'[i][j];\n'
                designCpp += '\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t}\n'
            designCpp += '\t\t\t}\n'
            designCpp += '\t\t}\n'
            designCpp += '\t}\n'
            
            if modelDict[i]['activation'] == 'Softmax':
                designCpp += '\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
                designCpp += '\t\tfor (j=0 ; j<LENGTH'+str(i)+' ; ++j){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS PIPELINE\n'
                designCpp += '\t\t\tif ((isInference == true && i==0)||(isInference == false)){\n'
                designCpp += '\t\t\t\tP'+str(i)+'[i][j] /= softmaxA'+str(i)+'[i];\n'
                if i == len(modelDict) - 1:
                    designCpp += '\t\t\t\tPOUT[i][j] = P'+str(i)+'[i][j];\n'
                designCpp += '\t\t}\t\n'
                designCpp += '\t\t}\n'
                designCpp += '\t}\n'
    
    designCpp += '\n\n\tif (isInference == true) return;\n\n'
    
    designCpp += '\t//Backward Pass\n\n'
    
    designCpp += '\t//Backward Dense and Flatten Storage Arrays\n'
    
    for i in range(len(modelDict)-1, -1, -1):
        if modelDict[i]['layerType'] in ('Dense', 'Flatten'):
            designCpp += '\ttype_weight BDFSA'+str(i)+'[BATCHSIZE][LENGTH'+str(i)+'] = {0};\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS ARRAY_PARTITION variable=BDFSA'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'   
    designCpp += '\n\t//Backward Dense and Flatten Loss Gradient Arrays\n'
    
    for i in range(len(modelDict)-1, -1, -1):
        if modelDict[i]['layerType'] in ('Dense'):
            designCpp += '\ttype_weight BDFLGA'+str(i)+'[LENGTHPREV'+str(i)+'][LENGTH'+str(i)+'] = {0};\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS ARRAY_PARTITION variable=BDFLGA'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
        
    designCpp += '\n\t//Backward Conv2D Kernel and Bias Weight Loss Gradient Arrays\n'        
    
    for i in range(len(modelDict)-1, -1, -1):
        if modelDict[i]['layerType'] in ('Conv2D'):
            designCpp += '\ttype_weight BKLGA'+str(i)+'[KERNELX'+str(i)+'][KERNELY'+str(i)+'][FILTERAMOUNTPREV'+str(i)+'][FILTERAMOUNT'+str(i)+'] = {0};\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS ARRAY_PARTITION variable=BKLGA'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
            if modelDict[i]['useBias'] == True:
                designCpp += '\ttype_weight BKBLGA'+str(i)+'[FILTERAMOUNT'+str(i)+'] = {0};\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS ARRAY_PARTITION variable=BKBLGA'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
            
 
    designCpp += '\n\t//Backward Conv2D MaxPool2D AvgPool2D Product Loss Gradient Arrays\n'  
    
    for i in range(len(modelDict)-1, -1, -1):
        if modelDict[i]['layerType'] in ('Conv2D', 'MaxPool2D', 'AvgPool2D'): 
            designCpp += '\ttype_weight BCMLGA'+str(i)+'[BATCHSIZE][PRODUCTX'+str(i)+'][PRODUCTY'+str(i)+'][PRODUCTZ'+str(i)+'] = {0};\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS ARRAY_PARTITION variable=BCMLGA'+str(i)+' cyclic factor='+str(parallelFactor)+' dim=0\n'
            
            
            
    if (modelDict[len(modelDict)-1]['optimizerType'] == 'ADAM'):  
        designCpp += '\n\t//ADAM Step-wise Pows and Count\n'
        designCpp += '\tstatic type_index stepCount = 1;\n'
        designCpp += '\tif (stepCount>NTRAIN/BATCHSIZE) stepCount = 1;\n\n'
        
        designCpp += '\ttype_weight powB1 = powf(BETA1, stepCount);\n'
        designCpp += '\ttype_weight powB2 = powf(BETA2, stepCount);\n'
        
    
    designCpp += '\n\t//Backward Loss Gradient Calculation\n'
    
    for i in range(len(modelDict)-1, 0, -1):
        if modelDict[i]['layerType'] in ('Dense', 'Flatten'):
            if i == len(modelDict)-1 and modelDict[i]['applyTraining'] == True:
                designCpp += '\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
                designCpp += '\t\tfor (j=0 ; j<LENGTH'+str(i)+' ; ++j){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS PIPELINE\n'
                
                designCpp += '\t\t\tBDFSA'+str(i)+'[i][j] = P'+str(i)+'[i][j] - outActual[i][j];\n'
                
                designCpp += '\t\t}\n'
                designCpp += '\t}\n'
            if i != len(modelDict)-1 and modelDict[i+1]['applyTraining'] == True:
                designCpp += '\tfor (k=0 ; k<LENGTH'+str(i+1)+' ; ++k){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
                designCpp += '\t\tfor (j=0 ; j<LENGTH'+str(i)+' ; ++j){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
                designCpp += '\t\t\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS PIPELINE\n'
                #if i+1 != len(modelDict)-1:
                designCpp += '\t\t\t\tBDFSA'+str(i)+'[i][j] += BDFSA'+str(i+1)+'[i][k]*W'+str(i+1)+'[j][k];\n'
                #else:
                #   designCpp += '\t\t\t\tBDFSA'+str(i)+'[i][j] = BDFSA'+str(i+1)+'[i][k];\n'
                
                designCpp += '\t\t\t\tBDFLGA'+str(i+1)+'[j][k] += P'+str(i)+'[i][j]*BDFSA'+str(i+1)+'[i][k];\n'
                designCpp += '\t\t\t\tif (i == BATCHSIZE-1){\n'
                
                if (modelDict[len(modelDict)-1]['optimizerType'] == 'ADAM'):
                    designCpp += '\t\t\t\t\tmW'+str(i+1)+'[j][k] = BETA1*mW'+str(i+1)+'[j][k] + (1-BETA1)*BDFLGA'+str(i+1)+'[j][k];\n'
                    designCpp += '\t\t\t\t\tvW'+str(i+1)+'[j][k] = BETA2*vW'+str(i+1)+'[j][k] + (1-BETA2)*BDFLGA'+str(i+1)+'[j][k]*BDFLGA'+str(i+1)+'[j][k];\n'
                    
                    
                    designCpp += '\t\t\t\t\tW'+str(i+1)+'[j][k] -= ETA*(mW'+str(i+1)+'[j][k]/(1-powB1)) / ((EPSILON+sqrt(vW'+str(i+1)+'[j][k]/(1-powB2)))*BATCHSIZE);\n'
                
                
                if (modelDict[len(modelDict)-1]['optimizerType'] == 'SGD'):
                    designCpp += '\t\t\t\t\tW'+str(i+1)+'[j][k] -= ETA*BDFLGA'+str(i+1)+'[j][k]/BATCHSIZE;\n'
                
                
                designCpp += '\t\t\t\t}\n'
                
                if modelDict[i+1]['useBias'] == True:
                    if (modelDict[len(modelDict)-1]['optimizerType'] == 'ADAM'):
                        designCpp += '\t\t\t\tif (j == LENGTH'+str(i)+'-1){\n'
                        designCpp += '\t\t\t\t\tmBW'+str(i+1)+'[i][k] = BETA1*mBW'+str(i+1)+'[i][k] + (1-BETA1)*BDFSA'+str(i+1)+'[i][k];\n'
                        designCpp += '\t\t\t\t\tvBW'+str(i+1)+'[i][k] = BETA2*vBW'+str(i+1)+'[i][k] + (1-BETA2)*BDFSA'+str(i+1)+'[i][k]*BDFSA'+str(i+1)+'[i][k];\n'
                        
                        designCpp += '\t\t\t\t\tBW'+str(i+1)+'[i][k] -= ETA*(mBW'+str(i+1)+'[i][k]/(1-powB1)) / ((EPSILON+sqrt(vBW'+str(i+1)+'[i][k]/(1-powB2)))*BATCHSIZE);\n'
                        designCpp += '\t\t\t\t}\n'
                        
                    if (modelDict[len(modelDict)-1]['optimizerType'] == 'SGD'):
                        designCpp += '\t\t\t\tif (j == LENGTH'+str(i)+'-1){\n'
                        
                        designCpp += '\t\t\t\t\tBW'+str(i+1)+'[i][k] -= ETA*BDFSA'+str(i+1)+'[i][k]/BATCHSIZE;\n'
                        designCpp += '\t\t\t\t}\n'
                
                designCpp += '\t\t\t}\n'
                designCpp += '\t\t}\n'
                designCpp += '\t}\n'
        
    
    for i in range(len(modelDict)-2, 0, -1):
        if modelDict[i+1]['layerType'] == 'Flatten':
            designCpp += '\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\tfor (j=0 ; j<PRODUCTZ'+str(i)+' ; ++j){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\tfor (k=0 ; k<PRODUCTX'+str(i)+' ; ++k){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\tfor (l=0 ; l<PRODUCTY'+str(i)+' ; ++l){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS PIPELINE\n'
            
            designCpp += '\t\t\t\t\tBCMLGA'+str(i)+'[i][k][l][j] = BDFSA'+str(i+1)+'[i][j*PRODUCTX'+str(i)+'*PRODUCTY'+str(i)+'+k*PRODUCTX'+str(i)+'+l];\n'
            
            designCpp += '\t\t\t\t}\n'   
            designCpp += '\t\t\t}\n'
            designCpp += '\t\t}\n'
            designCpp += '\t}\n'
            
        if modelDict[i+1]['layerType'] == 'MaxPool2D':
            designCpp += '\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\tfor (j=0 ; j<PRODUCTZ'+str(i+1)+' ; ++j){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\tfor (k=0 ; k<PRODUCTX'+str(i+1)+' ; ++k){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\tfor (l=0 ; l<PRODUCTY'+str(i+1)+' ; ++l){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS PIPELINE\n'
            
            designCpp += '\t\t\t\t\tBCMLGA'+str(i)+'[i][IndexP'+str(i+1)+'[i][k][l][j][0]][IndexP'+str(i+1)+'[i][k][l][j][1]][j] = BCMLGA'+str(i+1)+'[i][k][l][j];\n'
            
            designCpp += '\t\t\t\t}\n'   
            designCpp += '\t\t\t}\n'
            designCpp += '\t\t}\n'
            designCpp += '\t}\n'
        
        if modelDict[i+1]['layerType'] == 'AvgPool2D':
            designCpp += '\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\tfor (j=0 ; j<PRODUCTZ'+str(i+1)+' ; ++j){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\tfor (k=0 ; k<PRODUCTX'+str(i+1)+' ; ++k){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\tfor (l=0 ; l<PRODUCTY'+str(i+1)+' ; ++l){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\t\tfor (m=0 ; m<POOLX'+str(i+1)+' ; ++m){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\t\t\tfor (n=0 ; n<POOLY'+str(i+1)+' ; ++n){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS PIPELINE\n'
            
            designCpp += '\t\t\t\t\t\t\tBCMLGA'+str(i)+'[i][k*POOLX'+str(i+1)+'+m][l*POOLY'+str(i+1)+'+n][j] = BCMLGA'+str(i+1)+'[i][k][l][j]/(POOLX'+str(i+1)+'*POOLY'+str(i+1)+');\n'
            
            designCpp += '\t\t\t\t\t\t}\n' 
            designCpp += '\t\t\t\t\t}\n' 
            designCpp += '\t\t\t\t}\n'   
            designCpp += '\t\t\t}\n'
            designCpp += '\t\t}\n'
            designCpp += '\t}\n'
            
        if modelDict[i]['layerType'] == 'Conv2D':
            designCpp += '\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\tfor (j=0 ; j<PRODUCTZ'+str(i-1)+' ; ++j){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\tfor (k=0 ; k<KERNELX'+str(i)+' ; ++k){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\tfor (l=0 ; l<KERNELY'+str(i)+' ; ++l){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\t\tfor (m=0 ; m<PRODUCTZ'+str(i)+' ; ++m){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\t\t\tfor (n=0 ; n<PRODUCTX'+str(i)+' ; ++n){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
            designCpp += '\t\t\t\t\t\t\tfor (p=0 ; p<PRODUCTY'+str(i)+' ; ++p){\n'
            if parallelFactor != 0:
                designCpp += '#pragma HLS PIPELINE\n'
            
            if i-1 == 0:
                designCpp += '\t\t\t\t\t\t\t\tBKLGA'+str(i)+'[k][l][j][m] += IArr[i][1 + j*IMAGEX*IMAGEY + (k-PADDING'+str(i)+'+n)*IMAGEX + (l-PADDING'+str(i)+'+p)] * BCMLGA'+str(i)+'[i][n][p][m];\n'
            else:
                designCpp += '\t\t\t\t\t\t\t\tBKLGA'+str(i)+'[k][l][j][m] += P'+str(i)+'[i][k-PADDING'+str(i)+'+n][l-PADDING'+str(i)+'+p][j] * BCMLGA'+str(i)+'[i][n][p][m];\n'
            
            if modelDict[i]['useBias'] == True:
                designCpp += '\t\t\t\t\t\t\t\tif (j==PRODUCTZ'+str(i-1)+'-1 && k==KERNELX'+str(i)+'-1 && l==KERNELY'+str(i)+'-1)\n'
                designCpp += '\t\t\t\t\t\t\t\t\tBKBLGA'+str(i)+'[m] += BCMLGA'+str(i)+'[i][n][p][m];\n'
            
            
            if modelDict[i]['applyTraining'] == True:
                if (modelDict[len(modelDict)-1]['optimizerType'] == 'ADAM'):
                    designCpp += '\t\t\t\t\t\t\t\tif (i==BATCHSIZE-1 &&  n==PRODUCTX'+str(i)+'-1 && p==PRODUCTY'+str(i)+'-1){\n'
                    
                    designCpp += '\t\t\t\t\t\t\t\t\tmF'+str(i)+'[k][l][j][m] = BETA1*mF'+str(i)+'[k][l][j][m] + (1-BETA1)*BKLGA'+str(i)+'[k][l][j][m];\n'
                    designCpp += '\t\t\t\t\t\t\t\t\tvF'+str(i)+'[k][l][j][m] = BETA2*vF'+str(i)+'[k][l][j][m] + (1-BETA2)*BKLGA'+str(i)+'[k][l][j][m]*BKLGA'+str(i)+'[k][l][j][m];\n'
                    
                    designCpp += '\t\t\t\t\t\t\t\t\tF'+str(i)+'[k][l][j][m] -= ETA*(mF'+str(i)+'[k][l][j][m]/(1-powB1))/((EPSILON+sqrt(vF'+str(i)+'[k][l][j][m]/(1-powB2)))*BATCHSIZE);\n'
                    
                    
                    if modelDict[i]['useBias'] == True:
                        designCpp += '\t\t\t\t\t\t\t\t\tif (i==BATCHSIZE-1 &&  n==PRODUCTX'+str(i)+'-1 && p==PRODUCTY'+str(i)+'-1 && l==KERNELY'+str(i)+'-1 && k==KERNELX'+str(i)+'-1 && j==PRODUCTZ'+str(i-1)+'-1){\n'
                        designCpp += '\t\t\t\t\t\t\t\t\t\tmBF'+str(i)+'[m] = BETA1*mBF'+str(i)+'[m]+ (1-BETA1)*BKBLGA'+str(i)+'[m];\n'
                        designCpp += '\t\t\t\t\t\t\t\t\t\tvBF'+str(i)+'[m] = BETA2*vBF'+str(i)+'[m] + (1-BETA2)*BKBLGA'+str(i)+'[m]*BKBLGA'+str(i)+'[m];\n'
                    
                        designCpp += '\t\t\t\t\t\t\t\t\t\tBF'+str(i)+'[m] -= ETA*(mBF'+str(i)+'[m]/(1-powB1))/((EPSILON+sqrt(vBF'+str(i)+'[m]/(1-powB2)))*BATCHSIZE);\n'
                        designCpp += '\t\t\t\t\t\t\t\t\t}\n'
                        
                    designCpp += '\t\t\t\t\t\t\t\t}\n'
                
                if (modelDict[len(modelDict)-1]['optimizerType'] == 'SGD'):
                    designCpp += '\t\t\t\t\t\t\t\tif (i==BATCHSIZE-1 &&  n==PRODUCTX'+str(i)+'-1 && p==PRODUCTY'+str(i)+'-1){\n'
                    
                    
                    designCpp += '\t\t\t\t\t\t\t\t\tF'+str(i)+'[k][l][j][m] -= ETA*BKLGA'+str(i)+'[k][l][j][m]/BATCHSIZE;\n'
                    
                    
                    if modelDict[i]['useBias'] == True:
                        designCpp += '\t\t\t\t\t\t\t\t\tif (i==BATCHSIZE-1 &&  n==PRODUCTX'+str(i)+'-1 && p==PRODUCTY'+str(i)+'-1 && l==KERNELY'+str(i)+'-1 && k==KERNELX'+str(i)+'-1 && j==PRODUCTZ'+str(i-1)+'-1){\n'
                        
                        designCpp += '\t\t\t\t\t\t\t\t\t\tBF'+str(i)+'[m] -= ETA*BKBLGA'+str(i)+'[m]/BATCHSIZE;\n'
                        designCpp += '\t\t\t\t\t\t\t\t\t}\n'
                        
                    designCpp += '\t\t\t\t\t\t\t\t}\n'
            
            designCpp += '\t\t\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t\t}\n'
            designCpp += '\t\t\t\t}\n'   
            designCpp += '\t\t\t}\n'
            designCpp += '\t\t}\n'
            designCpp += '\t}\n'
            
            if i!=1:
        
                designCpp += '\tfor (i=0 ; i<BATCHSIZE ; ++i){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
                designCpp += '\t\tfor (j=0 ; j<PRODUCTZ'+str(i-1)+' ; ++j){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
                designCpp += '\t\t\tfor (k=-1 ; k<PRODUCTX'+str(i-1)+'+1 ; ++k){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
                designCpp += '\t\t\t\tfor (l=-1 ; l<PRODUCTY'+str(i-1)+'+1 ; ++l){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
                designCpp += '\t\t\t\t\tfor (m=0 ; m<FILTERAMOUNT'+str(i)+' ; ++m){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
                designCpp += '\t\t\t\t\t\tfor (n=KERNELX'+str(i)+'-1 ; n>=0 ; --n){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS UNROLL factor='+str(parallelFactor)+'\n'
                designCpp += '\t\t\t\t\t\t\tfor (p=KERNELY'+str(i)+'-1 ; p>=0 ; --p){\n'
                if parallelFactor != 0:
                    designCpp += '#pragma HLS PIPELINE\n'
                
                designCpp += '\t\t\t\t\t\t\t\tif (k>=0 && k<PRODUCTX'+str(i-1)+' && l>=0 && l<PRODUCTY'+str(i-1)+')\n'
    
                designCpp += '\t\t\t\t\t\t\t\t\tBCMLGA'+str(i-1)+'[i][k][l][j] += BCMLGA'+str(i)+'[i][k-PADDING'+str(i)+'+n][l-PADDING'+str(i)+'+p][m] * F'+str(i)+'[n][p][j][m];\n'
                
                designCpp += '\t\t\t\t\t\t\t}\n'
                designCpp += '\t\t\t\t\t\t}\n'
                designCpp += '\t\t\t\t\t}\n'
                designCpp += '\t\t\t\t}\n'   
                designCpp += '\t\t\t}\n'
                designCpp += '\t\t}\n'
                designCpp += '\t}\n'
                
                
    if (modelDict[len(modelDict)-1]['optimizerType'] == 'ADAM'):            
        designCpp += '\t++stepCount;\n\n'
                
    designCpp += '}\n\n'
    
    
    #testbench.cpp building
    testbenchCpp = ''
    
    testbenchCpp += '#include "design.hpp"\n\n\n'
    testbenchCpp += 'int main(){\n\n'
    
    
    testbenchCpp += '\tint i, j, k, l;\n\n'
    testbenchCpp += '\tstd::random_device rd;\n'
    testbenchCpp += '\tstd::mt19937 gen(rd());\n'
    testbenchCpp += '\tstd::normal_distribution<double> d(0, 0.1);\n\n'
    for i in range(1, len(modelDict)):
        layerDict = modelDict[i]
        
        
        if (layerDict['layerType'] == 'Conv2D'):
            testbenchCpp += '\tfor (i=0 ; i<KERNELX'+str(i)+' ; ++i){\n'
            testbenchCpp += '\t\tfor (j=0 ; j<KERNELY'+str(i)+' ; ++j){\n'
            testbenchCpp += '\t\t\tfor (k=0 ; k<FILTERAMOUNTPREV'+str(i)+' ; ++k){\n'
            testbenchCpp += '\t\t\t\tfor (l=0 ; l<FILTERAMOUNT'+str(i)+' ; ++l){\n'
            testbenchCpp += '\t\t\t\t\tF'+str(i)+'[i][j][k][l] = d(gen);\n'
            if (layerDict['useBias'] == True):
                testbenchCpp += '\t\t\t\t\tif(i==0 && j==0 && k==0)\n'
                testbenchCpp += '\t\t\t\t\t\tBF'+str(i)+'[l]= d(gen);\n'
             
            testbenchCpp += '\t\t\t\t}\n'
            testbenchCpp += '\t\t\t}\n'
            testbenchCpp += '\t\t}\n'
            testbenchCpp += '\t}\n'
        
        if (layerDict['layerType'] == 'Dense'):
            testbenchCpp += '\tfor (i=0 ; i<LENGTHPREV'+str(i)+' ; ++i){\n'
            testbenchCpp += '\t\tfor (j=0 ; j<LENGTH'+str(i)+' ; ++j){\n'
            testbenchCpp += '\t\t\tfor (k=0 ; k<BATCHSIZE ; ++k){\n'
            testbenchCpp += '\t\t\t\tif(k==0)\n'
            testbenchCpp += '\t\t\t\t\tW'+str(i)+'[i][j] = d(gen);\n'
            if (layerDict['useBias'] == True):
                testbenchCpp += '\t\t\t\tif(i==0)\n'
                testbenchCpp += '\t\t\t\t\tBW'+str(i)+'[k][j] = d(gen);\n'
    
            testbenchCpp += '\t\t\t}\n'
            testbenchCpp += '\t\t}\n'
            testbenchCpp += '\t}\n'
    
    
    testbenchCpp += '\n\ttype_index epochIndex, nIndex;\n\n'
    
    testbenchCpp += '\ttype_weight I[BATCHSIZE][1+IMAGEX*IMAGEY*IMAGECHANNEL];\n'
    testbenchCpp += '\ttype_weight POUT[BATCHSIZE][CLASSSIZE];\n'
    testbenchCpp += '\tchar imageRow[1+4*IMAGEX*IMAGEY*IMAGECHANNEL];\n\n'
    
    if modelDict[len(modelDict)-1]['useShuffle'] == True:
        testbenchCpp += '\tlong rowBeginningPositions[NTRAIN] = {0};\n\n'

        testbenchCpp += '\tFILE* f_train = fopen("mnist_train.csv", "r+");\n\n'
    
        testbenchCpp += '\tfor (nIndex = 1; nIndex <= NTRAIN; ++nIndex) {\n'
        testbenchCpp += '\t\trowBeginningPositions[nIndex - 1] = ftell(f_train);\n'
        testbenchCpp += '\t\tfscanf(f_train, "%s\\n", imageRow);\n'
        testbenchCpp += '\t}\n\n'
    
        testbenchCpp += '\tfclose(f_train);\n\n'
    
    testbenchCpp += '\tfor (epochIndex = 0; epochIndex < EPOCH ; ++epochIndex){\n'
    
    testbenchCpp += '\t\tFILE* f_train = fopen("'+modelDict[0]['trainLocation']+'","r+");\n\n'
    
    if modelDict[len(modelDict)-1]['useShuffle'] == True:
        testbenchCpp += '\t\tvector<int> indices(NTRAIN);\n'
        testbenchCpp += '\t\tfor (i = 0; i < NTRAIN; ++i) indices[i] = i;\n'
        testbenchCpp += '\t\tshuffle(indices.begin(), indices.end(), default_random_engine(time(NULL)));\n\n'
    
    testbenchCpp += '\t\tfor (nIndex = 1; nIndex <= NTRAIN ; ++nIndex){\n'
    if modelDict[len(modelDict)-1]['useShuffle'] == True:
        testbenchCpp += '\t\t\tfseek(f_train, rowBeginningPositions[indices[nIndex-1]], SEEK_SET);\n\n'
    testbenchCpp += '\t\t\tfscanf(f_train, "%s\\n", imageRow);\n'
    testbenchCpp += '\t\t\tchar* token;\n'


    testbenchCpp += '\t\t\tint ind = 0;\n'
    testbenchCpp += '\t\t\ttoken = strtok(imageRow, ",");\n'
    testbenchCpp += '\t\t\tI[(nIndex%BATCHSIZE-1+BATCHSIZE)%BATCHSIZE][ind] = atoi(token);\n'
    testbenchCpp += '\t\t\tind += 1;\n'
    testbenchCpp += '\t\t\ttoken = strtok(NULL, ",");\n'
    testbenchCpp += '\t\t\twhile (token != NULL){\n'
    testbenchCpp += '\t\t\t\tI[(nIndex%BATCHSIZE-1+BATCHSIZE)%BATCHSIZE][ind] = atoi(token)/255.0;\n'
    testbenchCpp += '\t\t\t\tind += 1;\n'
    testbenchCpp += '\t\t\t\ttoken = strtok(NULL, ",");\n'
    testbenchCpp += '\t\t\t}\n\n'
    
    testbenchCpp += '\t\t\tif (nIndex%BATCHSIZE != 0)\n'
    testbenchCpp += '\t\t\t\tcontinue;\n\n'
    
    
    testbenchCpp += '\t\t\ttrain(I, POUT, false);\n\n'
    
    
    testbenchCpp += '\t\t\tif (nIndex % 1000 == 0){\n'
    testbenchCpp += '\t\t\t\tprintf("Epoch: %d, nIndex: %d\\n", epochIndex+1, nIndex);\n'
    testbenchCpp += '\t\t\t}\n'
    
    testbenchCpp += '\t\t}\n\n'
    
    if mode == 1:
        testbenchCpp += '\t\t//Testing\n'
        testbenchCpp += '\t\tprintf("TESTING\\n");\n'
        testbenchCpp += '\t\tdouble trueCnt = 0.0, falseCnt = 0.0;\n'
        testbenchCpp += '\n\n\t\tFILE* f_test = fopen("'+modelDict[0]['testLocation']+'", "r+");\n\n'
        testbenchCpp += '\t\tfor (nIndex = 1; nIndex <= NTEST ; ++nIndex){\n'
    
        testbenchCpp += '\t\t\tfscanf(f_test, "%s\\n", imageRow);\n'
        testbenchCpp += '\t\t\tchar* token;\n'
        testbenchCpp += '\t\t\tint classImage;\n\n'


        testbenchCpp += '\t\t\tint ind = 0;\n'
        testbenchCpp += '\t\t\ttoken = strtok(imageRow, ",");\n'
        testbenchCpp += '\t\t\tI[0][ind] = atoi(token);\n'
        testbenchCpp += '\t\t\tind += 1;\n'
        testbenchCpp += '\t\t\ttoken = strtok(NULL, ",");\n'
        testbenchCpp += '\t\t\twhile (token != NULL){\n'
        testbenchCpp += '\t\t\t\tI[0][ind] = atoi(token)/255.0;\n'
        testbenchCpp += '\t\t\t\tind += 1;\n'
        testbenchCpp += '\t\t\t\ttoken = strtok(NULL, ",");\n'
        testbenchCpp += '\t\t\t}\n\n'

        testbenchCpp += '\t\t\ttrain(I, POUT, true);\n\n'

        testbenchCpp += '\t\t\tdouble val = 0;\n'
        testbenchCpp += '\t\t\tclassImage = I[0][0];\n\n'
        testbenchCpp += '\t\t\tint valInd = 0;\n\n'

        testbenchCpp += '\t\t\tfor (i=0 ; i<CLASSSIZE ; i++){\n'
        testbenchCpp += '\t\t\t\tif (POUT[0][i]>=val){\n'
        testbenchCpp += '\t\t\t\t\tval = POUT[0][i];\n'
        testbenchCpp += '\t\t\t\t\tvalInd = i;\n'
        testbenchCpp += '\t\t\t\t}\n'
        testbenchCpp += '\t\t\t}\n\n'

        testbenchCpp += '\t\t\tif (valInd == classImage)\n'
        testbenchCpp += '\t\t\t\ttrueCnt += 1;\n'
        testbenchCpp += '\t\t\telse\n'
        testbenchCpp += '\t\t\t\tfalseCnt += 1;\n'

        testbenchCpp += '\t\t}\n'

        testbenchCpp += '\t\tprintf("Accuracy: %f\\n\\n", (1.0*trueCnt)/(trueCnt + falseCnt));\n'
        
        testbenchCpp += '\n\n\t\tfclose(f_test);\n'
        
    testbenchCpp += '\t\tfclose(f_train);\n'
    
    testbenchCpp += '\n\t}\n\n'
    

    testbenchCpp += '\treturn 0;\n\n'
    testbenchCpp += '}\n\n'
    
    
    
    designHppTextF = open("design.hpp", "w")
    designHppTextF.write(designHpp)
    designHppTextF.close()
    
    designCppTextF = open("design.cpp", "w")
    designCppTextF.write(designCpp)
    designCppTextF.close()
    
    testbenchCppTextF = open("testbench.cpp", "w")
    testbenchCppTextF.write(testbenchCpp)
    testbenchCppTextF.close()
    
    
    
    
    
