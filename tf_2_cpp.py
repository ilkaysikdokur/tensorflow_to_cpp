import nn_code_generator as nn


model = nn.Sequential()
model.add(nn.Input(imageX=32, imageY=32, imageChannel=3, epoch=1, batchSize=2, classSize=10, nTrain=50000, nTest=10000))
model.add(nn.Conv2D(kernelX=5, kernelY=5, filterAmount=2, activation='RELU'))
model.add(nn.MaxPool2D())
model.add(nn.Conv2D(kernelX=5, kernelY=5, filterAmount=2, activation='RELU'))
model.add(nn.MaxPool2D())
model.add(nn.Flatten())
model.add(nn.Dense(64, activation='RELU'))
model.add(nn.Dense(10))
model.optimizer(optimizerType='SGD')

model.setDataset(trainLocation='mnist_train.csv', testLocation='mnist_test.csv')

#0: train
#1: train + test(accuracy)
#2: train + inference(index given, on one example from dataTest) ////to be added
nn.exportHLS(model.modelDict, mode=1)