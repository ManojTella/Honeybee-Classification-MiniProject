# Honeybee-Classification-MiniProject
### Aim:
Multi-Classification  of Honey bees  using Convolutional NEURAL NETWORK
### Introduction:
#### 1.1. Need of Multiclassification of Honeybees:
       hii
       
           In order to classify honeybees, multiple groups must be created based on their shared qualities and characteristics. 
This categorization serves a number of crucial functions. 
           
#### 1.2. Different Methods of Classification
Deep Learning Techniques:

• Convolutional Neural Networks (CNNs): CNNs are effective at classifying honeybees based on pictures of them and other image-based multiclassification tasks.

• Recurrent neural networks (RNNs): RNNs are useful for sequential data, such as time-series information on the activity of honeybees or environmental variables.

• Multilayer Perceptrons (MLPs): Using structured data and a predetermined number of features, MLPs may be used for multiclass classification.

#### 1.1. Figures:
![image](https://github.com/ManojTella/Honeybee-Classification-MiniProject/assets/94883876/ecc70a7a-af1d-465b-b50c-bed384d7df1d)

 			
#### 1.1.2 Code or Program
```
DatasetPath = fullfile('bees');
imds = imageDatastore(DatasetPath, 'IncludeSubfolders', true, 'LabelSource','foldername');
imds.Files
numObs=length(imds.Labels)
numObsToShow = 20;
class = "bees";
figure;
histogram(imds.Labels)
set(gca,'TickLabelInterpreter','none')
img = readimage(imds,1); 
size(img)
imds.ReadSize = numpartitions(imds)
num_partitions = numpartitions(imds);
num_partitions
imds.ReadFcn = @(loc)imresize(imread(loc),[224,224])
img = readimage(imds,22); 
imshow(img); 
size(img)
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');
layers = [
imageInputLayer([224 224 3])
convolution2dLayer(3,8,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,16,'Padding','same')
batchNormalizationLayer
reluLayerA
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,32,'Padding','same')
batchNormalizationLayer
reluLayer
fullyConnectedLayer(4)
softmaxLayer
classificationLayer];
analyzeNetwork(layers)
options = trainingOptions('sgdm', ...
'InitialLearnRate',0.01, ...
'MaxEpochs',4, ...
'Shuffle','every-epoch', ...
'ValidationData',imdsValidation, ...
'ValidationFrequency',30, ...
'Verbose',false, ...
'Plots','training-progress');
net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsValidation)
YValidation = imdsValidation.Labels
plotconfusion(YValidation,YPred)
accuracy = sum(YPred == YValidation)/numel(YValidation)
```
### Result:


