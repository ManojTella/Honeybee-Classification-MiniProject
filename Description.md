### Introduction:
#### 1.1. Need of Multiclassification of Honeybees:  
In order to classify honeybees, multiple groups must be created based on their shared qualities and characteristics. This categorization serves a number of crucial functions. Scientists may investigate honeybees' evolutionary history, genetics, behaviour, and ecological functions through classification. Researchers can compare comparable honeybee species and come to conclusions about their biology and ecology by doing so. Threats to honeybee populations include habitat loss, pesticide use, illness, and climate change. Classifying honeybees makes it easier to distinguish between many species and subspecies, assisting conservation efforts by emphasizing those that are in danger or in decline. Different species and subspecies of honeybee exhibit a range of behaviours, feeding patterns, and pest and disease resistance.  

For sustainable agricultural methods that rely on honeybee pollination, understanding these distinctions is essential. Certain illnesses and pests affect some honeybee species more than others. A better understanding of which honeybees may be more immune to these dangers will assist beekeepers and researchers direct their efforts to create disease-resistant breeds. For many crops and untamed plants, honeybees are essential pollinators. The precise categorization of honeybee species is critical for maximizing pollination services in various environments because distinct honeybee species could have preferences for particular types of flowers.

By pollinating plants that serve as habitat and food for other species, honeybees play a critical role in ecosystems. Researcher comprehension of these complex linkages and ecological processes is aided by proper categorization. Our knowledge of the taxonomy and evolutionary history of the animal kingdom is larger thanks to the classification of honeybees. It helps us understand honeybees' origins and adaptations by putting them in the perspective of other similar creatures.

### Different Methods of Classification
Deep Learning Techniques:

• Convolutional Neural Networks (CNNs): CNNs are effective at classifying honeybees based on pictures of them and other image-based multiclassification tasks.

• Recurrent neural networks (RNNs): RNNs are useful for sequential data, such as time-series information on the activity of honeybees or environmental variables.

• Multilayer Perceptrons (MLPs): Using structured data and a predetermined number of features, MLPs may be used for multiclass classification.

### Figures:
![image](https://github.com/ManojTella/Honeybee-Classification-MiniProject/assets/94883876/ecc70a7a-af1d-465b-b50c-bed384d7df1d)
#### 1.5. Flow Diagram.
![image](https://github.com/ManojTella/Honeybee-Classification-MiniProject/assets/94883876/6548efa9-3ddd-43a8-ba85-fb71c6b04690)

 			
### Code or Program
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
### Literature Review:

![image](https://github.com/ManojTella/Honeybee-Classification-MiniProject/assets/94883876/63ebcac2-19f9-4893-ad0a-dad33b019647)
![image](https://github.com/ManojTella/Honeybee-Classification-MiniProject/assets/94883876/05c3a016-6b0f-4ef9-8595-c9b50ac4c7a4)
![image](https://github.com/ManojTella/Honeybee-Classification-MiniProject/assets/94883876/2f70007b-7372-4a3f-8c00-6140f63b7e43)

### Result:
#### Validation and Testing
![image](https://github.com/ManojTella/Honeybee-Classification-MiniProject/assets/94883876/33bb4032-0c98-4ece-971a-237cd90e77cc)

#### Accuracy
![image](https://github.com/ManojTella/Honeybee-Classification-MiniProject/assets/94883876/44aa6780-6fe1-4e6d-8d6e-dcf4cc2854d4)

#### Accuracy Curve
![image](https://github.com/ManojTella/Honeybee-Classification-MiniProject/assets/94883876/d7252166-6469-4d96-a3ce-2b311e1e464c)

#### Confusion Matrix
![image](https://github.com/ManojTella/Honeybee-Classification-MiniProject/assets/94883876/9d0504c7-53ff-46a7-a04f-c26f551c7a1e)


### References
[1] V. A. Kulyukin and S. K. Reka, “Toward sustainable electronic beehive monitoring: Algorithms for omnidirectional bee counting from images and harmonic analysis of buzzing signals.” Engineering Letters, vol. 24, no. 3, 2016.

[2] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” in Advances in neural information processing systems, 2012, pp. 1097–1105.

[3] Z. Babic, R. Pilipovic, V. Risojevic, and G. Mirjanic. Pollen bearing honey bee detection in hive entrance video recorded by remote embedded system for pollination monitoring. ISPRS Annals of Photogrammetry, Remote Sensing and Spatial Information Sciences, III-7:51–57, 2016.

[4] G. J. Tu, M. K. Hansen, and P. A. Per Kryger. Automatic behaviour analysis system for honeybees using computer vision. Computer and Electronics in Agriculture, 122, 2016.

[5]Rahman Tashakkori, Nathaniel P. Hernandez, Ahmad Ghadiri, Aleksander P. Ratzloff, and Michael B. Crawford. "A Honeybee Hive Monitoring System: From Surveillance Cameras to Raspberry Pis." ResearchGate, March 1, 2017.

[6] Donald Howard and Gordon Hunter. "Progress Towards an Intelligent Beehive: Building an Intelligent Environment to Promote the Well-being of Honeybees." Proceedings of the 12th International Conference on Intelligent Environments (IE), 262-265. IEEE, 2016.

[7] Aleksandra Zlatkova. "Honeybees swarming detection approach by sound signal processing." 28th Telecommunications Forum (TELFOR), 1-4. IEEE, 2020.

[8] Karthiga M. "A Deep Learning Approach to classify the Honeybee Species and health Identification." Proceedings of the 2023 International Conference on Computer Science and Artificial Intelligence (CSAI), 1-6. IEEE, 2023.

[9]Khaldi, M., Sriti, M., & Hmida, R. (2021). Honey Bee Classification 
      Based on Deep Learning Techniques. In International Conference on 
      Practical Applications of Agents and Multi-Agent Systems (pp. 271
      281). Springer.
      
[10]Aly, M., Hassanein, A. I., & Abu El-Yazeed, A. A. (2019). Honeybee 
      Classification Based on Deep Learning. In 2019 4th International 
      Conference on Computing, Communications and Security (ICCCS) 
      (pp. 1-6). IEEE. 
      
 [11]Salari, E., Vahedi, E., Bahmanzadegan, A., & Jafari Navimipour, N. 
      (2019). Honey bee species classification using transfer learning. In 
       2019 16th CSI International Symposium on Computer Architecture 
       & Digital Systems (CADS) (pp. 1-6). IEEE.
       
[12]Dreiss, A. N., & Thomson, J. D. (2017). Honey bee navigation: 
       Nature's guidance systems. Naturwissenschaften, 104(9-10), 40. 
       
[13]Ramsey, M. (2019). Beekeeping in the United States. U.S. 
       Department of Agriculture, National Agricultural Statistics Service.
