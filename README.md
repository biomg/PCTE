# PCTE
Parallel convolutional neural networks and Transforemer encoders (PCTE)
We propose a deep learning model based on parallel convolutional neural networks and Transforemer encoders (PCTE), and a new TCRs preprocessing method. We use word vectors that can participate in training to replace the fixed vectors used by Beshnova et al., making it easier for the model to extract time series features of TCRs. We cut the redundant amino acids in TCRs and fill the shorter TCRs with meaningless word vectors, so that all TCRs have the same length and can be trained using the same model, which greatly improves the utilization rate of the dataset. PCTE consists of four parts: a primary feature extractor composed of convolutions, a time series feature extractor composed of Transformer encoders, an advanced feature extractor composed of convolutions, and a feature classifier composed of fully connected layers. PCTE can effectively extract the features of TCRs, and after ten times triple fold cross validation, an AUC of 0.86 was obtained on Wong et al.'s dataset.
# Dependency:
Python 3.7 <br>
Pytorch 1.13.1 <br>
Numpy 1.18.5 <br>
# Data 
The dataset required for PCW is contained in the https://github.com/cew88/AutoCAT, transfer all files in the DeepCatinput in AutoCAT to the DeepCat TrainingData directory, place the PCW.py file in the DeepCat root directory, and run the PCW.py file to train and test PCW. DeepCat can be found on the https://github.com/s175573/DeepCAT
# Usage:
python train_and_test.py
