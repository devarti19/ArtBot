# ArtBot

This repository contains the chatbot program for Artworks called ArtBot. The chatbot does conversation with the user and provide information about famous artwork, artist, various genres of art, art movements and history. The chatbot remembers art related facts and check the facts as well. There is one more functionality added to the chatbot where the artwork image is selected from local computer and the chatbot identifies which artist created that artwork.

The chatbot is made in two approach. 
<li> Using for loop in python.
<li> Using AIML xml file.
  
First, one sample conversation csv file is read, and the questions are lemmatised to find the root form of the words, transformed into TF-IDF and using cosine similarity the question asked by the user are related to the already present conversation. 

First order logic is implemented next. In this, one knowledgebase file is created using nltk format and converted into a list. Then the fact written by the user is first converted into the readable format and compared with the already present facts. If the fact is already present in the knowledgebase, message is displayed that the fact is already present. If the fact is not present, it is added to the knowledgebase and if the fact contradicts the already present fact, message is displayed accordingly. Same goes for checking the facts from knowledgebase. This will display whether the fact is correct or incorrect. 

Third functionality includes image classification. The dataset of artwork and their respective artists is taken from Kaggle. The dataset is augmentated so that the dataset is increased and can be fitted to the model. Deep learning model is used for classifying the images. Pretrained ResNet model is used to train with 20 epoch using Adam optimizer. The trained model is then evaluated, and it achieves 85% accuracy. This trained model is saved here (<a>https://drive.google.com/file/d/1WknkUJGMhED3Rctx15TDTLsS5gBGczOh/view?usp=sharing</a>). 

The image of artwork is selected from local computer using tkinter library, where a new window is opened and after selecting the image, the Machine Learning model is loaded, and it will predict the artist who created the selected artwork. This window can be closed from the close button in tkinter. 
