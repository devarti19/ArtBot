import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image
import keras.utils as image
from tensorflow.keras.models import load_model



labels = {0: 'Piet_Mondrian',1: 'Albrecht_Du╠êrer',2: 'Edgar_Degas',3: 'Diego_Velazquez',4: 'Georges_Seurat',5: 'Andrei_Rublev',
 6: 'Francisco_Goya',7: 'Alfred_Sisley',8: 'Rene_Magritte',9: 'Michelangelo',10: 'Titian',11: 'Giotto_di_Bondone',12: 'Jan_van_Eyck',
 13: 'Eugene_Delacroix',14: 'Andy_Warhol',15: 'Pieter_Bruegel',16: 'El_Greco',17: 'Edouard_Manet',18: 'Paul_Klee',19: 'Paul_Gauguin',
 20: 'Claude_Monet',21: 'Marc_Chagall',22: 'Sandro_Botticelli',23: 'Camille_Pissarro',24: 'Paul_Cezanne',25: 'Kazimir_Malevich',
 26: 'Henri_de_Toulouse-Lautrec',27: 'Salvador_Dali',28: 'Diego_Rivera',29: 'Gustav_Klimt',30: 'Vasiliy_Kandinskiy',31: 'Vincent_van_Gogh',
 32: 'Gustave_Courbet',33: 'Amedeo_Modigliani',34: 'Henri_Matisse',35: 'Pablo_Picasso',36: 'Peter_Paul_Rubens',37: 'Pierre-Auguste_Renoir',
 38: 'Jackson_Pollock',39: 'Edvard_Munch',40: 'Frida_Kahlo',41: 'Joan_Miro',42: 'Hieronymus_Bosch',43: 'Caravaggio',44: 'Mikhail_Vrubel',45: 'Raphael',
 46: 'Rembrandt',47: 'Leonardo_da_Vinci',48: 'Henri_Rousseau',49: 'William_Turner',50: 'Albrecht_Dürer'}

read_expr = Expression.fromstring

#Task 1: Conversation with finding similarity
def conversation_QnA(user_input):
    #reading sampleQA file
    df = pd.read_csv('conversation.csv')
    # Preprocess the data
    lemmatizer = WordNetLemmatizer()
    df['question_lemmatized'] = df['question'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x.lower())]))
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['question_lemmatized'])

    user_input_lemmatized = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(user_input.lower())])
    user_input_tfidf = tfidf.transform([user_input_lemmatized])
    similarities = cosine_similarity(user_input_tfidf, tfidf_matrix)
    closest_match_idx = np.argmax(similarities)
    if similarities[0][closest_match_idx] == 0:
        return "I'm sorry, I don't understand."
    else:
        return df['answer'][closest_match_idx]


# Task 2: First order logic KnowledgeBase
def read_kb():
    # reading knowledgebase csv file and appending it into list
    clauses = []
    kb_file = 'kb.csv'

    with open(kb_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                clauses.append(read_expr(item))
    return clauses

def exp_formatting(obj, sub):
    # Removing spaces
    obj = obj.replace(" ", "")
    sub = sub.replace(" ", "")
    return obj, sub

def obj_formatting(user_input):
    # Forming first logic sentence
    if "displayed" in user_input:
        sub = "displayed"
        obj1, obj2 = user_input.split('is displayed at')
        obj1, obj2 = exp_formatting(obj1, obj2)

    elif "painted" in user_input:
        sub = "art"
        obj1, obj2 = user_input.split('painted')
        obj1, obj2 = exp_formatting(obj1, obj2)

    expr = read_expr(sub + ' (' + obj1 + "," + obj2 + ')')
    return expr


#Task 3: Image Classification
def upload_file():
    f_types = [('Jpg Files', '*.jpg'),('PNG Files','*.png')]   # type of files to select
    filename = tk.filedialog.askopenfilename(multiple=False,filetypes=f_types)
    return filename


def image_classification():
    model =load_model('img_class.h5')

    root = tk.Tk()
    root.geometry("200x100")  # Size of the window
    root.title('ArtBot')
    # b1 = tk.Button(root, text='Select an image',width =20, command = tk.filedialog.askopenfilename(multiple=False,filetypes=f_types)).pack(side= TOP)
    b2 = tk.Button(root, text='Close the window', width=20, command=root.quit).pack(side=TOP)

    filename = upload_file()
    img = Image.open(filename)
    img = img.resize((224, 224))

    test_image = image.img_to_array(img)
    test_image /= 255.
    test_image = np.expand_dims(test_image, axis=0)

    prediction = model.predict(test_image)
    prediction_probability = np.amax(prediction)
    prediction_idx = np.argmax(prediction)

    title = "Predicted artist = {}\nPrediction probability = {:.2f} %" \
                .format(labels[prediction_idx].replace('_', ' '), prediction_probability*100)

    plt.imshow(img)
    plt.title(title)
    plt.show()
    root.mainloop()
    return title


# Chatbot Function
# function to handle user input
def respond_to_user_input(user_input):
    # Read Knowledgebase
    kb = read_kb()

    # I know that...is...
    if "I know that" in user_input:
        user_input = user_input.replace("I know that", "")
        if "displayed" in user_input:  # I know that The Birth of Venus is displayed at The Uffizi Gallery Florence
            expr = obj_formatting(user_input)
        elif "painted" in user_input:  # I know that Vincent Van Gogh painted The Starry Night
            expr = obj_formatting(user_input)
        else:  # I know that The Scream is artwork
            obj, sub = user_input.split(' is ')
            obj, sub = exp_formatting(obj, sub)
            expr = read_expr(sub + ' (' + obj + ')')
        res_expr = ResolutionProver().prove(expr, kb, verbose=False)



    # Check that...is..
    elif "Check that" in user_input:
        user_input = user_input.replace("Check that", "")
        if "displayed" in user_input:  # Check that The Birth of Venus is displayed at The Uffizi Gallery Florence
            expr = obj_formatting(user_input)
        elif "painted" in user_input:  # Check that Vincent Van Gogh painted The Starry Night
            expr = obj_formatting(user_input)
        else:  # Check that The Scream is artwork
            obj, sub = user_input.split(' is ')
            obj, sub = exp_formatting(obj, sub)
            expr = read_expr(sub + ' (' + obj + ')')
        res_expr = ResolutionProver().prove(expr, kb, verbose=False)

        # check in knowledgebase
        #if expr not in kb:
        #    res = "Sorry, I don't know"
        if res_expr:
            res = "Correct."
        else:
            res = "This is incorrect."

    elif "Select image" in user_input:
        res = image_classification()

    else:
        res = conversation_QnA(user_input)

    return res


# Run the chatbot in main function
def main():
    print("Bot: Hello! I'm an artbot.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'hi' or user_input.lower() == 'hello':
            print("Bot: Ask me anything about art and artist or I can suggest something.")
        elif user_input.lower() == "suggest something":
            print( "Bot: 1. Facts about Art \n 2. Make me remember any facts \n 3. Check any facts \n 4.Select any image to find it's artist")
        elif user_input.lower() == "check facts" or user_input.lower() == "remember facts":
            print("Bot: Enter the fact")
        elif user_input.lower() == 'bye':
            print("Bot: Goodbye!")
            break
        else:
            response = respond_to_user_input(user_input)
            print("Bot: " + response)

main()