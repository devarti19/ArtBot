import json, requests
import aiml
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

kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

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

def check_kb(expr, res_expr, kb):
    # check in knowledgebase or add into it
    if expr in kb:
        r = "This fact is already within my knowledge set! Try something else."
    elif res_expr:
        kb.append(expr)
        r = "OK, I will remember that."
    else:
        r = "Sorry this contradicts with what I know!"

    return r

#Task 3: Image Classification
def image_classification():
    model = load_model('img_class.h5')

    root = tk.Tk()

    f_types = [('Jpg Files', '*.jpg'), ('PNG Files', '*.png')]  # type of files to select
    root.filename = tk.filedialog.askopenfilename(multiple=False, filetypes=f_types)
    root.destroy()
    img_path = root.filename

    img = Image.open(img_path)
    img = img.resize((224, 224))

    test_image = image.img_to_array(img)
    test_image /= 255.
    test_image = np.expand_dims(test_image, axis=0)

    prediction = model.predict(test_image)
    prediction_idx = np.argmax(prediction)

    title = "Artist = {}".format(labels[prediction_idx].replace('_', ' '))

    plt.imshow(img)
    plt.title(title)
    plt.show()

    return title



# Run the chatbot in main function
def main():
    # Read Knowledgebase
    kb = read_kb()
    while True:
        try:
            userInput = input("> ")
        except (KeyboardInterrupt, EOFError) as e:
            print("Bye!")
            break
        responseAgent = 'aiml'
        if responseAgent == 'aiml':
            answer = kern.respond(userInput)
        if answer == "":
            continue
        if answer[0] == '#':
            params = answer[1:].split('$')
            cmd = int(params[0])

            if cmd == 0:  # Chatbot exit
                print(params[1])
                break

            elif cmd == 20: # I know that The Scream is artwork
                params[1] = params[1].replace("I know that", "")
                obj, sub = params[1].split(' is ')
                obj, sub = exp_formatting(obj, sub)
                expr = read_expr(sub + ' (' + obj + ')')
                res_expr = ResolutionProver().prove(expr, kb, verbose=False)
                print(check_kb(expr, res_expr, kb))

            elif cmd == 21:  # I know that The Birth of Venus is displayed at The Uffizi Gallery Florence
                params[1] = params[1].replace("I know that", "")
                expr = obj_formatting(params[1])
                res_expr = ResolutionProver().prove(expr, kb, verbose=False)
                print(check_kb(expr, res_expr, kb))

            elif cmd == 22: # I know that Vincent Van Gogh painted The Starry Night
                params[1] = params[1].replace("I know that", "")
                expr = obj_formatting(params[1])
                res_expr = ResolutionProver().prove(expr, kb, verbose=False)
                print(check_kb(expr, res_expr, kb))

            elif cmd == 50: # Check that The Scream is artwork
                params[1] = params[1].replace("Check that", "")
                obj, sub = params[1].split(' is ')
                obj, sub = exp_formatting(obj, sub)
                expr = read_expr(sub + ' (' + obj + ')')
                res_expr = ResolutionProver().prove(expr, kb, verbose=False)
                if res_expr:
                    res = "Correct."
                else:
                    res = "This is incorrect."
                print(res)

            elif cmd == 51:  # Check that The Birth of Venus is displayed at The Uffizi Gallery Florence
                params[1] = params[1].replace("Check that", "")
                expr = obj_formatting(params[1])
                res_expr = ResolutionProver().prove(expr, kb, verbose=False)
                if res_expr:
                    res = "Correct."
                else:
                    res = "This is incorrect."
                print(res)

            elif cmd == 52: # Check that Vincent Van Gogh painted The Starry Night
                params[1] = params[1].replace("Check that", "")
                expr = obj_formatting(params[1])
                res_expr = ResolutionProver().prove(expr, kb, verbose=False)
                if res_expr:
                    res = "Correct."
                else:
                    res = "This is incorrect."
                print(res)

            elif cmd == 25: #Image classification
                res = image_classification()
                print(res)

            elif cmd == 99: #cosine similarity
                res = conversation_QnA(userInput)
                print(res)
        else:
            print(answer)


main()