import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # gets rid of harmless error messages
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'
import json
import numpy as np
import tkinter as tk
from tkinter.messagebox import showinfo
from tensorflow import keras
from tkinter import filedialog as fd
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from keras.models import load_model
from plot_raw_signal import *
from RUL_model import *


import colorama
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

type_names = ["No Fault","exciting_Class1","exciting_Class2","exciting_Class3","exciting_Class4","exciting_Class5", "exciting_Class6","exciting_Class7","exciting_Class8","exciting_Class9","exciting_Class10", "exciting_Class11","exciting_tt","exciting_ww",
              'series_Class1','series_Class2','series_Class3','series_Class4','series_Class5','series_Class6','series_Class7','series_Class8','series_Class9','series_Class10','series_Class11','series_tt','series_ww',
               'transformer_Class1','transformer_Class2','transformer_Class3','transformer_Class4','transformer_Class5','transformer_Class6','transformer_Class7','transformer_Class8','transformer_Class9','transformer_Class10','transformer_Class11','transformer_tt','transformer_ww',
                "Capacitor_Switch", "external_fault","ferroresonance",  "Magnetic_Inrush","Non_Linear_Load_Switch","Sympathetic_inrush"]

with open('FPL_Datasets/Chatbot/Simple_Chatbot/intents.json') as file:
    data = json.load(file)

loaded_transformer_model = load_model('FPL_Datasets/ML_TIME/vanilla_transformer_model.keras')

def prediction(input, loaded_model): # shape(726, 3), /file.keras
    pred = loaded_model.predict(input) # predict for a whole dataset shape(num_files, 726, 3)
    pred_fault = type_names[np.argmax(pred)]
    return pred_fault

def select_file(initialdir):
    filetypes = (('text files', '*'),('All files', '*.*'))
    filepath = fd.askopenfilename(title='Please select your file for Fault',initialdir=initialdir, filetypes=filetypes)
    showinfo(title='Selected File',message=filepath)
    return filepath

def load_fault_data(initialdir):
    X = []
    fault_path = select_file(initialdir)
    fault_signal = pd.read_csv(fault_path, header=None)
    signal_is = [fault_signal[z].values[:726] for z in range(1, 4)]
    X.append(np.stack(signal_is, axis=-1))
    X = np.array(X, dtype = np.float32)
    plot_signal(fault_path)
    return X

def fault_detection(i):
    response = "Performing fault detection...\n" 
    X = load_fault_data(initialdir='/home/isense/Transformer/FPL_Datasets/PSCAD_datasets/Total_Multiclass')
    pred_fault = prediction(X, loaded_transformer_model)
    return (response + np.random.choice(i['responses']) + f'\n{pred_fault}', "FD")

def RUL_detection(i):
    response = "Performing Remaining Useful Life prediction... \n" 
    X = select_file(initialdir='/home/isense/Transformer/FPL_Datasets/ETT_Datasets/Dataset')
    pred_RUL = remaining_life(X)
    return (response + np.random.choice(i['responses']) + f'\n{pred_RUL}' + ' Days', "RUL")

def chat(message):
    # load trained model
    model = keras.models.load_model('FPL_Datasets/Chatbot/Simple_Chatbot/chat_model')
    # load tokenizer object
    with open('FPL_Datasets/Chatbot/Simple_Chatbot/tokenizer.pickle', 'rb') as enc:
        tokenizer = pickle.load(enc)
    # load label encoder object
    with open('FPL_Datasets/Chatbot/Simple_Chatbot/label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    # parameters
    max_len = 20
    while True:
        inp = message
        if inp.lower() == ('quit' or 'exit'):
            exit()
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                                          truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        for i in data['intents']:
            if i['tag'] == tag:
                if 'goodbye' == tag:
                    return np.random.choice(i['responses']), exit()
                elif 'fault_detection' == tag:
                    return fault_detection(i)
                elif "rul_estimation" == tag:
                    #RUL
                    return RUL_detection(i)
                else:
                    return (np.random.choice(i['responses']), None)



# if __name__ == "__main__":



