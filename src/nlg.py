from fis import *
import sys
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn 
import pandas as pd


import warnings

# Ignorar advertencias específicas
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

# Tu código aquí


class NLG_Genesys():

    def __init__(self):

        self.fis = fis()
        
        with open('tokenizer.json', 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
            self.tokenizer = tokenizer_from_json(tokenizer_json)


        #print(self.tokenizer.word_index)

    
        #self.Scaler = joblib.load('src/scaler.joblib')
        self.num_decoder_tokens = len(self.tokenizer.word_index) + 1

        self.model = seq2seqLSTM(4, 64, self.num_decoder_tokens)
        self.model.load_state_dict(torch.load('seq2seqLSTM_model.pt'))



    def predict(self,input:list):
        fuzzy_set=[]

        # get Fuzzy set names 
        for i in range(len(input)):
            if i == 0:
                name,_= self.fis.get_membership(antecedent_name='proporcional',value=input[i])
                fuzzy_set.append(name)
            elif i == 1:
                name,_= self.fis.get_membership(antecedent_name='derivativo',value=input[i])
                fuzzy_set.append(name)
            elif i == 2:
                name,_= self.fis.get_membership(antecedent_name='salida',value=input[i])
                fuzzy_set.append(name)
            elif i == 3:
                name,_= self.fis.get_membership(antecedent_name='salida',value=input[i])
                fuzzy_set.append(name)
        #print('Fuzzy sets')
        #print(fuzzy_set)
        # Normalize and get tensor for decoder inputs        
        #enc_test_in = self.Scaler.transform(np.array(input).reshape(1,-1))
        enc_test_in = np.array(input).astype((float)).reshape(1,-1)   
        enc_test_in = torch.tensor(enc_test_in.reshape(1,1,-1)).float()

        #Tokenize fuzzy set names and padding
        fuzzy_set=' '.join(fuzzy_set)
        dec_test_in = self.tokenizer.texts_to_sequences([fuzzy_set])
        #print('Tokens sequences')
        #print(dec_test_in)
        dec_test_in = pad_sequences(dec_test_in, maxlen=self.num_decoder_tokens, padding='post', truncating='post')
        #print(dec_test_in)

        # get decoder input tensor
        dec_test_in = torch.tensor(dec_test_in).long()

        self.model.eval()
        # Hacer predicciones con el modelo cargado
        with torch.no_grad():
            preds = self.model(enc_test_in, dec_test_in)
            _, predicts = torch.max(preds.data, 1)

        predicciones_finales = self.__predictions(predicts.numpy())
        return predicciones_finales


    def __predictions(self,preds):
        preds_words = []
        for i in range(preds.shape[0]):
            preds_row = preds[i]
            preds_row_words = []
            for idx in preds_row:
                if idx in self.tokenizer.index_word:
                    preds_row_words.append(self.tokenizer.index_word[idx])
            preds_words.append(' '.join(preds_row_words))

        return ' '.join(preds_words)
    

class seq2seqLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_decoder_tokens, num_layers=1):
        super(seq2seqLSTM, self).__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder_embedding = nn.Embedding(num_decoder_tokens, 128)
        self.decoder_lstm = nn.LSTM(128, hidden_size, num_layers=1, batch_first=True)
        self.linear_layer = nn.Linear(hidden_size, num_decoder_tokens)

    def forward(self, inputs, dec_inputs):
        enc_output, (state_h, state_c) = self.encoder_lstm(inputs)
        embedding = self.decoder_embedding(dec_inputs)
        decoder_output, _ = self.decoder_lstm(embedding, (state_h, state_c))
        decoder_output = decoder_output.reshape(-1, decoder_output.shape[2])
        linear_output = self.linear_layer(decoder_output)
        # Remodela para que tenga el mismo número de pasos de tiempo que dec_inputs
        linear_output = linear_output.reshape(dec_inputs.shape[0], dec_inputs.shape[1], -1)
        return linear_output


####--------------------------------------------------
# PARA EJECUTAR DESDE FUERA

data = input('Ingrese la ruta del archivo a procesar: ')

try:
    with open(data, 'r') as file:
        rows = file.readlines()
        dataset = [row.strip().split(" / ") for row in rows]
        # Imprimir el dataset
        for row in dataset:
            print(row)
except FileNotFoundError:
    print("Ruta incorrecta o no existe archivo")



## Obtener la lista
#numeros_str = sys.argv[1]  # el primer parámetro
#pdlr = [[float(n) for n in numeros_str.split(",")]]
#
## Obtener el nombre de archivo
#nombre_archivo = sys.argv[2]  # el segundo parámetro
#


modelo = NLG_Genesys()

#for i in range(len(dataset)):
#    resultado = modelo.predict(dataset[i])
#    print(resultado)
nombre_archivo = input('Nombre archivo de salida: ')
with open(fr'C:\Users\leand\Documents\LEANDRO\UBA\CEIA\PROYECTO DE GRADO\src\/{nombre_archivo}', 'w') as f:
    for i in range(len(dataset)):
        resultado = modelo.predict(dataset[i])
        f.write(resultado + '\n')
print('instrucciones finalizadas')