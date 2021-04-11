# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 21:50:15 2020

@author: dhamuk
"""

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding,BatchNormalization, Dropout, TimeDistributed, Dense,Concatenate, RepeatVector, Activation, Input, add
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input
from tqdm import tqdm
from keras.preprocessing import image
from keras import backend 
from keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import Utils as util
from keras.utils import to_categorical



###set path
root_path_dataset = 'Y:/Flickr_Data/'
root_path_features = 'Y:/lab2/ShowTell_Vinit_5thMorn/showtell/data'



###load training description dataset ######
trainfile=root_path_dataset+'Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'
train = util.loadDataset(trainfile)

###load encoded photo features ####
features=pickle.load(open(root_path_features+'/featuresNew.pkl', 'rb'))
print('Features length :',len(features))

####get descriptions
descriptions_map = util.getDescriptions_dataset(root_path_features+'/descriptions.txt',train)
samples=len(descriptions_map)
print('Number of samples :',samples)
all_words=util.desc_list(descriptions_map)
no_words=len(all_words)
print('Number of words :' ,no_words)

### read vocab file ###
vocab= [] 
vocab = list(set(vocab))
vocab= pickle.load(open(root_path_features+'/vocab_new.p', 'rb'))
vocab_size = len(vocab)
print('Vocab size : ',vocab_size)
###

### max caption size
max_length = util.max_length(descriptions_map)
print('max length caption : ',max_length)

### train and dump tokenizer
tokenizer = util.fit_tokenizer(descriptions_map)
pickle.dump( tokenizer, open( "tokernizer.p", "wb" ) )
###
    
#### this method creates the instances with image,input word and next word as output
def createInstancesByWords(max_length, desc_list, photo):
    img_in1,word_in2, word_y = list(), list(), list()	
    
    for desc in desc_list:
		# get embeddings
        word_seq = tokenizer.texts_to_sequences([desc])[0]       
		
        for i in range(1, len(word_seq)):           
			
            in_seq, out_seq = word_seq[:i], word_seq[i] #break into words sequence till now and the next word			
            in_seq = sequence.pad_sequences([in_seq], maxlen=max_length)[0] # pad			
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0] # embed outout
			#create list 
            img_in1.append(photo)
            word_in2.append(in_seq)
            word_y.append(out_seq)
            
    return np.asarray(img_in1),np.asarray(word_in2),np.asarray(word_y)
                
                
    

### Create batches for training
def data_generator():
    
    while 1:
        ##for all images get img and captions
        for key, caption in descriptions_map.items():              
           
            photo = features[key] #get feature for img
            in1,in2,out=createInstancesByWords( max_length,caption,photo ) #get instances of all words sequences
            
            yield [[in1, in2], out]

      
            
       


# In[67]:


# make model
def get_model(input_dim=2048):
    
    #image    
    embed_dim = 300 
    image_inputs = Input(shape=(input_dim,))
    img_layer = Dropout(0.5)(image_inputs)
    img_model_D = Dense(256, activation='relu')(img_layer)
    #sentence
    caption_in = Input(shape=(max_length,))
    caption_embed = Embedding(vocab_size, embed_dim, mask_zero=True)(caption_in)
    caption_dropout = Dropout(0.5)(caption_embed)
    lstm_model = LSTM(256)(caption_dropout)
    #combine inputs to decoder
    decoder = add([img_model_D, lstm_model])
    decoder_final = Dense(256, activation='relu')(decoder)
    #output layer
    output = Dense(vocab_size, activation='softmax')(decoder_final)
    #final model
    model = Model(inputs=[image_inputs, caption_in], outputs=output)    
    model.summary()
    
    return model
    
    

 
    


# In[ ]:

 
#####training #####
    
epoch = 5
#batch_size = 512
model=get_model()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


### train per epoch

for i in range(epoch): 	
     generator = data_generator() #generate data for epoch
     #train
     history=model.fit_generator(generator, epochs=1, steps_per_epoch=samples, verbose=1)
     #save
     model.save('showTellFulltrain_12jan' + str(i) + '.h5')  
     pickle.dump( history, open( "hist"+ str(i)+".p", "wb" ) )
    
    
   

