# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:38:08 2020

@author: dhamuk
"""

from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


###constants for sentence start and end
START='<start> '
END=' <end>'

def readDoc(filename):
	
	file = open(filename, 'r')	
	doc = file.read()	
	file.close()
	return doc
 

def loadDataset(filename):
    doc = readDoc(filename)
    dataset = list()
    
    for l in doc.split('\n'):    	
            if len(l) < 1: #skip empty
                continue
        		
            text=l
            dataset.append(text)
    return set(dataset)
 

def getDescriptions_dataset(filename, dataset):

	document = readDoc(filename)
	descriptions_all = dict()
	for line in document.split('\n'):	
		words = line.split()		
		imgId, img_desc = words[0], words[1:]	
		if imgId in dataset:
		
			if imgId not in descriptions_all:
				descriptions_all[imgId] = list()
		
			desc =  START+ ' '.join(img_desc) +END
		
			descriptions_all[imgId].append(desc)
	return descriptions_all
 


###get all photo features for a dataset
def getFeatures_dataset(file, dataset):	
	feature_all = load(open(file, 'rb'))	
	feature = {f: feature_all[f] for f in dataset}
	return feature
 
####list of descriptions
def desc_list(descriptions):
	descList = list()
	for k in descriptions.keys():
		[descList.append(d) for d in descriptions[k]]
	return descList
 

def fit_tokenizer(descriptions):
	desc = desc_list(descriptions)
    ##User keras Tokenizer
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(desc)
	return tokenizer
 
####get max lenght of caption
def max_length(descriptions):
	captions = desc_list(descriptions)    
	return max(len(c.split()) for c in captions)
 


# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
