
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import pickle
import random

import nltk
import itertools
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences


# In[39]:


def get_id2line():
    lines=open('./movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line


# In[40]:


def get_conversations():
    conv_lines = open('./movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
    convs = [ ]
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))
    return convs


# In[41]:


def extract_conversations(convs,id2line,path=''):
    idx = 0
    for conv in convs:
        f_conv = open(path + str(idx)+'.txt', 'w')
        for line_id in conv:
            f_conv.write(id2line[line_id])
            f_conv.write('\n')
        f_conv.close()
    idx += 1


# In[42]:


def gather_dataset(convs, id2line):
    questions = []; answers = []

    for conv in convs:
        if len(conv) %2 != 0:
            conv = conv[:-1]
        for i in range(len(conv)):
            if i%2 == 0:
                questions.append(id2line[conv[i]])
            else:
                answers.append(id2line[conv[i]])

    return questions, answers


# In[43]:


conv=get_conversations()
id2line=get_id2line()
questions,answers=gather_dataset(conv, id2line)
print(questions[:10])


# In[44]:


from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer


# In[45]:


print(word_tokenize(answers[0]))


# In[46]:


# change to lower case (just for en)
questions = [ line.lower() for line in questions ]
answers = [ line.lower() for line in answers ]


# In[47]:


EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''


# In[48]:


'''
 remove anything that isn't in the vocabulary
    return str(pure en)
'''
def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])


# In[49]:


# filter out unnecessary characters
print('\n>> Filter lines')
questions = [ filter_line(line, EN_WHITELIST) for line in questions ]
answers = [ filter_line(line, EN_WHITELIST) for line in answers ]


# In[50]:



'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )
'''
limit = {
        'maxq' : 25,
        'minq' : 2,
        'maxa' : 25,
        'mina' : 2
        }

UNK = 'unk'
VOCAB_SIZE = 8000
def filter_data(qseq, aseq):
    filtered_q, filtered_a = [], []
    raw_data_len = len(qseq)

    assert len(qseq) == len(aseq)

    for i in range(raw_data_len):
        qlen, alen = len(qseq[i].split(' ')), len(aseq[i].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(qseq[i])
                filtered_a.append(aseq[i])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


# In[52]:


# filter out too long or too short sequences
print('\n>> 2nd layer of filtering')
qlines, alines = filter_data(questions, answers)

for q,a in zip(qlines[141:145], alines[141:145]):
    print('q : [{0}]; a : [{1}]'.format(q,a))


# In[55]:





# In[53]:


# convert list of [lines of text] into list of [list of words ]
print('\n>> Segment lines into words')
qtokenized = [ [w.strip() for w in wordlist.split(' ') if w] for wordlist in qlines ]
atokenized = [ [w.st                rip() for w in wordlist.split(' ') if w] for wordlist in alines ]
print('\n:: Sample from segmented list of words')
for q,a in zip(qtokenized[141:145], atokenized[141:145]):
    print('q : [{0}]; a : [{1}]'.format(q,a))


# In[56]:


def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


# In[57]:


# indexing -> idx2w, w2idx
print('\n >> Index words')
idx2w, w2idx, freq_dist = index_( qtokenized + atokenized, vocab_size=VOCAB_SIZE)
np.save('models/word-input-word2idx.npy', w2idx)
np.save('models/word-input-idx2word.npy', idx2w)


# In[58]:


def filter_unk(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)

    filtered_q, filtered_a = [], []

    for qline, aline in zip(qtokenized, atokenized):
        unk_count_q = len([ w for w in qline if w not in w2idx ])
        unk_count_a = len([ w for w in aline if w not in w2idx ])
        if unk_count_a <= 2:
            if unk_count_q > 0:
                if unk_count_q/len(qline) > 0.2:
                    pass
            filtered_q.append(qline)
            filtered_a.append(aline)

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((data_len - filt_data_len)*100/data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


# In[59]:


# filter out sentences with too many unknowns
print('\n >> Filter Unknowns')
qtokenized, atokenized = filter_unk(qtokenized, atokenized, w2idx)
print('\n Final dataset len : ' + str(len(qtokenized)))


# In[62]:



'''
 create the final dataset :
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )
'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]
'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))


# In[63]:


print('\n >> Zero Padding')
idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

print('\n >> Save numpy arrays to disk')
# save them
np.save('./models/idx_q.npy', idx_q)
np.save('./models/idx_a.npy', idx_a)

# let us now save the necessary dictionaries
metadata = {
        'w2idx' : w2idx,
        'idx2w' : idx2w,
        'limit' : limit,
        'freq_dist' : freq_dist
            }

# write to disk : data control dictionaries
with open('./models/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)


# In[64]:


# count of unknowns
unk_count = (idx_q == 1).sum() + (idx_a == 1).sum()
# count of words
word_count = (idx_q > 1).sum() + (idx_a > 1).sum()

print('% unknown : {0}'.format(100 * (unk_count/word_count)))
print('Dataset count : ' + str(idx_q.shape[0]))


print ('>> gathered questions and answers.\n')


# In[68]:



import numpy as np
from random import sample

'''
 split data into train (70%), test (15%) and valid(15%)
    return tuple( (trainX, trainY), (testX,testY), (validX,validY) )
'''
def split_dataset(x, y, ratio = [0.7, 0.15, 0.15] ):
    # number of examples
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX,trainY), (testX,testY), (validX,validY)
def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + './models/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + './models/idx_q.npy')
    idx_a = np.load(PATH + './models/idx_a.npy')
    return metadata, idx_q, idx_a


# In[69]:


metadata, idx_q, idx_a=load_data(PATH='')
train,test,val=split_dataset(idx_q, idx_a, ratio = [0.7, 0.15, 0.15] )
ques=train[0]
ans=train[1]


# In[70]:


from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from keras.callbacks import ModelCheckpoint


# In[71]:


word_embedding_size = 100
sentence_embedding_size = 300
dictionary_size =7000
maxlen_input = 25
maxlen_output = 25
num_subsets = 1
Epochs = 100
BatchSize = 128  #  Check the capacity of your GPU
Patience = 0
dropout = .25
n_test = 100


# In[73]:


from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Bidirectional, Dropout, merge
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
import keras.backend as K
import os
WEIGHT_FILE_PATH='./models/test.h5'


# In[74]:


#define encoder artitecture
encoder_inputs = Input(shape=(None,25), name='encoder_inputs')
print(encoder_inputs)
encoder = LSTM(units=256, return_state=True, name="encoder_lstm")
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

#put encoder state-h and state-c in Decoder
decoder_inputs = Input(shape=(None,25), name='decoder_inputs')
print(decoder_inputs)
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(25, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)




model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
ques=ques.reshape((ques.shape[0], 1,ques.shape[1]))
ans1=ans.reshape((ans.shape[0], 1,ans.shape[1]))
checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)
model.fit([ques, ans1], ans1, batch_size=60, epochs=10,validation_split=0.2,callbacks=[checkpoint])


# In[75]:


model.save_weights(WEIGHT_FILE_PATH)
json = model.to_json()
open('./models/char-architecture.json', 'w').write(json)
