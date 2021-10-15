# ------------------------------------------------------------------------------
# Project 2
# Written by Aria Adibi, Student id: ****
# For COMP 6721 Section F – Fall 2019
# ------------------------------------------------------------------------------
import os
import traceback

import numpy as np
import pandas as pd

import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import math

import nltk
nltk.download('punkt')
nltk.download('wordnet')
import nltk.tokenize
import nltk.stem
import nltk.corpus

import sklearn
import sklearn.metrics           # For accuracy_score

#Global variables for accuracy and memory/speed management----------------------

## Global & Hyper parameters ---------------------------------------------------
RND_SEED= 10238642

DATA_DIR_PATH= '../'
DATA_FILE_NAME= 'hn2018_2019 copy 100.csv'
pd.set_option('display.max_colwidth', -1)

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

TARGET_COL_NAME= 'Post Type'
POST_TYPES= ['story', 'ask_hn', 'show_hn', 'poll']

RM_PUNCTUATIONS= set( [':', '!', '.', ',', ';'] ) #? not included because I think it can be useful
CONVERT= { 'n\'t' : 'not' }
PREPROC_RM_STOP_WORDS= set( ['\'s', '_', '-'] ).union( RM_PUNCTUATIONS )
# PREPROC_RM_STOP_WORDS= PREPROC_RM_STOP_WORDS.union( set( nltk.corpus.stopwords.words('english') ) )
REMOVED= set()
VOCAB= set()

PSEUDOCOUNT= 0.5

FREQ= 0 #TODO
LOG10PROB= 1 #TODO
Y_SCORES= {} #TODO

#-------------------------------------------------------------------------------
def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text

#TODO Sort out Computer Science .......
def lowercase_tokenize_lemmatize(text):
    global VOCAB, REMOVED

    text= text.lower()
    token_words= nltk.tokenize.word_tokenize(text)

    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized= []
    for word in token_words:
        if word not in PREPROC_RM_STOP_WORDS:

            if word in CONVERT:
                word= CONVERT[word]

            word_lemma= wordnet_lemmatizer.lemmatize(word, pos= 'v')
            if word not in lemmatized:
                lemmatized.append(word_lemma)

        elif word not in REMOVED:
            REMOVED.add(word)

    VOCAB= VOCAB.union(lemmatized)
    return lemmatized

def read_the_data():
    '''
    No missing value is assumed. --> No imputation is necessary
    '''
    # names of columns, as per description
    cols_names= [ 'ID', 'Object ID', 'Title', TARGET_COL_NAME, 'Author', 'Created At', 'URL', 'Points', 'Number of Comments']

    # read the data
    hn_18_19= pd.read_csv(DATA_DIR_PATH + DATA_FILE_NAME, header= 0, names= cols_names,
                          converters= { 'Title' : lowercase_tokenize_lemmatize,\
                                        TARGET_COL_NAME : strip,\
                                        'Created At' : strip\
                                      }\
                         )

    hn_18_19= hn_18_19.drop(columns= ['ID', 'Object ID', 'Author', 'URL', 'Points', 'Number of Comments'])

    X_f_test= hn_18_19[ hn_18_19['Created At'].str.contains('2019') ].drop( columns= 'Created At' )
    y_f_test= X_f_test[TARGET_COL_NAME].copy()
    X_f_test= X_f_test.drop( columns= TARGET_COL_NAME ).copy()

    X= hn_18_19[ hn_18_19['Created At'].str.contains('2018') ].drop( columns= 'Created At' )
    y= X[TARGET_COL_NAME].copy()
    X= X.drop( columns= TARGET_COL_NAME ).copy()

    del hn_18_19

    return X_f_test, y_f_test, X, y

#-------------------------------------------------------------------------------
def log10_additive_smoothing(n_occurrence, n_trials, n_pos_observations, pseudocount):
    # if(pseudocount == 0): #TODO
        # print('here, n_occurrence= {}, n_trials= {}, n_pos_observations= {}, pseudocount= {}'.format(n_occurrence, n_trials, n_pos_observations, pseudocount))
    return math.log10( (n_occurrence + pseudocount)/(n_trials + pseudocount * n_pos_observations) )

#TODO
def inc_freq(title, word):
    global FREQ #TODO
    if word in title:
        FREQ += 1

def calc_vocab_ptype_freq_log10prob(vocab, ptypes, X, y): #TODO naming
    '''
    P(wi | cj)
    '''
    global FREQ #TODO
    global LOG10PROB

    vocab_ptype_freq_log10prob= dict()

    for ptype in ptypes:
        for word in vocab:
            X_ptype= X[ y==ptype ]
            FREQ= 0;
            [ inc_freq(title, word) for title in X_ptype['Title'] ]
            LOG10PROB= log10_additive_smoothing( FREQ, len(X_ptype), len(vocab), PSEUDOCOUNT )
            vocab_ptype_freq_log10prob.update( { (word, ptype) : (FREQ, LOG10PROB) } )

    return vocab_ptype_freq_log10prob

def calc_vocab_log10prob(vocab, X): #TODO naming
    global FREQ #TODO
    global LOG10PROB

    vocab_log10prob= dict()

    for word in vocab:
        FREQ= 0
        [ inc_freq(title, word) for title in X['Title'] ]
        LOG10PROB= log10_additive_smoothing( FREQ, len(X), len(vocab), PSEUDOCOUNT )
        vocab_log10prob.update( { word : (FREQ, LOG10PROB) } )

    return vocab_log10prob

def calc_ptype_log10prob(ptypes, y): #TODO naming
    global FREQ #TODO
    global LOG10PROB

    ptype_log10prob= dict()

    for ptype in ptypes:
        FREQ= sum( y == ptype )
        LOG10PROB= log10_additive_smoothing( FREQ, len(y), len(ptypes), PSEUDOCOUNT )
        ptype_log10prob.update( { ptype : (FREQ, LOG10PROB) } )

    return ptype_log10prob

#-------------------------------------------------------------------------------

def calc_classtype_log10prob(title, classtype, vocab_ptype_freq_log10prob, vocab_log10prob, ptype_log10prob):
    global Y_SCORES #TODO
    global LOG10PROB #TODO
    LOG10PROB= ptype_log10prob[classtype][1]

    for word in title:
        if word not in vocab_log10prob: #ignore unknown words
            continue
        LOG10PROB= LOG10PROB + vocab_ptype_freq_log10prob[ (word, classtype) ][1] - vocab_log10prob[word][1]

    Y_SCORES[classtype].append(LOG10PROB)

def predict(X_test, vocab_ptype_freq_log10prob, vocab_log10prob, ptype_log10prob):
    global Y_SCORES # TODO
    Y_SCORES= {}
    for ptype in POST_TYPES:
        Y_SCORES.update( {ptype : []} )
        [ calc_classtype_log10prob(title, ptype, vocab_ptype_freq_log10prob, vocab_log10prob, ptype_log10prob) for title in X_test['Title'] ]

    #TODO #TODO
    y_predict= []
    for i in range( len(X_test) ):
        ypsi= -np.inf
        prediction= POST_TYPES[0]
        for ptype in POST_TYPES:
            if ypsi <  Y_SCORES[ptype][i]:
                ypsi= Y_SCORES[ptype][i]
                prediction= ptype

        y_predict.append(prediction)

    return y_predict

def report_model_and_result(X_f_test, y_f_test, X, y, vocab, model_name, result_name):
    t_start_report= time.time()

    vocab_ptype_freq_log10prob= calc_vocab_ptype_freq_log10prob(vocab, POST_TYPES, X, y)
    vocab_log10prob= calc_vocab_log10prob(vocab, X)
    ptype_log10prob= calc_ptype_log10prob(POST_TYPES, y)

    #write the model
    with open('./' + model_name, 'w') as file:
        sorted_vocab= sorted(vocab)
        line_num= 0
        for word in sorted_vocab:
            line_num += 1
            file.write('{the_line_num}  {the_word}'.format(the_line_num= line_num, the_word= word))
            for ptype in POST_TYPES:
                file.write('  {}  {:.10f}'.format(vocab_ptype_freq_log10prob[ (word, ptype) ][0], vocab_ptype_freq_log10prob[ (word, ptype) ][1])) #freq word_ptype, log10prob word_ptype
            file.write('\n')


    y_predict= predict(X_f_test, vocab_ptype_freq_log10prob, vocab_log10prob, ptype_log10prob)
    num_correct= 0
    #write the result
    with open('./' + result_name, 'w') as file: #TODO original title is wanted?
        for i in range( len(X_f_test) ):

            correctness= 'wrong'
            if y_predict[i] == y_f_test.iloc[i]:
                num_correct += 1
                correctness= 'right'

            file.write('{the_line_num}  {the_title}  {prediction}  {ptype1_score}  {ptype2_score}  {ptype3_score}  {ptype4_score}  {org_ptype}  {correctness}\n'\
            .format(    the_line_num= i + 1, the_title= X_f_test['Title'].iloc[i], prediction= y_predict[i],\
                        ptype1_score= Y_SCORES[POST_TYPES[0]][i], ptype2_score= Y_SCORES[POST_TYPES[1]][i], ptype3_score= Y_SCORES[POST_TYPES[2]][i], ptype4_score= Y_SCORES[POST_TYPES[3]][i],\
                        org_ptype= y_f_test.iloc[i], correctness= correctness)   )

    accuracy= (num_correct / len(y_f_test)) * 100
    return accuracy, time.time()-t_start_report

if (__name__ == '__main__'):
    t_start= time.time()

    X_f_test, y_f_test, X, y= read_the_data()
    with open('./vocabulary.txt', 'w') as file:
        for word in VOCAB:
            file.write('{}\n'.format(word))

    with open('./remove_word.txt', 'w') as file:
        for word in REMOVED:
            file.write('{}\n'.format(word))

    #Baseline
    vocab= VOCAB
    accuray, b_time= report_model_and_result(X_f_test, y_f_test, X, y, vocab, 'model-2018.txt', 'baseline-result.txt')
    print('Hi from Baseline:')
    print('accuracy= {}, time= {}(s)'.format(accuray, b_time))

    #Stop_word filtering
    stop_words= set()
    with open('../Stopwords.txt') as file:
        for line in file:
            line= line.strip();
            stop_words.add(line)

    vocab= VOCAB.difference(stop_words)
    accuray, stop_w_time= report_model_and_result(X_f_test, y_f_test, X, y, vocab, 'stopword-model.txt', 'stopword-result.txt')
    print('Hi from Stopwords:')
    print('accuracy= {}, time= {}(s)'.format(accuray, stop_w_time))

    #Word length filtering
    vocab= set()
    for word in VOCAB:
        if (len(word) > 2) and (len(word) < 9):
            vocab.add(word)

    accuray, w_len_time= report_model_and_result(X_f_test, y_f_test, X, y, vocab, 'wordlength-model.txt', 'wordlength-result.txt')
    print('Hi from Word length filtering:')
    print('accuracy= {}, time= {}(s)'.format(accuray, w_len_time))

    #Infrequent word filtering

    #Different Pseudocount
    # global PSEUDOCOUNT # TODO

    pcount_accuracy= []
    for PSEUDOCOUNT in np.linspace(0.1, 1, 10):
        accuray, p_time= report_model_and_result(X_f_test, y_f_test, X, y, vocab, 'smoothing{}-model.txt'.format(PSEUDOCOUNT), 'smoothing{}-result.txt'.format(PSEUDOCOUNT))
        pcount_accuracy.append(accuray)

    #-------------------
    fig= plt.figure('')

    main_ax= fig.add_subplot(1, 1, 1, facecolor= 'white', title= 'Effect of pseudocount')

    main_ax.set_xlim(0, 1)
    # main_ax.set_ylim(0, 100)
    main_ax.set_xlabel('Pseudocount')
    main_ax.set_ylabel('Accuracy')

    main_ax.plot(np.linspace(0.1, 1, 10), pcount_accuracy, color= 'red', linewidth= 3, alpha= 1)

    # main_ax.legend()
    plt.show()
    #-------------------

    print('Done', '{}(s)'.format( time.time()-t_start ) )
