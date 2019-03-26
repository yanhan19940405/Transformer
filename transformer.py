

from attentionlevel import Attention
from pos_emb import pos_emb
from data_db import TextData

import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
from gensim.models import word2vec
import gensim
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from gensim.test.utils import datapath, get_tmpfile, common_texts
from gensim.corpora import LowCorpus
from gensim.corpora import Dictionary
from keras import optimizers
import re
from sklearn.metrics import classification_report
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers.merge import concatenate
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import f1_score
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
import pickle
from keras.utils import plot_model
import  tensorflow as tf
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
class Transformer:
    def creat_model(wordindex,wordindex1,matrix0,maxlen0,X_train, X_test, y_train, y_test):
        embedding_layer0 = Embedding(len(wordindex) + len(wordindex1) + 2, 256, weights=[matrix0], input_length=maxlen0)
        main_input0 = Input(shape=(maxlen0,))
        embed = embedding_layer0(main_input0)
        print("embed", K.int_shape(embed))
        pos=pos_emb(d=256,output_dim=K.int_shape(embed)[-1])(embed)
        added = keras.layers.Add()([embed, pos])
        added_0 = keras.layers.concatenate([added, added, added], axis=-1)
        att_layer1 = Attention(h=1,output_dim=K.int_shape(added)[-1])(added)
        att_layer2 = Attention(h=1, output_dim=K.int_shape(added)[-1])(added)
        att_layer3 = Attention(h=1, output_dim=K.int_shape(added)[-1])(added)
        merge_1=keras.layers.concatenate([att_layer1,att_layer2,att_layer3], axis=-1)
        added_2=keras.layers.Add()([added_0,merge_1])

        normal=keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                      beta_initializer='zeros', gamma_initializer='ones',
                                                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                                      gamma_constraint=None)(added_2)
        BP1=Dense(units=K.int_shape(normal)[-1],  activation='relu')(normal)
        added_3=keras.layers.Add()([normal,BP1])
        normal_1=keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                      beta_initializer='zeros', gamma_initializer='ones',
                                                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                                      gamma_constraint=None)(added_3)
        flatten_layer2=Flatten()(normal_1)
        main_output = Dense(3, activation='softmax')(flatten_layer2)
        model = Model(inputs=main_input0, outputs=main_output)
        optimizer = optimizers.Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file="./image/model.png", show_shapes=True)
        earlystopping = EarlyStopping(monitor='val_acc', min_delta=1e-2, patience=3, verbose=2, mode='auto')
        history=model.fit(X_train, y_train, verbose=1, batch_size=batch_size, epochs=n_epoch, validation_data=(X_test, y_test),
                  callbacks=[earlystopping])
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # 绘制训练 & 验证的损失值
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        filepath = "./sen_model_10.h5"
        model.save(filepath=filepath, include_optimizer=True)
        score, acc = model.evaluate(X_test, y_test, verbose=1, batch_size=batch_size)
        return model
    def handle_data(rpath):
        datadf = pd.read_excel(rpath)
        dd = datadf['情感修正'].value_counts()
        print(dd)
        ylist = list(datadf['情感修正'])
        titlelist=list(datadf['标题'])
        abstractlist=list(datadf['摘要'])

        return ylist,titlelist,abstractlist
if __name__ == "__main__":
    model=Transformer()
    # model.creat_model()