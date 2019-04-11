

from attentionlevel import Attention
from layernormalize import LayerNormalize
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


def pos_matrix(maxlen: int, d_emb: int) -> np.array:
    pos_enc = np.array(
        [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in
         range(maxlen)], dtype=np.float32)
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
    return pos_enc
class Transformer:
    def creat_model(wordindex,matrix0,maxlen0,X_train, X_test, y_train, y_test,d_model):
        embedding_layer0 = Embedding(len(wordindex)+ 1, d_model, weights=[matrix0], input_length=maxlen0)
        main_input0 = Input(shape=(maxlen0,))
        embed = embedding_layer0(main_input0)
        pos = keras.layers.Embedding(len(wordindex)+1, d_model, trainable=False, input_length=maxlen0,
                                     name='PositionEmbedding',weights=[pos_matrix(maxlen=len(wordindex) +1, d_emb=d_model)])(main_input0)
        added = keras.layers.Add()([embed, pos])

        #第一层Transformer
        added_0 = keras.layers.concatenate([added, added, added], axis=-1)
        att_layer1 = Attention(h=1, output_dim=K.int_shape(added)[-1])(added)
        att_layer2 = Attention(h=1, output_dim=K.int_shape(added)[-1])(added)
        att_layer3 = Attention(h=1, output_dim=K.int_shape(added)[-1])(added)
        merge_1 = keras.layers.concatenate([att_layer1, att_layer2, att_layer3], axis=-1)
        merge_1 = Dropout(0.5)(merge_1)
        added_2 = keras.layers.Add()([added_0, merge_1])
        normal = LayerNormalize(output_dim=K.int_shape(added_2)[-1])(added_2)
        BP1 = Dense(units=K.int_shape(normal)[-1], activation='relu')(normal)
        added_3 = keras.layers.Add()([normal, BP1])
        normal_1 = LayerNormalize(output_dim=K.int_shape(added_3)[-1])(added_3)

        # 第二层Transformer
        added_0 = keras.layers.concatenate([normal_1, normal_1, normal_1], axis=-1)
        att_layer1 = Attention(h=1, output_dim=K.int_shape(normal_1)[-1])(normal_1)
        att_layer2 = Attention(h=1, output_dim=K.int_shape(normal_1)[-1])(normal_1)
        att_layer3 = Attention(h=1, output_dim=K.int_shape(normal_1)[-1])(normal_1)
        merge_1 = keras.layers.concatenate([att_layer1, att_layer2, att_layer3], axis=-1)
        merge_1 = Dropout(0.5)(merge_1)
        added_2 = keras.layers.Add()([added_0, merge_1])
        normal = LayerNormalize(output_dim=K.int_shape(added_2)[-1])(added_2)
        BP1 = Dense(units=K.int_shape(normal)[-1], activation='relu')(normal)
        added_3 = keras.layers.Add()([normal, BP1])
        normal_1 = LayerNormalize(output_dim=K.int_shape(added_3)[-1])(added_3)

        # 第三层Transformer
        added_0 = keras.layers.concatenate([normal_1, normal_1, normal_1], axis=-1)
        att_layer1 = Attention(h=1, output_dim=K.int_shape(normal_1)[-1])(normal_1)
        att_layer2 = Attention(h=1, output_dim=K.int_shape(normal_1)[-1])(normal_1)
        att_layer3 = Attention(h=1, output_dim=K.int_shape(normal_1)[-1])(normal_1)
        merge_1 = keras.layers.concatenate([att_layer1, att_layer2, att_layer3], axis=-1)
        merge_1 = Dropout(0.5)(merge_1)
        added_2 = keras.layers.Add()([added_0, merge_1])
        normal = LayerNormalize(output_dim=K.int_shape(added_2)[-1])(added_2)
        BP1 = Dense(units=K.int_shape(normal)[-1], activation='relu')(normal)
        added_3 = keras.layers.Add()([normal, BP1])
        normal_1 = LayerNormalize(output_dim=K.int_shape(added_3)[-1])(added_3)
        flatten_layer2 = Flatten()(normal_1)
        main_output = Dense(7, activation='softmax')(flatten_layer2)
        model = Model(inputs=main_input0, outputs=main_output)
        optimizer = optimizers.Adam(lr=0.001)
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

if __name__ == "__main__":
    model=Transformer()
    # model.creat_model()