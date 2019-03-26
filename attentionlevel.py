from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer
class Attention(Layer):

    def __init__(self,h,output_dim,**kwargs):
        self.output_dim=output_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[-1],self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[-1],self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[-1], self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
    def call(self, x):
        Q_seq, K_seq, V_seq = x,x,x
        Q_seq = K.dot(Q_seq, self.WQ)
        print(K.int_shape(x))
        print(K.int_shape(self.WQ))
        K_seq = K.dot(K_seq, self.WK)
        V_seq = K.dot(V_seq, self.WV)
        A = K.batch_dot( Q_seq,K_seq,axes=[2, 2]) / K.int_shape(x)[-1]** 0.5
        A = K.softmax(A)
        O_seq = K.batch_dot(A, V_seq, axes=[2, 1])
        print(K.int_shape(O_seq))
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)