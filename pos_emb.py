from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer
import tensorflow as tf

class pos_emb(Layer):

    def __init__(self,d,output_dim,**kwargs):
        self.d=d
        self.output_dim=output_dim
        super(pos_emb, self).__init__(**kwargs)

    def build(self, input_shape):
        super(pos_emb, self).build(input_shape)
    def call(self, x):
        # u=x[-1][-1]
        var =[]
        var_2k=[]
        v1 = K.permute_dimensions(x, pattern=(2, 1, 0))
        i=0
        for i in range(int(K.int_shape(x)[-1]/2)):
            var.append(2*(i+1)-1)
            var_2k.append(2*(i+1)-2)
            i=i+1
        print(len(var_2k))
        print(len(var))
        pos_2k=K.gather(v1,var)
        pos_2k_1=K.gather(v1,var_2k)
        pos_emb_1=K.sin(pos_2k/(10000**(2*i/self.d)))
        pos_emb_2= K.cos(pos_2k_1 / (10000 ** (2 * i / self.d)))
        pos_emb=K.concatenate([pos_emb_2,pos_emb_1], axis=0)
        pos_emb=K.permute_dimensions(pos_emb, pattern=(2, 1, 0))
        print("1",K.int_shape(pos_emb))
        return pos_emb

        return pos_emb

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)