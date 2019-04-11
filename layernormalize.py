from keras import backend as K
from keras.engine.topology import Layer
class LayerNormalize(Layer):

    def __init__(self,output_dim,**kwargs):
        self.output_dim=output_dim
        super(LayerNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(name='a',
                                  shape=(input_shape[1],self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
        self.b = self.add_weight(name='b',
                                  shape=(input_shape[1],self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
        super(LayerNormalize, self).build(input_shape)
    def call(self, x):
        noise=1e-5
        x_mean=K.mean(x, axis=-1, keepdims=True)
        var=K.var(x,axis=-1,keepdims=True)
        normalized= (x - x_mean) / K.sqrt(var + noise)
        normalized=self.a*normalized+self.b
        return normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)