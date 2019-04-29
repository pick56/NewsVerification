from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import activations,RepeatVector
from keras import initializers
import tensorflow as tf

class AttentionLayer(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
    def __init__(self,init='glorot_uniform',attn_activation='tanh',weights=None,**kwargs):
        self.init=initializers.get(init)
        self.attn_activation=activations.get(attn_activation)
        if weights:
            self.initial_weights=weights
        super(AttentionLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        w_dim = input_shape[-1]
        self.W_s = self.add_weight(shape=(w_dim,w_dim),
                                   initializer =self.init,
                                   name='{}_Ws'.format(self.name),
                                   trainable=True)
        self.B_s = self.add_weight((w_dim,),
                                   initializer='zero',
                                   name='{}_bs'.format(self.name))
        self.Attention_vec = self.add_weight((w_dim,),
                                             initializer='normal',
                                             name='{}_att_vec'.format(self.name))
        super(AttentionLayer,self).build(input_shape)

    def call(self, x, mask=None):
        #transform (None,length,word_dim)*(word_dim,w_dim) --> (None,length,w_dim)
        u = self.attn_activation(K.dot(x,self.W_s) + self.B_s)

        #transform (None,length,w_dim)*(w_dim,) --> (None,length,)-->(None,length)
        e = K.exp(K.sum(u*self.Attention_vec,axis=-1))
        #weight
        a = e / tf.expand_dims(K.sum(e,axis=1),-1)
        return tf.expand_dims(a,-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],1)

class AttentionLayerTF(Layer):
    ''' Attention Layer over LSTM

    '''
    def __init__(self, init='glorot_uniform', attn_activation='tanh', weights=None, **kwargs):
        self.init = initializers.get(init)
        self.attn_activation = activations.get(attn_activation)
        #self.get_weights = get_weights
        if weights:
            self.initial_weights = weights
        super(AttentionLayerTF, self).__init__(**kwargs)

    def build(self, input_shape): # (batch, steps, dim)
        w_dim = input_shape[-1]
        self.W_s = self.add_weight(shape=(w_dim, w_dim),
                initializer=self.init,
                name='{}_Ws'.format(self.name),
                trainable=True)
        self.B_s = self.add_weight((w_dim,),
                initializer='zero',
                name='{}_bs'.format(self.name))
        self.Attention_vec = self.add_weight((w_dim,),
                initializer='normal',
               name='{}_att_vec'.format(self.name))
        super(AttentionLayerTF, self).build(input_shape)

    def call(self, x, mask=None):
        # 1. transform, (None, steps, idim)*(idim, wdim) -> (None, steps, wdim), (to hidden dim)
        u = self.attn_activation(K.dot(x, self.W_s) + self.B_s)
        # 2. {(None, steps, outdim) *(outdim), axis = 2} -> (None, steps)
        e = K.exp(K.sum(u*self.Attention_vec, axis=-1))
        # 3. weights:  (None, steps)
        # (None, steps) / (None,)
        a = e / tf.expand_dims(K.sum(e, axis = 1), -1)
        #if self.get_weights:
        #    return a
        #a = e/K.sum(e, axis=1).dimshuffle(0,'x')
        # 4. (None, steps, dim) * (None, steps, 1) -> (None, steps, dim)
        #print x.get_shape(), tf.expand_dims(a, -1).get_shape()
        weighted_input = x * tf.expand_dims(a, -1)
        #weighted_input = x*a.dimshuffle(0,1,'x')
        return K.sum(weighted_input, axis=1)
        #return a#x#tf.expand_dims(a, -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])