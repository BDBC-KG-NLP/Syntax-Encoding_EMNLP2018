"""Embedding layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.legacy import interfaces
import numpy as np
import tensorflow as tf

__author__ = 'Zhiyaun Hu'
__email__  = "zhiyuan.hu.bj@gmail.com"
__time__   = "19/4 2018"


class Syntax_Embedding(Layer):
    """Turns positive integers (indexes) into dense vectors of fixed size.

    eg.
    [
      [[1,2,3,0], [1,3,0,0], [1,4,3,0]],   
      [[1,3,2,0], [1,2,0,0], [0,0,0,0]],
      [[1,2,3,4], [1,2,4,3], [0,0,0,0]],
      [[1,2,3,4], [1,2,4,3], [0,0,0,0]]
    ]
    (batch_size, sequence_length,path_of_word_in_syntax)
    
    --->>> 
    [
      [[0.5,0.1,0.2,0.4], [0.3,0.3,0.2,0.1], [0.1,0.4,0.3,0.0]],   
      [[0.1,0.3,0.2,0.0], [0.1,0.2,0.0,0.3], [0.2,0.1,0.4,0.5]],
      [[0.1,0.2,0.3,0.4], [0.1,0.2,0.4,0.3], [0.6,0.7,0.8,0.9]],
      [[0.6,0.2,0.3,0.7], [0.1,0.3,0.8,0.9], [0.1,0.3,0.6,0.9]]
    ]
    (batch_size, sequence_length,output_dim)
    

    This layer can only be used as the first layer in a model.
    # Example
    ```python
      model = Sequential()
      model.add(Embedding(1000, 64, input_length=10))
      # the model will take as input an integer matrix of size (batch, input_length).
      # the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
      # now model.output_shape == (None, 10, 64), where None is the batch dimension.
      input_array = np.random.randint(1000, size=(32, 10))
      model.compile('rmsprop', 'mse')
      output_array = model.predict(input_array)
      assert output_array.shape == (32, 10, 64)
    ```

    # Arguments
      
      input_dim: int > 0. Size of the vocabulary(how many labels in syntax tree),
          i.e. maximum integer index + 1.
      
      output_dim: int >= 0. Dimension of the dense embedding.
      
      syntax_tree_label_dim: int > 0. Dimension of each lable in syntax tree
      especially, the syntax_tree_label_dim should be equal with the output_dim
      because the there is a action for multiply

      depth_of_syntax_tree: int > 0.  The maximum of path in syntax tree

      embeddings_initializer: Initializer for the `embeddings` matrix
          (see [initializers](../initializers.md)).
      
      embeddings_regularizer: Regularizer function applied to
          the `embeddings` matrix
          (see [regularizer](../regularizers.md)).
      
      embeddings_constraint: Constraint function applied to
          the `embeddings` matrix
          (see [constraints](../constraints.md)).
      
      mask_zero: Whether or not the input value 0 is a special "padding"
          value that should be masked out.
          This is useful when using [recurrent layers](recurrent.md)
          which may take variable length input.
          If this is `True` then all subsequent layers
          in the model need to support masking or an exception will be raised.
          If mask_zero is set to True, as a consequence, index 0 cannot be
          used in the vocabulary (input_dim should equal size of
          vocabulary + 1).
      
      input_length: Length of input sequences, when it is constant.
          This argument is required if you are going to connect
          `Flatten` then `Dense` layers upstream
          (without it, the shape of the dense outputs cannot be computed).
   
    # Input shape
        3D tensor with shape: `(batch_size, sequence_length,path_of_word_in_syntax)`.
    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """



    @interfaces.legacy_embedding_support
    def __init__(self, input_dim, syntax_tree_label_dim,depth_of_syntax_tree,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        output_dim = syntax_tree_label_dim
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,depth_of_syntax_tree)
            else:
                kwargs['input_shape'] = (None,)
        
        super(Syntax_Embedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.syntax_tree_label_dim = syntax_tree_label_dim
        self.depth_of_syntax_tree = depth_of_syntax_tree
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.input_length = input_length

    def build(self, input_shape):
        self.embeddings = self.add_weight(
                                      shape=(self.input_dim, self.syntax_tree_label_dim),
                                      initializer=self.embeddings_initializer,
                                      name='embeddings',
                                      regularizer=self.embeddings_regularizer,
                                      constraint=self.embeddings_constraint,
                                      dtype=self.dtype)
        self.elemt_wise = self.add_weight(
                                      shape=(self.depth_of_syntax_tree, self.output_dim),
                                      initializer=self.embeddings_initializer,
                                      name='elemt_wise',
                                      regularizer=self.embeddings_regularizer,
                                      constraint=self.embeddings_constraint,
                                      dtype=self.dtype)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(inputs, 0)

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        zero = tf.zeros((1,self.syntax_tree_label_dim))

        embeddings = tf.concat([zero, self.embeddings], axis=0)

        parser_label_vec = tf.nn.embedding_lookup(embeddings, inputs)

        element_wise_multiply = tf.multiply(parser_label_vec,self.elemt_wise)
        out = tf.reduce_sum(element_wise_multiply,2)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.input_length,self.output_dim)