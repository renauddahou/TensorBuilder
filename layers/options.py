"""
THIS FILE DICTATES SOME OF THE OPTIONS THE LAYERS TENSORFLOW ACCEPTS,
AND THEIR CORRESPONDING DEFAULT VALUES IF APPLICABLE, AS WELL AS THE INPUT TYPE
"""

import tensorflow as tf

from abc import ABC, abstractmethod
from typing import *
from tensorflow.keras.layers import *


class AbstractOptions(ABC):
    """Abstract Class that all Layer Options must implement"""

    ACTIVATIONS = (
        'elu', 'exponential', 'gelu', 'get', 'hard_sigmoid', 'linear', 'relu',
        'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh'
    )

    INITIALIZERS = (
        'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'identity',
        'lecun_normal', 'lecun_uniform', 'ones', 'orthogonal', 'random_normal',
        'random_uniform', 'truncated_normal', 'variance_scaling', 'zeros'
    )

    REGULARIZERS = (
        'l1', 'l2', 'l1_l2', 'orthogonal_regularizer'
    )

    CONSTRAINTS = (
        'max_norm', 'min_max_norm', 'non_neg', 'radial_constraint', 'unit_norm'
    )

    TYPES = (
        'bfloat16', 'bool', 'complex128', 'complex64', 'double', 'float16', 'float32',
        'float64', 'half', 'int16', 'int32', 'int64', 'int8', 'qint16', 'qint32', 'qint8',
        'quint16', 'quint8', 'resource', 'string', 'uint16', 'uint32', 'uint64', 'uint8', 'variant'
    )

    NEURON_RANGE = range(0, 10000)

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.parameters = {}
        self.filled_parameters = {k: -1 for k in self.parameters.keys()}

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abstractmethod
    def dispatch_to_layer(self):
        """Returns the complete layer to the caller"""

        if len(list(filter(lambda x: x[1] == -1, self.filled_parameters.items()))) > 0:
            raise AssertionError('Cannot dispatch to layer if there are any blanks')
        raise NotImplementedError

    def insert_parameter(self, param_name: str, alias: Optional[str], param_type: Any,
                         param_range: Sequence[Any], default_value: Any):
        """
        Inserts a new parameter into the parameter table

        Parameters
        ----------
        param_name:                     Symbolic name of the parameter for the Layer
        alias:                          Alias name of the parameter which will be displayed to the User
        param_type:                     Permitted type for parameter
        param_range:                    Permitted range of values for parameter
        default_value:                  Default Value
        """

        if param_name not in self.parameters:
            if not isinstance(default_value, param_type):
                raise TypeError('Default value is not in the permitted types for parameter')

            self.parameters[param_name] = {
                'alias': alias,
                'type': param_type,
                'range': param_range,
                'default': default_value
            }
            self.filled_parameters[param_name] = -1
        else:
            raise ValueError('Cannot add in a parameter that already exists')

    def fulfill_parameter(self, param_name: str, value: Any):
        """
        Adds the parameter to the table of filled parameter

        Parameters
        ----------
        param_name:                     Name of the parameter
        value:                          Parameter value
        """

        if param_name not in self.filled_parameters or param_name not in self.parameters:
            raise ValueError('Parameter name is invalid')
        else:
            param_spec = self.parameters.get(param_name)
            if value in param_spec['range'] and isinstance(value, param_spec['type']):
                self.filled_parameters[param_name] = value
            else:
                raise AssertionError('Parameter is of invalid type or value')

    def delete_parameter(self, param_name: str):
        """
        Removes a parameter from the parameter table

        Parameters
        ----------
        param_name:                     Name of the parameter
        """

        if param_name in self.parameters:
            del self.parameters[param_name]
        else:
            raise ValueError('Cannot delete a parameter that is not part of this set of parameters')


class InputOptions(AbstractOptions):
    """
    Options for Input Layer
    """

    def __init__(self, *args, **kwargs):
        self.parameters = {
            'shape': {
                'alias': 'Input Shape',
                'type': (tuple, type(None)),
                'range': None,
                'default': None
            },
            'batch_size': {
                'alias': 'Batch Size',
                'type': (int, ),
                'range': self.NEURON_RANGE,
                'default': None
            },
            'name': {
                'alias': 'Name of Model',
                'type': (str, ),
                'range': None,
                'default': None
            },
            'dtype': {
                'alias': 'Data Type',
                'type': (str, ),
                'range': self.TYPES,
                'default': None
            },
            'sparse': {
                'alias': 'Sparse Placeholder Creation',
                'type': (bool, ),
                'range': (True, False),
                'default': None
            },
            'tensor': {
                'alias': 'Existing Tensor to Wrap into Input Layer',
                'type': (tf.Tensor, ),
                'range': None,
                'default': None
            },
            'ragged': {
                'alias': 'Sparse Placeholder Creation',
                'type': (bool, ),
                'range': (True, False),
                'default': None
            },
            'type_spec': {
                'alias': 'Type Spec of Input',
                'type': (tf.TypeSpec, ),
                'range': None,
                'default': None
            }
        }
        self.filled_parameters = {k: v['default'] for k, v in self.parameters.items()}

    def __str__(self):
        return 'Input'

    def __repr__(self):
        return 'Input'

    def dispatch_to_layer(self) -> Input:
        if self.filled_parameters.get('sparse') != -1 and self.filled_parameters.get('ragged') != -1:
            raise ValueError('Both sparse and ragged cannot be filled')
        else:
            return Input(
                shape=self.filled_parameters.get('shape'),
                batch_size=self.filled_parameters.get('batch_size'),
                name=self.filled_parameters.get('name'),
                dtype=self.filled_parameters.get('dtype'),
                sparse=self.filled_parameters.get('sparse'),
                tensor=self.filled_parameters.get('tensor'),
                ragged=self.filled_parameters.get('ragged'),
                type_spec=self.filled_parameters.get('type_spec')
            )


class DenseOptions(AbstractOptions):
    def __init__(self):
        self.parameters = {
            'neurons': {
                'alias': 'Neurons',
                'type': int,
                'range': self.NEURON_RANGE,
                'default': 32
            },
            'activation': {
                'alias': 'Activation Function',
                'type': str,
                'range': self.ACTIVATIONS,
                'default': None
            },
            'kernel_initializer': {
                'alias': 'Kernel Initializer Function',
                'type': str,
                'range': self.INITIALIZERS,
                'default': 'glorot_uniform'
            },
            'bias_initializer': {
                'alias': 'Bias Initializer Function',
                'type': str,
                'range': self.INITIALIZERS,
                'default': 'zeros'
            },
            'kernel_regularizer': {
                'alias': 'Kernel Regularizer Function',
                'type': str,
                'range': self.REGULARIZERS,
                'default': None
            },
            'bias_regularizer': {
                'alias': 'Bias Regularizer Function',
                'type': str,
                'range': self.REGULARIZERS,
                'default': None
            },
            'activity_regularizer': {
                'alias': 'Activity Regularizer Function',
                'type': str,
                'range': self.REGULARIZERS,
                'default': None
            },
            'kernel_constraint': {
                'alias': 'Kernel Constraint Function',
                'type': str,
                'range': self.CONSTRAINTS,
                'default': None
            },
            'bias_constraint': {
                'alias': 'Bias Constraint Function',
                'type': str,
                'range': self.CONSTRAINTS,
                'default': None
            }
        }
        self.filled_parameters = {k: v['default'] for k, v in self.parameters.items()}

    def __str__(self):
        return 'Dense'

    def __repr__(self):
        return 'Dense'

    def dispatch_to_layer(self) -> Dense:
        return Dense(
            self.filled_parameters.get('neurons'),
            activation=self.filled_parameters.get('activation'),
            use_bias=self.filled_parameters.get('use_bias'),
            kernel_initializer=self.filled_parameters.get('kernel_initializer'),
            bias_initializer=self.filled_parameters.get('bias_initializer'),
            kernel_regularizer=self.filled_parameters.get('kernel_regularizer'),
            bias_regularizer=self.filled_parameters.get('bias_regularizer'),
            activity_regularizer=self.filled_parameters.get('activity_regularizer'),
            kernel_constraint=self.filled_parameters.get('kernel_constraint'),
            bias_constraint=self.filled_parameters.get('bias_constraint'),
        )


class BidirectionalOptions(AbstractOptions):
    def __init__(self, *args, **kwargs):
        self.parameters = {
            'layer': {
                'alias': 'Compatible Layer',
                'type': (tf.keras.layers.RNN, tf.keras.layers.Layer),
                'range': lambda: self.OPTIONS.keys(),
                'default': None
            },
            'merge_mode': {
                'alias': 'Bidirectional Merge Mode',
                'type': (str, ),
                'range': ('sum', 'mul', 'concat', 'ave', None),
                'default': 'concat'
            },
            'weights': {
                'alias': 'Initialising Weights',
                'type': (type(None), ),
                'range': None,
                'default': None
            },
            'backward_layer': {
                'alias': 'Backward Layer',
                'type': (tf.keras.layers.RNN, tf.keras.layers.Layer, type(None)),
                'range': lambda: self.OPTIONS.keys(),
                'default': None
            }
        }
        self.filled_parameters = {k: v['default'] for k, v in self.parameters.items()}

    def __str__(self):
        return 'Bidirectional'

    def __repr__(self):
        return 'Bidirectional'

    def dispatch_to_layer(self):
        return Bidirectional(
            self.filled_parameters.get('layer'),
            merge_mode=self.filled_parameters.get('merge_mode'),
            weights=self.filled_parameters.get('weights'),
            backward_layer=self.filled_parameters.get('backward_layer')
        )

class Conv1DOptions(AbstractOptions):
    def __init__(self, *args, **kwargs):
        self.parameters = {
            'filters': {
                'alias': 'Dimensionality of output space',
                'type': int,
                'range': self.NEURON_RANGE,
                'default': None
            },
            'kernel_size': {
                'alias': 'Length of the 1D Convolution Window',
                'type': (int, tuple, list),
                'range': self.NEURON_RANGE,
                'default': None
            },
            'strides': {
                'alias': 'Stride Length of Convolution',
                'type': (int, tuple, list),
                'range': self.NEURON_RANGE,
                'default': 1
            },
            'padding': {
                'alias': 'How to pad inputs',
                'type': str,
                'range': ('valid', 'same', 'causal'),
                'default': 'valid'
            },
            'data_format': {
                'alias': 'Specify how to process date format',
                'type': str,
                'range': ('channels_last', 'channels_first'),
                'default': 'channels_last'
            },
            'dilation_rate': {
                'alias': 'Dilation Rate for Dilated Convolution',
                'type': (int, tuple, list),
                'range': self.NEURON_RANGE,
                'default': 1
            },
            'groups': {
                'alias': 'Number of groups in which the input is split along the channel axis',
                'type': (int, ),
                'range': self.NEURON_RANGE,
                'default': 1
            },
            'activation': {
                'alias': 'Activation Function',
                'type': str,
                'range': self.ACTIVATIONS,
                'default': None
            },
            'use_bias': {
                'alias': 'Use Biases for Layer',
                'type': bool,
                'range': (True, False),
                'default': True
            },
            'kernel_initializer': {
                'alias': 'Kernel Initializer Function',
                'type': str,
                'range': self.INITIALIZERS,
                'default': 'glorot_uniform'
            },
            'bias_initializer': {
                'alias': 'Bias Initializer Function',
                'type': str,
                'range': self.INITIALIZERS,
                'default': 'zeros'
            },
            'kernel_regularizer': {
                'alias': 'Kernel Regularizer Function',
                'type': str,
                'range': self.REGULARIZERS,
                'default': None
            },
            'bias_regularizer': {
                'alias': 'Bias Regularizer Function',
                'type': str,
                'range': self.REGULARIZERS,
                'default': None
            },
            'activity_regularizer': {
                'alias': 'Activity Regularizer Function',
                'type': str,
                'range': self.REGULARIZERS,
                'default': None
            },
            'kernel_constraint': {
                'alias': 'Kernel Constraint Function',
                'type': str,
                'range': self.CONSTRAINTS,
                'default': None
            },
            'bias_constraint': {
                'alias': 'Bias Constraint Function',
                'type': str,
                'range': self.CONSTRAINTS,
                'default': None
            }
        }
        self.filled_parameters = {k: v['default'] for k, v in self.parameters.items()}

    def __str__(self):
        return 'Conv1D'

    def __repr__(self):
        return 'Conv1D'

    def dispatch_to_layer(self) -> Conv1D:
        if self.filled_parameters.get('dilation_rate') != 1 and self.filled_parameters.get('strides') != 1:
            raise ValueError('Both Dilation Rate and Strides cannot be != 1')
        else:
            return Conv1D(
                self.filled_parameters.get('filters'),
                self.filled_parameters.get('kernel_size'),
                strides=self.filled_parameters.get('strides'),
                padding=self.filled_parameters.get('padding'),
                data_format=self.filled_parameters.get('data_format'),
                dilation_rate=self.filled_parameters.get('dilation_rate'),
                groups=self.filled_parameters.get('groups'),
                activation=self.filled_parameters.get('activation'),
                use_bias=self.filled_parameters.get('use_bias'),
                kernel_initializer=self.filled_parameters.get('kernel_initializer'),
                bias_initializer=self.filled_parameters.get('bias_initializer'),
                kernel_regularizer=self.filled_parameters.get('kernel_regularizer'),
                bias_regularizer=self.filled_parameters.get('bias_regularizer'),
                activity_regularizer=self.filled_parameters.get('activity_regularizer'),
                kernel_constraint=self.filled_parameters.get('kernel_constraint'),
                bias_constraint=self.filled_parameters.get('bias_constraint')
            )


class Conv2DOptions(AbstractOptions):
    def __init__(self, *args, **kwargs):
        self.parameters = {
            'filters': {
                'alias': 'Dimensionality of output space',
                'type': int,
                'range': self.NEURON_RANGE,
                'default': None
            },
            'kernel_size': {
                'alias': 'Length of the 1D Convolution Window',
                'type': (int, tuple, list),
                'range': self.NEURON_RANGE,
                'default': None
            },
            'strides': {
                'alias': 'Stride Length of Convolution',
                'type': (tuple, list),
                'range': self.NEURON_RANGE,
                'default': (1, 1)
            },
            'padding': {
                'alias': 'How to pad inputs',
                'type': str,
                'range': ('valid', 'same', 'causal'),
                'default': 'valid'
            },
            'data_format': {
                'alias': 'Specify how to process date format',
                'type': str,
                'range': ('channels_last', 'channels_first'),
                'default': None
            },
            'dilation_rate': {
                'alias': 'Dilation Rate for Dilated Convolution',
                'type': (int, tuple, list),
                'range': self.NEURON_RANGE,
                'default': (1, 1)
            },
            'groups': {
                'alias': 'Number of groups in which the input is split along the channel axis',
                'type': (int, ),
                'range': self.NEURON_RANGE,
                'default': (1, 1)
            },
            'activation': {
                'alias': 'Activation Function',
                'type': str,
                'range': self.ACTIVATIONS,
                'default': None
            },
            'use_bias': {
                'alias': 'Use Biases for Layer',
                'type': bool,
                'range': (True, False),
                'default': True
            },
            'kernel_initializer': {
                'alias': 'Kernel Initializer Function',
                'type': str,
                'range': self.INITIALIZERS,
                'default': 'glorot_uniform'
            },
            'bias_initializer': {
                'alias': 'Bias Initializer Function',
                'type': str,
                'range': self.INITIALIZERS,
                'default': 'zeros'
            },
            'kernel_regularizer': {
                'alias': 'Kernel Regularizer Function',
                'type': str,
                'range': self.REGULARIZERS,
                'default': None
            },
            'bias_regularizer': {
                'alias': 'Bias Regularizer Function',
                'type': str,
                'range': self.REGULARIZERS,
                'default': None
            },
            'activity_regularizer': {
                'alias': 'Activity Regularizer Function',
                'type': str,
                'range': self.REGULARIZERS,
                'default': None
            },
            'kernel_constraint': {
                'alias': 'Kernel Constraint Function',
                'type': str,
                'range': self.CONSTRAINTS,
                'default': None
            },
            'bias_constraint': {
                'alias': 'Bias Constraint Function',
                'type': str,
                'range': self.CONSTRAINTS,
                'default': None
            }
        }
        self.filled_parameters = {k: v['default'] for k, v in self.parameters.items()}

    def __str__(self):
        return 'Conv2D'

    def __repr__(self):
        return 'Conv2D'

    def dispatch_to_layer(self) -> Conv2D:
        if self.filled_parameters.get('dilation_rate') != 1 and self.filled_parameters.get('strides') != 1:
            raise ValueError('Both Dilation Rate and Strides cannot be != 1')
        else:
            return Conv2D(
                self.filled_parameters.get('filters'),
                self.filled_parameters.get('kernel_size'),
                strides=self.filled_parameters.get('strides'),
                padding=self.filled_parameters.get('padding'),
                data_format=self.filled_parameters.get('data_format'),
                dilation_rate=self.filled_parameters.get('dilation_rate'),
                groups=self.filled_parameters.get('groups'),
                activation=self.filled_parameters.get('activation'),
                use_bias=self.filled_parameters.get('use_bias'),
                kernel_initializer=self.filled_parameters.get('kernel_initializer'),
                bias_initializer=self.filled_parameters.get('bias_initializer'),
                kernel_regularizer=self.filled_parameters.get('kernel_regularizer'),
                bias_regularizer=self.filled_parameters.get('bias_regularizer'),
                activity_regularizer=self.filled_parameters.get('activity_regularizer'),
                kernel_constraint=self.filled_parameters.get('kernel_constraint'),
                bias_constraint=self.filled_parameters.get('bias_constraint')
            )


class EmbeddingOptions(AbstractOptions):
    def __init__(self, *args, **kwargs):
        self.parameters = {
            'input_dim': {
                'alias': 'Size of Vocabulary',
                'type': int,
                'range': self.NEURON_RANGE,
                'default': None
            },
            'output_dim': {
                'alias': 'Dimension of Dense Embedding',
                'type': int,
                'range': self.NEURON_RANGE,
                'default': None
            },
            'embeddings_initializer': {
                'alias': 'Embedding Layer Initializer',
                'type': str,
                'range': self.INITIALIZERS,
                'default': 'zeros'
            },
            'embeddings_regularizer': {
                'alias': 'Embedding Layer Regularizer',
                'type': str,
                'range': self.REGULARIZERS,
                'default': None
            },
            'embeddings_constraint': {
                'alias': 'Embedding Layer Constraint',
                'type': str,
                'range': self.CONSTRAINTS,
                'default': None
            },
            'mask_zero': {
                'alias': 'Inputs of value 0 is masked',
                'type': bool,
                'range': (True, False),
                'default': False
            },
            'input_length': {
                'alias': 'Length of Input Sequences',
                'type': int,
                'range': self.NEURON_RANGE,
                'default': None
            },
        }
        self.filled_parameters = {k: v['default'] for k, v in self.parameters.items()}

    def __str__(self):
        return 'Embedding'

    def __repr__(self):
        return 'Embedding'

    def dispatch_to_layer(self):
        return Embedding(
            self.filled_parameters.get('input_dim'),
            self.filled_parameters.get('output_dim'),
            embeddings_initializer=self.filled_parameters.get('embeddings_initializer'),
            embeddings_regularizer=self.filled_parameters.get('embeddings_regularizer'),
            activity_regularizer=self.filled_parameters.get('activity_regularizer'),
            embeddings_constraint=self.filled_parameters.get('embeddings_constraint'),
            mask_zero=self.filled_parameters.get('mask_zero'),
            input_length=self.filled_parameters.get('input_length')
        )


NAME_TO_CLASS = {
    'Input': InputOptions,
    'Dense': DenseOptions,
    'Bidirectional': BidirectionalOptions,
    'Conv1D': Conv1DOptions,
    'Conv2D': Conv2DOptions,
    'Embedding': EmbeddingOptions,
}

CLASS_TO_NAME = {
    v: k for k, v in NAME_TO_CLASS.items()
}
