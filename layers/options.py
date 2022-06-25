"""
THIS FILE DICTATES SOME OF THE OPTIONS THE LAYERS TENSORFLOW ACCEPTS,
AND THEIR CORRESPONDING DEFAULT VALUES IF APPLICABLE, AS WELL AS THE INPUT TYPE
"""

import tensorflow as tf
from typing import *


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


# ALL SUPPORTED LAYER OPTIONS
OPTIONS = {
    # Input Layer Parameters
    'Input': {
        'shape': {
            'alias': 'Input Shape',
            'type': Tuple[Union[None, int]],
            'range': None,
            'default': None
        },
        'batch_size': {
            'alias': 'Batch Size',
            'type': int,
            'range': NEURON_RANGE,
            'default': None
        },
        'name': {
            'alias': 'Name of Model',
            'type': str,
            'range': None,
            'default': None
        },
        'dtype': {
            'alias': 'Data Type',
            'type': str,
            'range': TYPES,
            'default': None
        },
        'sparse': {
            'alias': 'Sparse Placeholder Creation',
            'type': bool,
            'range': (True, False),
            'default': None
        },
        'tensor': {
            'alias': 'Existing Tensor to Wrap into Input Layer',
            'type': tf.Tensor,
            'range': None,
            'default': None
        },
        'ragged': {
            'alias': 'Sparse Placeholder Creation',
            'type': bool,
            'range': (True, False),
            'default': None
        },
        'type_spec': {
            'alias': 'Type Spec of Input',
            'type': tf.TypeSpec,
            'range': None,
            'default': None
        },
    },

    # Dense Layer Input Parameters
    'Dense': {
        'neurons': {
            'alias': 'Neurons',
            'type': int,
            'range': NEURON_RANGE,
            'default': 32
        },
        'activation': {
            'alias': 'Activation Function',
            'type': str,
            'range': ACTIVATIONS,
            'default': None
        },
        'kernel_initializer': {
            'alias': 'Kernel Initializer Function',
            'type': str,
            'range': INITIALIZERS,
            'default': 'glorot_uniform'
        },
        'bias_initializer': {
            'alias': 'Bias Initializer Function',
            'type': str,
            'range': INITIALIZERS,
            'default': 'zeros'
        },
        'kernel_regularizer': {
            'alias': 'Kernel Regularizer Function',
            'type': str,
            'range': REGULARIZERS,
            'default': None
        },
        'bias_regularizer': {
            'alias': 'Bias Regularizer Function',
            'type': str,
            'range': REGULARIZERS,
            'default': None
        },
        'activity_regularizer': {
            'alias': 'Activity Regularizer Function',
            'type': str,
            'range': REGULARIZERS,
            'default': None
        },
        'kernel_constraint': {
            'alias': 'Kernel Constraint Function',
            'type': str,
            'range': CONSTRAINTS,
            'default': None
        },
        'bias_constraint': {
            'alias': 'Bias Constraint Function',
            'type': str,
            'range': CONSTRAINTS,
            'default': None
        }
    },


}
