'''This module is used to programtically create custom archetectures, it creates input embedders
   and middle ware.
'''
from markov.architecture.constants import SchemeInfo, ActivationFunctions, EmbedderType
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.layers import Conv2d, Dense, BatchnormActivationDropout

def create_scheme(info_dict):
    '''Creates a custom scheme whose first layers are convolutional layers and
       last layers are dense layers.
       info_dict- dictionary containing the following entries:
       conv_info_list - List of list where the embedded list represent the
                        num of filter, kernel size, and stride. Embedded list
                        of size less than 3 will produce an exception. The size
                        of the non-imbedded list is interpreted and the desired number
                        convolutional layers.
       dense_layer__hidden_unit_list = List where the size represents the number of desired
                                       dense layers to be used after the convolution layer, the
                                       value of the list represents the number of hidden units
    '''
    try:
        scheme = list()
        # Add the convolutional layers first
        for conv_info in info_dict[SchemeInfo.CONV_INFO_LIST.value]:
            num_filters, kernel_size, strides = tuple(conv_info)
            scheme.append(Conv2d(num_filters, kernel_size, strides))

        for hindden_units in info_dict[SchemeInfo.DENSE_LAYER_INFO_LIST.value]:
            scheme.append(Dense(hindden_units))

        return scheme
    except KeyError as err:
        raise Exception("Info, key {} not found".format(err.args[0]))
    except ValueError as err:
        raise Exception("Error while unpacking info: {}".format(err))
    except Exception as err:
        raise Exception("Error while creating scheme: {}".format(err))

def create_batchnorm_scheme(info_dict):
    '''Creates a scheme where every other layer is a batchnorm layer, convolutional layers
       are first then dense layers.
        info_dict- dictionary containing the following entries:
        conv_info_list - List of list where the embedded list represent the
                         num of filter, kernel size, and stride. Embedded list
                         of size less than 3 will produce an exception. The size
                         of the non-imbedded list is interpreted and the desired number
                         convolutional layers.
        dense_layer__hidden_unit_list = List where the size represents the number of desired
                                        dense layers to be used after the convolution layer, the
                                        value of the list represents the number of hidden units
        bn_info_conv - List containing bool whether or not to use batchnorm for the convolutional
                       part of the archetecture, string for desired activation function, and dropout
                       rate, list with less than 3 items will cause an excpetion.
        bn_info_dense - List containing bool whether or not to use batchnorm for the dense
                        part of the archetecture, string for desired activation function,
                        and dropout rate, list with less than 3 items will cause an excpetion.
        is_first_layer_bn - True if the first layer of the scheme should be a batchnorm
                            layer.
    '''
    try:
        batchnorm, activation_function, dropout_rate = \
            tuple(info_dict[SchemeInfo.BN_INFO_CONV.value])

        if not ActivationFunctions.has_activation_function(activation_function):
            raise Exception("Invalid activation function for batchnorm scheme")

        scheme = list()

        if info_dict[SchemeInfo.IS_FIRST_LAYER_BN.value]:
            scheme.append(BatchnormActivationDropout(batchnorm=batchnorm,
                                                     activation_function=activation_function,
                                                     dropout_rate=dropout_rate))
        # Add the convolutional layers first
        for conv_info in info_dict[SchemeInfo.CONV_INFO_LIST.value]:
            # Add the convolutional filters followed by batchnorms
            num_filters, kernel_size, strides = tuple(conv_info)
            scheme.append(Conv2d(num_filters, kernel_size, strides))
            scheme.append(BatchnormActivationDropout(batchnorm=batchnorm,
                                                     activation_function=activation_function,
                                                     dropout_rate=dropout_rate))

        batchnorm, activation_function, dropout_rate = \
            tuple(info_dict[SchemeInfo.BN_INFO_DENSE.value])

        if not ActivationFunctions.has_activation_function(activation_function):
            raise Exception("Invalid activation function for batchnorm scheme")

        for hindden_units in info_dict[SchemeInfo.DENSE_LAYER_INFO_LIST.value]:
            scheme.append(Dense(hindden_units))
            scheme.append(BatchnormActivationDropout(batchnorm=batchnorm,
                                                     activation_function=activation_function,
                                                     dropout_rate=dropout_rate))
        return scheme
    except KeyError as err:
        raise Exception("Info, key {} not found".format(err.args[0]))
    except ValueError as err:
        raise Exception("Error while unpacking info: {}".format(err))
    except Exception as err:
        raise Exception("Error while creating scheme: {}".format(err))

# Dictionary of available scheme's
SCHEME_TYPE = {EmbedderType.SCHEME.value: create_scheme,
               EmbedderType.BN_SCHEME.value: create_batchnorm_scheme}

def create_input_embedder(scheme_dict, embedder_type, activation_function):
    '''Creates an rl coach input embedder
       scheme_dict - Dictionary where the key is the observation and the value is
                     a dictionary containing all the information required by
                     the scheme creation methods defined above.
       embedder_type - String indicating desired embedder type, available
                        types are defined in SCHEME_TYPE
       activation_function - Desired activationfunction for the embdedder
    '''
    try:
        if not ActivationFunctions.has_activation_function(activation_function):
            raise Exception("Invalid activation function for input embedder")

        embedder_types_parameters = dict()

        for observation, info in scheme_dict.items():
            scheme = SCHEME_TYPE[embedder_type](info)
            embedder_types_parameters[observation] = \
                InputEmbedderParameters(scheme=scheme, activation_function=activation_function)

        return embedder_types_parameters
    except KeyError as err:
        raise Exception("Input embedder, key {} not found".format(err.args[0]))
    except Exception as err:
        raise Exception("Error while creating input emmbedder: {}".format(err))

def create_middle_embedder(scheme_dict, embedder_type, activation_function):
    '''Creates rl coach middleware
       scheme_dict - Dictionary containing all the information required by
                     the scheme creation methods defined above.
       embedder_type - String indicating desired inputembedder type, available
                        types are defined in SCHEME_TYPE
       activation_function - Desired activationfunction for the embdedder
    '''
    try:
        if not ActivationFunctions.has_activation_function(activation_function):
            raise Exception("Invalid activation function for middleware")

        scheme = SCHEME_TYPE[embedder_type](scheme_dict)

        return FCMiddlewareParameters(scheme=scheme, activation_function=activation_function)

    except KeyError as err:
        raise Exception("Middleware, key {} not found".format(err.args[0]))
    except Exception as err:
        raise Exception("Error while creating middleware: {}".format(err))
