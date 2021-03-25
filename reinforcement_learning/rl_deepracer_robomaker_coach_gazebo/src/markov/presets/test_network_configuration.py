# Default network with only one camera
neural_network_settings = {
    'input_embedders': 'DEFAULT_INPUT_EMBEDDER',
    'middleware_embedders': 'DEFAULT_MIDDLEWARE',
    'embedder_type': 'scheme',
    'activation_function': 'relu'
}

# For shallow network with only stereo
neural_network_settings = {
    'input_embedders': 'SHALLOW_LEFT_STEREO_INPUT_EMBEDDER',
    'middleware_embedders': 'DEFAULT_MIDDLEWARE',
    'embedder_type': 'scheme',
    'activation_function': 'relu'
}

# For shallow network with left camera and stereo
neural_network_settings = {
    'input_embedders': 'SHALLOW_STEREO_INPUT_EMBEDDER',
    'middleware_embedders': 'DEFAULT_MIDDLEWARE',
    'embedder_type': 'scheme',
    'activation_function': 'relu'
}

# For shallow network with left and stereo batch norms
neural_network_settings = {
    'input_embedders': 'SHALLOW_LEFT_STEREO_WITH_BN_INPUT_EMBEDDER',
    'middleware_embedders': 'DEFAULT_MIDDLEWARE',
    'embedder_type': 'bn_scheme',
    'activation_function': 'none'
}

# For deeper network with left camera and stereo
neural_network_settings = {
    'input_embedders': 'DEEP_LEFT_STEREO_INPUT_EMBEDDER',
    'middleware_embedders': 'DEFAULT_MIDDLEWARE',
    'embedder_type': 'scheme',
    'activation_function': 'relu'
}

# For shallow network only LIDAR without batch norms
neural_network_settings = {
    'input_embedders': 'SHALLOW_LIDAR_INPUT_EMBEDDER',
    'middleware_embedders': 'DEFAULT_MIDDLEWARE',
    'embedder_type': 'bn_scheme',
    'activation_function': 'none'
}

# For shallow network only LIDAR with batch norms
neural_network_settings = {
    'input_embedders': 'SHALLOW_LIDAR_BN_INPUT_EMBEDDER',
    'middleware_embedders': 'SHALLOW_LIDAR_MIDDLEWARE',
    'embedder_type': 'bn_scheme',
    'activation_function': 'none'
}

