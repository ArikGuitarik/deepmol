import tensorflow as tf


def fc_nn(x, layer_dims, activation=tf.nn.leaky_relu, output_activation=None):
    """A simple fully connected neural network

    Args:
        x: input tensor
        layer_dims: list specifying the dimensions of hidden layers and output
        activation: activation function to apply after each hidden layer
        output_activation: activation function to apply to the output
    """
    for hidden_dim in layer_dims[:-1]:
        x = tf.layers.dense(x, hidden_dim, activation)

    output_tensor = tf.layers.dense(x, layer_dims[-1], output_activation)

    return output_tensor
