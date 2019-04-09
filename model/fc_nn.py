import tensorflow as tf
from .abstract_model import Model


class FullyConnectedNN(Model):
    """A simple fully connected neural network

    Args:
        layer_dims: list specifying the dimensions of hidden layers and output
        activation: activation function to apply after each hidden layer
        output_activation: activation function to apply to the output
    """

    def __init__(self, hparams, layer_dims, activation=tf.nn.leaky_relu, output_activation=None):
        super(FullyConnectedNN, self).__init__(hparams)
        self.layer_dims = layer_dims
        self.activation = activation
        self.output_activation = output_activation

    def _forward(self, x):
        """Forward pass of the network.

        Args:
            x: input tensor
        """
        for hidden_dim in self.layer_dims[:-1]:
            x = tf.layers.dense(x, hidden_dim, self.activation)

        output_tensor = tf.layers.dense(x, self.layer_dims[-1], self.output_activation)

        return output_tensor
