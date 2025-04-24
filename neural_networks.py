import numpy as np
import tensorflow as tf
from tensorflow import keras

def create_simple_nn(input_shape, hidden_units=32, output_units=1, activation='relu'):
    """
    Create a simple feedforward neural network.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (excluding batch dimension)
    hidden_units : int
        Number of neurons in the hidden layer
    output_units : int
        Number of output neurons
    activation : str
        Activation function for hidden layer
        
    Returns:
    --------
    model : keras.Model
        Compiled Keras model
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(hidden_units, activation=activation),
        keras.layers.Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy' if output_units == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_deep_nn(input_shape, hidden_layers=[64, 32], output_units=1, activation='relu', dropout_rate=0.2):
    """
    Create a deep feedforward neural network with multiple hidden layers.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (excluding batch dimension)
    hidden_layers : list
        List of integers representing number of units in each hidden layer
    output_units : int
        Number of output neurons
    activation : str
        Activation function for hidden layers
    dropout_rate : float
        Dropout rate between hidden layers
        
    Returns:
    --------
    model : keras.Model
        Compiled Keras model
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    
    # Add hidden layers
    for units in hidden_layers:
        model.add(keras.layers.Dense(units, activation=activation))
        model.add(keras.layers.Dropout(dropout_rate))
    
    # Add output layer
    model.add(keras.layers.Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy' if output_units == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_cnn(input_shape, conv_layers=[(32, 3), (64, 3)], dense_layers=[128], output_units=10):
    """
    Create a Convolutional Neural Network (CNN).
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (height, width, channels)
    conv_layers : list of tuples
        List of (filters, kernel_size) for each convolutional layer
    dense_layers : list
        List of units for each dense layer after convolutional layers
    output_units : int
        Number of output neurons
        
    Returns:
    --------
    model : keras.Model
        Compiled Keras model
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    
    # Add convolutional layers
    for filters, kernel_size in conv_layers:
        model.add(keras.layers.Conv2D(filters, kernel_size, activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
    
    # Flatten and add dense layers
    model.add(keras.layers.Flatten())
    
    for units in dense_layers:
        model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
    
    # Add output layer
    model.add(keras.layers.Dense(output_units, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_rnn(input_shape, rnn_units=50, rnn_layers=1, dense_layers=[32], output_units=1, rnn_type='lstm'):
    """
    Create a Recurrent Neural Network (RNN).
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (time_steps, features)
    rnn_units : int
        Number of units in RNN layers
    rnn_layers : int
        Number of RNN layers
    dense_layers : list
        List of units for each dense layer after RNN layers
    output_units : int
        Number of output neurons
    rnn_type : str
        Type of RNN cell ('lstm', 'gru', or 'simple')
        
    Returns:
    --------
    model : keras.Model
        Compiled Keras model
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    
    # Choose RNN type
    if rnn_type.lower() == 'lstm':
        rnn_layer = keras.layers.LSTM
    elif rnn_type.lower() == 'gru':
        rnn_layer = keras.layers.GRU
    else:
        rnn_layer = keras.layers.SimpleRNN
    
    # Add RNN layers
    for i in range(rnn_layers):
        return_sequences = i < rnn_layers - 1  # Return sequences for all but last RNN layer
        model.add(rnn_layer(rnn_units, return_sequences=return_sequences))
    
    # Add dense layers
    for units in dense_layers:
        model.add(keras.layers.Dense(units, activation='relu'))
    
    # Add output layer
    if output_units == 1:
        model.add(keras.layers.Dense(output_units))  # No activation for regression
    else:
        model.add(keras.layers.Dense(output_units, activation='softmax'))  # Softmax for classification
    
    # Compile for regression or classification
    if output_units == 1:
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_autoencoder(input_shape, encoding_dim=32, hidden_layers=[64]):
    """
    Create an autoencoder for dimensionality reduction or feature learning.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (excluding batch dimension)
    encoding_dim : int
        Dimension of the encoded representation
    hidden_layers : list
        List of units for hidden layers in encoder (decoder is symmetric)
        
    Returns:
    --------
    autoencoder : keras.Model
        Full autoencoder model
    encoder : keras.Model
        Encoder part of the model
    """
    # Build encoder
    encoder_inputs = keras.layers.Input(shape=input_shape)
    x = encoder_inputs
    
    # Add encoder hidden layers
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    
    # Encoded representation
    encoded = keras.layers.Dense(encoding_dim, activation='relu')(x)
    
    # Build encoder model
    encoder = keras.Model(encoder_inputs, encoded, name='encoder')
    
    # Build decoder
    decoder_inputs = keras.layers.Input(shape=(encoding_dim,))
    x = decoder_inputs
    
    # Add decoder hidden layers (in reverse)
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    
    # Output layer
    decoded = keras.layers.Dense(input_shape[0], activation='sigmoid')(x)
    
    # Build decoder model
    decoder = keras.Model(decoder_inputs, decoded, name='decoder')
    
    # Build autoencoder
    autoencoder_outputs = decoder(encoder(encoder_inputs))
    autoencoder = keras.Model(encoder_inputs, autoencoder_outputs, name='autoencoder')
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder
