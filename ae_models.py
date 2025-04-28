import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_autoencoder(input_dim, latent_dim=10):
    """Builds a standard Autoencoder model."""
    input_layer = keras.Input(shape=(input_dim,))
    # Encoder
    encoded = layers.Dense(256, activation='relu', name='encoder_dense_1')(input_layer)
    encoded = layers.Dense(32, activation='relu', name='encoder_dense_2')(encoded)
    encoded = layers.Dense(latent_dim, activation='relu', name='latent_space')(encoded)
    # Decoder
    decoded = layers.Dense(32, activation='relu', name='decoder_dense_1')(encoded)
    decoded = layers.Dense(256, activation='relu', name='decoder_dense_2')(decoded)
    decoded = layers.Dense(input_dim, activation='linear', name='reconstruction')(decoded) # Linear activation for reconstruction

    # Autoencoder model
    autoencoder = keras.Model(input_layer, decoded, name='autoencoder')
    # Encoder model
    encoder = keras.Model(input_layer, encoded, name='encoder')
    # Decoder model (standalone)
    decoder_input = keras.Input(shape=(latent_dim,))
    deco = autoencoder.get_layer('decoder_dense_1')(decoder_input)
    deco = autoencoder.get_layer('decoder_dense_2')(deco)
    deco = autoencoder.get_layer('reconstruction')(deco)
    decoder = keras.Model(decoder_input, deco, name='decoder')


    return autoencoder, encoder, decoder

def build_conditional_autoencoder(input_dim, latent_dim=10):
    """Builds a Conditional Autoencoder model (predicts next step)."""
    input_layer = keras.Input(shape=(input_dim,))
    # Encoder
    encoded = layers.Dense(256, activation='relu', name='encoder_dense_1')(input_layer)
    encoded = layers.Dropout(0.2)(encoded) # Add dropout
    encoded = layers.Dense(32, activation='relu', name='encoder_dense_2')(encoded)
    encoded = layers.Dropout(0.2)(encoded) # Add dropout
    encoded = layers.Dense(latent_dim, activation='relu', name='latent_space')(encoded) # No dropout before latent space usually
    # Decoder
    decoded = layers.Dense(32, activation='relu', name='decoder_dense_1')(encoded)
    decoded = layers.Dropout(0.2)(decoded) # Add dropout
    decoded = layers.Dense(256, activation='relu', name='decoder_dense_2')(decoded)
    decoded = layers.Dropout(0.2)(decoded) # Add dropout
    decoded = layers.Dense(input_dim, activation='linear', name='reconstruction')(decoded) # Linear activation for reconstruction, no dropout on final output

    # Autoencoder model
    autoencoder = keras.Model(input_layer, decoded, name='conditional_autoencoder')
    # Encoder model
    encoder = keras.Model(input_layer, encoded, name='conditional_encoder')
    # Decoder model (standalone)
    decoder_input = keras.Input(shape=(latent_dim,))
    deco = autoencoder.get_layer('decoder_dense_1')(decoder_input)
    deco = autoencoder.get_layer('decoder_dense_2')(deco)
    deco = autoencoder.get_layer('reconstruction')(deco)
    decoder = keras.Model(decoder_input, deco, name='conditional_decoder')

    return autoencoder, encoder, decoder
