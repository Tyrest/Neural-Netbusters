import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def train_autoencoder(model, data, epochs=50, batch_size=32, validation_split=0.2, random_state=42, save_path=None):
    """Trains a standard Autoencoder."""
    X_train, X_val = train_test_split(data, test_size=validation_split, random_state=random_state)
    history = model.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_val, X_val),
                        verbose=1)
    if save_path:
        model.save(save_path)
    return history

def train_conditional_autoencoder(model, data, epochs=50, batch_size=32, validation_split=0.2, random_state=None, save_path=None):
    """Trains a Conditional Autoencoder (predicts next step)."""
    # Split data for training and validation, maintaining temporal order if random_state is None
    X_train, X_val = train_test_split(data, test_size=validation_split, shuffle=(random_state is not None), random_state=random_state)

    # Prepare input (X) and target (Y) pairs for conditional prediction
    Y_train_offset = X_train[1:]
    X_train_offset = X_train[:-1]
    Y_val_offset = X_val[1:]
    X_val_offset = X_val[:-1]

    keras.utils.set_random_seed(42) # For reproducibility in training

    history = model.fit(X_train_offset, Y_train_offset,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_val_offset, Y_val_offset),
                        verbose=1)
    if save_path:
        model.save(save_path)
    return history
