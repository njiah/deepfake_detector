import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout

def create_model():
    model = tf.keras.models.Sequential([
        EfficientNetB0(include_top=False, input_shape=(128, 128, 3)),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
