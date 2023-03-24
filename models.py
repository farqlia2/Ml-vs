import tensorflow as tf

def build_model(n_hidden=1, n_neurons=30, 
                learning_rate=3e-3, input_shape=[8]):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    # Do not use any activation function since it's regression
    # problem and we want a numerical output
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='mse')
    return model

