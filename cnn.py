from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Reshape, Dense, Bidirectional, LSTM, Lambda
from keras.models import Model
from keras.backend import ctc_batch_cost
from keras.optimizers import Adam


# Kreiranje modela - konvolucione (rekurentne) neuronske mreze
def create_model(chars):
    input_layer = Input(shape=(256, 64, 1))

    hidden_layers = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(input_layer)
    hidden_layers = MaxPooling2D(pool_size=(2, 2))(hidden_layers)
    hidden_layers = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(hidden_layers)
    hidden_layers = MaxPooling2D(pool_size=(2, 2))(hidden_layers)
    hidden_layers = Dropout(0.25)(hidden_layers)
    hidden_layers = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(hidden_layers)
    hidden_layers = MaxPooling2D(pool_size=(1, 2))(hidden_layers)
    hidden_layers = Dropout(0.25)(hidden_layers)

    # Konverzija u rekurentnu neuronsku mrezu
    hidden_layers = Reshape((64, 1024))(hidden_layers)
    hidden_layers = Dense(64, activation="relu")(hidden_layers)
    hidden_layers = Bidirectional(LSTM(128, return_sequences=True))(hidden_layers)
    hidden_layers = Bidirectional(LSTM(64, return_sequences=True))(hidden_layers)

    output_layer = Dense(len(chars) + 1, activation="softmax")(hidden_layers)

    model = Model(inputs=input_layer, outputs=output_layer)
    return input_layer, model, output_layer


# Connectionist Temporal Classification (CTC) loss funkcija
def ctc_loss_function(args):
    output_layer, input_layers = args
    output_layer = output_layer[:, 2:, :]  # Uklanjanje pocetnih vrednosti iz RNN-a
    return ctc_batch_cost(input_layers[0], output_layer, input_layers[1], input_layers[2])


# Kompajliranje kreiranog modela
def compile_model(chars, max_len, input_train, output_train, input_validation, output_validation):
    input_layer, model, output_layer = create_model(chars)

    input_layers = [Input(shape=[max_len]), Input(shape=[1]), Input(shape=[1])]
    ctc_loss = Lambda(ctc_loss_function, output_shape=(1, ), name="ctc")([output_layer, input_layers])

    model_ctc = Model(inputs=[input_layer, input_layers[0], input_layers[1], input_layers[2]], outputs=ctc_loss)
    model_ctc.compile(loss={"ctc": lambda true_y, predict_y: predict_y}, optimizer=Adam(), metrics=["accuracy"])
    model_ctc.fit(x=input_train, y=output_train, validation_data=(input_validation, output_validation), epochs=15, batch_size=128)

    return model.predict(input_validation[0])
