from data_manipulation import *
from cnn import *
from keras.backend import get_value, ctc_decode


# Provera ispravnosti rezultata modela
def check_results(results, limit):
    # Dobavljanje i konvertovanje predvidjenih brojeva u labele
    input_len = np.ones(results.shape[0]) * results.shape[1]
    numbers = get_value(ctc_decode(results, input_length=input_len, greedy=True)[0][0])

    true_labels = str(validation_data.loc[0:limit, "IDENTITY"])
    predict_labels = []
    for i in range(limit):
        predict_labels.append(number_to_char(numbers[i], chars))

    # Poredjenje svake predvidjene labele sa tacnom
    char_cnt = 0
    label_cnt = 0
    for i in range(limit):
        true_label = true_labels[i]
        true_len = len(true_label)
        predict_label = predict_labels[i]
        predict_len = len(predict_label)
        len_limit = min(true_len, predict_len)

        for j in range(len_limit):
            if true_label[j] == predict_label[j]:
                char_cnt += 1

        # Sve preko 85% tacnosti je uspesno prepoznato
        if char_cnt >= true_len * 0.85:
            label_cnt += 1

    return label_cnt


if __name__ == '__main__':
    # Ucitavanje podataka
    train_data = get_data("train")
    validation_data = get_data("validation")
    test_data = get_data("test")
    chars, max_len = get_chars_and_max(train_data)

    # Zbog slabijih performansi racunara, korisceno je 15% podataka
    train_limit = 49645
    validation_limit = 6205
    test_limit = 6205

    # Generisanje podataka za treniranje i validaciju modela
    x_train, y_train, len_true_train = get_x_and_y("train", train_data, train_limit, max_len, chars)
    x_validation, y_validation, len_true_validation = get_x_and_y("validation", validation_data, validation_limit, max_len, chars)
    x_test, y_test, len_true_test = get_x_and_y("test", test_data, test_limit, max_len, chars)

    # Generisanje matrica koje cuvaju duzine predvidjenih labela
    len_predict_train = np.ones([train_limit, 1]) * max_len
    len_predict_validation = np.ones([validation_limit, 1]) * max_len
    len_predict_test = np.ones([test_limit, 1]) * max_len

    # Generisanje matrica koje cuvaju vrednosti CTC loss funkcije
    output_train = np.zeros([train_limit])
    output_validation = np.zeros([validation_limit])
    output_test = np.zeros([validation_limit])

    input_train = [x_train, y_train, len_predict_train, len_true_train]
    input_validation = [x_validation, y_validation, len_predict_validation, len_true_validation]
    input_test = [x_test, y_test, len_predict_test, len_true_test]

    # Kompajliranje i provera modela za validacioni skup
    results = compile_model(chars, max_len, input_train, output_train, input_validation, output_validation)
    print("System accuracy - validation: {.2f}".format(check_results(results, validation_limit) / validation_limit * 100))

    # Kompajliranje i provera modela za test skup
    results = compile_model(chars, max_len, input_train, output_train, input_test, output_test)
    print("System accuracy - test: {.2f}".format(check_results(results, test_limit) / test_limit * 100))

    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
