import pandas as pd
import cv2
import numpy as np


# Dobavljanje svih podataka iz datog .csv fajla
def get_data(file_name):
    data = pd.read_csv("data/written_name_" + file_name + ".csv")

    # Uklanjanje null i "UNREADABLE" vrednosti
    data = data.dropna()
    data = data[data["IDENTITY"] != ""]
    data = data[data["IDENTITY"] != "UNREADABLE"]

    # Postavljanje svih vrednosti u uppercase
    data["IDENTITY"] = data["IDENTITY"].str.upper()

    # Izmena redosleda redova u random i povratak indexa na pocetak dataframe-a
    data = data.sample(frac=1).reset_index(drop=True)

    return data


# Dobavljanje svih koriscenih karaktera u labelama i maksimalne duzine labele
def get_chars_and_max(data):
    labels = []
    chars = set()
    for name in data["IDENTITY"]:
        labels.append(str(name))
        for char in str(name):
            chars.add(char)
    chars = sorted(chars)
    return {index: char for index, char in enumerate(chars)}, len(max(labels, key=len))


# Konvertuje labelu u niz brojeva
def char_to_number(label, chars):
    numbers = []
    for char in label:
        for key, value in chars.items():
            if value == char:
                numbers.append(key)
    return numbers


# Konvertuje niz brojeva u niz karaktera
def number_to_char(numbers, chars):
    label = ""
    for number in numbers:
        label += chars[number]
    return label


# Dobavljanje i pretprocesiranje slika iz datog foldera, uz odgovarajuce labele
def get_x_and_y(dir_name, data, limit, max_len, chars):
    x = []
    y = np.ones([limit, max_len]) * -1
    len_true = np.zeros([limit, 1])

    for i in range(limit):
        image_name = data.loc[i, "FILENAME"]
        label = str(data.loc[i, "IDENTITY"])

        # Ucitavanje slike u grayscale modu i podesavanje dimenzije na 256x64
        image = cv2.imread("data/" + dir_name + "/" + image_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 64))

        # Normalizacija podataka
        image = image.astype("float32")
        image /= 255

        x.append(image)

        # Racunanje duzine labele i dodavanje labele u y matricu
        y[i, 0:len(label)] = char_to_number(label, chars)
        len_true[i] = len(label)

    return np.array(x).reshape(-1, 256, 64, 1), y, len_true
