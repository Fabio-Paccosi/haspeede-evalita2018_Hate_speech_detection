import numpy as np  # Serie di tools utili alla computazione numerica (https://numpy.org)
import csv #Modulo per la lettura di file .csv e .tsv
import pandas as pd  # Tools per processare e manipolare file di dati (https://pandas.pydata.org)
import re  # Modulo per regex
from nltk.corpus import stopwords  # Import del vocabolario di stopwords della libreria Natural Language ToolKit per il Language Processing
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Conv1D, Dropout, MaxPooling1D, concatenate
from sklearn.model_selection import train_test_split
import spacy
import os

# COME FUNZIONA IL PROGRAMMA:
# 1] Prendo in input i file da analizzare
# 2] Pulisco i dati (le frasi) da caratteri singoli, url, stopwords (parole che non hanno particolare senso) tramite la libreria NLTK etc.
# 3] Tokenizzo i dati (le frasi) per parola con la funzione sent_tokenize() di NLTK
# 4]

# CONSTANTI
PROJECT_TITLE = "haspeede@evalita 2018 Project by Fabio Paccosi matr. 307616"
VERSION_NUMBER = "0.1";
EMBEDDING_DIM = 64
# MAX_SEQUENCE_LENGTH = 500
MAX_SEQUENCE_LENGTH = 408
# MAX_SEQUENCE_LENGTH = 12
# MAX_SEQUENCE_LENGTH = 396
# EPOCHS = 100
EPOCHS = 2
BATCH_SIZE = 32
NO_EMBEDDING = True
MULTIPLE_CONV_LEVEL = True

# Configurazioni Top-level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Nascondo dalla console i messaggi che manda TensorFlow in cui ci avverte che la nostra esecuzione potrebbe essere più veloce utilizzano hardware specifico
spacy.prefer_gpu() # Esegue le operazioni di spacy su GPU, se disponibile.

#
def conv_net(max_sequence_length, num_words, embedding_dim, no_embedding_input, trainable=False):
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                input_length=max_sequence_length,
                                trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='float32') if not no_embedding_input else Input(
        shape=(max_sequence_length, 1), dtype='float32')
    embedded_sequences = embedding_layer(sequence_input) if not no_embedding_input else sequence_input

    convs = []
    filter_sizes = [3, 4, 5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)
    x = Dropout(0.5)(l_merge)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='sigmoid')(x)
    x = Dense(10, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.summary()
    return model

#
def get_conv_model(max_sequence_length, size, embedding_dim, no_embedding_input):
    model = Sequential()
    if not no_embedding_input:
        model.add(Embedding(size, embedding_dim, input_length=max_sequence_length))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(max_sequence_length, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.summary()
    return model

#
def exec_ml(data, multiple_conv=True, no_embedding_input=False):
    print('Start ml')

    x_train, x_test, y_train, y_test = data

    x_train = tf.keras.preprocessing.sequence.pad_sequences(list(x_train), maxlen=MAX_SEQUENCE_LENGTH, value=0.0, dtype='float32', truncating='post')
    x_test = tf.keras.preprocessing.sequence.pad_sequences(list(x_test), maxlen=MAX_SEQUENCE_LENGTH, value=0.0, dtype='float32', truncating='post')

    if no_embedding_input:
        x_train = np.expand_dims(x_train, axis=2)
        x_test = np.expand_dims(x_test, axis=2)

    print('Train shape: '+str(x_train.shape))
    print('Test shape: '+str(x_test.shape))

    #try:
    model = conv_net(MAX_SEQUENCE_LENGTH, len(x_train) + 1, EMBEDDING_DIM, no_embedding_input,
                         trainable=True) if multiple_conv else get_conv_model(MAX_SEQUENCE_LENGTH, len(x_train) + 1,
                                                                              EMBEDDING_DIM, no_embedding_input)

    x_array = np.array(x_train)
    y_array = np.array(y_train)
    history = model.fit(x_array, y_array, validation_split=0.25, epochs=EPOCHS, verbose=1)

    plt.plot()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('Model metrics')
    plt.ylabel('Metrics')
    plt.xlabel('Epoch')
    plt.legend(['Train Acc', 'Test Acc', 'Loss Train', 'Loss Test'], loc='upper left')
    plt.show()

    print(model.metrics_names)
    #print(model.evaluate(x_test, y_test))
    print(model.evaluate(x_array, y_array))
    #except Exception as e:
    #    print('--- ML ERROR ---')

#
def store_features(data, storage_name):
    with open('data/features/'+storage_name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

#
def load_features(storage_name):
    with open('data/features/'+storage_name + '.pkl', 'rb') as f:
        return pickle.load(f)

#
def to_data(data_matrix):
    X = list(map(lambda x: x['features'], data_matrix))
    y = list(map(lambda x: 0 if x['tag'].lower() == '0' else 1, data_matrix))
    print('splitting....')
    # Return splitted arrays or matrices into random train and test subsets.
    return train_test_split(X, y, test_size=0.33)

#
def tree_height(root):
    """
    Find the maximum depth (height) of the dependency parse of a spacy sentence by starting with its root
    Code adapted from https://stackoverflow.com/questions/35920826/how-to-find-height-for-non-binary-tree
    :param root: spacy.tokens.token.Token
    :return: int, maximum height of sentence's dependency parse tree
    """
    if not list(root.children):
        return 1
    else:
        return 1 + max(tree_height(x) for x in root.children)

#
def get_heights_measure(paragraph, nlp):
    """
    Computes average height of parse trees for each sentence in paragraph.
    :param paragraph: spacy doc object or str
    :return: float
    """
    try:
        if type(paragraph) == str:
            doc = nlp(paragraph)
        else:
            doc = paragraph
        roots = [sent.root for sent in doc.sents]
        heights = [tree_height(root) for root in roots]
        return np.max(heights), np.min(heights), np.mean(heights)
    except Exception as e:
        return 0

# La tokenizzazione suddivide il contenuto di una frase (o di un testo) in parole (o in frasi) chiamate appunto token.
# La tokenizzazione aiuta a interpretare il significato del testo analizzando la sequenza delle parole.
'''
def tokenize_data(data):
    # Installiamo il modulo desiderato con il comando 'spacy download [nome_pipeline]'
    # Creiamo l'oggetto 'nlp' importando una tra le pipeline di spacy già addestrate per la lingua italiana.
    nlp = spacy.load("it_core_news_sm") # Modello indicato se si vuole efficienza nella computazione (13 MB)
    #nlp = spacy.load("it_core_news_md") # (43 MB)
    #nlp = spacy.load("it_core_news_lg") # Modello indicato se si vuole accuratezza nella computazione (544 MB)

    vector = []
    counter_verb = 0
    counter_adj = 0
    counter_punct = 0
    counter_stop = 0
    counter_sym = 0

    # Per ogni elemento in data, richiamiamo la funzione nlp
    # L'elaborazione delle frasi con l'oggetto nlp restituisce un oggetto doc che contiene le informazioni sui token, le loro caratteristiche linguistiche e le loro relazioni.
    for item in data:
        print(item)
        try:
            tokens = nlp(item["post"])
            for token in tokens:
                print(token)
                vector.append(token.vector_norm)
                if not token.is_stop:
                    counter_stop += 1
                if token.is_punct:
                    counter_punct += 1
                if token.pos_ == 'ADJ':
                    counter_adj += 1
                if token.pos_ == 'VERB':
                    counter_verb += 1
                if token.pos_ == 'SYM':
                    counter_sym += 1

            max_of_vector = tokens.vector.max()
            min_of_vector = tokens.vector.min()
            avg_of_vector = np.mean(tokens.vector)
            paragraph_height_max, paragraph_height_min, paragraph_height_avg = get_heights_measure(tokens, nlp)
            size = len(vector)
            features = [
                size,
                max_of_vector,
                min_of_vector,
                avg_of_vector,
                # counter_punct,
                counter_punct / size,
                # counter_stop,
                counter_stop / size,
                # counter_verb,
                counter_verb / size,
                # counter_adj,
                counter_adj / size,
                # counter_sym,
                counter_sym / size,
                # paragraph_height_max,
                paragraph_height_max / size,
                # paragraph_height_min,
                paragraph_height_min / size,
                # paragraph_height_avg,
                paragraph_height_avg / size,
            ]
            # print("Features:")
            # print(features)
            # print("Tokens vector:")
            # print(tokens.vector)
            # data.append(np.concatenate([features, tokens.vector]))
            data['features'] = np.concatenate([features, tokens.vector])
            # doc['features'] = features
            # doc['features'] = tokens.vector
            # return data
        except Exception as e:
            print(e)

    return data
'''

def tokenizer_func(nlp):
    def inner(doc):
        vector = []
        counter_verb = 0
        counter_adj = 0
        counter_punct = 0
        counter_stop = 0
        counter_sym = 0
        text = doc.get('post', None)
        tokens = nlp(text)
        for token in tokens:
            vector.append(token.vector_norm)
            if not token.is_stop:
                counter_stop += 1
            if token.is_punct:
                counter_punct += 1
            if token.pos_ == 'ADJ':
                counter_adj += 1
            if token.pos_ == 'VERB':
                counter_verb += 1
            if token.pos_ == 'SYM':
                counter_sym += 1

        max_of_vector = tokens.vector.max()
        min_of_vector = tokens.vector.min()
        avg_of_vector = np.mean(tokens.vector)
        paragraph_height_max, paragraph_height_min, paragraph_height_avg = get_heights_measure(tokens, nlp)
        size = len(vector)
        features = [
            size,
            max_of_vector,
            min_of_vector,
            avg_of_vector,
            # counter_punct,
            counter_punct / size,
            # counter_stop,
            counter_stop / size,
            # counter_verb,
            counter_verb / size,
            # counter_adj,
            counter_adj / size,
            # counter_sym,
            counter_sym / size,
            # paragraph_height_max,
            paragraph_height_max / size,
            # paragraph_height_min,
            paragraph_height_min / size,
            # paragraph_height_avg,
            paragraph_height_avg / size,
        ]
        doc['features'] = np.concatenate([features, tokens.vector])
        # doc['features'] = features
        # doc['features'] = tokens.vector
        return doc

    return inner


def tokenize_data(data):
    nlp = spacy.load("it_core_news_sm")  # Modello indicato se si vuole efficienza nella computazione (13 MB)
    # nlp = spacy.load("it_core_news_md") # (43 MB)
    # nlp = spacy.load("it_core_news_lg") # Modello indicato se si vuole accuratezza nella computazione (544 MB)
    return list(map(tokenizer_func(nlp), list(filter(lambda x: not x.get('post', None) is None, data))))

# Utilizzando la librerie RE e il modulo stopwords di nltk, pulisco le frasi della tabella dai dati non necessari
# Perchè farlo? Un post di FB o un Tweet potrebbe essere differente da un testo tradizionale: uso di caratteri speciali, parole abbreviate o non corrette etc.
# L'obiettivo della funzione 'sentence_clean' è quello di rimuovere quindi i caratteri speciali, link e tutto quello che non aggiunge significato alla frase
def sentence_clean(sent):
    # Rimuovo i caratteri di punteggiatura e i numeri
    sentence = re.sub('[^a-zA-Z]', ' ', sent)
    # Rimuovo i caratteri singoli
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Rimuovo le stopweods dopo aver installato il pacchetto di risorse 'stopwords' di nltk con il comando 'nltk.download()'
    stops = stopwords.words('italian')
    for word in sentence.split():
        if word in stops:
            sentence = sentence.replace(word, '')
    #Rimuovo gli spazi multipli
    sentence = re.sub(r'\s+', ' ', sentence)
    #Rimuovo gli spazi iniziali e finali
    sentence = sentence.strip()
    #Ritorna la frase con i caratteri minuscoli
    return sentence.lower()

# Effettuo il parsing dei dati e ritorno un array di coppie di valori [post, tag] dove post rappresenta il testo dell'utente e tag i valori di 1 o 0, se presente un sentimento di odio nel testo o meno
def parse_data(file_path):
    try:
        rows = []
        # Utilizzo il separatore '\t' proprio del file .tsv
        file = pd.read_csv(file_path,
                           header=None, sep="\t",
                           engine='python',
                           error_bad_lines=False,
                           warn_bad_lines=False)

        print("-- Training Set File infos:" % file.columns, file.shape, len(file))

        with open(file_path, "r", encoding ="utf8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                row = {}
                row['post'] = sentence_clean(line[1])
                row['tag'] = line[2]
                if len(row['post']) > 0 and len(row['tag']) > 0:
                    rows.append(row)
        return rows
    except Exception as e:
        print(e)

def get_features():
    input_files = input(
        "Inserisci i path dei dataset (uno o più) che desidere analizzare, divisi dal carattere \' :\n=> ")
    features_storage_name = input("Inserisci il nome del file che contiene le features estratte:\n=> ")
    data = []
    paths = input_files.split(", ")  # Esempio di path: data/train/haspeede_FB-train.tsv
    for file_path in paths:
        print('- Parsing data: ' + file_path)
        data = data + parse_data(file_path)
    # print(data) # Vediamo i dati parsati nell'oggetto 'data'
    data = tokenize_data(data)
    # print(data)
    store_features(data, features_storage_name)
    return features_storage_name

#
def get_user_input():
    # Ottengo l' input degli utenti
    print("Lista dei comandi eseguibili:")
    print("[1] Estrazione delle features e addestramento ml")
    print("[2] Estrazione e salvataggio delle features")
    print("[3] Addestramento ml da set features esistente")

    try:
        selected_option = int(input("Digita il numero di una tra le opzioni disponibili => "))
        if (selected_option == 1):
            loaded_features = load_features(get_features())
            dataset = to_data(loaded_features)
            exec_ml(dataset, multiple_conv=MULTIPLE_CONV_LEVEL, no_embedding_input=NO_EMBEDDING)
        elif (selected_option == 2):
            feature_file = get_features()
            print("I risultati dell\'estrazione delle features sono stati salvati nel file \'" + feature_file + ".pkl\'")
        elif (selected_option == 3):
            feature_file = input()
            loaded_features = load_features(feature_file)
            dataset = to_data(loaded_features)
            exec_ml(dataset, multiple_conv=MULTIPLE_CONV_LEVEL, no_embedding_input=NO_EMBEDDING)
        else:
            print("Scelta non corretta!\n")
            get_user_input()
    except Exception as e:
        print("Errore! Devi inserire un carattere numerico...\n\n")
        get_user_input()

    print("========================================")

# Entry point
if __name__ == '__main__':
    print(PROJECT_TITLE + ", program version: " + VERSION_NUMBER)
    get_user_input()