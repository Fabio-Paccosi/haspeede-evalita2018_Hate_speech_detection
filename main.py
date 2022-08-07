import numpy as np  # Serie di tools utili alla computazione numerica (https://numpy.org)
import csv #Modulo per la lettura di file .csv e .tsv
import pandas as pd  # Tools per processare e manipolare file di dati (https://pandas.pydata.org)
import re  # Modulo per regex
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords  # Import del vocabolario di stopwords della libreria Natural Language ToolKit per il Language Processing
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Dropout, MaxPooling1D, Embedding
from simplemma import simplemma
from sklearn.model_selection import train_test_split
import spacy
import os

# CONSTANTI
PROJECT_TITLE = "haspeede@evalita 2018 Project by Fabio Paccosi matr. 307616"
VERSION_NUMBER = "1.0"
EMBEDDING_DIM = 256 #128
MAX_WORD_LENGTH = -1
BATCH_SIZE = 32

vocab = {}

# Configurazioni Top-level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Nascondo dalla console i messaggi che manda TensorFlow in cui ci avverte che la nostra esecuzione potrebbe essere più veloce utilizzano hardware specifico
spacy.prefer_gpu()# Esegue le operazioni di spacy su GPU, se disponibile.

# Setup della Convolutional Neural Network (CNN)
# La rete di convoluzione è un tipo di rete neurale artificiale feed-forward adatta, come nel nostro caso, nell'elaborazione del linguaggio naturale
def setup_convolution_net(activation_choice, optimizer_choice):
    print("Setup della CNN... ")

    # Definiamo il modello
    model = Sequential()

    # Il primo layer effettua l'addestramento basato sula tecnica word embedding
    model.add(Embedding(len(vocab), EMBEDDING_DIM, input_length=MAX_WORD_LENGTH))

    # Aggiungiamo il layer di convoluzione
    # Con questo livello creiamo un kernel di convoluzione di una singola dimensione spaziale per produrre un tensore di output.
    # Argomenti utilizzati:
    # filters = la dimensione dello spazio di output, ossia il numero di filtri di output nella convoluzione
    # kernel_size = specifica la lunghezza della finestra di convoluzione 1D
    # activation = la funzione di attivazione da utilizzare
    # input_shape = tensore 3+D
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(MAX_WORD_LENGTH, 1)))

    # Aggiungiamo un pooling layer, che viene utilizzato per ridurre le dimensioni della mappa delle features
    model.add(MaxPooling1D(pool_size=2))

    # Applichimo un ulteriore layer di convoluzione
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

    # Applichiamo un ulteriore livello di pooling
    model.add(MaxPooling1D(pool_size=2))

    # Il layer dropout imposta in modo casuale le unità di input su 0 con una frequenza di 0.5 a ogni passaggio durante il training
    # Questo passaggio aiuta a prevenire l'overfitting, ossia quando il modello risulta complesso e c'è un'alta variabilità nella classificazione
    model.add(Dropout(0.5))

    # Rimuoviamo tutte le dimensioni tranne una
    model.add(Flatten())

    # Impostiamo l'ottimizzatore in base alle scelte dell'utente
    if (activation_choice == 1):
        activation_choosen = "softmax"
        logits_value = False
    elif (activation_choice == 2):
        activation_choosen = "sigmoid"
        logits_value = False
    else:
        activation_choosen = None
        logits_value = True

    # I layer Dense è il naturale livello di rete neurale connessa.
    # È il layer più comune ed esegue l'operazione seguente sull'input e restituisce l'output.
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation=activation_choosen))

    # Compiliamo il modello specificando:
    # loss = funzione loss
    # optimizer = ottimizzatore
    # metrics = lista di metriche valutate dal modello

    # Impostiamo l'ottimizzatore in base alle scelte dell'utente
    if(optimizer_choice==0):
        optimizer_choosen = "adam"
    else:
        optimizer_choosen = "adadelta"


    # Compiliamo il modello
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=logits_value),
                  optimizer=optimizer_choosen,
                  metrics=["accuracy"])
    #model.compile(loss='binary_crossentropy', optimizer=optimizer_choosen, metrics=['accuracy'])

    print(model.summary())

    model.summary()
    return model

# Eseguo la procedura di Machine Learning
def do_machine_learning(data, log_level, activation_choice, optimizer_choice, epochs):
    print('Inizio la procedura di Machine Learning...')

    x_train, x_test, y_train, y_test = data

    # Assicuriamoci che tutte le sequenze nella lista hanno la stessa lunghezza
    x_train = pad_sequences(list(x_train), maxlen=MAX_WORD_LENGTH, value=0.0, dtype='float32', truncating='post')
    x_test = pad_sequences(list(x_test), maxlen=MAX_WORD_LENGTH, value=0.0, dtype='float32', truncating='post')

    # Inserisce un nuovo asse che apparirà nella posizione scelta nella matrice.
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    # Visualizziamo la forma delle matrici di addestramento e test
    if log_level == 1:
        print('Train shape: '+str(x_train.shape))
        print('Test shape: '+str(x_test.shape))

    try:
        model = setup_convolution_net(activation_choice, optimizer_choice)

        x_array = np.array(x_train)
        y_array = np.array(y_train)
        # Addestriamo il modello
        history = model.fit(x_array, y_array, validation_split=0.33, epochs=epochs, verbose=1)

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

        if log_level == 1:
            print(model.metrics_names)
            print(model.evaluate(x_array, y_array))
    except Exception as e:
        print('--- ML ERROR ---')
        print(str(e))


# Metodo per il salvataggio del file .pkl contenente le features estratte
def store_features(data, storage_name):
    with open('data/features/'+storage_name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


# Metodo per il caricamento del file .pkl contenente le features estratte
def load_features(storage_name):
    with open('data/features/'+storage_name + '.pkl', 'rb') as f:
        return pickle.load(f)


# Metodo che ritorna le matrici divise in set di addestramento e test casuali
def to_data(data_matrix):
    X = list(map(lambda x: x['features'], data_matrix))
    y = list(map(lambda x: 0 if x['tag'].lower() == '0' else 1, data_matrix))
    print('- Eseguo lo la divisione dei dati...')
    # Richiamo il metodo della libreria sklearn per dividere i set passando la dimensione desiderata del set di test
    return train_test_split(X, y, test_size=0.25)


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
def tokenizer_func(nlp):
    def inner(doc):
        vocab_index = 1
        vector = []
        counter_verb = 0
        counter_adj = 0
        counter_adp = 0
        counter_aux = 0
        counter_noun = 0
        counter_num = 0
        counter_propn = 0
        counter_punct = 0
        counter_stop = 0
        counter_sym = 0
        text = doc.get('post', None)
        tokens = nlp(text) #Tokenizzo il testo del post

        # Imposto la variabile maxlen
        global MAX_WORD_LENGTH
        if MAX_WORD_LENGTH < len(tokens):
            MAX_WORD_LENGTH = len(tokens)

        # Analizzo ogni singolo token e lo inserisco (se non presente) nel vocabolario globale
        for token in tokens:
            if token not in vocab:
                vocab[token] = vocab_index
                vocab_index += 1
            vector.append(token.vector_norm) #La norma di un vettore complesso
            #Analizzo la PoS
            if not token.is_stop:
                counter_stop += 1
            if token.is_punct:
                counter_punct += 1
            if token.pos_ == 'PROPN':
                counter_propn += 1
            if token.pos_ == 'ADJ':
                counter_adj += 1
            if token.pos_ == 'ADP':
                counter_adp += 1
            if token.pos_ == 'AUX':
                counter_aux += 1
            if token.pos_ == 'VERB':
                counter_verb += 1
            if token.pos_ == 'SYM':
                counter_sym += 1
            if token.pos_ == 'NOUN':
                counter_noun += 1
            if token.pos_ == 'NUM':
                counter_num += 1

        max_of_vector = tokens.vector.max()
        min_of_vector = tokens.vector.min()
        avg_of_vector = np.mean(tokens.vector)
        paragraph_height_max, paragraph_height_min, paragraph_height_avg = get_heights_measure(tokens, nlp)
        size = len(vector)
        features = [
            abs(size),
            abs(max_of_vector),
            abs(min_of_vector),
            abs(avg_of_vector),
            counter_punct / size,
            counter_stop / size,
            counter_propn / size,
            counter_verb / size,
            counter_adp / size,
            counter_adj / size,
            counter_aux / size,
            counter_noun / size,
            counter_sym / size,
            counter_num / size,
            paragraph_height_max / size,
            paragraph_height_min / size,
            paragraph_height_avg / size,
        ]
        doc['features'] = np.concatenate([features])

        return doc

    return inner


# La funzione carica un modello della libreria spacy nell'oggetto chiamato 'nlp' e richiama la funzione 'tokenizer_func' con quell'oggetto su ogni elemento/frase del dataset
# I seguenti modelli di spacy, devono essere prima scaricati e installati con il comando 'spacy download [nome_modello]' per poter essere utilizzati
def tokenize_data(data, model_level):
    if model_level == 0:
        nlp = spacy.load("it_core_news_sm")  # Modello indicato se si vuole efficienza nella computazione (13 MB)
    elif model_level == 1:
        nlp = spacy.load("it_core_news_md")  # (43 MB)
    else:
        nlp = spacy.load("it_core_news_lg")  # Modello indicato se si vuole accuratezza nella computazione (544 MB)

    return list(map(tokenizer_func(nlp), list(filter(lambda x: not x.get('post', None) is None, data))))


# Utilizzando la librerie RE e il modulo stopwords di nltk, pulisco le frasi della tabella dai dati non necessari
# Perchè farlo? Un post di FB o un Tweet potrebbe essere differente da un testo tradizionale: uso di caratteri speciali, parole abbreviate o non corrette etc.
# L'obiettivo è quello di rimuovere quindi i caratteri speciali, link e tutto quello che non aggiunge significato alla frase
def sentence_clean(sent):
    # Rimuovo i caratteri di punteggiatura e i numeri
    sentence = re.sub('[^a-zA-Z]', ' ', sent)
    # Rimuovo i caratteri singoli
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Rimuovo le stopweods dopo aver installato il pacchetto di risorse 'stopwords' di nltk con il comando 'nltk.download()'
    stops = stopwords.words('italian')
    for word in sentence.split():
        # Uso la lemmatizzazione per ricondurre le parole al tema
        if word in stops:
            sentence = sentence.replace(word, '')
        #Uso la lemmatizzazione per ricondurre le parole al tema
        lemma = simplemma.lemmatize(word, lang='it')
        sentence = sentence.replace(word, lemma)
    #Rimuovo gli spazi multipli
    sentence = re.sub(r'\s+', ' ', sentence)
    #Rimuovo gli spazi iniziali e finali
    sentence = sentence.strip()
    #Ritorna la frase con i caratteri minuscoli
    return sentence.lower()

# Effettuo il parsing dei dati e ritorno un array di coppie di valori [post, tag] dove post rappresenta il testo dell'utente e tag i valori di 1 o 0, se presente un sentimento di odio nel testo o meno
def parse_data(file_path, log_level=0):
    try:
        rows = []
        # Utilizzo il separatore '\t' proprio del file .tsv
        file = pd.read_csv(file_path,
                           header=None, sep="\t",
                           engine='python',
                           error_bad_lines=False,
                           warn_bad_lines=False)
        if log_level == 1:
            print("-- Training Set File infos:" % file.columns, file.shape, len(file))

        with open(file_path, "r", encoding ="utf8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                row = {'post': sentence_clean(line[1]), 'tag': line[2]}
                if len(row['post']) > 0 and len(row['tag']) > 0:
                    rows.append(row)
        return rows
    except Exception as e:
        print(e)

# Estraggo le features dai dataset selezionati e salvo il risultato nel file .pkl scelto
def get_features(log_level, model_level):
    input_files = input(
        "- Inserisci i path dei dataset (uno o più) che desidere analizzare, divisi dal carattere \',\' : => ")
    features_storage_name = input("- Inserisci il nome del file che contiene le features estratte: => ")
    data = []
    paths = input_files.split(", ")
    for file_path in paths:
        print('-> Eseguo il parsing del set: ' + file_path)
        data = data + parse_data(file_path)

    # Vediamo i dati parsati nell'oggetto 'data'
    if log_level == 1:
        print(data)

    # Tokenizziamo i dati dell'array
    data = tokenize_data(data, model_level)

    # Visualizziamo i dati tokenizzati
    if log_level == 1:
        print("Vocab lenght: "+str(len(vocab)));
        print("Vocab: "+str(vocab))
        print("Tokenize data: " + str(data))

    # Salvo il risultato dell'estrazione
    store_features(data, features_storage_name)
    return features_storage_name

# Scegliamo il comportamento del codice e forniamo in input i dati necessari
#  Esempio di path: data/train/haspeede_FB-train.tsv
def get_user_input():
    # Ottengo l' input degli utenti
    print("##### Lista dei comandi eseguibili #####")
    print("[1] Estrazione delle features e addestramento ml")
    print("[2] Estrazione e salvataggio delle features")
    print("[3] Addestramento ml da set features esistente")

    try:
        selected_option = int(input("- Digita il numero di una tra le opzioni disponibili => "))
        log_level = int(input("- Digita il livello di log che voi attivare [0 = nessuno, 1 = attivo] => "))
        if (selected_option == 1):
            model_level = int(input("- Digita il livello di accuratezza del modello della pipeline di addestramento [0 = efficiente, 1 = intemedio, 2 = accurato] => "))
            activation_choice = int(input("- Digita la funzione di attivazione che vuoi utilizzare [0 = nessuna, 1 = softmax, 2 = sigmoid] => "))
            optimizer_type = int(input("- Digita quale ottimizzatore vuoi utilizzare [0 = adam, 1 = adadelta] => "))
            epochs = int(input("- Digita il numero di epoche di addestramento => "))
            loaded_features = load_features(get_features(log_level, model_level))
            dataset = to_data(loaded_features)
            do_machine_learning(dataset, log_level, activation_choice, optimizer_type, epochs)
        elif (selected_option == 2):
            model_level = int(input("- Digita il livello di accuratezza del modello della pipeline di addestramento [0 = efficiente, 1 = intemedio, 2 = accurato] => "))
            feature_file = get_features(log_level, model_level)
            print("I risultati dell\'estrazione delle features sono stati salvati nel file \'" + feature_file + ".pkl\'")
        elif (selected_option == 3):
            feature_file = input("- Digita il nome del file che contiene le features estratte => ")
            activation_choice = int(input("- Digita la funzione di attivazione che vuoi utilizzare [0 = nessuna, 1 = softmax, 2 = sigmoid] => "))
            optimizer_type = int(input("- Digita quale ottimizzatore vuoi utilizzare [0 = adam, 1 = adadelta] => "))
            epochs = int(input("- Digita il numero di epoche di addestramento => "))
            loaded_features = load_features(feature_file)
            dataset = to_data(loaded_features)
            do_machine_learning(dataset, log_level, activation_choice, optimizer_type, epochs)
        else:
            print("Scelta non corretta!\n")
            get_user_input()
    except Exception as e:
        print("=========== ERRORE ===========")
        print(str(e))
        print("==============================")
        get_user_input()

    print("========================================")

# Entry point
if __name__ == '__main__':
    print(PROJECT_TITLE + ", program version: " + VERSION_NUMBER)
    vocab['pad'] = 0
    get_user_input()
