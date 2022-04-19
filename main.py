import numpy as np  # Serie di tools utili alla computazione numerica (https://numpy.org)
import csv #Modulo per la lettura di file .csv e .tsv
import pandas as pd  # Tools per processare e manipolare file di dati (https://pandas.pydata.org)
import click  # Tools per accettare parametri in input da riga di comando
import re  # Modulo per regex
import nltk  # Libreria per il Language Processing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# CONSTANTI
PROJECT_TITLE = "haspeede@evalita 2018 Project by Fabio Paccosi matr. 307616"
VERSION_NUMBER = "0.1";

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


# Recupero i parametri dalla libreria Click e li processo
@click.command()
@click.option('--input-files', '-m', prompt='Insert input file paths divided them by \',\' character')
def preprocessing(input_files):
    paths = input_files.split(", ")
    for file_path in paths:
        print("Reading Training Set File at: " + file_path)
        # Utilizzo il separatore '\t' proprio del file .tsv
        file = pd.read_csv(file_path,
                           header=None, sep="\t",
                           engine='python',
                           error_bad_lines=False,
                           warn_bad_lines=False)  # Esempio di path: data/train/haspeede_FB-train.tsv
        # Cancello la prima colonna (che contiene gli ID, non utili alla nostra analisi)
        file.drop(file.columns[[0, 0]], axis=1, inplace=True)
        # save it as txt file
        #file.to_csv("data/train/pippo.txt", index=False, header=False)
        print("- Training Set File infos:" % file.columns, file.shape, len(file))
        # Preprocesso il file
        print("-PREPROCESSO IL FILE:")
        with open("data/train/haspeede_FB-train.tsv", "r", encoding ="utf8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                sent = sentence_clean(line[1])
                print(sent +" "+ line[2])


# @click.option('--feature-storage', '-s', prompt='Give feature storage files', help='Feature storage data file.')
# @click.option('--only-train', is_flag=False)
# @click.option('--only-features', is_flag=True)
def start(input_files, feature_storage, only_train, only_features):
    data = []
    if not only_train:
        for file in input_files:
            print('parsing... ' + file)
            data = data  # + parse_xml(file)
        data = []  # tokenize_data(data)
        # store_features(data, feature_storage)
    else:
        data = []  # load_features(feature_storage)

    if not only_features:
        dataset = []  # to_data(data)
        # dataset = to_cross_data(data,
        #                         train_filter=['children', 'diary', 'journalism', 'twitter', ],
        #                         test_filter=['youtube', ])
        # dataset = to_cross_data(data,
        #                         train_filter=['youtube', 'diary', 'journalism', 'twitter', ],
        #                         test_filter=['children', ])
        # dataset = to_cross_data(data,
        #                         train_filter=['children', 'youtube', 'journalism', 'twitter', ],
        #                         test_filter=['diary', ])
        # dataset = to_cross_data(data,
        #                         train_filter=['children', 'diary', 'youtube', 'twitter', ],
        #                         test_filter=['journalism', ])
        # dataset = to_cross_data(data,
        #                         train_filter=['children', 'diary', 'journalism', 'youtube', ],
        #                         test_filter=['twitter', ])
        # dataset = to_cross_data(data,
        #                         train_filter=['children', 'diary', 'journalism', 'youtube', ],
        #                         test_filter=['twitter', ])
    # exec_ml(dataset, multiple_conv=MULTIPLE_CONV_LEVEL, no_embedding_input=NO_EMBEDDING)


# COME FUNZIONA IL PROGRAMMA:
# 1] Prendo in input i file da analizzare
# 2] Pulisco i dati (le frasi) da caratteri singoli, url, stopwords (parole che non hanno particolare senso) tramite la libreria NLTK etc.
# 3] Tokenizzo i dati (le frasi) per parola con la funzione sent_tokenize() di NLTK
# 4]
if __name__ == '__main__':
    print(PROJECT_TITLE + ", program version: " + VERSION_NUMBER)
    preprocessing()
