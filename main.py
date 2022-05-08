import numpy as np  # Serie di tools utili alla computazione numerica (https://numpy.org)
import csv #Modulo per la lettura di file .csv e .tsv
import pandas as pd  # Tools per processare e manipolare file di dati (https://pandas.pydata.org)
import click  # Tools per accettare parametri in input da riga di comando
import re  # Modulo per regex
from nltk.corpus import stopwords  # Import del vocabolario di stopwords della libreria Natural Language ToolKit per il Language Processing
import pickle
#from nltk.tokenize import word_tokenize # Import della funzione di NLTK per fare la tokenizzazione per parole
#from nltk.stem import WordNetLemmatizer

import spacy
from spacy.lang.it.examples import sentences

# CONSTANTI
PROJECT_TITLE = "haspeede@evalita 2018 Project by Fabio Paccosi matr. 307616"
VERSION_NUMBER = "0.1";

"""
# Il processo di 'tokenizzazione' è la suddivisione di un testo o una frase in più parti, dette appunto token.
# E' possibile suddivire un testo in frasi, oppure una frase in parole (il nostro caso specifico).
def sentence_tokeninze(sent):
    word_tokenizer_output = word_tokenize(sent)
    return word_tokenizer_output;

# Il processo di 'lemmatizzazione' delle parole di una frase viene utilizzato per cercare di ricondurre diverse forme flesse allo stesso tema.
# Ad esempio:se nella frase troviamo le parole “camminare”, “cammino”, “camminiamo”, “cammineremo”, la lemmatization le ricondurrà tutte al lemma (forma base, quella che troviamo sul vocabolario) “camminare”.
def sentence_lemmatize(sent):
    # Prima di utilizzare l'oggetto WornNetLemmatizer è necessarion installare i pacchetti di risorse 'wordnet' e 'omw-1.4' di nltk con il comando 'nltk.download()'
    lemmatizer = WordNetLemmatizer()
    # Prima di utilizzare la funzione word_tokenize dobbiamo installare il pacchetto di risorse 'punkt'
    #word_list = nltk.word_tokenize(sent)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(word) for word in sent])
    return lemmatized_output

# Recupero i parametri dalla libreria Click e li processo
#@click.command()
#@click.option('--input-files', '-m', prompt='Insert input file paths divided them by \',\' character')
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
        print("- Training Set File infos:" % file.columns, file.shape, len(file))

        # Preprocesso il file
        print("-PREPROCESSO IL FILE:")
        with open(file_path, "r", encoding ="utf8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                #FIN QUI
                sent = sentence_clean(line[1])
                tokens = sentence_tokeninze(sent)
                print(sent +" -> "+ sentence_lemmatize(tokens) + " == "+ line[2])
                #for t in tokens:
                #    print("- "+t)
"""

# COME FUNZIONA IL PROGRAMMA:
# 1] Prendo in input i file da analizzare
# 2] Pulisco i dati (le frasi) da caratteri singoli, url, stopwords (parole che non hanno particolare senso) tramite la libreria NLTK etc.
# 3] Tokenizzo i dati (le frasi) per parola con la funzione sent_tokenize() di NLTK
# 4]

#
def store_features(data, storage_name):
    with open(storage_name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

#
def load_features(storage_name):
    with open(storage_name + '.pkl', 'rb') as f:
        return pickle.load(f)


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
def tokenize_data(data):
    # Installiamo il modulo desiderato con il comando 'spacy download [nome_pipeline]'
    # Creiamo l'oggetto 'nlp' importando una tra le pipeline di spacy già addestrate per la lingua italiana.
    #nlp = spacy.load("it_core_news_sm") # Modello indicato se si vuole efficienza nella computazione (13 MB)
    #nlp = spacy.load("it_core_news_md") # (43 MB)
    nlp = spacy.load("it_core_news_lg") # Modello indicato se si vuole accuratezza nella computazione (544 MB)

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
            print("Features:")
            print(features)
            print("Tokens vector:")
            print(tokens.vector)
            data.append(np.concatenate([features, tokens.vector]))
        except Exception as e:
            print(e)

    return data

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

# Recupero i parametri dalla libreria Click e li processo
@click.command()
@click.option('--input-files', '-m', prompt='Insert input file paths divided them by \',\' character')
def get_input_files(input_files):
    data = []
    paths = input_files.split(", ") # Esempio di path: data/train/haspeede_FB-train.tsv
    for file_path in paths:
        print('- Parsing data: ' + file_path)
        data = data + parse_data(file_path)
    print(data) # Vediamo i dati parsati nell'oggetto 'data'
    data = tokenize_data(data)
    #print(data)
    store_features(data, 'ciao')

# Entry point
if __name__ == '__main__':
    print(PROJECT_TITLE + ", program version: " + VERSION_NUMBER)
    get_input_files()