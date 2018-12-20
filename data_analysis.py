# Usual Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import operator
import textstat
import warnings
import os
import nltk
from nltk.corpus import stopwords
from collections import defaultdict

# Plotly based imports for visualization
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

# spaCy based imports
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# ---------------------------------------- Data Preparation ---------------------------------------
# plotly account info
plotly.tools.set_credentials_file(username='JackyWang1992', api_key='xhgyjvYlY4v2e3wiPAOb')

warnings.filterwarnings("ignore")

print(os.listdir("input"))
quora_train = pd.read_csv("input/train.csv")

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#',
          '*', '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â',
          '█', '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―',
          '¥', '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸',
          '¾', 'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
          '¹', '≤', '‡', '√', ]

quora_train["question_text"] = quora_train["question_text"].apply(lambda x: clean_text(x))

global insincere, sincere
insincere = quora_train[quora_train["target"] == 1]
sincere = quora_train[quora_train["target"] == 0]



stop_words = set(stopwords.words('english'))


def read_data_header():
    quora_train.head()


# SpaCy Parser for questions
punctuations = string.punctuation
stopwords = list(STOP_WORDS)

parser = English()


def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


# apply tokenizer and show the progress bar
tqdm.pandas()
sincere_questions = quora_train["question_text"][quora_train["target"] == 0].progress_apply(spacy_tokenizer)
insincere_questions = quora_train["question_text"][quora_train["target"] == 1].progress_apply(spacy_tokenizer)


# ---------------------------------------------- Data Analysis ----------------------------------------------------
# the function to build our table of data analysis
def build_table(lst_1, lst_2):
    mean_1 = np.mean(lst_1)
    std_1 = np.std(lst_1)
    md_1 = np.median(lst_1)
    mx_1 = np.max(lst_1)
    mn_1 = np.min(lst_1)

    mean_2 = np.mean(lst_2)
    std_2 = np.std(lst_2)
    md_2 = np.median(lst_2)
    mx_2 = np.max(lst_2)
    mn_2 = np.min(lst_2)
    # create tables
    data_matrix = [['Statistical Measures', 'Insincere Question', 'Sincere Question'],
                   ['Mean', mean_1, mean_2],
                   ['Standard Deviation', std_1, std_2],
                   ['Median', md_1, md_2],
                   ['Maximum', mx_1, mx_2],
                   ['Minimum', mn_1, mn_2]]
    table = ff.create_table(data_matrix)
    return table


# syllable analysis
def build_syllable_dct(lst):
    dct = {}
    for t in lst:
        if textstat.syllable_count(t) in dct:
            dct[textstat.
                syllable_count(t)] = dct[textstat.syllable_count(t)] + 1
        else:
            dct[textstat.syllable_count(t)] = 1
    for key in dct.keys():
        dct[key] = dct[key] / len(lst)
    sorted_tuple = sorted(dct.items(), key=operator.itemgetter(0))
    return sorted_tuple


def syllable_plot():
    sorted_tuple_1 = build_syllable_dct(insincere_questions.tolist())
    sorted_tuple_2 = build_syllable_dct(sincere_questions.tolist())

    # Create traces
    trace0 = go.Scatter(
        x=[x[0] for x in sorted_tuple_1],
        y=[y[1] for y in sorted_tuple_1],
        mode='lines',
        name='insincere questions'
    )
    trace1 = go.Scatter(
        x=[x[0] for x in sorted_tuple_2],
        y=[y[1] for y in sorted_tuple_2],
        mode='lines',
        name='sincere questions'
    )
    data = [trace0, trace1]
    py.plot(data, filename='syllable analysis')


def syllable_table():
    lst_1 = [textstat.syllable_count(text) for text in insincere_questions.tolist()]
    lst_2 = [textstat.syllable_count(text) for text in sincere_questions.tolist()]
    table = build_table(lst_1, lst_2)
    py.plot(table, filename='syllable_table')


# lexicon analysis
def build_lexicon_dict(lst):
    dct = {}
    for t in lst:
        if textstat.lexicon_count(t) in dct:
            dct[textstat.lexicon_count(t)] = dct[textstat.lexicon_count(t)] + 1
        else:
            dct[textstat.lexicon_count(t)] = 1
    for key in dct.keys():
        dct[key] = dct[key] / len(lst)
    sorted_tuple = sorted(dct.items(), key=operator.itemgetter(0))
    return sorted_tuple


def lexicon_plot():
    sorted_tuple_1 = build_lexicon_dict(insincere_questions.tolist())
    sorted_tuple_2 = build_lexicon_dict(sincere_questions.tolist())

    # Create traces
    trace0 = go.Scatter(
        x=[x[0] for x in sorted_tuple_1],
        y=[y[1] for y in sorted_tuple_1],
        mode='lines',
        name='insincere questions'
    )
    trace1 = go.Scatter(
        x=[x[0] for x in sorted_tuple_2],
        y=[y[1] for y in sorted_tuple_2],
        mode='lines',
        name='sincere questions'
    )
    data = [trace0, trace1]
    py.plot(data, filename='lexicon analysis')


def lexicon_table():
    lst_1 = [textstat.lexicon_count(text) for text in insincere_questions.tolist()]
    lst_2 = [textstat.lexicon_count(text) for text in sincere_questions.tolist()]
    table = build_table(lst_1, lst_2)
    py.plot(table, filename='lexicon_table')


# Question Length Analysis
def question_length_dict(lst):
    dct = {}
    for t in lst:
        if len(t) in dct:
            dct[len(t)] = dct[len(t)] + 1
        else:
            dct[len(t)] = 1
    for key in dct.keys():
        dct[key] = dct[key] / len(lst)
    sorted_tuple = sorted(dct.items(), key=operator.itemgetter(0))
    return sorted_tuple


def question_length_plot():
    sorted_tuple_1 = question_length_dict(insincere_questions.tolist())
    sorted_tuple_2 = question_length_dict(sincere_questions.tolist())

    # Create traces
    trace0 = go.Scatter(
        x=[x[0] for x in sorted_tuple_1],
        y=[y[1] for y in sorted_tuple_1],
        mode='lines',
        name='insincere questions'
    )
    trace1 = go.Scatter(
        x=[x[0] for x in sorted_tuple_2],
        y=[y[1] for y in sorted_tuple_2],
        mode='lines',
        name='sincere questions'
    )
    data = [trace0, trace1]
    py.plot(data, filename='question length analysis')


def question_length_table():
    lst_1 = [len(text) for text in insincere_questions.tolist()]
    lst_2 = [len(text) for text in sincere_questions.tolist()]
    table = build_table(lst_1, lst_2)
    py.plot(table, filename='question_length_table')


# ---------------------------- N-gram analysis ---------------------------------------------------
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in stop_words if token not in puncts]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]


def ngram_plot(df, subtitle):
    freq_dict = defaultdict(int)
    for sent in df["question_text"]:
        for word in generate_ngrams(sent):
            freq_dict[word] += 1

    fd_sorted = sorted(freq_dict.items(), key=lambda x: x[1])[::-1]
    x = [w[0] for w in fd_sorted][:30]
    y = [w[1] for w in fd_sorted][:30]

    plt.rcdefaults()
    x = tuple(x)
    y_pos = np.arange(30)

    plt.barh(y_pos, y, align='center', alpha=0.7)
    plt.yticks(y_pos, x)
    plt.ylabel(subtitle)
    plt.title('Most used ' + subtitle + ' in insincere questions')

    plt.show()


def top_sincere_words():
    # Top Sincere words
    ngram_plot(sincere, "words")


def top_insincere_words():
    # Top Insincere words
    ngram_plot(insincere, "words")


def top_sincere_bigrams():
    ngram_plot(sincere, "bigrams")


def top_insincere_bigrams():
    ngram_plot(insincere, "bigrams")


def top_sincere_trigrams():
    ngram_plot(sincere, "trigrams")


def top_insincere_trigrams():
    ngram_plot(insincere, "trigrams")


if __name__ == '__main__':
    syllable_plot()
    syllable_table()
    lexicon_plot()
    lexicon_table()
    question_length_plot()
    question_length_table()
    top_sincere_words()
    top_insincere_words()
    top_sincere_bigrams()
    top_insincere_bigrams()
    top_sincere_trigrams()
    top_insincere_trigrams()
