# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import nltk

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

## some config values
embed_size = 300  # how big is each word vector
max_features = 95000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70  # max number of words in a question to use
extra_features = 1  # num of extra features

import os
import time
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import concatenate
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams
from nltk import word_tokenize
from collections import defaultdict


# import re

def load_and_prec():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    print("Train shape : ", train.shape)
    print("Test shape : ", test.shape)

    # data cleaning: remove punctuations
    train["question_text"] = train["question_text"].apply(lambda x: clean_text(x))
    test["question_text"] = test["question_text"].apply(lambda x: clean_text(x))

    # fill up the missing values
    train_X = train["question_text"].fillna("_##_").values
    test_X = test["question_text"].fillna("_##_").values

    train_text = train_X
    test_text = test_X

    # Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    train_sin = train[train["target"] == 0]
    global sin_unilist, sin_bilist, sin_trilist
    sin_unilist = sin_list(train_sin, 1)
    sin_bilist = sin_list(train_sin, 2)
    sin_trilist = sin_list(train_sin, 3)

    global insin_unilist, insin_bilist, insin_trilist
    train_insin = train[train["target"] == 1]
    insin_unilist = insin_list(train_insin, 1)
    insin_bilist = insin_list(train_insin, 2)
    insin_trilist = insin_list(train_insin, 3)


    train_features = extract_features(train, train_text, train_X)
    test_features = extract_features(test, test_text, test_X)

    # Pad the sentences
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    # Get the target values
    train_y = train['target'].values

    # shuffling the data
    np.random.seed(2018)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]

    return train_X, test_X, train_y, tokenizer.word_index, train_features, test_features


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#',
          '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥',
          '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░',
          '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■',
          '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、',
          '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

comparison = ['better', 'worse', 'than', 'more', 'less', 'compared', 'between', 'among', 'whereas', 'contrast']

exaggerated = ['best', 'worst', 'super', 'extreme', 'most', 'least', 'never', 'no way', 'impossible', 'no chance',
               'must']

groups = ['united states of america', 'afghanistan', 'albania', 'algeria', 'andorra', 'angola', 'antigua & deps',
          'argentina', 'armenia', 'australia', 'austria', 'azerbaijan', 'bahamas', 'bahrain', 'bangladesh', 'barbados',
          'belarus', 'belgium', 'belize', 'benin', 'bhutan', 'bolivia', 'bosnia herzegovina', 'botswana', 'brazil',
          'brunei', 'bulgaria', 'burkina', 'burma', 'burundi', 'cambodia', 'cameroon', 'canada', 'cape verde',
          'central african rep', 'chad', 'chile', 'china', 'republic of china', 'colombia', 'comoros',
          'democratic republic of the congo', 'republic of the congo', 'costa rica', 'croatia', 'cuba', 'cyprus',
          'czech republic', 'danzig', 'denmark', 'djibouti', 'dominica', 'dominican republic', 'east timor', 'ecuador',
          'egypt', 'el salvador', 'equatorial guinea', 'eritrea', 'estonia', 'ethiopia', 'fiji', 'finland', 'france',
          'gabon', 'gaza strip', 'the gambia', 'georgia', 'germany', 'ghana', 'greece', 'grenada', 'guatemala',
          'guinea', 'guinea-bissau', 'guyana', 'haiti', 'holy roman empire', 'honduras', 'hungary', 'iceland', 'india',
          'indonesia', 'iran', 'iraq', 'republic of ireland', 'israel', 'italy', 'ivory coast', 'jamaica', 'japan',
          'jonathanland', 'jordan', 'kazakhstan', 'kenya', 'kiribati', 'north korea', 'south korea', 'kosovo', 'kuwait',
          'kyrgyzstan', 'laos', 'latvia', 'lebanon', 'lesotho', 'liberia', 'libya', 'liechtenstein', 'lithuania',
          'luxembourg', 'macedonia', 'madagascar', 'malawi', 'malaysia', 'maldives', 'mali', 'malta',
          'marshall islands', 'mauritania', 'mauritius', 'mexico', 'micronesia', 'moldova', 'monaco', 'mongolia',
          'montenegro', 'morocco', 'mount athos', 'mozambique', 'namibia', 'nauru', 'nepal', 'newfoundland',
          'netherlands', 'new zealand', 'nicaragua', 'niger', 'nigeria', 'norway', 'oman', 'ottoman empire', 'pakistan',
          'palau', 'panama', 'papua new guinea', 'paraguay', 'peru', 'philippines', 'poland', 'portugal', 'prussia',
          'qatar', 'romania', 'rome', 'russian federation', 'rwanda', 'st kitts & nevis', 'st lucia', 'saint vincent',
          'grenadines', 'samoa', 'san marino', 'sao tome & principe', 'saudi arabia', 'senegal', 'serbia', 'seychelles',
          'sierra leone', 'singapore', 'slovakia', 'slovenia', 'solomon islands', 'somalia', 'south africa', 'spain',
          'sri lanka', 'sudan', 'suriname', 'swaziland', 'sweden', 'switzerland', 'syria', 'tajikistan', 'tanzania',
          'thailand', 'togo', 'tonga', 'trinidad & tobago', 'tunisia', 'turkey', 'turkmenistan', 'tuvalu, uganda',
          'ukraine', 'united arab emirates', 'united kingdom', 'uruguay', 'uzbekistan', 'vanuatu', 'vatican city',
          'venezuela', 'vietnam', 'yemen', 'zambia', 'zimbabwe', 'chinese', 'english', 'american', 'indian', 'japanese',
          'korean', 'british', 'spanish', 'austrian', 'cuban', 'canadian', 'french', 'germany', 'italian', 'muslim',
          'christian', 'buddhist']

dirty = ['ass', 'assfucker', 'asshole', 'assholes', 'asswhole', 'ballbag', 'balls', 'ballsack', 'bastard', 'beastial',
         'beastiality', 'bellend', 'bestial', 'bestiality', 'bi+ch', 'biatch', 'bitch', 'bitcher', 'bitchers',
         'bitches', 'bitchin', 'bitching', 'bloody', 'blow job', 'blowjob', 'blowjobs', 'boiolas', 'bollock', 'bollok',
         'boner', 'boob', 'breasts', 'buceta', 'bugger', 'bum', 'bunny fucker', 'butt', 'butthole', 'buttmuch',
         'buttplug', 'c0ck', 'c0cksucker', 'carpet muncher', 'cawk', 'chink', 'cipa', 'cl1t', 'clit', 'clitoris',
         'clits', 'cnut', 'cock', 'cock-sucker', 'cockface', 'cockhead', 'cockmunch', 'cockmuncher', 'cocks',
         'cocksuck', 'cocksucked', 'cocksucker', 'cok', 'cokmuncher', 'coksucka', 'coon', 'cox', 'crap', 'cum',
         'cumshot', 'cunilingus', 'cunt', 'cuntlick', 'cyalis', 'cyberfuck', 'damn', 'dick', 'dickhead', 'dildo',
         'dink', 'dog-fucker', 'donkeyribber', 'doosh', 'duche', 'dyke', 'ejaculate', 'fag', 'fagot', 'fanny',
         'fannyflaps', 'fannyfucker', 'fanyy', 'fatass', 'fellate', 'fellatio', 'fingerfuck', 'fistfuck', 'flange',
         'fook', 'fuck', 'fudge packer', 'fudgepacker', 'gangbang', 'gaylord', 'gaysex', 'goddamn', 'hardcoresex',
         'hell', 'hore', 'horny', 'hotsex', 'jackoff', 'jap', 'jism', 'jiz', 'kawk', 'knob', 'kock', 'kondum', 'kum',
         'kunilingus', 'labia', 'lmfao', 'lust', 'masochist', 'masterbate', 'mofo', 'mother fucker', 'motherfuck',
         'motherfucker', 'muff', 'nigga', 'nigger', 'nob', 'numbnuts', 'nutsack', 'orgasim', 'pawn', 'pecker', 'penis',
         'penisfucker', 'phonesex', 'pigfucker', 'piss', 'poop', 'porn', 'pornography', 'pussi', 'pussy', 'sex', 'shit',
         'shitdick', 'shite', 'shitfuck', 'shitfull', 'shithead', 'shitty', 'skank', 'slut', 'smegma', 'smut', 'snatch',
         'son-of-a-bitch', 'teets', 'teez', 'tit', 'titfuck', 'twat', 'twathead', 'vagina', 'viagra', 'vulva', 'wang',
         'wank', 'whoar', 'whore', 'willies', 'xrated', 'xxx']

stop_words = set(stopwords.words('english'))


def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')

    # for stop_word in stop_words:
    #     x = x.replace(stop_word, '')
    return x


def lemmalized(tokens):
    lmtzr = WordNetLemmatizer()
    lm_lst = []
    for t in tokens:
        t_lst = []
        for w in t:
            w = str(w)
            t_lst.append(lmtzr.lemmatize(w))
        lm_lst.append(t_lst)
    return lm_lst


def find_comparison(lst):
    for w in lst:
        if w.lower() in comparison:
            return 1
        else:
            return 0


def find_exaggerated(lst):
    for w in lst:
        if w.lower() in exaggerated:
            return 1
        else:
            return 0


def find_groups(lst):
    nums = 0
    for w in lst:
        if w.lower() in groups:
            nums = nums + 1
    return nums


def find_dirty(lst):
    nums = 0
    for w in lst:
        if w.lower() in dirty:
            nums = nums + 1
    return nums


def extract_features(train, texts, tokens):
    features = np.zeros((len(tokens), 1))
    lm_lst = lemmalized(tokens)
    for i in range(len(tokens)):
        num = 0
        for j in tokens[i]:
            if j in stop_words:
                num += 1
        # features[i][0] = num
        # features[i][0] = find_comparison(lm_lst[i])
        # features[i][1] = find_exaggerated(lm_lst[i])
        # features[i][2] = find_groups(lm_lst[i])
        # features[i][0] = find_dirty(lm_lst[i])
        # features[i][0] = sin_ngrams(tokens[i], 1)
        # features[i][1] = sin_ngrams(texts[i], 2)
        # features[i][2] = sin_ngrams(texts[i], 3)
        # features[i][3] = insin_ngrams(tokens[i], 1)
        # features[i][4] = insin_ngrams(texts[i], 2)
        # features[i][5] = insin_ngrams(texts[i], 3)

    return features


def sin_ngrams(question, number):
    count = 0
    if number == 1:
        for w in question:
            if w in sin_unilist:
                count = count + 1
        return count
    if number == 2:
        token = nltk.word_tokenize(question)
        bigrams = ngrams(token, 2)
        bigrams = [str(w[0] + " " + w[1]) for w in bigrams]
        for w in bigrams:
            if w in sin_bilist:
                count = count + 1
        return count
    if number == 3:
        token = nltk.word_tokenize(question)
        trigrams = ngrams(token, 3)
        trigrams = [str(w[0] + " " + w[1] + " " + w[2]) for w in trigrams]
        for w in trigrams:
            if w in sin_trilist:
                count = count + 1
        return count
    return 0


def insin_ngrams(question, number):
    count = 0
    if number == 1:
        for w in question:
            if w in insin_unilist:
                count = count + 1
        return count
    if number == 2:
        token = nltk.word_tokenize(question)
        bigrams = ngrams(token, 2)
        bigrams = [str(w[0] + " " + w[1]) for w in bigrams]
        for w in bigrams:
            if w in insin_bilist:
                count = count + 1
        return count
    if number == 3:
        token = nltk.word_tokenize(question)
        trigrams = ngrams(token, 3)
        trigrams = [str(w[0] + " " + w[1] + " " + w[2]) for w in trigrams]
        for w in trigrams:
            if w in insin_trilist:
                count = count + 1
        return count
    return 0


def generate_ngrams(text, n_gram):
    token = [token for token in text.lower().split(" ") if token != "" if token not in stop_words if
             token not in puncts]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]


def sin_list(train_sin, n_gram):
    freq_dict = defaultdict(int)
    for sent in train_sin["question_text"]:
        for word in generate_ngrams(sent, n_gram):
            freq_dict[word] += 1
    sin_list = sorted(freq_dict.items(), key=lambda x: x[1])[::-1]
    sin_list = [w[0] for w in sin_list][:100]
    return set(sin_list)


def insin_list(train_insin, n_gram):
    freq_dict = defaultdict(int)
    for sent in train_insin["question_text"]:
        for word in generate_ngrams(sent, n_gram):
            freq_dict[word] += 1
    insin_list = sorted(freq_dict.items(), key=lambda x: x[1])[::-1]
    insin_list = [w[0] for w in insin_list][:100]
    return set(insin_list)

def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf-8"))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

        def build(self, input_shape):
            assert len(input_shape) == 3

            self.W = self.add_weight((input_shape[-1],),
                                     initializer=self.init,
                                     name='{}_W'.format(self.name),
                                     regularizer=self.W_regularizer,
                                     constraint=self.W_constraint)
            self.features_dim = input_shape[-1]

            if self.bias:
                self.b = self.add_weight((input_shape[1],),
                                         initializer='zero',
                                         name='{}_b'.format(self.name),
                                         regularizer=self.b_regularizer,
                                         constraint=self.b_constraint)
            else:
                self.b = None

            self.built = True

        def compute_mask(self, input, input_mask=None):
            return None

        def call(self, x, mask=None):
            features_dim = self.features_dim
            step_dim = self.step_dim

            eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                                  K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

            if self.bias:
                eij += self.b

            eij = K.tanh(eij)

            a = K.exp(eij)

            if mask is not None:
                a *= K.cast(mask, K.floatx())

            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

            a = K.expand_dims(a)
            weighted_input = x * a
            return K.sum(weighted_input, axis=1)

        def compute_output_shape(self, input_shape):
            return input_shape[0], self.features_dim


def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)

    atten_1 = Attention(maxlen)(x)  # skip connect
    atten_2 = Attention(maxlen)(y)
    avg_pool_1 = GlobalAveragePooling1D()(atten_1)
    max_pool_1 = GlobalMaxPooling1D()(atten_1)

    avg_pool_2 = GlobalAveragePooling1D()(atten_2)
    max_pool_2 = GlobalMaxPooling1D()(atten_2)

    # a = Dense(16, activation="relu")(atten_1)

    conc = concatenate([avg_pool_1, max_pool_1, avg_pool_2, max_pool_2])
    conc = Dense(16, activation="relu")(conc)

    inp2 = Input(shape=(extra_features,))
    ex = Dense(16, activation="relu")(inp2)
    ex = Dropout(0.1)(conc)
    ex = Dense(16, activation="relu")(ex)
    # ex = Dropout(0.1)(conc)
    # ex = Dense(16, activation="relu")(ex)

    conc = concatenate([conc, ex])
    # conc = Dropout(0.1)(conc)
    # conc = Dense(16, activation="relu")(conc)

    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=[inp, inp2], outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    return model


def train_pred(model, train_X, train_y, val_X, val_y, train_F, val_F, epochs=2, callback=None):
    for e in range(epochs):
        model.fit([train_X, train_F], train_y, batch_size=512, epochs=1, validation_data=([val_X, val_F], val_y),
                  callbacks=callback, verbose=0)
        pred_val_y = model.predict([val_X, val_F], batch_size=1024, verbose=0)

        best_score = metrics.f1_score(val_y, (pred_val_y > 0.33).astype(int))
        print("Epoch: ", e, "-    Val F1 Score: {:.4f}".format(best_score))

    pred_test_y = model.predict([test_X, test_features], batch_size=1024, verbose=0)
    print('=' * 60)
    return pred_val_y, pred_test_y, best_score


if __name__ == '__main__':

    train_X, test_X, train_y, word_index, train_features, test_features = load_and_prec()
    embedding_matrix = load_glove(word_index)

    np.shape(embedding_matrix)


    def threshold_search(y_true, y_proba):
        best_threshold = 0
        best_score = 0
        for threshold in [i * 0.01 for i in range(100)]:
            score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
            if score > best_score:
                best_threshold = threshold
                best_score = score
        search_result = {'threshold': best_threshold, 'f1': best_score}
        return search_result


    DATA_SPLIT_SEED = 2018
    # clr = CyclicLR(base_lr=0.001, max_lr=0.002,
    #                step_size=300., mode='exp_range',
    #                gamma=0.99994)

    train_meta = np.zeros(train_y.shape)
    test_meta = np.zeros(test_X.shape[0])
    splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=DATA_SPLIT_SEED).split(train_X, train_y))
    for idx, (train_idx, valid_idx) in enumerate(splits):
        X_train = train_X[train_idx]
        y_train = train_y[train_idx]
        X_val = train_X[valid_idx]
        y_val = train_y[valid_idx]
        F_train = train_features[train_idx]
        F_val = train_features[valid_idx]
        model = model_lstm_atten(embedding_matrix)
        pred_val_y, pred_test_y, best_score = train_pred(model, X_train, y_train, X_val, y_val, F_train, F_val,
                                                         epochs=8,
                                                         callback=None)
        train_meta[valid_idx] = pred_val_y.reshape(-1)
        test_meta += pred_test_y.reshape(-1) / len(splits)

    sub = pd.read_csv('../input/sample_submission.csv')
    sub.prediction = test_meta > 0.33
    sub.to_csv("submission.csv", index=False)

    f1_score(y_true=train_y, y_pred=train_meta > 0.33)
