import pandas as pd
from tqdm import tqdm
import numpy as np
import operator
import re

tqdm.pandas()

mispell_dict = {
    'redmi': 'cell phone',
    'quorans': 'people on quora',
    'brexit': 'british exit',
    'cryptocurrencies': 'digital asset',
    'kvpy': 'indian scholarship',
    'paytm': ' e-commerce payment',
    'iiser': 'indian science education and research institutes',
    'ethereum': 'blockchain platform',
    'iisc': 'indian university',
    'jinping': 'chinese president',
    'viteee': 'indian exam',
    'iocl': 'indian oil corporation',
    'nmims': 'indian university',
    'upes': 'indian university',
    'rohingya': 'stateless district'
}


def load_and_prep():
    train = pd.read_csv("input/train.csv")
    test = pd.read_csv("input/test.csv")
    print("Train shape : ", train.shape)
    print("Test shape : ", test.shape)

    train["question_text"] = train["question_text"].str.lower()
    test["question_text"] = test["question_text"].str.lower()

    train["question_text"] = train["question_text"].apply(lambda t: clean_text(t))
    test["question_text"] = test["question_text"].apply(lambda t: clean_text(t))

    # train["question_text"] = train["question_text"].apply(lambda num: clean_numbers(num))
    # test["question_text"] = test["question_text"].apply(lambda num: clean_numbers(num))

    # # fill up the missing values
    # train = train["question_text"].fillna("_##_").values
    # test = test["question_text"].fillna("_##_").values

    train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
    sentences = train["question_text"].progress_apply(lambda x: x.split())
    # sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]

    # sentences = train["question_text"].progress_apply(lambda x: x.split()).values
    vocab = build_vocab(sentences)
    print({k: vocab[k] for k in list(vocab)[:5]})
    return vocab


def clean_text(x):
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

    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


# def clean_numbers(num):
#     num = re.sub('[0-9]{5,}', '#####', num)
#     num = re.sub('[0-9]{4}', '####', num)
#     num = re.sub('[0-9]{3}', '###', num)
#     num = re.sub('[0-9]{2}', '##', num)
#     return num


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


def replace_typical_misspell(text):
    mispellings, mispellings_re = _get_mispell(mispell_dict)

    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


def build_vocab(sentences, verbose=True):
    """
    :param verbose:
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def embedding_res():
    EMBEDDING_FILE = 'input/embeddings/glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf-8"))
    return embeddings_index


def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x


if __name__ == '__main__':
    voc = load_and_prep()
    em_idx = embedding_res()
    oov = check_coverage(voc, em_idx)
    print(oov[:50])
