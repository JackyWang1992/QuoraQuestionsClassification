import nltk
import pandas as pd
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk import word_tokenize
import sys

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


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#',
          '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥',
          '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░',
          '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■',
          '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、',
          '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

stop_words = set(stopwords.words('english'))

def load_data():
    train_df = pd.read_csv("input/train.csv")[:10000]
    test_df = pd.read_csv("input/test.csv")[:100]
    print("Train shape : ", train_df.shape)
    print("Test shape : ", test_df.shape)

    # fill up the missing values
    train_texts = train_df["question_text"].fillna("_##_").values
    test_texts = test_df["question_text"].fillna("_##_").values

    # Get the target values
    train_label = train_df['target'].values

    # shuffling the data
    np.random.seed(2018)
    trn_idx = np.random.permutation(len(train_texts))

    train_texts = train_texts[trn_idx]
    train_label = train_label[trn_idx]

    global insincere_questions, sincere_questions
    insincere_questions = train_df[train_df["target"] == 1]
    sincere_questions = train_df[train_df["target"] == 0]

    global sin_unilist, sin_bilist, sin_trilist
    sin_unilist= sin_list(sincere_questions, 1)
    sin_bilist = sin_list(sincere_questions, 2)
    sin_trilist = sin_list(sincere_questions, 3)

    global insin_unilist, insin_bilist, insin_trilist
    insin_unilist = insin_list(insincere_questions, 1)
    insin_bilist = insin_list(insincere_questions, 2)
    insin_trilist = insin_list(insincere_questions, 3)

    return train_texts, train_label, test_texts


def create_feature_sets(texts, labels):
    val_set_size = int(len(texts) / 10)
    train_set_size = val_set_size * 9

    train_features = [wsd_features(text) for text in texts[:train_set_size]]
    val_features = [wsd_features(text) for text in texts[val_set_size:]]

    return train_features, val_features, labels[:train_set_size], labels[val_set_size:]


def wsd_features(text):
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    lm_lst = lemmalized(tokens)
    features = {}

    # length of question
    features['length'] = len(text)
    features['sin_uni'] = sin_ngrams(tokens, 1)
    features['sin_bi'] = sin_ngrams(text, 2)
    features['sin_tri'] = sin_ngrams(text, 3)
    features['insin_uni'] = insin_ngrams(tokens, 1)
    features['insin_bi'] = insin_ngrams(text, 2)
    features['insin_tri'] = insin_ngrams(text, 3)

    # number of tokens
    features['token_num'] = len(tokens)

    # groups
    features['group'] = find_groups(lm_lst)

    # exaggerated
    features['exaggerated'] = find_exaggerated(lm_lst)

    # comparison
    features['comparison'] = find_comparison(lm_lst)

    # dirty words
    features['dirty'] = find_dirty(lm_lst)

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



def lemmalized(tokens):
    lmtzr = WordNetLemmatizer()
    lm_lst = []
    for t in tokens:
        lm_lst.append(lmtzr.lemmatize(t))
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


def bayes_train_classifier(train_set):
    # create the classifier
    return nltk.NaiveBayesClassifier.train(train_set)


def decision_tree_train_classifier(train_set):
    # create the classifier
    return nltk.DecisionTreeClassifier.train(train_set)


def evaluate_classifier(classifier, val_set):
    # get the accuracy and print it
    print(nltk.classify.accuracy(classifier, val_set))


def run_classifier(classifier, test_set):
    labels = [classifier.classify(wsd_features(text)) for text in test_set]
    sub = pd.read_csv('input/output.csv')
    sub.prediction = labels
    sub.to_csv("output/output.csv", index=False)


if __name__ == '__main__':
    classifier = sys.argv[0]
    train_X, train_y, test_X = load_data()
    print('loaded data')
    train_F, val_F, train_l, val_l = create_feature_sets(train_X, train_y)
    train_set = [(train_F[i], train_l[i]) for i in range(len(train_F))]
    val_set = [(val_F[i], val_l[i]) for i in range(len(val_F))]
    print('created features')

    if classifier == 'N':
        bayes_classifier = bayes_train_classifier(train_set)
        print('trained')
        evaluate_classifier(bayes_classifier, val_set)
        run_classifier(bayes_classifier, test_X)
    else:
        decision_tree_classifier = decision_tree_train_classifier(train_set)
        print('trained')
        evaluate_classifier(decision_tree_classifier, val_set)
        run_classifier(decision_tree_classifier, test_X)
