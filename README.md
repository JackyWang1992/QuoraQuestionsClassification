# Quora Insincere Questions Classificaton
This project is from [kaggle Competition](https://www.kaggle.com/c/quora-insincere-questions-classification)

## Brief Introduction
**nerual_network.py**: this is our neural network model to train and test data

**naive_bayes&decision_tree.py**: this part contains the bayes and decision tree models to train and test data

**data_analysis**.py: this part we mainly automate analysis of our data and output our tables and plots

**embeddings_analysis.py**: this part is where we analyze the embedding coverage for our neural network models

## Packages we need to install
Since we used several packages related with neural networks and linguistic computation.
you may need to pre-install them to make sure you can run our code! :)

All you have to do is run following command in terminal:
```bash
bash compile.sh
```
We care about TA as we always do! :)
## How to run our neural_network.py?
Unfortunately, since we used LSTM and GRU neural network model, both training and prediction need GPU's help :(
So we cannot make our main.py run locally without powerful GPU and we only run code here in Kaggle kernel.
But you can still browse our code and get a good idea about how we clean our data, how we implement our models and how we used our 
features to improve our model. It's very interesting and shows how we think about linguistic and deep learning.
## How to run embeddings_analysis.py?
You can run embeddings_analysis.py to see how we do data cleaning to increase embedding coverage.
But we only build a small part(500 entries) of glove embeddings since embedding files are few GB large!! 
If you want to run embedding_analysis.py, you need to download the our embedding file
 [glove.840B.300d](https://www.kaggle.com/c/quora-insincere-questions-classification/download/embeddings.zip). 
After unzip, you can put embeddings directory inside our project's input directory. And you are good to play with embeddings_analysis.py! 
## How to run others?
You can run data_analysis.py to see how we make data analysis and create beautiful plots.
Most importantly, you can run naive_bayes&decision_tree.py using command line to see the similar features as we used in our neural network and you can see the results here:
Use this command to train and predict with Naive Bayes classfier:
```bash
python3 naive_bayes&decision_tree.py 'N'
```
Use this command to train and predict with Decision Tree classfier:
```bash
python3 naive_bayes&decision_tree.py 'D'
```
We add a lot of features for both neural network and bayes/decision tree model. Although not each of them worked well, we still learn a lot 
and find the best feature which help us to rank top 15% of all teams in competition! Cheers! 

## Division of work

Shulin Chan:
1. analyzed insincere questions and created list of features to use
2. implemented bayes and decision tree classifier set up for features
3. wrote report on features and general introduction and goals
4. reviewed entire paper

Jiaqi Wang: 
1. designed the embeddings analysis alogorithm and implemented in embedding_analysis.py
2. designed the data analysis algorithm( except the n-gram analysis part) and implemented in data_analysis.py
3. employ feature engineering related with context: dirty word, comparison, exagerrated word, groups.
4. implemented features above in both neural network models and bayes/decision models
5. write papers related with what I did above
6. write Readme for our project

Minghui Zhu:
1. choose the final project topic and lead the team
2. designed and implemented codes for neural network models, LSTM, GRU, MLP.
3. revised Naive Bayes Classifier and Decision Tree Classifier
4. designed and implemented the raw-text based features:length of question, number of punctuations, number of stopwords, number of tokens, number of types and average length of tokens.
5. write the models part in paper
6. cleaning and polishing all the code, and make most of them runnable locally

Yue: 
1. designed n-gram analysis in data analysis part, implemented and visualized in data_analysis.py
2. designed features of most frequently used unigrams/bigrams/trigrams in sincere questions, and insincere questions.
3. implemented the n-gram features in neural network model, naive bayes and decision tree classifier.
4. designed the paper's structure, write part of introduction, data analysis section, feature section and conclusion 
5. polish the whole paper
