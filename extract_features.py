import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from pyarabic.araby import tokenize

# get unigram features TF-IDF
def get_unigram_features(train_data):
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),max_features=1000,token_pattern=r"(?u)\b[أ-ي]*\b")

    unigramdataGet = word_vectorizer.fit_transform(train_data['text'].astype('str'))
    unigramdataGet = unigramdataGet.toarray()

    vocab = word_vectorizer.get_feature_names_out()
    unigramdata_features=pd.DataFrame(np.round(unigramdataGet, 1), columns=vocab)
    unigramdata_features[unigramdata_features>0] = 1
    return unigramdata_features, word_vectorizer, vocab

# get word embedding features
def get_word_embedding_features(train_data, test_data):
    train_data_tokenized=train_data["text"].apply(tokenize)
    w2v_model = Word2Vec(train_data_tokenized, vector_size=100, window=5, min_count=1, workers=4)

    # Replace the words in each text message with the learned word vector
    words = set(w2v_model.wv.index_to_key)
    X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                            for ls in train_data["text"]])
    X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                            for ls in test_data['text']])

    # Average the word vectors for each sentence (and assign a vector of zeros if the model
    # did not learn any of the words in the text message during training
    X_train_vect_avg = []
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(100, dtype=float))
            
    X_test_vect_avg = []
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(100, dtype=float))

    return X_train_vect_avg, X_test_vect_avg