import re
import arabicstopwords.arabicstopwords as ast
from tashaphyne.stemming import ArabicLightStemmer

punctuations_list = '''^_-`$%&÷×؛<=>()*&^%][،/;:"؟.,'{}~¦+|!”…“'''
#english_punctuations = string.punctuation
#print(english_punctuations)
#punctuations_list = arabic_punctuations

def remove_punctuations(text):
    translator = str.maketrans(punctuations_list, ' '*len(punctuations_list))
    return text.translate(translator)
    
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text) # no emoji

# a small function to remove stop words
def remove_stop_words(text):
    #getting a stopwords_list
    stop_words = ast.stopwords_list()
    return ' '.join(word for word in text.split() if word not in stop_words)

# a small function to lemmatize the text
def lemmatiz_word(text):
    # lemmer = qalsadi.lemmatizer.Lemmatizer()
    # return ' '.join(lemmer.lemmatize(word) for word in text.split()) 
    #---------
    #st = ISRIStemmer()
    #return ' '.join(st.stem(word) for word in text.split())
    #---------
    ArListem = ArabicLightStemmer()
    return ' '.join(ArListem.light_stem(word) for word in text.split())
    
def processPost(tweet): 

    #Remove <LF> from tweet
    tweet = re.sub('<LF>', ' ', tweet)
    
    #Replace @username with empty string
    tweet = re.sub('@[^\s]+', ' ', tweet)
    
    #remove url
    tweet = re.sub('(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})',' ',tweet)
    
    #remove hashtage #
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    # remove punctuations
    tweet = remove_punctuations(tweet)
    
    # remove emoji  (not sure with this step)
    tweet = remove_emoji(tweet)
    
    # normalize the tweet
    # tweet= normalize_arabic(tweet)
    
    # remove repeated letters
    tweet=re.sub(r'(.)\1+', r'\1', tweet)

    #remove stop words
    tweet = remove_stop_words(tweet)

    tweet=lemmatiz_word(tweet)
    
    return tweet