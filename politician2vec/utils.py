import re
from collections import defaultdict
import string
import nltk
for dependency in ['punkt', 'wordnet', 'omw-1.4', 'stopwords', 'averaged_perceptron_tagger']:
    nltk.download(dependency)

def preproc_docs(text):
    #Lowercasing words
    text = text.lower()
    
    #Removing HTML tag
    text = re.sub(r'&amp', '', text)

    #Replace "&" with "and"
    text = re.sub(r'&','and', text)
    
    #Removing mentions 
    text = re.sub(r'@\w+ ', '', text)
    
    #Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation.replace('-',''))) #Taking hyphens out of punctuation to remove
    text = re.sub(r' - ','', text) #removing dash lines bounded by whitespace (and therefore not part of a word)
    text = re.sub(r'[â€˜â€™â€œâ€â€”]','',text) #removing punctuation that is not captured by string.punctuation
    
    #Removing 'RT' and 'via'
    text = re.sub(r'(^rt|^via)((?:\\b\\W*@\\w+)+): ', '', text)
    
    # Removing mentions
    text = re.sub(r'(@[A-zÃ¦Ã¸Ã¥0-9]{1,15})', '', text)
    
    #Removing odd special characters
    text = re.sub(r"[â”»â”ƒâ”â”³â”“â”â”›â”—]","", text)
    text = re.sub(r"\u202F|\u2069|\u200d|\u2066|\U0001fa86","", text)
    
    #Removing URLs
    text = re.sub(r'http\S+', '', text)
    #text = re.sub(r'https:\/\/t\.co\/[a-zA-Z0-9\-\.]+', '', text) # Shortened "https://t.co/..." Twitter URLs
    #text = re.sub(r'https:\/\/[A-z0-9?\.\/-_=!]+', '', text) # Remove other URLs

    #Removing numbers
    text = re.sub(r'[0-9.]','', text)

    # Removing idiosynchratic characters in our data
    text = re.sub(r'-\n|\n-|\na-|\nb-|â€“|Â«|--', '', text)
    text = re.sub(r'- ', ' ', text)

    #Removing separators and superfluous whitespace
    text = text.strip()
    text = re.sub(r' +',' ',text)

    #Tokenizing
    tokenizer = nltk.TweetTokenizer() 
    tokens = tokenizer.tokenize(text)

    #Removing stopwords
    stop_words_list = nltk.corpus.stopwords.words('danish')
    tokens = [i for i in tokens if i not in stop_words_list]
    
    # Removing generic Europe- and nuclear-related words
    stop_words_list = []
    stop_words_list.extend(['mink'])
    tokens = [i for i in tokens if i not in stop_words_list]

    return tokens