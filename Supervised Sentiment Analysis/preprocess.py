import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def preprocess(review_lst):
    docs = []
    labels = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    for line in review_lst:
       sent = line.lower()
       labels.append(int(sent[-2])) # extracting labels 1 or 0 
       sent = sent[:len(sent)-2]
       alnum_string = re.sub(r"[^a-zA-Z0-9]"," ",sent) # removing any special characters.
       filtered_string = ""
       for word in alnum_string.split():
           if word not in stop_words:    # removing stopwords
               filtered_string+=word+" "
       lemmatised_string = ""
       for word in filtered_string:
           lemmatised_string+= lemmatizer.lemmatize(word)   # lemmatizing
       string_lst = word_tokenize(lemmatised_string) # tokenize
       docs.append(string_lst)
    return docs,labels
 
def listToString(s): 

    string = " " 
    return (string.join(s))


def string_lst(doc):
    lst = []
    for i in range(len(doc)):
        string = listToString(doc[i])
        lst.append(string)
    return lst