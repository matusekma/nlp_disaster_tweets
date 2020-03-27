# Applying a first round of text cleaning techniques
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
import nltk
import re
import string
from bs4 import BeautifulSoup
from textblob import TextBlob


def clean_text(text):
    text = BeautifulSoup(text, 'lxml').get_text()
    eyes = "[8:=;]"
    nose = "['`\-]?"
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " ", text)

    text = re.sub("/", " / ", text)
    text = re.sub('@(\w+)', '', text)

    text = re.sub('#{eyes}#{nose}[)d]+|[)d]+#{nose}#{eyes}', "<smile>", text)
    text = re.sub('#{eyes}#{nose}p+', "<lolface>", text)
    text = re.sub('#{eyes}#{nose}\(+|\)+#{nose}#{eyes}', "<sadface>", text)
    text = re.sub('#{eyes}#{nose}[\/|l*]', "<neutralface>", text)
    text = re.sub('<3', "<heart>", text)
    # numbers
    text = re.sub('[-+]?[.\d]*[\d]+[:,.\d]*', " ", text)

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    # text = re.sub('\[.*?\]', '', text)
    # text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(
        string.punctuation.replace('<', '').replace('>', '')), ' ', text)
    text = re.sub('\n', ' ', text)

    # text = re.sub(r"[^a-zA-Z]", ' ', text)
    text = ''.join(filter(lambda x: x in string.printable, text))
    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

    # text = re.sub('\w*\d\w*', '', text)

    return text


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def text_preprocessing(text):
    tokenizer = nltk.tokenize.TweetTokenizer(
        strip_handles=True, reduce_len=True)

    lemmatizer = nltk.stem.WordNetLemmatizer()

    nopunc = clean_text(text)

    corrected = str(TextBlob(nopunc).correct())

    tokenized_text = tokenizer.tokenize(corrected)

    remove_stopwords = [
        w for w in tokenized_text if w not in stopwords.words('english')]

    lemmatized = [lemmatizer.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else lemmatizer.lemmatize(i)
                  for i, j in pos_tag(remove_stopwords)]

    combined_text = ' '.join(lemmatized)
    return combined_text


def clean_text_no_smiley(text):
    text = BeautifulSoup(text, 'lxml').get_text()
    eyes = "[8:=;]"
    nose = "['`\-]?"
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " ", text)

    text = re.sub("/", " / ", text)
    text = re.sub('@(\w+)', '', text)

    text = re.sub('#{eyes}#{nose}[)d]+|[)d]+#{nose}#{eyes}', "<smile>", text)
    text = re.sub('#{eyes}#{nose}p+', "<lolface>", text)
    text = re.sub('#{eyes}#{nose}\(+|\)+#{nose}#{eyes}', "<sadface>", text)
    text = re.sub('#{eyes}#{nose}[\/|l*]', "<neutralface>", text)
    text = re.sub('<3', "<heart>", text)
    # numbers
    text = re.sub('[-+]?[.\d]*[\d]+[:,.\d]*', " ", text)

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    # text = re.sub('\[.*?\]', '', text)
    # text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(
        string.punctuation.replace('<', '').replace('>', '')), ' ', text)
    text = re.sub('\n', ' ', text)

    # text = re.sub(r"[^a-zA-Z]", ' ', text)
    text = ''.join(filter(lambda x: x in string.printable, text))
    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

    # text = re.sub('\w*\d\w*', '', text)

    return text


def text_preprocessing_no_lemmatizer(text):
    tokenizer = nltk.tokenize.TweetTokenizer(
        strip_handles=True, reduce_len=True)

    lemmatizer = nltk.stem.WordNetLemmatizer()

    nopunc = clean_text(text)

    corrected = str(TextBlob(nopunc).correct())

    tokenized_text = tokenizer.tokenize(corrected)

    remove_stopwords = [
        w for w in tokenized_text if w not in stopwords.words('english')]

    # lemmatized = [lemmatizer.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else lemmatizer.lemmatize(i)
    # for i, j in pos_tag(remove_stopwords)]

    combined_text = ' '.join(remove_stopwords)
    return combined_text
