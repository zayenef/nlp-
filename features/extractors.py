import string
import nltk
from nltk.corpus import stopwords, opinion_lexicon
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nrclex import NRCLex
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('opinion_lexicon')
nltk.download('wordnet')

# Instantiate the necessary tools
sid = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stopwords_set = set(stopwords.words('english'))
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

# Define the emotion words
emotion_words = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']

# Function to calculate the total number of words in the text
def word_count(text):
    words = word_tokenize(text)
    return len(words)

# Function to calculate the total number of sentences in the text
def sentence_count(text):
    sentences = sent_tokenize(text)
    return len(sentences)

# Function to calculate the average word length in the text
def average_word_length(text):
    words = word_tokenize(text)
    total_length = sum(len(word) for word in words)
    return total_length / len(words)

# Function to calculate the total number of punctuation marks in the text
def punctuation_count(text):
    count = sum(1 for char in text if char in string.punctuation)
    return count

# Function to calculate the total number of capitalized words in the text
def capitalized_word_count(text):
    words = word_tokenize(text)
    count = sum(1 for word in words if word.isupper())
    return count

# Function to calculate the total number of stopwords in the text
def stopword_count(text):
    words = word_tokenize(text)
    count = sum(1 for word in words if word.lower() in stopwords_set)
    return count

# Function to calculate the total number of unique words in the text
def unique_word_count(text):
    words = word_tokenize(text)
    unique_words = set(words)
    return len(unique_words)

# Function to calculate the total number of verbs in the text
def verb_count(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    count = sum(1 for word, pos in tagged_words if pos.startswith('VB'))
    return count

# Function to calculate the total number of nouns in the text
def noun_count(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    count = sum(1 for word, pos in tagged_words if pos.startswith('NN'))
    return count

# Function to calculate the total number of adjectives in the text
def adjective_count(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    count = sum(1 for word, pos in tagged_words if pos.startswith('JJ'))
    return count

# Function to calculate the total number of adverbs in the text
def adverb_count(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    count = sum(1 for word, pos in tagged_words if pos.startswith('RB'))
    return count

# Function to calculate the total number of pronouns in the text
def pronoun_count(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    count = sum(1 for word, pos in tagged_words if pos.startswith('PRP'))
    return count

# Function to calculate the total number of conjunctions in the text
def conjunction_count(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    count = sum(1 for word, pos in tagged_words if pos.startswith('CC'))
    return count

# Function to calculate the total number of interjections in the text
def interjection_count(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    count = sum(1 for word, pos in tagged_words if pos.startswith('UH'))
    return count

# Function to calculate the total number of emotion words in the text
def emotion_word_count(text):
    lex = NRCLex(text)
    emotion_count = lex.affect_frequencies
    emotion_word_count = sum(emotion_count[word] for word in emotion_words)
    return emotion_word_count

# Function to calculate the frequency of positive words
def positive_word_count(text):
    words = word_tokenize(text)
    count = sum(1 for word in words if word.lower() in positive_words)
    return count

# Function to calculate the frequency of negative words
def negative_word_count(text):
    words = word_tokenize(text)
    count = sum(1 for word in words if word.lower() in negative_words)
    return count

# Function to analyze pronoun usage
def analyze_pronoun_usage(text):
    pronouns = {'first_person': ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'],
                'third_person': ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs']}
    pronoun_count = {pronoun_type: 0 for pronoun_type in pronouns}
    words = word_tokenize(text)
    for word in words:
        word_lemma = lemmatizer.lemmatize(word.lower())
        for pronoun_type, pronoun_list in pronouns.items():
            if word_lemma in pronoun_list:
                pronoun_count[pronoun_type] += 1
    return pronoun_count

# Function to analyze linguistic style
def analyze_linguistic_style(text):
    words = word_tokenize(text)
    word_count_value = len(words)
    if word_count_value > 0:
        average_word_length_value = sum(len(word) for word in words) / word_count_value
    else:
        average_word_length_value = 0
    # Add more linguistic style features as needed
    return word_count_value, average_word_length_value



#######################################Emotion_Lexicon############################################################################

features_data = r"C:\Users\DELL\OneDrive\Desktop\smart_data\Emotion_Lexicon.csv"
df2 = pd.read_csv(features_data)

# Initialize empty lists for each emotion
anger = []
anticipation = []
disgust = []
fear = []
joy = []
negative = []
positive = []
sadness = []
surprise = []
trust = []
charged = []

# Iterate over each row in df2
for index, row in df2.iterrows():
    emotion = row['Words']

    # Append the words to the corresponding emotion list
    if row['anger'] > 0:
        anger.append(emotion)
    if row['anticipation'] > 0:
        anticipation.append(emotion)
    if row['disgust'] > 0:
        disgust.append(emotion)
    if row['fear'] > 0:
        fear.append(emotion)
    if row['joy'] > 0:
        joy.append(emotion)
    if row['negative'] > 0:
        negative.append(emotion)
    if row['positive'] > 0:
        positive.append(emotion)
    if row['sadness'] > 0:
        sadness.append(emotion)
    if row['surprise'] > 0:
        surprise.append(emotion)
    if row['trust'] > 0:
        trust.append(emotion)
    if row['Charged'] > 0:
        charged.append(emotion)


def count_anger(text, anger):
    count = 0
    for word in text.split():
        if word in anger:
            count += 1
    return count

def count_anticipation(text, anticipation):
    count = 0
    for word in text.split():
        if word in anticipation:
            count += 1
    return count

def count_disgust(text, disgust):
    count = 0
    for word in text.split():
        if word in disgust:
            count += 1
    return count

def count_fear(text, fear):
    count = 0
    for word in text.split():
        if word in fear:
            count += 1
    return count

def count_joy(text, joy):
    count = 0
    for word in text.split():
        if word in joy:
            count += 1
    return count

def count_negative(text, negative):
    count = 0
    for word in text.split():
        if word in negative:
            count += 1
    return count

def count_positive(text, positive):
    count = 0
    for word in text.split():
        if word in positive:
            count += 1
    return count

def count_sadness(text, sadness):
    count = 0
    for word in text.split():
        if word in sadness:
            count += 1
    return count

def count_surprise(text, surprise):
    count = 0
    for word in text.split():
        if word in surprise:
            count += 1
    return count

def count_trust(text, trust):
    count = 0
    for word in text.split():
        if word in trust:
            count += 1
    return count

def count_charged(text, charged):
    count = 0
    for word in text.split():
        if word in charged:
            count += 1
    return count


##################################extract_features function#################################################################
def extract_features(text):
    # Calculate the feature counts
    anger_count = count_anger(text, anger)
    anticipation_count = count_anticipation(text, anticipation)
    disgust_count = count_disgust(text, disgust)
    fear_count = count_fear(text, fear)
    joy_count = count_joy(text, joy)
    negative_count = count_negative(text, negative)
    positive_count = count_positive(text, positive)
    sadness_count = count_sadness(text, sadness)
    surprise_count = count_surprise(text, surprise)
    trust_count = count_trust(text, trust)
    charged_count = count_charged(text, charged)
    positive_word_count_value = positive_word_count(text)
    negative_word_count_value = negative_word_count(text)
    pronoun_usage = analyze_pronoun_usage(text)
    word_count_value, average_word_length_value = analyze_linguistic_style(text)
    sentence_count_value = sentence_count(text)
    punctuation_count_value = punctuation_count(text)
    capitalized_word_count_value = capitalized_word_count(text)
    stopword_count_value = stopword_count(text)
    unique_word_count_value = unique_word_count(text)
    verb_count_value = verb_count(text)
    noun_count_value = noun_count(text)
    adjective_count_value = adjective_count(text)
    adverb_count_value = adverb_count(text)
    pronoun_count_value = pronoun_count(text)
    conjunction_count_value = conjunction_count(text)
    interjection_count_value = interjection_count(text)
    emotion_word_count_value = emotion_word_count(text)

    # Create the feature array
    feature_array = np.array([
        anger_count, anticipation_count, disgust_count, fear_count, joy_count,
        negative_count, positive_count, sadness_count, surprise_count, trust_count,
        charged_count, positive_word_count_value, negative_word_count_value,
        pronoun_usage['first_person'], pronoun_usage['third_person'], word_count_value,
        sentence_count_value, average_word_length_value, punctuation_count_value,
        capitalized_word_count_value, stopword_count_value, unique_word_count_value,
        verb_count_value, noun_count_value, adjective_count_value, adverb_count_value,
        pronoun_count_value, conjunction_count_value, interjection_count_value,
        emotion_word_count_value
    ])

    # Reshape the feature array to have a shape of (1, 30)
    feature_array = feature_array.reshape(1, -1)

    return feature_array

####################################example#############################################################################
#text = "Yesterday was a joyful day. I went for a walk in the park and felt a sense of peace and tranquility. The beautiful flowers and chirping birds added to the serenity of the surroundings. However, suddenly, a dog appeared out of nowhere and started barking loudly. I was startled and felt a surge of fear rushing through my body. It took a few moments to calm down and regain composure. Overall, it was an eventful day with a mix of positive and negative emotions."
#sentence_features = extract_features(text)
#print(sentence_features )
#print(sentence_features.shape)