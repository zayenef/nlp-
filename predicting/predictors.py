import sys
import os

# Add the project's root directory to the Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)




from myapp.features.extractors import extract_features
from myapp.embeddings.embedders import get_bert_embeddings
from myapp.trained_models.trained_model import models1, models2, meta_models


import numpy as np



def predict_traits(text):
    # Extract features from the text
    sentence_features = extract_features(text)

    # Generate word embeddings for the text
    sentence_embeddings = np.array(get_bert_embeddings(text))

    # Dictionary to store ensemble predictions for each trait
    ensemble_text_predictions = {}
    traits = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
    for trait in traits:
        # Obtain predictions from CNN and LSTM models for the new text
        cnn_text_predictions = models1[trait].predict([sentence_embeddings, sentence_features])
        lstm_text_predictions = models2[trait].predict([np.stack(sentence_embeddings), sentence_features])

        # Stack the CNN and LSTM predictions for the new text
        stacked_text_features = np.concatenate((cnn_text_predictions, lstm_text_predictions), axis=1)

        # Apply the corresponding meta-model for the trait to make predictions
        ensemble_text_predictions[trait] = meta_models[trait].predict_proba(stacked_text_features)[:, 1]

    return ensemble_text_predictions

######################################example with a text ##############################################################
#text = " I am happy and excited because i'am coming home "
#preds= predict_traits(text)
#print(preds)