import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

def get_bert_embeddings(text, batch_size=32):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertModel.from_pretrained('bert-base-cased')

    # Set maximum sequence length
    max_length = 512

    # Tokenize and encode the input text
    encoded_data = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)

    # Pad the sequence to the same length
    padded_data = tf.keras.preprocessing.sequence.pad_sequences([encoded_data], padding='post', maxlen=max_length)

    # Compute BERT embeddings for the padded sequence
    embeddings = model(padded_data)[0]

    return embeddings

##############################execution example##############################
#text = "Yesterday was a joyful day. I went for a walk in the park and felt a sense of peace and tranquility. The beautiful flowers and chirping birds added to the serenity of the surroundings. However, suddenly, a dog appeared out of nowhere and started barking loudly. I was startled and felt a surge of fear rushing through my body. It took a few moments to calm down and regain composure. Overall, it was an eventful day with a mix of positive and negative emotions."
#embeddings = get_bert_embeddings(text)
#print(embeddings)
#print(embeddings.shape)