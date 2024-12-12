import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the pre-trained BERT model with include_top=False to exclude the final classification layer
bert = TFBertModel.from_pretrained('bert-base-uncased', include_top=False)