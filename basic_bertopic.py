# Basic Bertopic using trip_advisor reviews of hotels

import os
import torch
import logging
import warnings
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import TextGeneration
from bertopic.representation import PartOfSpeech
from bertopic.vectorizers import ClassTfidfTransformer
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

from utils import loading_utils

# Use pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12==24.6.*pip install bertopic
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

logging.getLogger("spacy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="distutils Version classes are deprecated.*")
logging.getLogger("BERTopic").disabled = True

REQUIRED_COLUMNS = ["Review"]
path = "./data/hotel_reviews.csv"
dataset_subset = "all"
ds_path = f"./data/dataset_trip_advisor_{dataset_subset}"

ds = loading_utils.load_local_file(path=path, type="csv")
df = pd.DataFrame(data=ds)
df = (
    df.drop_duplicates(
        subset=REQUIRED_COLUMNS,
    )
    .dropna(subset=REQUIRED_COLUMNS)
    .reset_index(drop=True)
)

data = df["Review"].tolist()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Do embeddings in batches
import numpy as np
from tqdm.auto import tqdm

batch_size = 16
n = len(data)
embeds = np.zeros((n, embedding_model.get_sentence_embedding_dimension()))

for i in tqdm(range(0, n, batch_size)):
    i_end = min(i + batch_size, n)
    batch = data[i:i_end]
    batch_embed = embedding_model.encode(batch)
    embeds[i:i_end, :] = batch_embed

umap_model = UMAP(n_neighbors=3, n_components=3, min_dist=0.05)
hdbscan_model = HDBSCAN(
    min_cluster_size=15, min_samples=20, gen_min_span_tree=True, prediction_data=True
)

# # Note
#  The effect of reducing the term matrix size in BERTopic depends on how aggressively you reduce it and the characteristics of your data. While it can make the model faster and more efficient, it can also reduce topic quality, making the model less sensitive to subtle differences between topics.
#  It’s a trade-off between computational efficiency and model interpretability
stopwords = list(stopwords.words("english")) + ["http", "https", "amp", "com"]

# We add this to remove stopwords that can pollute topcs
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords)
mmr = MaximalMarginalRelevance(diversity=0.5)  # maximise diverstity of keyord selection
kbi = KeyBERTInspired()

# Prompt
prompt = """I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: '[KEYWORDS]'.
Based on the documents and keywords, what is this topic about?"""

ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=[mmr, kbi],
    language="english",
    # top_n_words=5,
    ctfidf_model=ctfidf_model,
    verbose=True,
).fit(data, embeds)

print("Original topics modelling results:")
print(topic_model.get_topic_info().head(10))

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")


def generate_label(topic_words):
    prompt = f"Given the following words: {', '.join(topic_words)}, provide a single word that best describes them. Ignore the word hotel"
    inputs = tokenizer.encode(
        prompt, return_tensors="pt", truncation=True, max_length=512
    )
    summary_ids = model.generate(
        inputs, max_length=10, num_beams=5, early_stopping=True
    )
    label = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return label


new_labels = {}
for topic_num, topic in topic_model.get_topics().items():
    topic_words = [word for word, _ in topic]
    new_labels[topic_num] = generate_label(topic_words)

# # Assign new labels to topics
topic_model.set_topic_labels(new_labels)
print(
    f"Given the following words: {', '.join(topic_words)}, provide a single word that best describes them:"
)
print(topic_model.get_topic_info().head(10))
