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

device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Do embeddings in batches
import numpy as np
from tqdm.auto import tqdm

embeds = embedding_model.encode(data, batch_size=32, show_progress_bar=True)

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
