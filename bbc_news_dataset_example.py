# A slightly different example with a large dataset from BBC news
import logging
import warnings
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import TextGeneration
from bertopic.representation import PartOfSpeech
from bertopic.vectorizers import ClassTfidfTransformer
from transformers import pipeline
from sentence_transformers import SentenceTransformer

import transformers

from utils import loading_utils

from keybert import KeyBERT

# Use pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12==24.6.*pip install bertopic

from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer


from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline


logging.getLogger("spacy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="distutils Version classes are deprecated.*")
logging.getLogger("BERTopic").disabled = True


REDUNDANT_COLUMNS = ["top_image", "authors"]
REQUIRED_COLUMNS = ["description", "title", "published_date", "content", "link"]

POS_TAGLIST = ["NOUN", "VERB"]
BT_POS_TAGLIST = [
    [{"POS": "VERB"}, {"POS": "NOUN"}],
    [{"POS": "NOUN"}],
    [{"POS": "VERB"}],
]

dataset_name = "RealTimeData/bbc_news_alltime"
dataset_subset = "2024-06"
ds_path = f"./data/bbc_dataset_{dataset_name}_{dataset_subset}"
# Load once

ds = loading_utils.load_data(
    path=dataset_name,
    dataset_path=ds_path,
    subset=dataset_subset,
)

ds = ds.remove_columns(REDUNDANT_COLUMNS)
df = pd.DataFrame(data=ds)
df = (
    df.drop_duplicates(
        subset=REQUIRED_COLUMNS,
    )
    .dropna(subset=REQUIRED_COLUMNS)
    .reset_index(drop=True)
)

# metadata title, published_data, section, link
bbc_data = df["description"].tolist()
print(f"Length {len(bbc_data)}")
length = [len(text) for text in bbc_data]
print(f"Longest text = {max(length)}")

metadata = df.to_dict(orient="records")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_kwargs = {"device": device}
encode_kwargs = {"normalize_embeddings": False}

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

umap_model = UMAP(
    n_neighbors=15, n_components=3, min_dist=0.0, metric="cosine", random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=20,
    gen_min_span_tree=True,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,  # Have to use with Bertopic
    min_samples=4,
)

mmr = MaximalMarginalRelevance(diversity=0.5)  # maximise diverstity of keyord selection
kbi = KeyBERTInspired()

kw_model = KeyBERT()
keywords = kw_model.extract_keywords(bbc_data)
# Create our vocabulary
vocabulary = [k[0] for keyword in keywords for k in keyword]
vocabulary = list(set(vocabulary))
vectorizer_model = CountVectorizer(vocabulary=vocabulary)

prompt = """I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: '[KEYWORDS]'.
Based on the documents and keywords, what is this topic about?"""

generator = pipeline("text2text-generation", model="google/flan-t5-small")
txtgen = TextGeneration(generator, prompt=prompt, tokenizer="whitespace")

ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
pos = PartOfSpeech("en_core_web_sm", pos_patterns=BT_POS_TAGLIST)
# print(f"PROCESSED_DOCS {processed_docs[:1]}")
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=[mmr, kbi, pos, txtgen],
    language="english",
    top_n_words=5,
    ctfidf_model=ctfidf_model,
    verbose=True,
)
topics, probs = topic_model.fit_transform(bbc_data)

metadata_df = pd.DataFrame(metadata)
print(topic_model.get_topic_info())

new_topics = topic_model.reduce_outliers(bbc_data, topics)
topic_model.update_topics(bbc_data, topics=new_topics)
print(topic_model.get_topic_info())
