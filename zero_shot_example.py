# Example demonstrating the use of zeroshot_topic_list in Bertopic
import logging
import warnings
import pandas as pd

from utils import loading_utils

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

logging.getLogger("spacy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="distutils Version classes are deprecated.*")
logging.getLogger("BERTopic").disabled = True

REQUIRED_COLUMNS = ["Review"]
path = "./data/hotel_reviews.csv"
dataset_subset = "all"
ds_path = f"./data/dataset_trip_advisor_{dataset_subset}"
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

model_kwargs = {"device": device}
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", model_kwargs=model_kwargs)

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
# # The effect of reducing the term matrix size in BERTopic depends on how aggressively you reduce it and the characteristics of your data. While it can make the model faster and more efficient, it can also reduce topic quality, making the model less sensitive to subtle differences between topics.
# # It’s a trade-off between computational efficiency and model interpretability
stopwords = list(stopwords.words("english")) + ["http", "https", "amp", "com"]

# we add this to remove stopwords that can pollute topcs
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords)
mmr = MaximalMarginalRelevance(diversity=0.5)  # maximise diverstity of keyord selection
kbi = KeyBERTInspired()

# Prompt
prompt = """I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: '[KEYWORDS]'.
Based on the documents and keywords, what is this topic about?"""


generator = pipeline("text2text-generation", model="google/flan-t5-small")
txtgen = TextGeneration(generator, prompt=prompt, tokenizer="whitespace")
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# We define a number of topics that we know are in the documents
zeroshot_topic_list = [
    "staff attitude and service",
    "hotel location",
    "food resturant bar",
    "clean immaculate spotless",
    "dirty stain soiled mold hygenic smell",
    "noise",
    "theft",
    "room size",
    "safety",
    "elevators stairs",
    "temperature",
    "bath shower basin taps toilet",
    "bed pillow sheets blankets duvet",
    "parking",
    "wifi",
    "fitness sport facilities",
    "cost price",
    "shuttle",
]
topic_model = BERTopic(
    embedding_model="thenlper/gte-small",
    min_topic_size=100,
    zeroshot_topic_list=zeroshot_topic_list,
    zeroshot_min_similarity=0.85,
    vectorizer_model=vectorizer_model,
    representation_model=[mmr, kbi],
).fit(data)

print(topic_model.get_topic_info().head(10))
