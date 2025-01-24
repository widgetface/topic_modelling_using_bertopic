from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from semantic_chunkers import StatisticalChunker

# note
# RecursiveCharacterTextSplitter: Focuses on maintaining context
# and relationships between text segments.
# CharacterTextSplitter: Splits text based solely on user-defined characters without
# considering the surrounding context. Splits by the size of the chunk only.


def split_texts(
    texts, chunk_size=100, chunk_overlap=10, length_func=len, recursive=True
):

    split_text = []
    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_overlap=chunk_overlap,
            chunk_size=chunk_size,
            length_function=length_func,
        )
        if recursive
        else CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )
    if type(texts) == str:
        split_text.append(splitter.split_text(texts))

    else:
        for text in texts:
            split_text.extend(splitter.split_text(text))

    return split_text


# breakpoint_threshold_type=
# "standard_deviation",
# any difference greater than X standard deviations is split.
# "interquartile",
# interquartile distance is used to split chunks
# "percentile"(default)
# all differences between sentences are calculated, and
# then any difference greater than the X percentile is split.
# "gradient"
# In this method, the gradient of distance is used to split
# chunks along with the percentile method. This method is
# useful when chunks are highly correlated with each other or specific to a domain e.g. legal or medical. The idea is to apply anomaly detection on gradient array so that the distribution become wider
# and easy to identify boundaries in highly semantic data.


# SEE https://python.langchain.com/docs/how_to/semantic-chunker/
# metadatas: Optional[List[dict]]
def semanitic_text_split(
    texts, embedder, metadatas=[{}], breakpoint_threshold_type="percentile"
):

    splitter = SemanticChunker(
        embedder, breakpoint_threshold_type=breakpoint_threshold_type
    )

    docs = splitter.create_documents(texts=texts, metadatas=metadatas)
    # print("split docs ------")
    # print(splitter.transform_documents(docs)[:3])
    return docs


def statistical_text_split(content, encoder):
    content = "".join(content) if type(content) == list else content
    splitter = StatisticalChunker(encoder=encoder)
    return splitter(docs=[content])
