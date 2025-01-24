# Topic modelling
## Experiments in topic modelling using Bertopic

[BERTopic](https://maartengr.github.io/BERTopic/index.html) is a topic modeling technique that leverages huggingface transformers and c-TF-IDF to implement [topic modelling](https://link.springer.com/article/10.1007/s10462-023-10661-7#Sec1).

Prequisites
Just pip install -r requirements.txt
You may need a HuggingFace token
cuML to speed up HDBSCAN through GPU acceleration. If you have issues installing it use 
```
pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12==24.6.*
```

I am adding to the examples and hopefully will be bringing out an article on this soon.
