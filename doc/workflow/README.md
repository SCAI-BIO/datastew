# Curation Workflow

Exemplary data harmonization: Existing data sources are used to compute embeddings that
are stored in a database. New data by users is being mapped based on the shortest embedding distances.
An end user curates data based on suggestions and may store mappings to improve the next mapping
iteration.

---

![Example Image](curation-workflow.png)

---

1. New data sources consisting of the raw study data and corresponding data dictionaries are collected and pre-processed by the end-user
2. The user can then specify the context in which the new data should be harmonized - this could be any kind of terminology that has been used for previous mappings or has been stored for retrieval in any terminology server, e.g. Ontology Lookup Service (OLS) [1]
3. Using a general purpose public LLM such as ChatGPT, a local variant such as LLaMA [2] or a domain-specific model like BioBERT [3], a high dimensional embedding can be computed for each of the previously curated data sources as well as for all relevant terminology sources
4. After the same embedding calculation is done for the unseen data, the data is then mapped to a concept of either the curated data or any terminology concept term in the matching pre-selected context based on its smallest Euclidean distance
5. The end-user will receive a list of the closest matching candidates, which can then again be manually curated with little additional effort compared to a full manual curation
6. The resulting mappings and embeddings are then stored together with the previously curated data
7. Based on the updated mappings, the upcoming iteration of mapping can benefit from an expanded baseline of underlying vector embeddings, thus potentially being able to utilize a more detailed and granular mapping model
---

[1]: [EMBL-EBI Ontology Lookup Service](https://www.ebi.ac.uk/) \
[2]: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) \
[3]: [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/abs/1901.08746) 
