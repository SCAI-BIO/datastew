from datastew.embedding import Vectorizer
from datastew.process.parsing import DataDictionarySource
from datastew.visualisation import plot_embeddings

# Variable and description refer to the corresponding column names in your excel sheet
data_dictionary_source_1 = DataDictionarySource("source1.xlsx", variable_field="var", description_field="desc")
data_dictionary_source_2 = DataDictionarySource("source2.xlsx", variable_field="var", description_field="desc")

vectorizer = Vectorizer()
plot_embeddings([data_dictionary_source_1, data_dictionary_source_2], vectorizer=vectorizer)
