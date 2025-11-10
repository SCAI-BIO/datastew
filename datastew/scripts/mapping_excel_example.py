"""
Example: Map variables between two data dictionaries using datastew

This script shows how to automatically map variables from a *source*
data dictionary to a *target* data dictionary based on semantic similarity
between their variable names and descriptions.

---

Quick Start

# 1. Prepare two Excel files:
#    source.xlsx — contains your original study variable names
#    target.xlsx — contains standardized variable names you want to map to
#    Each file must include at least:
#      - a column with variable names (e.g., "var")
#      - a column with variable descriptions (e.g., "desc")

# 2. Run the script:
python examples/map_data_dictionaries.py

# 3. The output will be written to:
#    result.xlsx — containing matched variables and similarity scores
"""

from datastew.process.mapping import map_dictionary_to_dictionary
from datastew.process.parsing import DataDictionarySource

# --------------------------------------------------------------------
# 1) Load source and target data dictionaries
# --------------------------------------------------------------------
# Each DataDictionarySource represents one Excel sheet containing variable names and descriptions.
# Adjust the column names below ("var" and "desc") to match your Excel file.
source = DataDictionarySource("source.xlxs", variable_field="var", description_field="desc")
target = DataDictionarySource("target.xlxs", variable_field="var", description_field="desc")

# --------------------------------------------------------------------
# 2) Perform automated mapping
# --------------------------------------------------------------------
# This uses LLM-based embeddings under the hood to identify semantically similar variables between the two dictionaries.
df = map_dictionary_to_dictionary(source, target)

# --------------------------------------------------------------------
# 3) Save results
# --------------------------------------------------------------------
# The result DataFrame includes the top matches and similarity scores.
output_path = "result.xlsx"
df.to_excel(output_path, index=False)
print(f"Mapped {len(df)} variables. Results saved to '{output_path}'.")
