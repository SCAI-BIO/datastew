"""
Example: Convert OHDSI vocabulary files to JSONL format for Weaviate import

This script demonstrates how to use the datastew WeaviateJsonlConverter to
transform an OHDSI-style vocabulary file (e.g., CONCEPT.csv) into a JSONL file
that can be directly imported into a Weaviate vector database.

Steps:
1. Initialize the converter with an output directory.
2. Convert the OHDSI CONCEPT.csv file to JSONL format.
3. The resulting JSONL files will be saved in the specified output directory.
"""

from datastew.process.jsonl_adapter import WeaviateJsonlConverter

# --------------------------------------------------------------------
# 1) Initialize the converter
# --------------------------------------------------------------------
# The output directory will contain the generated JSONL files.
output_directory = "resources/results"
jsonl_converter = WeaviateJsonlConverter(output_directory)

# --------------------------------------------------------------------
# 2) Convert the OHDSI CONCEPT.csv file
# --------------------------------------------------------------------
# The input file should follow the standard OHDSI vocabulary structure.
input_file = "resources/CONCEPT.csv"
jsonl_converter.from_ohdsi(input_file)

print(f"Conversion complete. JSONL files written to: {output_directory}")
