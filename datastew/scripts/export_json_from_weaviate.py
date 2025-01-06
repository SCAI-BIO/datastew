import weaviate
from weaviate.config import AdditionalConfig, Timeout
from weaviate.connect import ConnectionParams

from datastew.process.json_adapter import WeaviateJsonConverter
from datastew.repository import WeaviateRepository

repository = WeaviateRepository(mode='remote', path='localhost', port=8080)

converter = WeaviateJsonConverter(dest_path="datastew/export")

converter.from_repository(repository)
