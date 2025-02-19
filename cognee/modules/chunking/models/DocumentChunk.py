from typing import List

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from ...data.processing.document_types.Document import Document


class DocumentChunk(DataPoint):
    text: str
    word_count: int
    token_count: int
    chunk_index: int
    cut_type: str
    is_part_of: Document
    contains: List[DataPoint] = None

    metadata: dict = {"index_fields": ["text"]}
