from __future__ import annotations

from dataclasses import dataclass

from mcp_memory.models import Chunk, Document


@dataclass
class SimpleChunker:
    chunking_config: object

    def chunk_document(self, document: Document) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=f"{document.id}_chunk_0",
                doc_id=document.id,
                content=document.content,
                metadata=dict(document.metadata),
                chunk_index=0,
                header_path="",
                start_pos=0,
                end_pos=len(document.content),
                file_path=document.file_path,
                modified_time=document.modified_time,
            )
        ]


def get_chunker(chunking_config: object) -> SimpleChunker:
    return SimpleChunker(chunking_config)