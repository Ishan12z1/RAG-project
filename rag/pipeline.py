from __future__ import annotations
from typing import Sequence

from rag.model.model_Provider import ModelProvider
from rag.utils.contracts import RetrievedChunk,ParsedAnswer
from rag.prompt import build_evidence_block,build_prompt
from rag.parsing import parse_model_output

def answer_question(
        question:str,
        chunks:Sequence[RetrievedChunk],
        model:ModelProvider
)->ParsedAnswer:
    
    evidence_block, evidence_items = build_evidence_block(chunks)
    prompt=build_prompt(question,evidence_block)
    raw=model.get_response(prompt)

    parsed=parse_model_output(raw,evidence_items)

    return parsed


    