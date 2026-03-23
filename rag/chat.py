from __future__ import annotations

from rag.retrieval.retrieve import Retriever
# from rag.model.model_ollamma import OllamaModel
from rag.model.model_collab import CollabModel
from rag.abstain import should_abstain
from rag.retrieval import BM25Index,ChunkStore,HybridRerankRetriever,HybridRetriever
from rag.rerank.cross_encoder_reranker_API import CrossEncoderReranker
import yaml
from rag.prompt import (
    build_conversation_context,
    build_contextual_query,
    build_evidence_block,
    build_prompt,
    build_repair_prompt,
)
from rag.parsing import parse_model_output
from rag.abstain import should_abstain
from rag.utils.versioning import build_pipeline_versions
from rag.utils.contracts import PipelineResult, Citation, PipelineTimings,CacheHitInfo, ParsedAnswer
from error_handler.errors import RetrievalError,GenerationError
import time
import requests
from typing import Sequence


class RAGPipeline:
    def __init__(self,config_path:str="configs//chat_config.yaml"):
        self.config=self._load_config(path=config_path)
        
        self.hybrid_retriever=HybridRetriever(
            dense=Retriever(chunks_path=self.config["chunking_loc"],embeddings_dir=self.config["embedding_loc"],index_dir=self.config["indexing_loc"]),
            bm25=BM25Index.load(dir_path=self.config["bm25_path"]),
            chunk_store=ChunkStore(path=self.config["chunk_store_path"])
        )

        self.reranker=CrossEncoderReranker(config_path=self.config["reranker_config_path"])
        self.hybrid_retriever_reranker=HybridRerankRetriever(retriever=self.hybrid_retriever,reranker=self.reranker)
        timeout_cfg = self.config.get("timeouts", {}) or {}
        self.model = CollabModel(
            self.config["api_url"],
            max_new_tokens=int(self.config.get("generation", {}).get("max_new_tokens", 500)),
            temperature=float(self.config.get("generation", {}).get("temperature", 0.2)),
            timeout_s=float(timeout_cfg.get("generation_timeout_s", 60)),
            max_retries=int(timeout_cfg.get("generation_max_retries", 0)),
            backoff_s=float(timeout_cfg.get("generation_backoff_s", 0.0)),
)
        self.versions = build_pipeline_versions(
            config=self.config,
            dense_retriever=self.hybrid_retriever.dense,
            reranker=self.reranker,
        )
    def run(self, query: str, top_k: int = 5, history: Sequence[dict[str, str]] | None = None):

        total_start = time.perf_counter()
        conversation_context = build_conversation_context(history)
        retrieval_query = build_contextual_query(query, history)
        try:
            chunks, embed_ms, retrieve_ms, rerank_ms, embedding_cache_hit, retrieval_cache_hit = (
                    self.hybrid_retriever_reranker.retrieve(query=retrieval_query, top_k=top_k)
            )
        except Exception as e:
            raise RetrievalError("Failed to retrieve supporting context.") from e

        for ch in chunks:
            if not isinstance(ch.citation, Citation):
                raise RetrievalError(
                    f"Retrieved chunk {ch.chunk_id} has invalid citation format."
                )

        # abstain_decision = should_abstain(query=retrieval_query, retrieved=chunks)
        # if abstain_decision.abstain:
        #     total_ms = (time.perf_counter() - total_start) * 1000
        #     abstain_needs = [
        #         "Ask about information covered in the indexed source material",
        #         "Be more specific about the topic, entity, or time period you want",
        #     ]
        #     parsed_output = ParsedAnswer(
        #         mode="abstain",
        #         segments=[],
        #         needs=abstain_needs,
        #         raw_text='{"mode":"abstain","needs":["Ask about information covered in the indexed source material","Be more specific about the topic, entity, or time period you want"]}',
        #         parse_warnings=list(abstain_decision.reasons),
        #         schema_valid=True,
        #     )
        #     return PipelineResult(
        #         parsed_output=parsed_output,
        #         retrieved_chunks=chunks,
        #         timings_ms=PipelineTimings(
        #             embed=embed_ms,
        #             retrieve=retrieve_ms,
        #             rerank=rerank_ms,
        #             generate=None,
        #             total=total_ms,
        #         ),
        #         versions=self.versions,
        #         cache_hits=CacheHitInfo(
        #             embedding=embedding_cache_hit,
        #             retrieval=retrieval_cache_hit,
        #         ),
        #         cache_stats={
        #             "embedding": self.hybrid_retriever.dense.get_embedding_cache_stats(),
        #             "retrieval": self.hybrid_retriever.get_retrieval_cache_stats(),
        #         },
        #         context={
        #             "conversation_context": conversation_context,
        #             "effective_query": retrieval_query,
        #             "abstain_precheck": True,
        #             "abstain_reasons": abstain_decision.reasons,
        #             "abstain_signals": abstain_decision.signals,
        #         },
        #     )
        evidence_block, evidence_items = build_evidence_block(chunks)
        prompt=build_prompt(query, evidence_block, conversation_context)
        try:
            generate_start = time.perf_counter()
            raw=self.model.get_response(prompt)
            generate_ms = (time.perf_counter() - generate_start) * 1000
        except requests.Timeout as e:
            raise GenerationError(
                message="Generation timed out.",
                status_code=504,
                code="GENERATION_TIMEOUT",
            ) from e
        except requests.RequestException as e:
            raise GenerationError(
                message="Generation provider request failed.",
                status_code=502,
                code="GENERATION_FAILED",
            ) from e
        except Exception as e:
            raise GenerationError(
                message="Failed to generate answer.",
                status_code=500,
                code="GENERATION_FAILED",
            ) from e
        
        parsed_output=parse_model_output(raw,evidence_items)
        repair_attempted = False

        if parsed_output.mode == "parse_error":
            repair_attempted = True
            repair_prompt = build_repair_prompt(
                question=query,
                conversation_context=conversation_context or "None.",
                evidence_block=evidence_block,
                previous_output=raw,
                parse_warnings=parsed_output.parse_warnings,
            )
            try:
                repair_raw = self.model.get_response(repair_prompt)
                repaired_output = parse_model_output(repair_raw, evidence_items)
                if repaired_output.mode != "parse_error":
                    parsed_output = repaired_output
                else:
                    parsed_output.parse_warnings.append("repair_attempt_failed")
            except requests.Timeout as e:
                raise GenerationError(
                    message="Generation timed out.",
                    status_code=504,
                    code="GENERATION_TIMEOUT",
                ) from e
            except requests.RequestException as e:
                raise GenerationError(
                    message="Generation provider request failed.",
                    status_code=502,
                    code="GENERATION_FAILED",
                ) from e
            except Exception as e:
                raise GenerationError(
                    message="Failed to generate answer.",
                    status_code=500,
                    code="GENERATION_FAILED",
                ) from e

        total_ms = (time.perf_counter() - total_start) * 1000

        return PipelineResult(
            parsed_output=parsed_output,
            retrieved_chunks=chunks,
            timings_ms=PipelineTimings(
                embed=embed_ms,
                retrieve=retrieve_ms,
                rerank=rerank_ms,
                generate=generate_ms,
                total=total_ms,
            ),
            versions=self.versions,
            cache_hits=CacheHitInfo(
            embedding=embedding_cache_hit,
            retrieval=retrieval_cache_hit,
            ),
            cache_stats={
            "embedding": self.hybrid_retriever.dense.get_embedding_cache_stats(),
            "retrieval": self.hybrid_retriever.get_retrieval_cache_stats(),
            },
            context={
                "conversation_context": conversation_context,
                "effective_query": retrieval_query,
                "abstain_precheck": False,
                "generation_repair_attempted": repair_attempted,
            },
        )


    def _load_config(self,path:str):
        with open(path,"r") as f:
            return yaml.safe_load(f)
