"""Retrieval-Augmented Answer Generation using local LLMs (Flan-T5, Qwen).

Features:
- Explicit token budgeting and context truncation logging (instructions & question prioritized)
- Strict temperature handling (greedy decoding when temperature <= 0.0, sampling when > 0.0)
- Configurable model architecture (Seq2Seq like Flan-T5, CausalLM like Qwen) via GENERATOR_MODEL_NAME
- Configurable abstain path when retrieval confidence is below evidence threshold
- Strict citation extraction from generated text without artificial fallbacks
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.config import GENERATOR_MODEL_NAME, MAX_GENERATION_LENGTH, MAX_INPUT_CONTEXT_LENGTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ABSTAIN_MESSAGE = "not enough evidence in the indexed papers"


class ArxivRAGGenerator:
    """Generates grounded answers from retrieved arXiv abstracts with verifiable citations."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
        abstain_threshold: float = 0.0,
    ):
        resolved_name = model_name or os.getenv("GENERATOR_MODEL_NAME", GENERATOR_MODEL_NAME) or "google/flan-t5-base"
        self.model_name: str = str(resolved_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.abstain_threshold = abstain_threshold

        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
            self.is_seq2seq = hasattr(model, "config") and getattr(model.config, "is_encoder_decoder", False)
        else:
            logger.info(f"Loading generator model '{self.model_name}' onto {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            config = AutoConfig.from_pretrained(self.model_name)
            self.is_seq2seq = getattr(config, "is_encoder_decoder", False)

            if self.is_seq2seq:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                ).to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                ).to(self.device)  # type: ignore[arg-type]
            self.model.eval()

    def format_context_prompt(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        max_tokens: int = MAX_INPUT_CONTEXT_LENGTH,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Assemble structured context with strict token budgeting and truncation logging.

        Question and instructions are placed first so they are never truncated.
        Returns the formatted prompt and the subset of docs retained within the token budget.
        """
        header = (
            "Instructions: You are a scientific assistant answering research questions based ONLY on the provided arXiv abstracts. "
            "Cite supporting papers using brackets like [1] or [2]. "
            "If the provided context lacks sufficient information, reply exactly: 'not enough evidence in the indexed papers'.\n\n"
            f"Question: {question.strip()}\n\n"
            "Context Papers:\n"
        )
        header_tokens = len(self.tokenizer.encode(header, add_special_tokens=False))
        budget_remaining = max_tokens - header_tokens - 32  # 32 tokens buffer for safety

        retained_docs: List[Dict[str, Any]] = []
        context_blocks: List[str] = []
        dropped_count = 0

        for i, doc in enumerate(retrieved_docs):
            citation_num = i + 1
            title = doc.get("title", "Untitled")
            paper_id = doc.get("id", "Unknown")
            abstract = doc.get("abstract", "").strip()

            block = f"[{citation_num}] (arXiv:{paper_id}) {title}\nAbstract: {abstract}\n"
            block_tokens = len(self.tokenizer.encode(block, add_special_tokens=False))

            if block_tokens <= budget_remaining:
                context_blocks.append(block)
                retained_docs.append(doc)
                budget_remaining -= block_tokens
            else:
                # If budget is tight, retain at least a truncated snippet of the highest-ranked abstract
                if budget_remaining > 20 and not retained_docs:
                    words = abstract.split()
                    truncated_abs = " ".join(words[:max(10, budget_remaining // 2)]) + "..."
                    block = f"[{citation_num}] (arXiv:{paper_id}) {title}\nAbstract: {truncated_abs}\n"
                    context_blocks.append(block)
                    retained_docs.append(doc)
                    budget_remaining = 0
                else:
                    dropped_count += 1

        if dropped_count > 0:
            logger.warning(
                f"Context budget constraint: dropped {dropped_count} lowest-ranked abstracts "
                f"to respect maximum prompt length of {max_tokens} tokens."
            )

        full_prompt = header + "\n".join(context_blocks) + "\nAnswer:"
        return full_prompt, retained_docs

    def extract_citations(
        self, answer: str, retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse explicit citation numbers like [1], [2] from the generated text.

        Only includes documents explicitly referenced in the generated text.
        Does NOT inject artificial fallbacks.
        """
        found_nums = set(re.findall(r"\[(\d+)\]", answer))
        citations = []

        for num_str in sorted(found_nums, key=int):
            idx = int(num_str) - 1
            if 0 <= idx < len(retrieved_docs):
                doc = retrieved_docs[idx]
                citations.append(
                    {
                        "citation_id": f"[{num_str}]",
                        "paper_id": str(doc.get("id", "")),
                        "title": str(doc.get("title", "")),
                        "authors": str(doc.get("authors", "")),
                        "categories": str(doc.get("categories", "")),
                        "verified_by_nli": False,
                    }
                )

        return citations

    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        max_new_tokens: int = MAX_GENERATION_LENGTH,
        temperature: float = 0.0,
        abstain_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Synthesize answer using retrieved papers with abstain checking and explicit temperature."""
        threshold = self.abstain_threshold if abstain_threshold is None else abstain_threshold

        # Abstain Check 1: Empty retrieval
        if not retrieved_docs:
            return {
                "question": question,
                "answer": ABSTAIN_MESSAGE,
                "citations": [],
                "retrieved_docs": [],
                "abstained": True,
                "abstain_reason": "no_documents_retrieved",
            }

        # Abstain Check 2: Top retrieval score below confidence threshold
        top_score = float(retrieved_docs[0].get("score", 0.0))
        if threshold > 0.0 and top_score < threshold:
            logger.info(
                f"Abstain triggered: top retrieval score {top_score:.4f} is below threshold {threshold:.4f}."
            )
            return {
                "question": question,
                "answer": ABSTAIN_MESSAGE,
                "citations": [],
                "retrieved_docs": retrieved_docs,
                "abstained": True,
                "abstain_reason": "retrieval_score_below_threshold",
            }

        prompt, active_docs = self.format_context_prompt(question, retrieved_docs)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_INPUT_CONTEXT_LENGTH,
            truncation=True,
        ).to(self.device)

        # Strict temperature handling: greedy if temperature <= 0.0
        do_sample = temperature is not None and float(temperature) > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            if self.is_seq2seq:
                outputs = self.model.generate(**inputs, **gen_kwargs)
                raw_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            else:
                input_len = inputs["input_ids"].shape[1]
                outputs = self.model.generate(**inputs, **gen_kwargs)
                gen_tokens = outputs[0][input_len:]
                raw_answer = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Check if the model itself refused to answer
        refusal_keywords = [
            "not enough evidence",
            "insufficient information",
            "context does not contain",
            "cannot answer",
        ]
        abstained = any(rk in raw_answer.lower() for rk in refusal_keywords)

        citations = self.extract_citations(raw_answer, active_docs)

        return {
            "question": question,
            "answer": raw_answer,
            "citations": citations,
            "retrieved_docs": active_docs,
            "abstained": abstained,
            "abstain_reason": "model_refusal" if abstained else None,
        }
