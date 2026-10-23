"""Retrieval-Augmented Answer Generation using local Seq2Seq model (Flan-T5)."""

import logging
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import GENERATOR_MODEL_NAME, MAX_GENERATION_LENGTH, MAX_INPUT_CONTEXT_LENGTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ArxivRAGGenerator:
    """Generates grounded answers from retrieved arXiv abstracts with citations."""

    def __init__(
        self,
        model_name: str = GENERATOR_MODEL_NAME,
        device: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name

        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
        else:
            logger.info(f"Loading generator model '{model_name}' onto {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.model.eval()

    def format_context_prompt(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into structured context with citation tags."""
        context_blocks = []
        for i, doc in enumerate(retrieved_docs):
            citation_num = i + 1
            title = doc.get("title", "Untitled")
            paper_id = doc.get("id", "Unknown ID")
            abstract = doc.get("abstract", "").strip()
            # Truncate abstract if excessively long to keep prompt concise
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."

            block = f"[{citation_num}] (arXiv:{paper_id}) {title}: {abstract}"
            context_blocks.append(block)

        joined_context = "\n\n".join(context_blocks)

        prompt = (
            f"Context:\n{joined_context}\n\n"
            f"Question: {question}\n\n"
            "Instructions: Answer the question based on the context above. "
            "You must reference the source papers using citations like [1] or [2] to support your statements."
        )
        return prompt

    def extract_citations(
        self, answer: str, retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse citation numbers like [1], [2] from the generated text and return metadata."""
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
                    }
                )

        # Fallback: if model answered without explicit [N] tag, cite top retrieved document
        if not citations and retrieved_docs:
            top_doc = retrieved_docs[0]
            citations.append(
                {
                    "citation_id": "[1]",
                    "paper_id": str(top_doc.get("id", "")),
                    "title": str(top_doc.get("title", "")),
                    "authors": str(top_doc.get("authors", "")),
                    "categories": str(top_doc.get("categories", "")),
                }
            )

        return citations

    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        max_new_tokens: int = MAX_GENERATION_LENGTH,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Synthesize answer using retrieved papers and return text with citation links."""
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "No relevant arXiv papers found in the index to answer this question.",
                "citations": [],
                "retrieved_docs": [],
            }

        prompt = self.format_context_prompt(question, retrieved_docs)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_INPUT_CONTEXT_LENGTH,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            if temperature > 0.0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

        raw_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        citations = self.extract_citations(raw_answer, retrieved_docs)

        return {
            "question": question,
            "answer": raw_answer,
            "citations": citations,
            "retrieved_docs": retrieved_docs,
        }
