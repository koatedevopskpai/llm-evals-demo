"""answer_generator.py - the application adapter E01 needs.

Scoring needs *generated* answers. This module produces them from the retrieved
contexts of each golden case, using an OpenAI-compatible chat endpoint by
default, and accepts an injected ``complete_fn`` so the test suite never makes a
network call.

The prompt is deliberately strict about grounding: an answer that is not
supported by the supplied context is a failure the evaluation should be able to
detect, not something the generator should paper over.
"""

from __future__ import annotations

import os
from typing import Callable

from golden_set import GoldenCase

#: complete_fn(messages) -> assistant text
CompleteFn = Callable[[list[dict[str, str]]], str]

DEFAULT_MODEL = os.getenv("EVAL_GENERATION_MODEL", "gpt-5-nano")

SYSTEM_PROMPT = (
    "You answer questions using only the supplied context. "
    "If the context does not contain the answer, reply exactly: "
    "'The provided context does not contain this information.' "
    "Do not use outside knowledge and do not invent figures, dates or names."
)


def build_messages(question: str, contexts: list[str]) -> list[dict[str, str]]:
    """Build the chat messages for one case."""
    joined = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{joined}\n\nQuestion: {question}"},
    ]


def default_complete_fn(model: str = DEFAULT_MODEL) -> CompleteFn:
    """Return a chat completion function backed by an OpenAI-compatible endpoint."""

    def _complete(messages: list[dict[str, str]]) -> str:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0)
        return llm.invoke(messages).content

    return _complete


def generate_answer(
    case: GoldenCase,
    complete_fn: CompleteFn | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate the answer for a single case."""
    fn = complete_fn or default_complete_fn(model)
    text = fn(build_messages(case.question, case.retrieved_contexts))
    return (text or "").strip()


def generate_answers(
    cases: list[GoldenCase],
    complete_fn: CompleteFn | None = None,
    model: str = DEFAULT_MODEL,
) -> list[GoldenCase]:
    """Return the cases with ``generated_answer`` populated.

    A provider error on one case is recorded as an empty answer rather than
    aborting the run, so a partially failing application is still measurable.
    An empty answer is then refused by the evaluator, which is the intended
    behaviour: an unanswerable run must not score as a pass.
    """
    out: list[GoldenCase] = []
    for case in cases:
        try:
            answer = generate_answer(case, complete_fn, model)
        except Exception as exc:  # noqa: BLE001 - recorded, then scored as a failure
            answer = ""
            print(f"  generation failed for {case.question[:60]!r}: {type(exc).__name__}: {exc}")
        out.append(
            GoldenCase(
                question=case.question,
                reference_answer=case.reference_answer,
                retrieved_contexts=case.retrieved_contexts,
                generated_answer=answer,
            )
        )
    return out
