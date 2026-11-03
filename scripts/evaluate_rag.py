import argparse
import json
import os
from statistics import fmean
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.llm_cache import configure_llm_cache
from app.rag import retrieve_documents


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
        required = {"question", "reference_answer"}
        missing = required - case.keys()
        if missing:
            raise ValueError(f"Line {line_number} is missing: {', '.join(sorted(missing))}")
        cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}.")
    return cases


def write_results(
    output_path: Path,
    dataset_path: Path,
    model_name: str,
    top_k: int,
    cases: list[dict],
) -> dict:
    metric_names = tuple(cases[0]["scores"])
    report = {
        "dataset": str(dataset_path),
        "model": model_name,
        "top_k": top_k,
        "summary": {
            name: fmean(case["scores"][name] for case in cases)
            for name in metric_names
        },
        "results": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_evaluation(dataset_path: Path, top_k: int, output_path: Path) -> None:
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    except ImportError as exc:
        raise SystemExit(
            "RAGAS evaluation dependencies are unavailable in this environment. "
            "Create the evaluation environment and install `requirements-eval.txt`."
        ) from exc

    cases = load_cases(dataset_path)
    configure_llm_cache()
    client = AsyncOpenAI()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = llm_factory(model_name, client=client)
    embeddings = embedding_factory(
        "openai",
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        client=client,
    )
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }
    case_results = []
    for case in cases:
        documents = retrieve_documents(case["question"], k=top_k)
        contexts = [document.page_content for document in documents]
        if not contexts:
            raise SystemExit(f"No indexed documents were retrieved for: {case['question']}")
        context_text = "\n\n".join(contexts)
        prompt = (
            "Answer the question using only the supplied context. If the context does not contain "
            "the answer, say that the documents do not specify it.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {case['question']}"
        )
        response = str(ChatOpenAI(model=model_name, temperature=0).invoke(prompt).content)
        metric_inputs = {
            "user_input": case["question"],
            "response": response,
            "reference": case["reference_answer"],
            "retrieved_contexts": contexts,
        }
        case_scores = {
            "faithfulness": metrics["faithfulness"].score(**{
                key: metric_inputs[key] for key in ("user_input", "response", "retrieved_contexts")
            }),
            "answer_relevancy": metrics["answer_relevancy"].score(**{
                key: metric_inputs[key] for key in ("user_input", "response")
            }),
            "context_precision": metrics["context_precision"].score(**{
                key: metric_inputs[key] for key in ("user_input", "reference", "retrieved_contexts")
            }),
            "context_recall": metrics["context_recall"].score(**{
                key: metric_inputs[key] for key in ("user_input", "reference", "retrieved_contexts")
            }),
        }
        numeric_scores = {name: float(result.value) for name, result in case_scores.items()}
        case_results.append({
            "question": case["question"],
            "response": response,
            "scores": numeric_scores,
        })
        print(f"{case['question']} | " + " | ".join(
            f"{name}={float(result.value):.3f}" for name, result in case_scores.items()
        ))

    report = write_results(output_path, dataset_path, model_name, top_k, case_results)
    print("\nMean scores:")
    for name, score in report["summary"].items():
        print(f"{name}: {score:.3f}")
    print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the document RAG flow with RAGAS.")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL file with golden RAG cases.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved chunks per question.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/rag_results.json"),
        help="JSON output path (default: evals/rag_results.json).",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    load_dotenv()
    run_evaluation(args.dataset, args.top_k, args.output)


if __name__ == "__main__":
    main()