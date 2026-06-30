# local Gemma + local TriviaQA
# supports multiple sessions in one run and loads TriviaQA/Gemma only once

import argparse
import json
import os
import random
import re
import string
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "google/gemma-2-2b"
DATASET_PATH = "mandarjoshi/trivia_qa"
DATASET_NAME = "rc.nocontext"

MAX_NEW_TOKENS = 32

# If this file is saved as local_version/local_version.py, project root is parent of local_version/.
# If it is saved in the project root, project root is the file parent.
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent if FILE_DIR.name == "local_version" else FILE_DIR
RESULTS_DIR = PROJECT_ROOT / "results" / "local"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Shared helper functions
# ============================================================

def create_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def round_or_none(value: Optional[float], digits: int = 4):
    if value is None:
        return None
    return round(value, digits)


def percent_or_none(part: Optional[float], total: Optional[float]):
    if part is None or total is None or total == 0:
        return None
    return round((part / total) * 100, 2)


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())

    return text


def is_correct_answer(model_answer: str, aliases: List[str]) -> bool:
    normalized_model_answer = normalize_text(model_answer)

    for alias in aliases:
        normalized_alias = normalize_text(alias)

        if normalized_model_answer == normalized_alias:
            return True

        if normalized_alias and normalized_alias in normalized_model_answer:
            return True

    return False


def extract_short_answer(generated_text: str, prompt: str) -> str:
    answer = generated_text.replace(prompt, "").strip()
    answer = answer.split("\n")[0].strip()

    prefixes = ["Answer:", "answer:", "A:", "a:"]

    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    return answer


def get_question_indices(
    dataset_size: int,
    num_questions: int,
    mode: str,
    start_index: int,
    seed: int,
) -> List[int]:
    if num_questions <= 0:
        raise ValueError("num_questions must be positive.")

    if num_questions > dataset_size:
        raise ValueError("num_questions is larger than dataset size.")

    if mode == "ordered":
        start = start_index
        end = start + num_questions

        if start < 0:
            raise ValueError("start_index must be >= 0.")

        if end > dataset_size:
            raise ValueError(
                f"Ordered range is outside dataset. Dataset size is {dataset_size}."
            )

        return list(range(start, end))

    if mode == "random":
        random.seed(seed)
        return random.sample(range(dataset_size), num_questions)

    raise ValueError("mode must be either 'ordered' or 'random'.")


def build_session_configs(
    num_questions: int,
    num_questions_list: Optional[str],
    seed: int,
    seeds_list: Optional[str],
) -> List[Tuple[int, int]]:
    question_counts = parse_int_list(num_questions_list)
    seeds = parse_int_list(seeds_list)

    if question_counts is None:
        question_counts = [num_questions]

    if seeds is None:
        seeds = [seed]

    if len(question_counts) > 1 and len(seeds) == 1:
        seeds = seeds * len(question_counts)

    elif len(seeds) > 1 and len(question_counts) == 1:
        question_counts = question_counts * len(seeds)

    elif len(question_counts) != len(seeds):
        raise ValueError(
            "num_questions_list and seeds_list must have the same length, "
            "or one of them must contain only one value."
        )

    return list(zip(question_counts, seeds))


# ============================================================
# Local Gemma model loading and GPU stats
# ============================================================


def get_gpu_stats() -> Dict[str, Optional[float]]:
    stats = {
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "gpu_power_watts": None,
    }

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        ).strip()

        first_gpu = output.splitlines()[0]
        memory_used, memory_total, power = [
            x.strip() for x in first_gpu.split(",")
        ]

        stats["gpu_memory_used_mb"] = float(memory_used)
        stats["gpu_memory_total_mb"] = float(memory_total)
        stats["gpu_power_watts"] = float(power)

    except Exception:
        pass

    return stats


def load_local_gemma_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.getenv("HF_TOKEN")

    print("Loading Gemma model locally on lab PC...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model.eval()

    print("Gemma model loaded.")

    if torch.cuda.is_available():
        print(f"Local GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA GPU is not available.")

    return tokenizer, model, torch


# ============================================================
# Output columns
# ============================================================

RESPONSES_COLUMNS = [
    "session_id",
    "num_questions",
    "mode",
    "seed",
    "start_index",

    "question_number_in_session",
    "question_index_in_dataset",
    "question",
    "model_answer",
    "ground_truth_aliases",
    "is_correct",

    "processing_time",
    "inference_time",
    "local_total_time",

    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "gpu_power_watts",
    "energy_joules_estimate",
]

SUMMARY_COLUMNS = [
    "session_id",
    "num_questions",
    "mode",
    "seed",
    "start_index",
    "correct_answers",
    "accuracy",

    "processing_time",
    "inference_time",
    "local_total_time",

    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "gpu_power_watts",
    "energy_joules_estimate",
]

PERCENT_COLUMNS = [
    "session_id",
    "num_questions",
    "mode",
    "seed",
    "start_index",

    "local_total_percent",
    "processing_percent",
    "inference_percent",
]


# ============================================================
# Saving outputs
# ============================================================


def save_session_results(
    session_id: str,
    responses: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, str]:
    responses_path = RESULTS_DIR / f"responses_{session_id}.csv"
    summary_path = RESULTS_DIR / f"session_summary_{session_id}.csv"

    responses_df = pd.DataFrame(responses, columns=RESPONSES_COLUMNS)
    responses_df.to_csv(responses_path, index=False)

    ordered_summary = {column: summary.get(column) for column in SUMMARY_COLUMNS}
    summary_df = pd.DataFrame(
        list(ordered_summary.items()),
        columns=["metric", "value"],
    )
    summary_df.to_csv(summary_path, index=False)

    return {
        "responses_csv": str(responses_path),
        "summary_csv": str(summary_path),
    }


# ============================================================
# Running local sessions
# ============================================================


def run_single_local_session(
    dataset,
    tokenizer,
    model,
    torch_module,
    num_questions: int,
    mode: str,
    start_index: int,
    seed: int,
) -> Dict[str, Any]:
    session_id = create_session_id()

    print("\n" + "=" * 80)
    print(f"Starting local session {session_id}")
    print(
        f"num_questions={num_questions}, mode={mode}, seed={seed}, start_index={start_index}"
    )
    print("=" * 80)

    question_indices = get_question_indices(
        dataset_size=len(dataset),
        num_questions=num_questions,
        mode=mode,
        start_index=start_index,
        seed=seed,
    )

    responses = []

    for question_number, question_index in enumerate(question_indices, start=1):
        item = dataset[question_index]

        question = item["question"]
        aliases = item["answer"]["aliases"]

        # ----------------------------------------------------
        # local_total_time:
        # total time for local prompt processing + Gemma inference
        # + decoding/formatting + GPU stats reading.
        #
        # local_total_time = processing_time + inference_time
        # ----------------------------------------------------
        local_total_start = time.perf_counter()

        # processing_time part 1:
        # prepare prompt + tokenize input
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # inference_time:
        # Gemma generates answer
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()

        inference_start = time.perf_counter()

        with torch_module.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()

        inference_end = time.perf_counter()

        # processing_time part 2:
        # decode output + format answer + read GPU stats
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_answer = extract_short_answer(generated_text, prompt)

        gpu_stats = get_gpu_stats()

        local_total_end = time.perf_counter()

        inference_time = inference_end - inference_start
        local_total_time = local_total_end - local_total_start
        processing_time = local_total_time - inference_time

        energy_joules = None
        if gpu_stats["gpu_power_watts"] is not None:
            energy_joules = inference_time * gpu_stats["gpu_power_watts"]

        is_correct = is_correct_answer(model_answer, aliases)

        row = {
            "session_id": session_id,
            "num_questions": num_questions,
            "mode": mode,
            "seed": seed if mode == "random" else None,
            "start_index": start_index if mode == "ordered" else None,

            "question_number_in_session": question_number,
            "question_index_in_dataset": question_index,
            "question": question,
            "model_answer": model_answer,
            "ground_truth_aliases": json.dumps(aliases, ensure_ascii=False),
            "is_correct": is_correct,

            "processing_time": round_or_none(processing_time),
            "inference_time": round_or_none(inference_time),
            "local_total_time": round_or_none(local_total_time),

            "gpu_memory_used_mb": gpu_stats["gpu_memory_used_mb"],
            "gpu_memory_total_mb": gpu_stats["gpu_memory_total_mb"],
            "gpu_power_watts": gpu_stats["gpu_power_watts"],
            "energy_joules_estimate": round_or_none(energy_joules),
        }

        responses.append(row)

        print(
            f"{question_number}/{len(question_indices)} | "
            f"correct={is_correct} | "
            f"local_total_time={row['local_total_time']} | "
            f"processing_time={row['processing_time']} | "
            f"inference_time={row['inference_time']} | "
            f"gpu_power_watts={row['gpu_power_watts']}"
        )

    total_questions = len(responses)
    correct_answers = sum(1 for row in responses if row["is_correct"])
    accuracy = correct_answers / total_questions if total_questions > 0 else None

    def sum_column(column_name: str) -> Optional[float]:
        values = [
            row[column_name]
            for row in responses
            if row[column_name] is not None
        ]
        if not values:
            return None
        return sum(values)

    def avg_column(column_name: str) -> Optional[float]:
        values = [
            row[column_name]
            for row in responses
            if row[column_name] is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    processing_time = sum_column("processing_time")
    inference_time = sum_column("inference_time")
    local_total_time = sum_column("local_total_time")

    gpu_memory_total_values = [
        row["gpu_memory_total_mb"]
        for row in responses
        if row["gpu_memory_total_mb"] is not None
    ]

    gpu_memory_total = (
        gpu_memory_total_values[0]
        if gpu_memory_total_values
        else None
    )

    summary = {
        "session_id": session_id,
        "num_questions": total_questions,
        "mode": mode,
        "seed": seed if mode == "random" else None,
        "start_index": start_index if mode == "ordered" else None,

        "correct_answers": correct_answers,
        "accuracy": round_or_none(accuracy),

        "processing_time": round_or_none(processing_time),
        "inference_time": round_or_none(inference_time),
        "local_total_time": round_or_none(local_total_time),

        "gpu_memory_used_mb": round_or_none(
            avg_column("gpu_memory_used_mb"),
            2,
        ),
        "gpu_memory_total_mb": gpu_memory_total,
        "gpu_power_watts": round_or_none(
            avg_column("gpu_power_watts"),
            2,
        ),
        "energy_joules_estimate": round_or_none(
            sum_column("energy_joules_estimate")
        ),
    }

    files = save_session_results(
        session_id=session_id,
        responses=responses,
        summary=summary,
    )

    print("\nSession done.")
    print(json.dumps(summary, indent=2))
    print("\nSaved files:")
    print(files["responses_csv"])
    print(files["summary_csv"])

    return summary


def build_percent_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    local_total_time = summary.get("local_total_time")

    return {
        "session_id": summary.get("session_id"),
        "num_questions": summary.get("num_questions"),
        "mode": summary.get("mode"),
        "seed": summary.get("seed"),
        "start_index": summary.get("start_index"),

        "local_total_percent": 100.0 if local_total_time else None,
        "processing_percent": percent_or_none(
            summary.get("processing_time"),
            local_total_time,
        ),
        "inference_percent": percent_or_none(
            summary.get("inference_time"),
            local_total_time,
        ),
    }


def run_local_batch(
    session_configs: List[Tuple[int, int]],
    mode: str,
    start_index: int,
):
    batch_id = create_session_id()

    print("Loading TriviaQA locally once...")
    dataset = load_dataset(DATASET_PATH, DATASET_NAME, split="validation")
    print(f"TriviaQA size: {len(dataset)}")

    tokenizer, model, torch_module = load_local_gemma_model()

    all_summaries = []
    all_percent_summaries = []

    for num_questions, seed in session_configs:
        summary = run_single_local_session(
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            num_questions=num_questions,
            mode=mode,
            start_index=start_index,
            seed=seed,
        )
        all_summaries.append(summary)
        all_percent_summaries.append(build_percent_summary(summary))

    combined_summary_path = RESULTS_DIR / f"combined_session_summary_{batch_id}.csv"
    combined_df = pd.DataFrame(all_summaries, columns=SUMMARY_COLUMNS)
    combined_df.to_csv(combined_summary_path, index=False)

    combined_percent_summary_path = RESULTS_DIR / f"combined_percent_summary_{batch_id}.csv"
    combined_percent_df = pd.DataFrame(
        all_percent_summaries,
        columns=PERCENT_COLUMNS,
    )
    combined_percent_df.to_csv(combined_percent_summary_path, index=False)

    print("\n" + "=" * 80)
    print("All local sessions completed.")
    print(f"Combined summary saved to: {combined_summary_path}")
    print(f"Combined percent summary saved to: {combined_percent_summary_path}")
    print("=" * 80)


# ============================================================
# Command line
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local Gemma + local TriviaQA experiment"
    )

    parser.add_argument("--num_questions", type=int, default=100)

    parser.add_argument(
        "--num_questions_list",
        type=str,
        default=None,
        help='Comma-separated question counts, for example: "10,25,50,75,100"',
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["ordered", "random"],
    )

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--seeds_list",
        type=str,
        default=None,
        help='Comma-separated seeds, for example: "42,43,44,45,46"',
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    session_configs = build_session_configs(
        num_questions=args.num_questions,
        num_questions_list=args.num_questions_list,
        seed=args.seed,
        seeds_list=args.seeds_list,
    )

    run_local_batch(
        session_configs=session_configs,
        mode=args.mode,
        start_index=args.start_index,
    )
