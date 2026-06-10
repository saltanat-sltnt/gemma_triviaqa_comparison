import os
import re
import time
import json
import random
import string
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "google/gemma-2-2b"
DATASET_PATH = "mandarjoshi/trivia_qa"
DATASET_NAME = "rc.nocontext"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "local"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT COLUMNS
# ============================================================

SUMMARY_COLUMNS = [
    "session_id",
    "start_time",
    "end_time",
    "comment",
    "total_questions",
    "correct_answers",
    "accuracy",
    "total_elapsed_time_seconds",
    "avg_time_per_question_seconds",
    "avg_gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "total_energy_joules_estimate",
]

RESPONSES_COLUMNS = [
    "session_id",
    "question_number_in_session",
    "question_index_in_dataset",
    "mode",
    "seed",
    "start_index",
    "question",
    "model_answer",
    "ground_truth_aliases",
    "is_correct",
    "elapsed_time_seconds",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "gpu_power_watts",
    "energy_joules_estimate",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def get_gpu_stats() -> Dict[str, Optional[float]]:
    # get GPU memory and power using nvidia-smi

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
            x.strip() for x in first_gpu.split(",")]

        stats["gpu_memory_used_mb"] = float(memory_used)
        stats["gpu_memory_total_mb"] = float(memory_total)
        stats["gpu_power_watts"] = float(power)

    except Exception:
        pass

    return stats


def extract_short_answer(generated_text: str, prompt: str) -> str:
    answer = generated_text.replace(prompt, "").strip()
    answer = answer.split("\n")[0].strip()

    prefixes = ["Answer:", "answer:", "A:", "a:"]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    return answer


def ask_gemma_one_question(
    question: str,
    tokenizer,
    model,
) -> Dict[str, Any]:
    # ask Gemma one question directly on local GPU

    prompt = f"Question: {question}\nAnswer:"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.time()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    end_time = time.time()
    elapsed_time = end_time - start_time

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model_answer = extract_short_answer(generated_text, prompt)

    gpu_stats = get_gpu_stats()

    energy_joules = None
    if gpu_stats["gpu_power_watts"] is not None:
        energy_joules = elapsed_time * gpu_stats["gpu_power_watts"]

    return {
        "model_answer": model_answer,
        "elapsed_time_seconds": round(elapsed_time, 4),
        "gpu_memory_used_mb": gpu_stats["gpu_memory_used_mb"],
        "gpu_memory_total_mb": gpu_stats["gpu_memory_total_mb"],
        "gpu_power_watts": gpu_stats["gpu_power_watts"],
        "energy_joules_estimate": round(energy_joules, 4) if energy_joules is not None else None,
    }


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
                f"Ordered range is outside dataset. Dataset size is {dataset_size}.")

        return list(range(start, end))

    if mode == "random":
        random.seed(seed)
        return random.sample(range(dataset_size), num_questions)

    raise ValueError("mode must be either 'ordered' or 'random'.")


def save_results(
    session_id: str,
    responses: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, str]:
    responses_path = RESULTS_DIR / f"responses_{session_id}.csv"
    summary_path = RESULTS_DIR / f"session_summary_{session_id}.csv"

    responses_df = pd.DataFrame(responses, columns=RESPONSES_COLUMNS)
    responses_df.to_csv(responses_path, index=False)

    # vertical summary format: metric,value
    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["metric", "value"]
    )
    summary_df.to_csv(summary_path, index=False)

    return {
        "responses_csv": str(responses_path),
        "summary_csv": str(summary_path),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_local_session(
    num_questions: int,
    mode: str,
    start_index: int,
    seed: int,
    comment: str,
):
    session_id = create_session_id()

    print("Loading TriviaQA dataset...")
    dataset = load_dataset(DATASET_PATH, DATASET_NAME, split="validation")
    print(f"TriviaQA size: {len(dataset)}")

    print("Loading Gemma-2-2B model...")
    hf_token = os.getenv("HF_TOKEN")

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

    print("Model loaded.")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA GPU is not available. This will be very slow on CPU.")

    question_indices = get_question_indices(
        dataset_size=len(dataset),
        num_questions=num_questions,
        mode=mode,
        start_index=start_index,
        seed=seed,
    )

    start_time = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    responses = []

    for question_number, question_index in enumerate(question_indices, start=1):
        item = dataset[question_index]

        question = item["question"]
        aliases = item["answer"]["aliases"]

        gemma_result = ask_gemma_one_question(
            question=question,
            tokenizer=tokenizer,
            model=model,
        )

        model_answer = gemma_result["model_answer"]
        is_correct = is_correct_answer(model_answer, aliases)

        row = {
            "session_id": session_id,
            "question_number_in_session": question_number,
            "question_index_in_dataset": question_index,
            "mode": mode,
            "seed": seed if mode == "random" else None,
            "start_index": start_index if mode == "ordered" else None,
            "question": question,
            "model_answer": model_answer,
            "ground_truth_aliases": json.dumps(aliases, ensure_ascii=False),
            "is_correct": is_correct,
            "elapsed_time_seconds": gemma_result["elapsed_time_seconds"],
            "gpu_memory_used_mb": gemma_result["gpu_memory_used_mb"],
            "gpu_memory_total_mb": gemma_result["gpu_memory_total_mb"],
            "gpu_power_watts": gemma_result["gpu_power_watts"],
            "energy_joules_estimate": gemma_result["energy_joules_estimate"],
        }

        responses.append(row)

        print(
            f"Session {session_id}: "
            f"{question_number}/{len(question_indices)} done | "
            f"correct={is_correct}"
        )

    end_time = time.time()
    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_questions = len(responses)
    correct_answers = sum(1 for row in responses if row["is_correct"])
    accuracy = correct_answers / total_questions if total_questions > 0 else None

    total_elapsed_time = end_time - start_time
    avg_time_per_question = total_elapsed_time / \
        total_questions if total_questions > 0 else None

    total_energy = sum(
        row["energy_joules_estimate"]
        for row in responses
        if row["energy_joules_estimate"] is not None
    )

    gpu_memory_values = [
        row["gpu_memory_used_mb"]
        for row in responses
        if row["gpu_memory_used_mb"] is not None
    ]

    avg_gpu_memory_used = (
        sum(gpu_memory_values) / len(gpu_memory_values)
        if gpu_memory_values
        else None
    )

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
        "start_time": start_time_str,
        "end_time": end_time_str,
        "comment": comment,
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "total_elapsed_time_seconds": round(total_elapsed_time, 4),
        "avg_time_per_question_seconds": round(avg_time_per_question, 4) if avg_time_per_question is not None else None,
        "avg_gpu_memory_used_mb": round(avg_gpu_memory_used, 2) if avg_gpu_memory_used is not None else None,
        "gpu_memory_total_mb": gpu_memory_total,
        "total_energy_joules_estimate": round(total_energy, 4),
    }

    files = save_results(
        session_id=session_id,
        responses=responses,
        summary=summary,
    )

    print("\nDone.")
    print(json.dumps(summary, indent=2))
    print("\nSaved files:")
    print(files["responses_csv"])
    print(files["summary_csv"])


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Local Gemma-2-2B TriviaQA evaluation on PC GPU"
    )

    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument("--mode", type=str, default="random",
                        choices=["ordered", "random"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comment", type=str, default="Local GPU test")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_local_session(
        num_questions=args.num_questions,
        mode=args.mode,
        start_index=args.start_index,
        seed=args.seed,
        comment=args.comment,
    )
