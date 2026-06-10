import os
import re
import csv
import time
import json
import random
import string
import subprocess
from datetime import datetime
from typing import Optional, List, Dict, Any

import torch
import pandas as pd
from datasets import load_dataset
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "google/gemma-2-2b"
DATASET_PATH = "mandarjoshi/trivia_qa"
DATASET_NAME = "rc.nocontext"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Gemma-2-2B TriviaQA API")


# ============================================================
# GLOBAL VARIABLES
# ============================================================

tokenizer = None
model = None
triviaqa_dataset = None


# ============================================================
# REQUEST MODELS
# ============================================================

# POST /run_auto_session (multiple questions)
class AutoSessionRequest(BaseModel):
    num_questions: int = 100
    comment: str = ""
    mode: str = "ordered"      # "ordered" or "random"
    start_index: int = 0       # used only for ordered mode
    seed: int = 42             # used only for random mode

# POST /ask_gemma (single question)


class AskGemmaRequest(BaseModel):
    question: str


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

# unique session id: year-month-day_hour-minute-second
def create_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_text(text: str) -> str:
    # lowercase
    # remove punctuation
    # remove articles: a, an, the
    # remove extra spaces

    if text is None:
        return ""

    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())

    return text

# check Gemma's answer with ground-truth aliases


def is_correct_answer(model_answer: str, aliases: List[str]) -> bool:
    normalized_model_answer = normalize_text(model_answer)

    for alias in aliases:
        normalized_alias = normalize_text(alias)

        if normalized_model_answer == normalized_alias:
            return True

        # if correct answer is inside Gemma's answer
        if normalized_alias and normalized_alias in normalized_model_answer:
            return True

    return False


def get_gpu_stats() -> Dict[str, Optional[float]]:
    # get GPU info using nvidia-smi
    # gpu_memory_used_mb -> currently used GPU VRAM in MB
    # gpu_memory_total_mb -> total available GPU VRAM in MB
    # gpu_power_watts -> current GPU power draw in watts

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
        # if nvidia-smi isn't available, values remain as None
        pass

    return stats


def extract_short_answer(generated_text: str, prompt: str) -> str:
    # remove prompt from generated output, keep the first answer line only

    answer = generated_text.replace(prompt, "").strip()
    answer = answer.split("\n")[0].strip()

    prefixes = ["Answer:", "answer:", "A:", "a:"]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    return answer


def ask_gemma_one_question(question: str) -> Dict[str, Any]:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

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


def get_question_indices(request: AutoSessionRequest) -> List[int]:
    # ordered mode:
    #     start_index=0, num_questions=100 -> questions 0-99
    #     start_index=234, num_questions=100 -> questions 234-333
    #
    # random mode:
    #     seed=42, num_questions=100 -> random but reproducible 100 questions

    dataset_size = len(triviaqa_dataset)

    if request.num_questions <= 0:
        raise HTTPException(
            status_code=400, detail="num_questions must be positive.")

    if request.num_questions > dataset_size:
        raise HTTPException(
            status_code=400, detail="num_questions is larger than dataset size.")

    if request.mode == "ordered":
        start = request.start_index
        end = start + request.num_questions

        if start < 0:
            raise HTTPException(
                status_code=400, detail="start_index must be >= 0.")

        if end > dataset_size:
            raise HTTPException(
                status_code=400,
                detail=f"Ordered range is outside dataset. Dataset size is {dataset_size}."
            )

        return list(range(start, end))

    if request.mode == "random":
        random.seed(request.seed)
        return random.sample(range(dataset_size), request.num_questions)

    raise HTTPException(
        status_code=400, detail="mode must be either 'ordered' or 'random'.")


def save_results(
    session_id: str,
    responses: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, str]:
    # separate CSV files for every session:
    #   responses_{session_id}.csv
    #   session_summary_{session_id}.csv

    responses_path = os.path.join(RESULTS_DIR, f"responses_{session_id}.csv")
    summary_path = os.path.join(
        RESULTS_DIR, f"session_summary_{session_id}.csv")

    responses_df = pd.DataFrame(responses, columns=RESPONSES_COLUMNS)
    responses_df.to_csv(responses_path, index=False)

    summary_df = pd.DataFrame([summary], columns=SUMMARY_COLUMNS)
    summary_df.to_csv(summary_path, index=False)

    return {
        "responses_csv": responses_path,
        "summary_csv": summary_path,
    }


# ============================================================
# STARTUP: LOAD DATASET AND MODEL
# ============================================================

@app.on_event("startup")
def load_everything():
    # runs once when FastAPI starts.
    # loads:
    #   - TriviaQA validation dataset
    #   - Gemma-2-2B tokenizer
    #   - Gemma-2-2B model into GPU memory

    global tokenizer, model, triviaqa_dataset

    print("Loading TriviaQA dataset...")
    triviaqa_dataset = load_dataset(
        DATASET_PATH, DATASET_NAME, split="validation")

    print("Loading Gemma-2-2B model...")
    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # to use the model for inference, not training
    model.eval()

    print("Ready.")
    print(f"TriviaQA size: {len(triviaqa_dataset)}")


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    # simple health check endpoint

    return {
        "message": "Gemma-2-2B TriviaQA API is running",
        "model": MODEL_NAME,
        "dataset": f"{DATASET_PATH}/{DATASET_NAME}",
        "dataset_size": len(triviaqa_dataset) if triviaqa_dataset is not None else None,
        "docs": "/docs",
    }


@app.get("/retrieve_question")
def retrieve_question(question_index: int = 0):
    # returns one TriviaQA question and its ground-truth aliases

    if triviaqa_dataset is None:
        raise HTTPException(status_code=500, detail="Dataset is not loaded.")

    if question_index < 0 or question_index >= len(triviaqa_dataset):
        raise HTTPException(status_code=400, detail="Invalid question_index.")

    item = triviaqa_dataset[question_index]

    return {
        "question_index": question_index,
        "question": item["question"],
        "ground_truth_aliases": item["answer"]["aliases"],
    }


@app.post("/ask_gemma")
def ask_gemma(request: AskGemmaRequest):
    # sends one custom question to Gemma

    result = ask_gemma_one_question(request.question)

    return {
        "question": request.question,
        "model_answer": result["model_answer"],
        "elapsed_time_seconds": result["elapsed_time_seconds"],
        "gpu_memory_used_mb": result["gpu_memory_used_mb"],
        "gpu_memory_total_mb": result["gpu_memory_total_mb"],
        "gpu_power_watts": result["gpu_power_watts"],
        "energy_joules_estimate": result["energy_joules_estimate"],
    }


@app.post("/run_auto_session")
def run_auto_session(request: AutoSessionRequest):
    """
    Main experiment endpoint.

    It:
    1. chooses TriviaQA questions
    2. sends each question to Gemma
    3. checks answer correctness
    4. calculates accuracy, time, GPU memory, power, energy
    5. saves separate CSV files for this session
    """
    if triviaqa_dataset is None:
        raise HTTPException(status_code=500, detail="Dataset is not loaded.")

    session_id = create_session_id()
    question_indices = get_question_indices(request)

    start_time = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    responses = []

    for question_number, question_index in enumerate(question_indices, start=1):
        item = triviaqa_dataset[question_index]

        question = item["question"]
        aliases = item["answer"]["aliases"]

        gemma_result = ask_gemma_one_question(question)
        model_answer = gemma_result["model_answer"]

        is_correct = is_correct_answer(model_answer, aliases)

        row = {
            "session_id": session_id,
            "question_number_in_session": question_number,
            "question_index_in_dataset": question_index,
            "mode": request.mode,
            "seed": request.seed if request.mode == "random" else None,
            "start_index": request.start_index if request.mode == "ordered" else None,
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
        "comment": request.comment,
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

    return {
        "summary": summary,
        "files": files,
    }
