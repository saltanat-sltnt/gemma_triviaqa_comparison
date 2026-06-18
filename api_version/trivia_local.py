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
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from datasets import load_dataset
from fastapi import FastAPI, Request
from pydantic import BaseModel


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "google/gemma-2-2b"
DATASET_PATH = "mandarjoshi/trivia_qa"
DATASET_NAME = "rc.nocontext"

MAX_NEW_TOKENS = 32

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "api_local_triviaqa"
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


# ============================================================
# Cloud/server side
# RunPod/cloud loads only Gemma, not TriviaQA
# ============================================================

app = FastAPI(title="Gemma Cloud API")

TOKENIZER = None
MODEL = None
TORCH = None


class QuestionRequest(BaseModel):
    question: str


@app.middleware("http")
async def add_cloud_start_time(request: Request, call_next):
    request.state.cloud_start_time = time.perf_counter()
    response = await call_next(request)
    return response


def load_gemma_model():
    global TOKENIZER, MODEL, TORCH

    if MODEL is not None and TOKENIZER is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TORCH = torch

    hf_token = os.getenv("HF_TOKEN")

    print("Loading Gemma model on cloud GPU...")

    TOKENIZER = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token,
    )

    MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    MODEL.eval()

    print("Gemma model loaded.")

    if torch.cuda.is_available():
        print(f"Cloud GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA GPU is not available.")


@app.on_event("startup")
def startup_event():
    load_gemma_model()


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


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": MODEL is not None,
        "main_endpoint": "/ask_gemma",
    }


@app.post("/ask_gemma")
def ask_gemma(request_data: QuestionRequest, request: Request):
    if MODEL is None or TOKENIZER is None:
        load_gemma_model()

    cloud_start_time = request.state.cloud_start_time

    # processing_non_inference_time part 1:
    # receive request + prepare prompt + tokenize
    prompt_tokenize_start = time.perf_counter()

    prompt = f"Question: {request_data.question}\nAnswer:"
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)

    prompt_tokenize_end = time.perf_counter()

    # inference_time:
    # Gemma generates answer
    if TORCH.cuda.is_available():
        TORCH.cuda.synchronize()

    inference_start = time.perf_counter()

    with TORCH.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=TOKENIZER.eos_token_id,
        )

    if TORCH.cuda.is_available():
        TORCH.cuda.synchronize()

    inference_end = time.perf_counter()

    # processing_non_inference_time part 2:
    # decode + format answer + read GPU stats
    decode_format_start = time.perf_counter()

    generated_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    model_answer = extract_short_answer(generated_text, prompt)

    gpu_stats = get_gpu_stats()

    decode_format_end = time.perf_counter()

    inference_time = inference_end - inference_start
    cloud_total_time = decode_format_end - cloud_start_time
    processing_non_inference_time = cloud_total_time - inference_time

    energy_joules = None
    if gpu_stats["gpu_power_watts"] is not None:
        energy_joules = inference_time * gpu_stats["gpu_power_watts"]

    return {
        "model_answer": model_answer,

        "cloud_total_time": cloud_total_time,
        "processing_non_inference_time": processing_non_inference_time,
        "inference_time": inference_time,

        "gpu_memory_used_mb": gpu_stats["gpu_memory_used_mb"],
        "gpu_memory_total_mb": gpu_stats["gpu_memory_total_mb"],
        "gpu_power_watts": gpu_stats["gpu_power_watts"],
        "energy_joules_estimate": energy_joules,
    }


# ============================================================
# Local/client side
# Local PC loads TriviaQA and sends questions to cloud Gemma API
# ============================================================

RESPONSES_COLUMNS = [
    "question_number_in_session",
    "question_index_in_dataset",
    "question",
    "model_answer",
    "ground_truth_aliases",
    "is_correct",

    "client_api_total_time",
    "send_request_time",
    "network_latency",
    "cloud_total_time",
    "receive_response_time",

    "processing_non_inference_time",
    "inference_time",

    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "gpu_power_watts",
    "energy_joules_estimate",
]


def save_results(
    session_id: str,
    responses: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, str]:
    responses_path = RESULTS_DIR / f"responses_{session_id}.csv"
    summary_path = RESULTS_DIR / f"session_summary_{session_id}.csv"

    responses_df = pd.DataFrame(responses, columns=RESPONSES_COLUMNS)
    responses_df.to_csv(responses_path, index=False)

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["metric", "value"],
    )
    summary_df.to_csv(summary_path, index=False)

    return {
        "responses_csv": str(responses_path),
        "summary_csv": str(summary_path),
    }


def run_client_session(
    api_url: str,
    num_questions: int,
    mode: str,
    start_index: int,
    seed: int,
):
    session_id = create_session_id()

    print("Loading TriviaQA locally...")
    dataset = load_dataset(DATASET_PATH, DATASET_NAME, split="validation")
    print(f"TriviaQA size: {len(dataset)}")

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
        # client_api_total_time:
        # time after response - time before request
        # ----------------------------------------------------
        client_total_start = time.perf_counter()

        # ----------------------------------------------------
        # send_request_time:
        # local request payload preparation time
        # ----------------------------------------------------
        send_request_start = time.perf_counter()

        request_payload = json.dumps(
            {"question": question},
            ensure_ascii=False,
        ).encode("utf-8")

        send_request_end = time.perf_counter()

        send_request_time = send_request_end - send_request_start

        # ----------------------------------------------------
        # Send request to cloud API.
        # stream=True lets us measure response-body receive time separately.
        # ----------------------------------------------------
        response = requests.post(
            api_url,
            data=request_payload,
            headers={"Content-Type": "application/json"},
            timeout=300,
            stream=True,
        )

        response_headers_received = time.perf_counter()

        response.raise_for_status()

        # ----------------------------------------------------
        # receive_response_time:
        # receive response body + parse JSON
        # ----------------------------------------------------
        receive_response_start = time.perf_counter()

        response_body = response.content
        cloud_result = json.loads(response_body.decode("utf-8"))

        receive_response_end = time.perf_counter()

        receive_response_time = receive_response_end - receive_response_start

        client_total_end = receive_response_end

        client_api_total_time = client_total_end - client_total_start

        cloud_total_time = cloud_result["cloud_total_time"]

        # ----------------------------------------------------
        # network_latency:
        # residual part from the local formula:
        #
        # client_api_total_time =
        # send_request_time + network_latency
        # + cloud_total_time + receive_response_time
        # ----------------------------------------------------
        network_latency = (
            client_api_total_time
            - send_request_time
            - cloud_total_time
            - receive_response_time
        )

        if network_latency < 0:
            network_latency = 0

        model_answer = cloud_result["model_answer"]
        is_correct = is_correct_answer(model_answer, aliases)

        row = {
            "question_number_in_session": question_number,
            "question_index_in_dataset": question_index,
            "question": question,
            "model_answer": model_answer,
            "ground_truth_aliases": json.dumps(aliases, ensure_ascii=False),
            "is_correct": is_correct,

            "client_api_total_time": round_or_none(client_api_total_time),
            "send_request_time": round_or_none(send_request_time),
            "network_latency": round_or_none(network_latency),
            "cloud_total_time": round_or_none(cloud_total_time),
            "receive_response_time": round_or_none(receive_response_time),

            "processing_non_inference_time": round_or_none(
                cloud_result["processing_non_inference_time"]
            ),
            "inference_time": round_or_none(cloud_result["inference_time"]),

            "gpu_memory_used_mb": cloud_result["gpu_memory_used_mb"],
            "gpu_memory_total_mb": cloud_result["gpu_memory_total_mb"],
            "gpu_power_watts": cloud_result["gpu_power_watts"],
            "energy_joules_estimate": round_or_none(
                cloud_result["energy_joules_estimate"]
            ),
        }

        responses.append(row)

        print(
            f"{question_number}/{len(question_indices)} | "
            f"correct={is_correct} | "
            f"client_api_total_time={row['client_api_total_time']} | "
            f"send_request_time={row['send_request_time']} | "
            f"network_latency={row['network_latency']} | "
            f"cloud_total_time={row['cloud_total_time']} | "
            f"receive_response_time={row['receive_response_time']} | "
            f"processing_non_inference_time={row['processing_non_inference_time']} | "
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
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": round_or_none(accuracy),

        "client_api_total_time": round_or_none(
            sum_column("client_api_total_time")
        ),
        "send_request_time": round_or_none(
            sum_column("send_request_time")
        ),
        "network_latency": round_or_none(
            sum_column("network_latency")
        ),
        "cloud_total_time": round_or_none(
            sum_column("cloud_total_time")
        ),
        "receive_response_time": round_or_none(
            sum_column("receive_response_time")
        ),

        "processing_non_inference_time": round_or_none(
            sum_column("processing_non_inference_time")
        ),
        "inference_time": round_or_none(
            sum_column("inference_time")
        ),

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
# Command line
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gemma cloud API with local TriviaQA client"
    )

    parser.add_argument(
        "--role",
        choices=["server", "client"],
        required=True,
        help=(
            "server = run Gemma API on cloud; "
            "client = run TriviaQA locally and call cloud API"
        ),
    )

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument(
        "--api_url",
        type=str,
        default="http://127.0.0.1:8000/ask_gemma",
        help="Cloud API endpoint URL ending with /ask_gemma",
    )

    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["ordered", "random"],
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.role == "server":
        import uvicorn

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
        )

    elif args.role == "client":
        run_client_session(
            api_url=args.api_url,
            num_questions=args.num_questions,
            mode=args.mode,
            start_index=args.start_index,
            seed=args.seed,
        )
