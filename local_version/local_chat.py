import os
import torch
import time
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from datasets import load_dataset
import json, re, string, subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path

st.title("Local Gemma chat")

MODEL_NAME = "google/gemma-2-2b"
MAX_NEW_TOKENS = 32

DATASET_PATH = "mandarjoshi/trivia_qa"
DATASET_NAME = "rc.nocontext"

RESULTS_DIR = Path("results/local_ui")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


@st.cache_resource
def load_local_gemma_model():
    hf_token = os.getenv("HF_TOKEN")

    # print("Loading Gemma model locally on lab PC...")

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

    return tokenizer, model

@st.cache_resource
def load_triviaqa_dataset():
    return load_dataset(DATASET_PATH, DATASET_NAME, split="validation")


# display the text while code block executes (while loads Gemma)
with st.spinner("Loading Gemma model locally..."):
    tokenizer, model = load_local_gemma_model()

# if gpu available success message, otherwise warning
if torch.cuda.is_available():
    st.sidebar.success(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning(f"CUDA GPU isn't available.")


with st.spinner("Loading TriviaQA dataset..."):
    dataset = load_triviaqa_dataset()

###############################################

# leave clean answer


def extract_short_answer(generated_text, prompt):
    answer = generated_text.replace(prompt, "").strip()
    answer = answer.split("\n")[0].strip()

    for prefix in ["Answer:", "answer:", "A:", "a:"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    return answer

def get_gpu_stats():
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
        memory_used, memory_total, power = [x.strip() for x in first_gpu.split(",")]

        stats["gpu_memory_used_mb"] = float(memory_used)
        stats["gpu_memory_total_mb"] = float(memory_total)
        stats["gpu_power_watts"] = float(power)

    except Exception:
        pass

    return stats


def normalize_text(text):
    if text is None:
        return ""

    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())

    return text


def is_correct_answer(model_answer, aliases):
    normalized_model_answer = normalize_text(model_answer)

    for alias in aliases:
        normalized_alias = normalize_text(alias)

        if normalized_model_answer == normalized_alias:
            return True

        if normalized_alias and normalized_alias in normalized_model_answer:
            return True

    return False

# get response and timings

def generate_gemma_answer(question):
    local_total_start = time.perf_counter()

    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_end = time.perf_counter()

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = extract_short_answer(generated_text, prompt)

    gpu_stats = get_gpu_stats()

    local_total_end = time.perf_counter()

    inference_time = inference_end - inference_start
    local_total_time = local_total_end - local_total_start
    processing_time = local_total_time - inference_time

    energy_joules = None
    if gpu_stats["gpu_power_watts"] is not None:
        energy_joules = inference_time * gpu_stats["gpu_power_watts"]

    timing = {
        "processing_time": round(processing_time, 4),
        "inference_time": round(inference_time, 4),
        "local_total_time": round(local_total_time, 4),
        "gpu_memory_used_mb": gpu_stats["gpu_memory_used_mb"],
        "gpu_memory_total_mb": gpu_stats["gpu_memory_total_mb"],
        "gpu_power_watts": gpu_stats["gpu_power_watts"],
        "energy_joules_estimate": round(energy_joules, 4) if energy_joules is not None else None,
    }

    return answer, timing

# question indices from dataset
def get_question_indices(dataset_size, num_questions, mode, start_index, seed):
    if mode == "ordered":
        return list(range(start_index, start_index + num_questions))

    if mode == "random":
        random.seed(seed)
        return random.sample(range(dataset_size), num_questions)
    

def save_session_files(session_id, response_rows, summary):
    responses_path = RESULTS_DIR / f"responses_{session_id}.csv"
    summary_path = RESULTS_DIR / f"session_summary_{session_id}.csv"

    responses_df = pd.DataFrame(response_rows, columns=RESPONSES_COLUMNS)
    responses_df.to_csv(responses_path, index=False)

    ordered_summary = {column: summary.get(column) for column in SUMMARY_COLUMNS}
    summary_df = pd.DataFrame(
        list(ordered_summary.items()),
        columns=["metric", "value"]
    )
    summary_df.to_csv(summary_path, index=False)

    return responses_path, summary_path

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# automatic session logic

st.sidebar.header("Automatic TriviaQA session")

auto_num_questions = st.sidebar.number_input(
    "Number of questions",
    min_value=1,
    max_value=100,
    value=10,
    step=1
)

auto_mode = st.sidebar.selectbox(
    "Mode",
    ["ordered", "random"]
)

auto_start_index = st.sidebar.number_input(
    "Start index",
    min_value=0,
    value=0,
    step=1
)

auto_seed = st.sidebar.number_input(
    "Seed",
    min_value=0,
    value=42,
    step=1
)

run_auto = st.sidebar.button("Run automatic session")

if run_auto:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    indices = get_question_indices(
        dataset_size=len(dataset),
        num_questions=auto_num_questions,
        mode=auto_mode,
        start_index=auto_start_index,
        seed=auto_seed
    )

    response_rows = []
    progress_bar = st.progress(0)

    for question_number, question_index in enumerate(indices, start=1):
        item = dataset[question_index]

        question = item["question"]
        aliases = item["answer"]["aliases"]

        st.session_state.messages.append({
            "role": "user",
            "content": f"[TriviaQA question {question_index}] {question}"
        })

        with st.spinner(f"Running question {question_number}/{auto_num_questions}..."):
            response, timing = generate_gemma_answer(question)

        is_correct = is_correct_answer(response, aliases)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timing": timing
        })

        row = {
            "session_id": session_id,
            "num_questions": auto_num_questions,
            "mode": auto_mode,
            "seed": auto_seed if auto_mode == "random" else None,
            "start_index": auto_start_index if auto_mode == "ordered" else None,
            "question_number_in_session": question_number,
            "question_index_in_dataset": question_index,
            "question": question,
            "model_answer": response,
            "ground_truth_aliases": json.dumps(aliases, ensure_ascii=False),
            "is_correct": is_correct,
            "processing_time": timing["processing_time"],
            "inference_time": timing["inference_time"],
            "local_total_time": timing["local_total_time"],
            "gpu_memory_used_mb": timing["gpu_memory_used_mb"],
            "gpu_memory_total_mb": timing["gpu_memory_total_mb"],
            "gpu_power_watts": timing["gpu_power_watts"],
            "energy_joules_estimate": timing["energy_joules_estimate"],
        }

        response_rows.append(row)

        progress_bar.progress(question_number / auto_num_questions)

    correct_answers = sum(1 for row in response_rows if row["is_correct"])
    accuracy = correct_answers / auto_num_questions if auto_num_questions > 0 else None

    def sum_col(col):
        values = [row[col] for row in response_rows if row[col] is not None]
        return sum(values) if values else None

    def avg_col(col):
        values = [row[col] for row in response_rows if row[col] is not None]
        return sum(values) / len(values) if values else None

    gpu_total_values = [
        row["gpu_memory_total_mb"]
        for row in response_rows
        if row["gpu_memory_total_mb"] is not None
    ]

    summary = {
        "session_id": session_id,
        "num_questions": auto_num_questions,
        "mode": auto_mode,
        "seed": auto_seed if auto_mode == "random" else None,
        "start_index": auto_start_index if auto_mode == "ordered" else None,
        "correct_answers": correct_answers,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "processing_time": round(sum_col("processing_time"), 4),
        "inference_time": round(sum_col("inference_time"), 4),
        "local_total_time": round(sum_col("local_total_time"), 4),
        "gpu_memory_used_mb": round(avg_col("gpu_memory_used_mb"), 2),
        "gpu_memory_total_mb": gpu_total_values[0] if gpu_total_values else None,
        "gpu_power_watts": round(avg_col("gpu_power_watts"), 2),
        "energy_joules_estimate": round(sum_col("energy_joules_estimate"), 4),
    }

    responses_path, summary_path = save_session_files(
        session_id=session_id,
        response_rows=response_rows,
        summary=summary
    )

    st.success("Automatic session finished.")
    st.sidebar.success(f"Saved responses: {responses_path}")
    st.sidebar.success(f"Saved summary: {summary_path}")

# display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "timing" in message:
            timing = message["timing"]
            st.caption(
                f"processing_time = {timing['processing_time']}s | "
                f"inference_time = {timing['inference_time']}s | "
                f"local_total_time = {timing['local_total_time']}s"
            )

# accept user input
if prompt := st.chat_input("Ask Gemma something..."):
    # display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant"):
        # generate Gemma response
        with st.spinner("Gemma is generating..."):
            response, timing = generate_gemma_answer(prompt)

        # display Gemma response
        st.markdown(response)

        # display timings
        st.caption(
            f"processing_time = {timing['processing_time']}s | "
            f"inference_time = {timing['inference_time']}s | "
            f"local_total_time = {timing['local_total_time']}s"
        )

        # add Gemma response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timing": timing
        })
