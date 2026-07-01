import os
import torch
import time
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

st.title("Local Gemma chat")

MODEL_NAME = "google/gemma-2-2b"
MAX_NEW_TOKENS = 32


@st.cache_resource
def load_local_gemma_model():
    hf_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

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


# display the text while code block executes (while loads Gemma)
with st.spinner("Loading Gemma model locally..."):
    tokenizer, model = load_local_gemma_model()

# if gpu available success message, otherwise warning
if torch.cuda.is_available():
    st.sidebar.success(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning(f"CUDA GPU isn't available.")

###############################################

# leave clean answer


def extract_short_answer(generated_text, prompt):
    answer = generated_text.replace(prompt, "").strip()
    answer = answer.split("\n")[0].strip()

    for prefix in ["Answer:", "answer:", "A:", "a:"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    return answer

# get response and timings


def generate_gemma_answer(question):
    local_total_start = time.perf_counter()

    # 1. prepare prompt + tokenize
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # inference
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

    # 2. decode + format
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = extract_short_answer(generated_text, prompt)

    local_total_end = time.perf_counter()

    inference_time = inference_end - inference_start
    local_total_time = local_total_end - local_total_start
    processing_time = local_total_time - inference_time

    timing = {
        "processing_time": round(processing_time, 4),
        "inference_time": round(inference_time, 4),
        "local_total_time": round(local_total_time, 4),
    }

    return answer, timing


# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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
