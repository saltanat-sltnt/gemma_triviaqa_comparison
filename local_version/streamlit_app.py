import streamlit as st
from openai import OpenAI

st.title("ChatGPT-like clone")

# set OpenAI API key from secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# accept user input
# check if prompt is not None, assign to prompt
if prompt := st.chat_input("What's up?"):
    # display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    # add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # display assistant's response
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)

    # add assistant's response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
