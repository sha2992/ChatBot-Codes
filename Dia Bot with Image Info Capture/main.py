import streamlit as st
from PIL import Image
import tempfile
import concurrent.futures
import rag as l
import images as ic

# --- Streamlit Config ---
# st.set_page_config(page_title="Chat + Image Reader", layout="centered")
st.title("Diabot Assistant")

# --- Format Response ---
def format_response(response):
    response = response.replace("\n•", "\n\n•")
    response = response.replace("\n", "\n\n")
    return response

def fetch_response(prompt, l):
    raw_response = l.output(prompt)
    processed_response = l.ans_rep(raw_response)
    return format_response(processed_response)

# --- Session State Initialization ---
if '_initialized' not in st.session_state:
    st.session_state['_initialized'] = True

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'show_uploader' not in st.session_state:
    st.session_state.show_uploader = False

# --- Display Chat History ---
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# --- Toggle uploader button ---
if st.button("📎 Upload image", help="Click to upload an image"):
    st.session_state.show_uploader = not st.session_state.show_uploader

# --- Chat Input (must be outside container/columns) ---
prompt = st.chat_input("Pass your prompt here")

# --- Handle Image Upload ---
if st.session_state.show_uploader:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name

        with st.spinner("Extracting reading..."):
            try:
                reading = ic.extract_reading(tmp_path)

                if reading and str(reading).strip():
                    response = f"Extracted reading: {reading}"
                    st.success("Reading extracted successfully!")
                    st.chat_message("assistant").markdown(response)
                    st.session_state.messages.append({'role': 'assistant', 'content': response})
                else:
                    st.error("❌ Could not extract reading. Please upload a clearer image.")
            except Exception as e:
                st.error(f"⚠️ Error while processing the image: {e}")
 
# --- Handle Chat Prompt ---
if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': "user", 'content': prompt})

    with st.spinner("Generating Response..."):
        # formatted_response = fetch_response(prompt, l)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fetch_response, prompt, l)
            formatted_response = future.result()

    st.chat_message('assistant').markdown(formatted_response)
    st.session_state.messages.append({'role': "assistant", 'content': formatted_response})