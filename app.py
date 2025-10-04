import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import hashlib
from streamlit.components.v1 import html
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from urllib3.exceptions import MaxRetryError

# --- CONFIGURATION ---
METADATA_FILE = "form_metadata.json"
ADMIN_PASS = "hackathon2025"

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI FormGen", layout="wide")

# --- CUSTOM CSS ---
CUSTOM_CSS = """
<style>
/* --- GOOGLE FONTS --- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* --- GLOBAL VARIABLES --- */
:root {
    --bg-primary: #0D1117;
    --bg-secondary: #161B22;
    --text-primary: #E5E7EB;
    --text-secondary: #9CA3AF;
    --accent-gradient: linear-gradient(90deg, #6366F1, #8B5CF6);
    --card-bg: rgba(22, 27, 34, 0.7);
    --border-color: rgba(200, 200, 255, 0.15);
    --font-sans: 'Inter', sans-serif;
}

/* --- RESET & BASE --- */
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: var(--font-sans);
    background-color: var(--bg-primary);
    color: var(--text-primary);
}
.stApp { background: var(--bg-primary); }

/* Hide Streamlit default elements */
#MainMenu, header, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }

/* --- HEADER --- */
.header-container { display: flex; justify-content: space-between; align-items: center; padding: 1rem 3rem; border-bottom: 1px solid var(--border-color); }
.logo-text-gradient { font-size: 28px; font-weight: 800; background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.nav-links { display: flex; gap: 2rem; }
.nav-link { color: var(--text-secondary); text-decoration: none; font-weight: 600; transition: color 0.3s; }
.nav-link:hover { color: #fff; }

/* --- NAVIGATION (st.radio styled as buttons) --- */
[data-testid="stRadio"] {
    display: flex;
    justify-content: center;
    margin-bottom: 2rem;
}
[data-testid="stRadio"] > div {
    display: flex;
    flex-direction: row;
    gap: 1rem;
    border: 1px solid var(--border-color);
    background: var(--bg-secondary);
    padding: 0.5rem;
    border-radius: 12px;
}
[data-testid="stRadio"] label {
    cursor: pointer;
    padding: 0.5rem 1.5rem;
    border-radius: 8px;
    transition: all 0.2s ease-in-out;
    background-color: transparent;
    color: var(--text-secondary);
    font-weight: 600;
}
/* Hide the actual radio circle */
[data-testid="stRadio"] input[type="radio"] {
    display: none;
}
/* Style for the selected radio button's label */
[data-testid="stRadio"] input[type="radio"]:checked + div {
    background: var(--accent-gradient);
    color: white;
    font-weight: 700;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}


/* --- HERO SECTION --- */
.hero-section { display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 4rem 2rem 2rem 2rem; }
.hero-section h1 { font-size: 3rem; font-weight: 800; margin-bottom: 1rem; }
.hero-section p { font-size: 1.2rem; color: var(--text-secondary); max-width: 600px; }

/* --- PROMPT & FORM CONTAINER --- */
.prompt-container { max-width: 700px; width: 100%; margin: 2rem auto; padding: 2rem; border-radius: 16px; background: var(--bg-secondary); border: 1px solid var(--border-color); box-shadow: 0 8px 24px rgba(0,0,0,0.2); transition: all 0.3s ease; }
.prompt-container:focus-within { box-shadow: 0 0 30px rgba(99, 102, 241, 0.5); border-color: #6366F1;}

/* --- STYLING STREAMLIT WIDGETS --- */
[data-testid="stTextarea"] textarea, .stTextInput input, .stNumberInput input, .stDateInput input {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    font-size: 1rem;
    transition: all 0.3s ease;
}
[data-testid="stTextarea"] textarea:focus, .stTextInput input:focus, .stNumberInput input:focus, .stDateInput input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 8px rgba(99,102,241,0.5) !important;
}
.stCheckbox label { font-size: 1rem; }

/* --- BUTTONS --- */
.stButton>button { background: var(--accent-gradient) !important; color: white !important; font-weight: 700 !important; border: none !important; border-radius: 10px !important; padding: 0.75rem 1.5rem !important; transition: all 0.2s ease; }
.stButton>button:hover { transform: translateY(-3px); box-shadow: 0 12px 24px rgba(0,0,0,0.3); }

/* --- CARDS & DASHBOARD --- */
.dashboard-section-card {
    background: var(--bg-secondary); padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem; border: 1px solid var(--border-color);
}
.login-card {
    max-width: 500px;
    margin: 4rem auto;
}

/* --- DATAFRAMES --- */
[data-testid="stDataFrame"] { border-radius: 12px !important; border: 1px solid var(--border-color) !important; background-color: var(--bg-primary) !important; }
[data-testid="stDataFrame"] .row-header { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; font-weight: 600; }
[data-testid="stDataFrame"] .data-row:hover { background-color: var(--bg-secondary) !important; }

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- DATA & METADATA FUNCTIONS ---
def get_data_file_path(form_id: int) -> str: return f"form_data_{form_id}.csv"
def load_data(form_id: int) -> pd.DataFrame:
    data_file = get_data_file_path(form_id)
    try:
        if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
            df = pd.read_csv(data_file)
            if 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df
    except Exception: pass
    return pd.DataFrame()

def append_data(form_id: int, data: dict):
    data_file = get_data_file_path(form_id)
    try:
        df_new = pd.DataFrame([data])
        write_header = not (os.path.exists(data_file) and os.path.getsize(data_file) > 0)
        df_new.to_csv(data_file, mode='a', header=write_header, index=False)
    except Exception as e: st.error(f"Failed to save data: {e}")

def load_all_form_metadata(return_dict=False) -> dict:
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
                return metadata if return_dict else {int(k): v for k, v in metadata.items()}
    except Exception: pass
    return {}

def save_form_metadata(form_id: int, definition: dict, prompt: str):
    metadata = load_all_form_metadata(return_dict=True)
    metadata[str(form_id)] = {"id": form_id, "definition": definition, "prompt": prompt, "created_at": pd.Timestamp.now().isoformat()}
    with open(METADATA_FILE, 'w') as f: json.dump(metadata, f, indent=4)

# --- LLM CORE FUNCTIONS ---
JSON_SCHEMA = {"clarification": "string | null", "fields": [{"name": "string", "type": "string", "label": "string", "validation": "string"}]}
def generate_form_json(user_request: str) -> dict:
    try:
        llm = Ollama(model="llama3")
        schema_string = json.dumps(JSON_SCHEMA)
        system_prompt = "You are a precise AI form builder. Create a form definition as a single, valid JSON object following this structure: {schema}. If the request is contradictory, set 'clarification' to a message and 'fields' to []. No text before or after the JSON."
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{request}")])
        chain = prompt | llm | JsonOutputParser()
        return chain.invoke({"request": user_request, "schema": schema_string})
    except MaxRetryError:
        st.error("Connection Error: Could not connect to Ollama. Please ensure the Ollama application is running.")
        return None
    except Exception as e:
        st.error(f"An unexpected LLM error occurred: {e}")
        return None

# --- UI FUNCTIONS ---
def render_custom_header():
    st.markdown("""
    <div class="header-container">
        <div class="logo-area"><h2 class="logo-text-gradient">AI FormGen</h2></div>
        <div class="nav-links">
            <a class="nav-link" href="#">Home</a>
            <a class="nav-link" href="#">Templates</a>
            <a class="nav-link" href="#">Docs</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def copy_link_component(form_id):
    js_script = f"""
    <script>
    function copyToClipboard(formId) {{
        const url = `${{window.location.origin}}${{window.location.pathname}}?form_id=${{formId}}`;
        navigator.clipboard.writeText(url).then(() => {{
            const button = document.getElementById(`copyButton_${{formId}}`);
            button.innerText = 'Copied!';
            setTimeout(() => {{ button.innerText = 'Copy Link'; }}, 2000);
        }});
    }}
    </script>
    <button id="copyButton_{form_id}" onclick="copyToClipboard('{form_id}')" class="stButton">Copy Link</button>
    """
    html(js_script)

def render_form(form_id: int, data: dict, is_viewer_mode=False):
    st.markdown(f"<h2>{'Form Submission' if is_viewer_mode else 'Generated Form'}</h2>", unsafe_allow_html=True)
    if not is_viewer_mode:
        copy_link_component(form_id)
    with st.form(key=f"dynamic_form_{form_id}", clear_on_submit=True):
        st.markdown(f"**Form ID:** `{form_id}`")
        for field in data.get('fields', []):
            widget_map = {"text": st.text_input, "email": st.text_input, "number": st.number_input, "date": st.date_input, "checkbox": st.checkbox}
            widget_map.get(field['type'], st.text_input)(field['label'])
        if st.form_submit_button("Submit Form"):
            st.success("Form Submitted Successfully! Thank you.")

def render_dashboard():
    st.markdown("<h1>Data Insights Dashboard</h1>", unsafe_allow_html=True)
    all_forms = load_all_form_metadata()
    if not all_forms:
        st.warning("No forms generated yet.")
        return
    st.markdown('<div class="dashboard-section-card">', unsafe_allow_html=True)
    form_options = {f"ID: {m['id']} | {m['prompt'][:50]}...": m['id'] for m in all_forms.values()}
    selected_key = st.selectbox("Choose a form to analyze:", list(form_options.keys()))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dashboard-section-card">', unsafe_allow_html=True)
    df = load_data(form_options[selected_key])
    st.markdown("<h3>Latest Submissions</h3>", unsafe_allow_html=True)
    st.dataframe(df.tail(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def check_password_main_body():
    st.markdown('<div class="dashboard-section-card login-card">', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Admin Login</h2>", unsafe_allow_html=True)
    password = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Enter admin password...")
    if st.button("Access Dashboard", use_container_width=True):
        if password == ADMIN_PASS:
            st.session_state['password_correct'] = True
            st.rerun()
        else:
            st.error("Incorrect Password")
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN APP LOGIC ---
if "form_id" in st.query_params:
    try:
        view_form_id = int(st.query_params["form_id"])
        form_meta = load_all_form_metadata(return_dict=True).get(str(view_form_id))
        if form_meta:
            render_form(view_form_id, form_meta['definition'], is_viewer_mode=True)
        else:
            st.error("Form not found.")
    except (ValueError, TypeError):
        st.error("Invalid form ID.")
else:
    render_custom_header()

    # Use a styled st.radio for clear, top-level navigation instead of tabs
    page = st.radio(
        "Navigation", 
        ["Form Creator", "Admin Dashboard"], 
        horizontal=True, 
        label_visibility="collapsed"
    )

    if page == "Form Creator":
        st.markdown("<div class='hero-section'><h1>Craft Your Perfect Form with AI</h1><p>Describe the form you need in plain English. Our AI will build it for you in seconds.</p></div>", unsafe_allow_html=True)

        st.markdown("<div class='prompt-container'>", unsafe_allow_html=True)
        user_prompt = st.text_area("Prompt", placeholder="e.g., A simple contact form with name, email, and a message field.", label_visibility="collapsed")

        if st.button("Generate Form", use_container_width=True):
            st.session_state.pop('form_to_render', None)
            if user_prompt:
                with st.spinner("Generating form..."):
                    form_def = generate_form_json(user_prompt)
                if form_def and not form_def.get('clarification'):
                    json_str = json.dumps(form_def, sort_keys=True).encode('utf-8')
                    form_id = int(hashlib.sha256(json_str).hexdigest(), 16) % (10**10)
                    save_form_metadata(form_id, form_def, user_prompt)
                    st.session_state.form_to_render = (form_id, form_def)
                elif form_def:
                    st.warning(f"AI needs clarification: {form_def['clarification']}")
            else:
                st.error("Please enter a description for the form.")
        st.markdown("</div>", unsafe_allow_html=True)

        if 'form_to_render' in st.session_state:
            st.markdown("<hr style='border-color: var(--border-color); margin: 2rem 0;'>", unsafe_allow_html=True)
            form_id, form_def = st.session_state.form_to_render
            render_form(form_id, form_def)

    elif page == "Admin Dashboard":
        if st.session_state.get('password_correct', False):
            render_dashboard()
        else:
            check_password_main_body()