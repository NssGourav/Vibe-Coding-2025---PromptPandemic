import streamlit as st
import json
import re
import os
import requests
import hashlib 
from langchain_community.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date 
import pandas as pd

# --- Constants and Environment Setup ---
# CRITICAL FIX: Set LLM_MODEL_TO_USE based on OS check, not immediate API key check
# This ensures the app starts even if the secret hasn't fully loaded/registered
if 'GEMINI_API_KEY' in os.environ and os.environ['GEMINI_API_KEY']:
    GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
    LLM_MODEL_TO_USE = "gemini-2.5-flash-preview-05-20"
else:
    # Local/Fallback mode
    GEMINI_API_KEY = ""
    LLM_MODEL_TO_USE = "llama3" 

LLM_BASE_URL = "http://localhost:11434"

# --- Page Navigation State ---
if 'page' not in st.session_state:
    st.session_state.page = 'HOME'
if 'form_data_json' not in st.session_state:
    st.session_state.form_data_json = None
if 'current_form_id' not in st.session_state:
    st.session_state.current_form_id = None
if 'submissions_db' not in st.session_state:
    st.session_state['submissions_db'] = {}
    
def set_page(page_name):
    st.session_state.page = page_name

# --- Firestore Mock/Integration Placeholder (using session state) ---
def get_form_collection_path(form_id: str) -> str:
    """Generates a pseudo-unique path for form submission data."""
    return f"form_submissions/{form_id}"

def save_form_submission(form_id: str, data: dict):
    """Simulates saving data to Firestore using a unique ID for the collection."""
    path = get_form_collection_path(form_id)
    if path not in st.session_state['submissions_db']:
        st.session_state['submissions_db'][path] = []
    
    submission = {"timestamp": str(date.today()), **data} 
    st.session_state['submissions_db'][path].append(submission)

def get_all_submissions(form_id: str) -> List[dict]:
    """Retrieves all submissions for a specific form ID."""
    path = get_form_collection_path(form_id)
    return st.session_state.get('submissions_db', {}).get(path, [])


# --- 1. Pydantic Schema for Structured Output ---

class FieldDefinition(BaseModel):
    """A single field definition for the generated form."""
    name: str = Field(description="The unique, snake_case name for the field (e.g., 'full_name').")
    label: str = Field(description="The human-readable label for the field (e.g., 'Full Name').")
    type: str = Field(description="The input type (choose from: 'text', 'email', 'number', 'date', 'password', 'radio', 'checkbox', 'selectbox', 'textarea').")
    validation: str = Field(description="Comma-separated validation rules (e.g., 'required', 'email_format', 'min_length:5'). Use 'optional' if no strict rules apply.")
    options: Optional[List[str]] = Field(default=None, description="List of options for 'radio' or 'selectbox' types. Null otherwise.")

class FormSchema(BaseModel):
    """The root schema containing either the fields or a clarification message."""
    clarification: Optional[str] = Field(default=None, description="A message if the request is contradictory (e.g., anonymous but requires email). Should be null if fields are present.")
    fields: Optional[List[FieldDefinition]] = Field(default=None, description="List of fields if the request is valid. Null if clarification is present.")


# --- 2. Prompt Engineering and LLM Integration ---

parser = PydanticOutputParser(pydantic_object=FormSchema)
format_instructions = parser.get_format_instructions()

SYSTEM_PROMPT = f"""
You are a highly specialized AI assistant for building forms. Your task is to analyze a user's request for a form and output the form's schema in a precise JSON format that strictly conforms to the provided schema.

--- INSTRUCTION SET ---
1. STRICT JSON FORMAT: You MUST return a single JSON object that perfectly adheres to this schema. DO NOT include any introductory text, commentary, or markdown fences.
{format_instructions}

2. CONTRADICTION HANDLING: If the user's request is contradictory (e.g., asking for an anonymous form that requires personal details) or logically unsound:
    a. Set the 'clarification' field to a polite, clear sentence asking the user to resolve the conflict.
    b. Set the 'fields' field to null.

3. FIELD GENERATION: If the request is valid:
    a. Set the 'clarification' field to null.
    b. Generate the 'fields' list according to the schema. Infer the best 'type' and 'validation' based on the label.

Generate the JSON for the user's current request based on these instructions.
"""

@st.cache_data(show_spinner=f"Generating form structure with {LLM_MODEL_TO_USE.upper()}...")
def generate_form_json(prompt: str) -> str:
    """Invokes the appropriate LLM (local or cloud) to generate a structured JSON string."""
    full_prompt = SYSTEM_PROMPT + f"\nUser Request: {prompt}"

    if GEMINI_API_KEY:
        # --- CLOUD DEPLOYMENT (GEMINI API) ---
        return call_gemini_api(full_prompt)
    else:
        # --- LOCAL DEVELOPMENT (OLLAMA) ---
        # Note: This path should not be hit on the cloud but is the fast path for local testing.
        return call_ollama_local(full_prompt)

def call_gemini_api(prompt_text: str) -> str:
    """Makes a request to the Gemini API for structured JSON output."""
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': GEMINI_API_KEY
        }
        
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "config": {
                "responseMimeType": "application/json",
                "responseSchema": FormSchema.model_json_schema()
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if result.get('candidates') and result['candidates'][0].get('content'):
            json_text = result['candidates'][0]['content']['parts'][0]['text']
            return json_text

        return json.dumps({"clarification": "Cloud API returned an unexpected response structure or was blocked. Check API key status."})
        
    except requests.exceptions.RequestException as e:
        return json.dumps({"clarification": f"Cloud API Connection Error (Gemini): The service is currently inaccessible. Check API key status. Error: {e}"})
    except Exception as e:
        return json.dumps({"clarification": f"Gemini API Response Error: {e}"})

def call_ollama_local(prompt_text: str) -> str:
    """Makes a request to the local Ollama server."""
    try:
        # Instantiating here instead of globally prevents startup crashes
        llm = ChatOllama(model=LLM_MODEL_TO_USE, temperature=0.0, format="json", base_url=LLM_BASE_URL)
        messages = [HumanMessage(content=prompt_text)]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return json.dumps({"clarification": f"Ollama Local Error: Is the server running? Error: {e}"})


# --- UTILITY FUNCTION FOR CLEANING LLM OUTPUT ---

def clean_json_output(text: str) -> str:
    """Removes leading/trailing text and markdown fences (```json) from LLM output."""
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        return text[start:end]
    except ValueError:
        return text 

# --- 3. Dynamic UI Rendering Functions (Shared Validation) ---

def perform_validation(form_id: str, data: FormSchema) -> Dict[str, Any]:
    """Collects submitted data and performs client-side validation."""
    
    submitted_data = {
        f.name: st.session_state.get(f"{form_id}_{f.name}")
        for f in data.fields if f.name in st.session_state
    }
    errors = []
    
    for field in data.fields:
        rules = [r.strip() for r in field.validation.split(',')]
        value = submitted_data.get(field.name)
        str_value = str(value).strip() if value is not None and value is not date.today() else ""

        # Rule 1: Required check
        if "required" in rules and not str_value:
            errors.append(f"❌ {field.label} is required.")
            continue
        
        if value is None and "required" in rules:
             errors.append(f"❌ {field.label} is required.")
             continue


        if str_value:
            # Rule 2: Email format check
            if "email_format" in rules or field.type == 'email':
                if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", str_value):
                    errors.append(f"❌ {field.label} requires a valid email format.")

            # Rule 3 & 4: Length checks
            for rule in rules:
                if rule.startswith("min_length:"):
                    try:
                        min_len = int(rule.split(':')[1])
                        if len(str_value) < min_len:
                            errors.append(f"❌ {field.label} must be at least {min_len} characters long.")
                    except ValueError: pass
                
                if rule.startswith("exact_length:"):
                    try:
                        exact_len = int(rule.split(':')[1])
                        len_to_check = len(str_value.replace('.', '', 1)) if field.type == 'number' else len(str_value)
                        if len_to_check != exact_len: 
                            errors.append(f"❌ {field.label} must be exactly {exact_len} characters long.")
                    except ValueError: pass
                        
            # Rule 5: Number only check 
            if "number_only" in rules:
                if not str_value.replace('.', '', 1).isdigit():
                    errors.append(f"❌ {field.label} must contain only numeric digits.")
                    
    return {"errors": errors, "data": submitted_data}

def draw_form_widgets(form_id: str, data: FormSchema):
    """Draws the actual Streamlit widgets based on the parsed schema."""
    for field in data.fields:
        key = field.name
        label = field.label
        widget_key = f"{form_id}_{key}" 

        # Handle different field types
        if field.type in ['text', 'email', 'password']:
            st.text_input(label, type=field.type if field.type == 'password' else 'default', key=widget_key)
        elif field.type == 'textarea':
            st.text_area(label, key=widget_key)
        elif field.type == 'number':
            st.number_input(label, step=1, key=widget_key)
        elif field.type == 'date':
            st.date_input(label, key=widget_key, value=date.today())
        elif field.type == 'radio' and field.options:
            st.radio(label, field.options, key=widget_key)
        elif field.type == 'selectbox' and field.options:
            st.selectbox(label, field.options, key=widget_key)
        elif field.type == 'checkbox':
            st.checkbox(label, key=widget_key)

def render_live_form():
    """Renders the final, live, submission-ready form (Google Forms Style)."""
    form_id = st.session_state.current_form_id
    json_data = st.session_state.form_data_json
    
    if not form_id or not json_data:
        st.error("No form data found. Please generate or load a form first.")
        return
        
    try:
        # Re-parse data for safety
        data = FormSchema.model_validate_json(clean_json_output(json_data))
        
        st.title(f"Live Form View (ID: {form_id[:8]}...)")
        st.markdown("---")
        st.subheader("Please fill out this form.")
        
        with st.form(key=f"live_form_{form_id}"):
            # Draw the widgets based on the stored schema
            draw_form_widgets(form_id, data)

            submitted = st.form_submit_button("Submit Your Response", type="primary")

            if submitted:
                validation_result = perform_validation(form_id, data)
                
                if validation_result['errors']:
                    st.error("Please correct the following errors before submitting:")
                    for error in validation_result['errors']:
                        st.write(error)
                else:
                    st.success("Thank you! Your response has been saved.")
                    save_form_submission(form_id, validation_result['data'])
                    st.toast("Data saved successfully!")
                    
    except Exception as e:
        st.error(f"Error rendering live form: {e}")


# --- Form Generation/Editing Stage ---

def render_form_editor():
    """Renders the editor view, showing LLM output and submissions."""
    st.header("✍️ Form Editor & Analytics")
    st.caption("Review the generated JSON, make manual adjustments, and track submissions.")

    form_id = st.session_state.current_form_id
    json_data = st.session_state.form_data_json
    
    cols = st.columns([1, 1, 3])
    with cols[0]:
        st.button("↩️ Back to Generator", on_click=set_page, args=('HOME',))
    with cols[1]:
        st.button("🚀 Publish Live Form", on_click=set_page, args=('LIVE',), type="primary")
    
    st.markdown("---")

    col_json, col_preview = st.columns(2)

    with col_json:
        st.subheader("Generated JSON Schema")
        # Allow manual editing of the generated schema
        edited_json = st.text_area(
            "Edit Schema JSON:",
            value=clean_json_output(json_data),
            height=400,
            key='edited_json_input'
        )
        
        if st.button("Apply Changes & Update Preview"):
            try:
                # Validate the edited JSON structure against the Pydantic schema
                FormSchema.model_validate_json(edited_json)
                st.session_state.form_data_json = edited_json # Save validated JSON
                st.success("Schema updated! Preview on the right reflects changes.")
                st.rerun()
            except Exception as e:
                st.error(f"Invalid JSON Schema: {e}")

    with col_preview:
        st.subheader("Live Preview")
        st.caption("The form will look and behave like this when published.")
        
        try:
            preview_schema = FormSchema.model_validate_json(clean_json_output(st.session_state.form_data_json))
            
            with st.form(key=f"preview_form_{form_id}"):
                draw_form_widgets(form_id, preview_schema)
                st.form_submit_button("Preview Submit (No Save)", disabled=True)
                
            st.markdown("---")
            st.subheader("Submissions Data")
            all_submissions = get_all_submissions(form_id)
            if all_submissions:
                df = pd.DataFrame(all_submissions)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No submissions yet for this form.")
            
        except Exception:
            st.error("Cannot render preview. Please fix JSON syntax in the editor.")
            st.code(st.session_state.form_data_json)


# --- Form Generation Stage ---

def render_home_generator():
    """Renders the initial form generation interface."""
    st.header("🔮 Natural Language Form Generator")
    
    if st.session_state.current_form_id:
        st.button(f"➡️ Go to Editor (ID: {st.session_state.current_form_id[:8]}...)", 
                  on_click=set_page, args=('EDIT',))
    
    # Display the current mode
    if GEMINI_API_KEY:
        st.info(f"Deployment Mode: **Cloud (Gemini API)**. Ready for external sharing!")
    else:
        st.warning(f"Deployment Mode: **Local ({LLM_MODEL_TO_USE})**. Ensure `ollama serve` is running for use.")


    # Input area for the user's request
    user_prompt = st.text_area(
        "Describe your form here:",
        placeholder="e.g., A registration form for new club members with name, email, and favorite anime.",
        height=100,
        key="user_prompt_input"
    )

    # --- Generate Form ID based on Prompt ---
    user_prompt_clean = user_prompt.strip()
    form_id = hashlib.sha256(user_prompt_clean.encode('utf-8')).hexdigest() if user_prompt_clean else None
    
    # The "Generate" button to trigger the process
    if st.button("Generate Form", type="primary", disabled=not user_prompt_clean):
        st.session_state.current_form_id = form_id
        
        # 1. Clear cached LLM data if the prompt has changed
        st.cache_data.clear()

        # 2. Call the LLM function and store the result
        json_result = generate_form_json(user_prompt_clean)
        st.session_state.form_data_json = json_result
        
        # 3. Move to the Edit page automatically
        set_page('EDIT')
        st.rerun()

# --- Main Routing ---

def main_router():
    """Controls which interface is displayed based on session state."""
    
    st.sidebar.title("App Navigation")
    st.sidebar.button("Home (Generator)", on_click=set_page, args=('HOME',))
    if st.session_state.current_form_id:
        st.sidebar.button("Editor/Analytics", on_click=set_page, args=('EDIT',))
        st.sidebar.button("Live Form View", on_click=set_page, args=('LIVE',))

    if st.session_state.page == 'HOME':
        render_home_generator()
    elif st.session_state.page == 'EDIT':
        if st.session_state.form_data_json:
            render_form_editor()
        else:
            # Fallback if somehow landed here without data
            set_page('HOME')
            st.rerun()
    elif st.session_state.page == 'LIVE':
        render_live_form()
    
    
if __name__ == "__main__":
    main_router()
```
