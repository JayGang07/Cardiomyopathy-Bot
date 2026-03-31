# app.py
import streamlit as st
import os
import json
import asyncio
import re
import math
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import LLMChain, ConversationChain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
# Add this with your other imports
from google.api_core.exceptions import ResourceExhausted
# --- App Configuration ---
st.set_page_config(page_title="Cardiomyopathy Assistant", layout="centered")

# --- API KEY CONFIGURATION ---
# IMPORTANT: Ensure your API key is active.
#os.environ["GOOGLE_API_KEY"] = "AIzaSyCM4-BbwIOdv_jJitFhqovMJ6mZ-RBWsFw"
import random

# --- API KEY CONFIGURATION ---

# POOL 1: For Patient Educator & HCM Calculator (Lighter usage)
# POOL 1: For Patient Educator & HCM Calculator (Lighter usage)
EDUCATOR_HCM_KEYS = st.secrets["EDUCATOR_HCM_KEYS"]

# POOL 2: For Symptom Analyzer (Heavy usage - add more keys here)
ANALYZER_KEYS = st.secrets["ANALYZER_KEYS"]

#C:\Users\jayku\Downloads\Cardiomyopathy_bot\app.py
def get_api_key(service_type):
    """
    Returns a random key from the requested pool to distribute load.
    service_type: 'basic' (Educator/HCM) or 'analyzer' (Symptom Analyzer)
    """
    if service_type == 'analyzer':
        return random.choice(ANALYZER_KEYS)
    else:
        return random.choice(EDUCATOR_HCM_KEYS)
# --- Mobile-Friendly UI CSS ---
st.markdown("""
<style>
@media (max-width: 768px) {
    .main .block-container {
        padding-top: 2rem; padding-right: 1.5rem; padding-left: 1.5rem; padding-bottom: 2rem;
    }
    .stChatMessage { font-size: 1.05rem; }
}
</style>
""", unsafe_allow_html=True)


# --- SHARED HELPER CLASSES AND FUNCTIONS ---
# --- ADD THIS FUNCTION TO SHARED HELPER CLASSES ---

def execute_with_retry(chain, inputs, service_type):
    """
    Executes a chain. If a Quota Error occurs, it picks a new key and retries.
    """
    # Select the correct key pool
    pool = ANALYZER_KEYS if service_type == 'analyzer' else EDUCATOR_HCM_KEYS
    
    # Try as many times as we have keys + 1 buffer
    max_retries = len(pool) + 1
    current_key = chain.llm.google_api_key
    
    for attempt in range(max_retries):
        try:
            # Attempt to run the chain based on its type
            if isinstance(chain, ConversationChain):
                # ConversationChain usually takes a string input for 'input'
                if isinstance(inputs, str):
                    return chain.predict(input=inputs)
                return chain.predict(**inputs)
            else:
                # Standard LLMChain uses invoke
                return chain.invoke(inputs)
                
        except Exception as e:
            # Check if the error is a Quota/Limit error
            error_str = str(e)
            is_quota_error = "429" in error_str or "ResourceExhausted" in error_str or "quota" in error_str.lower()
            
            if is_quota_error and attempt < max_retries - 1:
                # 1. Pick a new key that is NOT the current failed key
                available_keys = [k for k in pool if k != current_key]
                if not available_keys: available_keys = pool # Fallback if only 1 key exists
                new_key = random.choice(available_keys)
                
                # 2. Re-initialize the LLM with the new key
                # We recreate the object to ensure the client resets completely
                chain.llm = ChatGoogleGenerativeAI(
                    model=chain.llm.model_name,
                    temperature=chain.llm.temperature,
                    google_api_key=new_key
                )
                
                # 3. Update tracker and loop again
                current_key = new_key
                continue 
            else:
                # If it's a different error (like a logic bug) or we ran out of keys, crash.
                raise e
class InMemoryChatHistory(BaseChatMessageHistory):
    def __init__(self): self.messages = []
    def add_message(self, message): self.messages.append(message)
    def add_user_message(self, message: str): self.add_message(HumanMessage(content=message))
    def add_ai_message(self, message: str): self.add_message(AIMessage(content=message))
    def clear(self): self.messages = []

@st.cache_data
def load_medical_data():
    """Loads the cardiomyopathy specific data to guide the AI."""
    file_path = "cardiomyopathy_medical_data.json"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# --- LOGIC FOR PAGE 1: PATIENT EDUCATOR ---

@st.cache_resource
def setup_educator_chain():
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            google_api_key=get_api_key('basic')
        )
    
    # Load medical data to inject into context
    medical_data = load_medical_data()
    # Format data as a string for the prompt, or use a default if empty
    medical_context = json.dumps(medical_data, indent=2) if medical_data else "No specific external data found. Use general medical knowledge about Cardiomyopathy."

    prompt_template = """System: You are a specialized cardiomyopathy educator providing evidence-based information.

    **Knowledge Base:**
    {medical_context}

    **Guidelines:**
    1. Use Reference Data to answer cardiomyopathy questions accurately
    2. Explain complex terms in simple language
    3. NEVER diagnose or suggest treatment - only educate
    4. Offer depth options: "Would you like a brief overview or detailed explanation?"
    5. Include relevant statistics/facts from Reference Data when appropriate
    6. Suggest reliable resources or next steps when relevant

    **Response Structure:**
    - Direct answer to the question in 3-4 lines , and ask the user for a detailed answer : if yes - give the detailed answer , and if no - move to the next question.
    - Key facts from Reference Data
    - Important considerations
    - Suggest speaking with healthcare provider for personal medical advice

    **Conversation History:**
    {history}

    **User Question:** {input}
    **Educational Response:**"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"], 
        partial_variables={"medical_context": medical_context},
        template=prompt_template
    )
    
    # Use ConversationChain with ConversationBufferMemory
    memory = ConversationBufferMemory(memory_key="history")
    chain = ConversationChain(llm=llm, prompt=prompt, memory=memory, verbose=False)
    return chain

def render_patient_educator():
    st.title("Patient Educator Chatbot 💬")
    st.markdown("This is a space for general questions about cardiomyopathy, its types, symptoms, and treatments.")
    
    if "educator_chain" not in st.session_state:
        st.session_state.educator_chain = setup_educator_chain()
    
    if "home_messages" not in st.session_state:
        st.session_state.home_messages = []

    for message in st.session_state.home_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a general question..."):
        st.session_state.home_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # --- FIND THIS SECTION IN render_patient_educator ---
        with st.spinner("Thinking..."):
            # OLD LINE: response = st.session_state.educator_chain.predict(input=prompt)
            
            # NEW REPLACEMENT LINE:
            response = execute_with_retry(st.session_state.educator_chain, prompt, 'basic')
            
            st.session_state.home_messages.append({"role": "assistant", "content": response})
            st.rerun()

# --- LOGIC FOR PAGE 2: SYMPTOM ANALYZER ---

class SymptomAnalyzerLogic:
    def __init__(self):
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
        except RuntimeError: pass
        #self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3) # Lower temp for strict logic
        self.medical_data = load_medical_data()
    
    def _get_llm(self):
        """Helper to get a fresh LLM instance with a random key from the ANALYZER pool."""
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.3,
            google_api_key=get_api_key('analyzer') # <--- UPDATED: USES POOL 2
        )

    def get_next_interaction(self, transcript: str) -> dict:
        # Prepare context from JSON
        medical_context_str = json.dumps(self.medical_data, indent=2) if self.medical_data else "Focus on Cardiomyopathy symptoms."

        # FIX: JSON examples now use double curly braces {{ }} to escape them. 
        # Actual variables {medical_context} and {transcript} keep single braces.
        template = """System: You are an AI Cardiologist specialized in CARDIOMYOPATHY. 
        Your job is to conduct a **highly structured, non-repetitive, deep clinical interview** and always return the next **single multiple-choice question** in strict JSON.

        -------------------------------------
        REFERENCE MEDICAL DATA:
        {medical_context}
        -------------------------------------

        GOAL:
        Ask the **next best clinical question** to fully understand symptoms, severity, functional impact, comorbidities, and risk factors related to cardiomyopathy.

        -------------------------------------
        INTERVIEW PROTOCOL (Strict Order):
        1. Chief Complaint  
        2. HPI - Onset  
        3. HPI - Nature/Quality  
        4. HPI - Severity  
        5. HPI - Timing, Duration, Progression  
        6. HPI - Triggers & Relievers  
        7. Associated Symptoms (Dyspnea, fatigue, palpitations, edema, syncope)  
        8. Red Flags (syncope on exertion, PND, orthopnea, chest pressure)  
        9. Medical History (HTN, diabetes, thyroid, anemia, viral illness, chemo, pregnancy)  
        10. Medication History  
        11. Family History (sudden death, genetic cardiomyopathy)  
        12. Lifestyle (smoking, alcohol, exercise levels)  
        13. Occupational/Stress Factors  
        14. Functional Capacity (NYHA-like assessment)

        -------------------------------------
        DETAILED QUESTIONING RULES:
        1. **Review entire transcript and NEVER repeat a question** (even indirectly).
        2. Ask **ONE** question at a time.
        3. **ALL QUESTIONS MUST BE MULTI-CHOICE**, always including **"Other"** for custom responses.
        4. If the patient answers a topic, ask the **next deeper sub-question**.
        Examples:
        - If “Yes, I smoke” → ask:  
            “How often do you smoke?” → ["Daily", "Weekly", "Occasionally", "Used to but quit", "Other"]
        - If they report chest pain → ask deeper:  
            “What activities trigger the pain?”
        5. If answer is vague, ask a **clarification question**, still multi-choice.
        6. Cover **every protocol topic in detail**, with layered depth.
        7. When ALL topics have been thoroughly explored → return:
        {{ "question": "INTERVIEW_COMPLETE", "type": "done" }}
        8. No commentary—ONLY return a JSON object.

        -------------------------------------
        TRANSCRIPT SO FAR:
        {transcript}

        -------------------------------------
        OUTPUT FORMAT:
        Return ONLY a valid JSON object:

        {{
        "question": "...",
        "options": ["...", "...", "Other"],
        "type": "multi_choice"
        }}
        """
        
        prompt = PromptTemplate(
            input_variables=["transcript"], 
            partial_variables={"medical_context": medical_context_str},
            template=template
        )
        
        chain = LLMChain(llm=self._get_llm(), prompt=prompt)
        
        for _ in range(3): # Retry loop for robustness
            try:
                result = execute_with_retry(chain, {'transcript': transcript}, 'analyzer')
                response_text = result['text']
                # Clean up potential markdown formatting from LLM (Gemini often adds ```json)
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_text)
                
                if "question" in data and "type" in data: 
                    return data
            except (json.JSONDecodeError, KeyError, AttributeError):
                continue
        
        # Fallback safe question
        return {"question": "Could you please describe your symptoms in more detail?", "type": "text"}
    
    def generate_summary(self, age: str, gender: str, transcript: str) -> str:
        template = """As an expert cardiologist, create a **Pre-Consultation Summary** based on the provided patient interview transcript.
        The summary should be well-organized, concise, and written in professional medical language.

        **Instructions:**
        1. Structure the summary using the following markdown sections:
           - **Chief Complaint**
           - **History of Presenting Illness**
           - **Past Medical History**
           - **Family History**
           - **Lifestyle & Social History**
           - **Clinical Impression**
        2. Do NOT add a separate "Patient Demographics" section.
        3. Conclude with a '---' horizontal line.

        **INTERVIEW TRANSCRIPT:**
        ---
        {transcript}
        ---

        **PRE-CONSULTATION SUMMARY:**"""
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(llm=self._get_llm(), prompt=prompt)
        return chain.invoke({'age': age, 'gender': gender, 'transcript': transcript})['text']

def render_symptom_analyzer():
    st.title("Guided Symptom Analyzer 🩺")
    st.markdown("This intelligent interview will ask dynamic, in-depth questions to build a comprehensive health summary.")
    
    if 'analyzer' not in st.session_state or st.session_state.analyzer is None:
        st.session_state.analyzer = {
            "logic": SymptomAnalyzerLogic(), 
            "stage": 'START', 
            "messages": [], 
            "context": {"transcript": ""}, 
            "summary": None, 
            "demographics_step": 'NAME', 
            "current_interaction_data": None
        }
    analyzer_state = st.session_state.analyzer

    ALL_COUNTRY_CODES = [
        "+93 (Afghanistan)", "+355 (Albania)", "+213 (Algeria)", "+1-684 (American Samoa)", "+376 (Andorra)", "+244 (Angola)", "+1-264 (Anguilla)", "+672 (Antarctica)",
        "+1-268 (Antigua and Barbuda)", "+54 (Argentina)", "+374 (Armenia)", "+297 (Aruba)", "+61 (Australia)", "+43 (Austria)", "+994 (Azerbaijan)", "+1-242 (Bahamas)",
        "+973 (Bahrain)", "+880 (Bangladesh)", "+1-246 (Barbados)", "+375 (Belarus)", "+32 (Belgium)", "+501 (Belize)", "+229 (Benin)", "+1-441 (Bermuda)", "+975 (Bhutan)",
        "+591 (Bolivia)", "+387 (Bosnia and Herzegovina)", "+267 (Botswana)", "+55 (Brazil)", "+246 (British Indian Ocean Territory)", "+1-284 (British Virgin Islands)",
        "+673 (Brunei)", "+359 (Bulgaria)", "+226 (Burkina Faso)", "+257 (Burundi)", "+855 (Cambodia)", "+237 (Cameroon)", "+1 (Canada)", "+238 (Cape Verde)",
        "+1-345 (Cayman Islands)", "+236 (Central African Republic)", "+235 (Chad)", "+56 (Chile)", "+86 (China)", "+57 (Colombia)", "+269 (Comoros)",
        "+682 (Cook Islands)", "+506 (Costa Rica)", "+385 (Croatia)", "+53 (Cuba)", "+599 (Curacao)", "+357 (Cyprus)", "+420 (Czech Republic)",
        "+243 (Democratic Republic of the Congo)", "+45 (Denmark)", "+253 (Djibouti)", "+1-767 (Dominica)", "+1-809 (Dominican Republic)", "+670 (East Timor)", "+593 (Ecuador)",
        "+20 (Egypt)", "+503 (El Salvador)", "+240 (Equatorial Guinea)", "+291 (Eritrea)", "+372 (Estonia)", "+251 (Ethiopia)", "+500 (Falkland Islands)",
        "+298 (Faroe Islands)", "+679 (Fiji)", "+358 (Finland)", "+33 (France)", "+689 (French Polynesia)", "+241 (Gabon)", "+220 (Gambia)", "+995 (Georgia)",
        "+49 (Germany)", "+233 (Ghana)", "+350 (Gibraltar)", "+30 (Greece)", "+299 (Greenland)", "+1-473 (Grenada)", "+1-671 (Guam)", "+502 (Guatemala)",
        "+44-1481 (Guernsey)", "+224 (Guinea)", "+245 (Guinea-Bissau)", "+592 (Guyana)", "+509 (Haiti)", "+504 (Honduras)", "+852 (Hong Kong)", "+36 (Hungary)",
        "+354 (Iceland)", "+91 (India)", "+62 (Indonesia)", "+98 (Iran)", "+964 (Iraq)", "+353 (Ireland)", "+44-1624 (Isle of Man)", "+972 (Israel)", "+39 (Italy)",
        "+225 (Ivory Coast)", "+1-876 (Jamaica)", "+81 (Japan)", "+44-1534 (Jersey)", "+962 (Jordan)", "+7 (Kazakhstan)", "+254 (Kenya)", "+686 (Kiribati)",
        "+383 (Kosovo)", "+965 (Kuwait)", "+996 (Kyrgyzstan)", "+856 (Laos)", "+371 (Latvia)", "+961 (Lebanon)", "+266 (Lesotho)", "+231 (Liberia)", "+218 (Libya)",
        "+423 (Liechtenstein)", "+370 (Lithuania)", "+352 (Luxembourg)", "+853 (Macau)", "+389 (Macedonia)", "+261 (Madagascar)", "+265 (Malawi)", "+60 (Malaysia)",
        "+960 (Maldives)", "+223 (Mali)", "+356 (Malta)", "+692 (Marshall Islands)", "+222 (Mauritania)", "+230 (Mauritius)", "+262 (Mayotte)", "+52 (Mexico)",
        "+691 (Micronesia)", "+373 (Moldova)", "+377 (Monaco)", "+976 (Mongolia)", "+382 ( Montenegro)", "+1-664 (Montserrat)", "+212 (Morocco)", "+258 (Mozambique)",
        "+95 (Myanmar)", "+264 (Namibia)", "+674 (Nauru)", "+977 (Nepal)", "+31 (Netherlands)", "+599 (Netherlands Antilles)", "+687 (New Caledonia)", "+64 (New Zealand)",
        "+505 (Nicaragua)", "+227 (Niger)", "+234 (Nigeria)", "+683 (Niue)", "+850 (North Korea)", "+1-670 (Northern Mariana Islands)", "+47 (Norway)", "+968 (Oman)",
        "+92 (Pakistan)", "+680 (Palau)", "+970 (Palestine)", "+507 (Panama)", "+675 (Papua New Guinea)", "+595 (Paraguay)", "+51 (Peru)", "+63 (Philippines)",
        "+48 (Poland)", "+351 (Portugal)", "+1-787 (Puerto Rico)", "+974 (Qatar)", "+242 (Republic of the Congo)", "+262 (Reunion)", "+40 (Romania)", "+7 (Russia)",
        "+250 (Rwanda)", "+590 (Saint Barthelemy)", "+290 (Saint Helena)", "+1-869 (Saint Kitts and Nevis)", "+1-758 (Saint Lucia)", "+508 (Saint Pierre and Miquelon)",
        "+1-784 (Saint Vincent and the Grenadines)", "+685 (Samoa)", "+378 (San Marino)", "+239 (Sao Tome and Principe)", "+966 (Saudi Arabia)", "+221 (Senegal)",
        "+381 (Serbia)", "+248 (Seychelles)", "+232 (Sierra Leone)", "+65 (Singapore)", "+1-721 (Sint Maarten)", "+421 (Slovakia)", "+386 (Slovenia)",
        "+677 (Solomon Islands)", "+252 (Somalia)", "+27 (South Africa)", "+82 (South Korea)", "+211 (South Sudan)", "+34 (Spain)", "+94 (Sri Lanka)", "+249 (Sudan)",
        "+597 (Suriname)", "+268 (Swaziland)", "+46 (Sweden)", "+41 (Switzerland)", "+963 (Syria)", "+886 (Taiwan)", "+992 (Tajikistan)", "+255 (Tanzania)",
        "+66 (Thailand)", "+228 (Togo)", "+690 (Tokelau)", "+676 (Tonga)", "+1-868 (Trinidad and Tobago)", "+216 (Tunisia)", "+90 (Turkey)", "+993 (Turkmenistan)",
        "+1-649 (Turks and Caicos Islands)", "+688 (Tuvalu)", "+1-340 (U.S. Virgin Islands)", "+256 (Uganda)", "+380 (Ukraine)", "+971 (United Arab Emirates)",
        "+44 (United Kingdom)", "+1 (United States)", "+598 (Uruguay)", "+998 (Uzbekistan)", "+678 (Vanuatu)", "+379 (Vatican)", "+58 (Venezuela)", "+84 (Vietnam)",
        "+681 (Wallis and Futuna)", "+967 (Yemen)", "+260 (Zambia)", "+263 (Zimbabwe)"
    ]

    def start_interview():
        st.session_state.analyzer = {
            "logic": SymptomAnalyzerLogic(), 
            "stage": 'DEMOGRAPHICS', 
            "messages": [{"role": "assistant", "content": "To begin, what is your full name?"}], 
            "context": {"transcript": ""}, 
            "summary": None, 
            "demographics_step": 'NAME', 
            "current_interaction_data": None
        }
        st.rerun()

    def process_user_response(question, answer):
        analyzer_state["messages"].append({"role": "user", "content": answer})
        analyzer_state["context"]["transcript"] += f"Q: {question}\nA: {answer}\n\n"
        analyzer_state["current_interaction_data"] = None # Clear old question to trigger generation of next
        analyzer_state["stage"] = 'IN_ANALYSIS'
        st.rerun()

    # Display Chat History
    for msg in analyzer_state["messages"]:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    current_stage = analyzer_state["stage"]

    # 1. Start Screen
    if current_stage == 'START':
        if st.button("Begin Guided Interview"): start_interview()

    # 2. Summary Generation
    elif current_stage == 'SUMMARY':
        if analyzer_state.get("summary") is None:
            with st.spinner("Analyzing your responses and creating the final summary..."):
                try:
                    summary = analyzer_state["logic"].generate_summary(
                        age=analyzer_state["context"].get('age', 'N/A'), 
                        gender=analyzer_state["context"].get('gender', 'N/A'), 
                        transcript=analyzer_state["context"]["transcript"]
                    )
                    if not summary or len(summary) < 20:
                        summary = "Could not generate a summary for this session."
                    analyzer_state["summary"] = summary
                    analyzer_state["messages"].append({"role": "assistant", "content": "Thank you. The guided interview is complete. Below is your structured summary."})
                    analyzer_state["messages"].append({"role": "assistant", "content": summary})
                except Exception as e:
                    st.error(f"An error occurred during summary generation: {e}")
                    analyzer_state["summary"] = "Error"
                st.rerun()

        if analyzer_state.get("summary") and analyzer_state["summary"] != "Error":
            name = analyzer_state["context"].get('name', 'N/A'); age = analyzer_state["context"].get('age', 'N/A'); gender = analyzer_state["context"].get('gender', 'N/A')
            email = analyzer_state["context"].get('email', 'N/A'); phone = analyzer_state["context"].get('phone', 'N/A')
            generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_header = f"""PATIENT DETAILS
===========================
Name: {name}
Age: {age}
Gender: {gender}
Email: {email}
Phone: {phone}
Report Generated: {generation_date}
"""
            report_content = report_header + f"\nPRE-CONSULTATION SUMMARY\n===========================\n\n{analyzer_state['summary']}\n\n\nFULL INTERVIEW TRANSCRIPT\n=========================\n\n{analyzer_state['context']['transcript']}"
            st.download_button(label="📥 Download Full Report", data=report_content, file_name=f"Symptom_Report_{name.replace(' ','_')}.txt", mime="text/plain")
        
        if st.button("Start New Interview"): start_interview()

    # 3. Main Analysis Loop (Demographics OR Symptom Analysis)
    else:
        # A. Analysis Phase - Generate Next Question
        if not analyzer_state["current_interaction_data"] and current_stage == 'IN_ANALYSIS':
            with st.spinner("Thinking..."):
                interaction_data = analyzer_state["logic"].get_next_interaction(transcript=analyzer_state["context"]["transcript"])
                
                if interaction_data.get("question") == "INTERVIEW_COMPLETE" or interaction_data.get("type") == "done":
                    analyzer_state["stage"] = 'SUMMARY'
                else:
                    analyzer_state["current_interaction_data"] = interaction_data
                    analyzer_state["messages"].append({"role": "assistant", "content": interaction_data["question"]})
                st.rerun()

        interaction_data = analyzer_state.get("current_interaction_data")

        # End Interview Button (always visible during analysis)
        if current_stage != 'DEMOGRAPHICS':
            if st.button("End Interview & Generate Summary", type="primary"):
                analyzer_state["stage"] = 'SUMMARY'
                analyzer_state["current_interaction_data"] = None 
                st.rerun()

        # B. Demographics Phase
        if current_stage == 'DEMOGRAPHICS':
            step = analyzer_state["demographics_step"]
            if step in ['NAME', 'AGE', 'GENDER', 'EMAIL']:
                prompt_text = f"Enter your {step.lower()}..."
                if prompt := st.chat_input(prompt_text):
                    if prompt.lower().strip() in ['quit', 'stop', 'end']:
                        analyzer_state["stage"] = 'SUMMARY'
                        st.rerun()
                    else:
                        analyzer_state["messages"].append({"role": "user", "content": prompt})
                        if step == 'NAME':
                            analyzer_state["context"]['name'] = prompt
                            analyzer_state["messages"].append({"role": "assistant", "content": "Thank you. And what is your age?"})
                            analyzer_state["demographics_step"] = 'AGE'
                        elif step == 'AGE':
                            analyzer_state["context"]['age'] = prompt
                            analyzer_state["messages"].append({"role": "assistant", "content": "Thank you. And what is your gender?"})
                            analyzer_state["demographics_step"] = 'GENDER'
                        elif step == 'GENDER':
                            analyzer_state["context"]['gender'] = prompt
                            analyzer_state["messages"].append({"role": "assistant", "content": "What is a good email address we can use to contact you?"})
                            analyzer_state["demographics_step"] = 'EMAIL'
                        elif step == 'EMAIL':
                            analyzer_state["context"]['email'] = prompt
                            analyzer_state["messages"].append({"role": "assistant", "content": "Finally, what is your phone number?"})
                            analyzer_state["demographics_step"] = 'PHONE'
                        st.rerun()

            elif step == 'PHONE':
                st.info("Please enter your phone number below.")
                with st.form("phone_form"):
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        try:
                            default_index = [i for i, s in enumerate(ALL_COUNTRY_CODES) if s.startswith("+91")][0]
                        except IndexError:
                            default_index = 0
                        country_code = st.selectbox("Country", ALL_COUNTRY_CODES, index=default_index)
                    with col2:
                        phone_num = st.text_input("Phone Number", placeholder="Enter number")

                    if st.form_submit_button("Submit Phone Number"):
                        if phone_num:
                            selected_code = country_code.split(' ')[0]
                            full_phone = f"{selected_code} {phone_num}"
                            analyzer_state["context"]['phone'] = full_phone
                            analyzer_state["messages"].append({"role": "user", "content": full_phone})
                            analyzer_state["stage"] = 'IN_ANALYSIS'
                            st.rerun()

        # C. Render Dynamic Question Form (Single/Multi/Text)
        elif current_stage == 'IN_ANALYSIS' and interaction_data:
            q_type = interaction_data.get("type")
            question = interaction_data["question"]
            
            with st.form(key=f"form_{len(analyzer_state['messages'])}"):
                # st.write(question) # Redundant, already shown in chat history above
                
                if q_type == "single_choice":
                    options = interaction_data.get("options", [])
                    answer = st.radio("Select one option:", options, label_visibility="collapsed", index=None)
                    if st.form_submit_button("Submit"):
                        if answer:
                            if answer == "Other":
                                analyzer_state["context"]["current_other_question"] = question
                                analyzer_state["stage"] = 'OTHER_INPUT'
                            else:
                                process_user_response(question, answer)
                            st.rerun()
                            
                elif q_type == "multi_choice":
                    options = interaction_data.get("options", [])
                    selections = {}
                    for option in options:
                        if option != "Other":
                            selections[option] = st.checkbox(option)
                    other_selected = st.checkbox("Other")
                    other_text = ""
                    if other_selected:
                        other_text = st.text_input("Please specify:")

                    if st.form_submit_button("Submit"):
                        final_answers = [opt for opt, checked in selections.items() if checked]
                        if other_selected and other_text:
                            final_answers.append(f"Other: {other_text}")
                        
                        if final_answers:
                            process_user_response(question, ", ".join(final_answers))
                            st.rerun()
                        else:
                            st.warning("Please make at least one selection.")
                            
                else: # Text/Default
                    answer = st.text_input("Your answer:")
                    if st.form_submit_button("Submit"):
                        if answer:
                            process_user_response(question, answer)
                            st.rerun()

        # D. Handling "Other" Input
        elif current_stage == 'OTHER_INPUT':
            question_for_other = analyzer_state["context"].get("current_other_question", "the previous question")
            st.info(f"Please provide more details for: **{question_for_other}**")
            if prompt := st.chat_input("Describe your answer..."):
                process_user_response(question_for_other, f"Other: {prompt}")


# --- LOGIC FOR PAGE 3: HCM RISK CALCULATOR ---

@st.cache_resource
def setup_hcm_llm():
    try:
        asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError: pass
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3,google_api_key=get_api_key('basic'))

def generate_hcm_summary(patient_data: dict, _llm) -> str:
    template = """System: You are a cardiologist. Your task is to generate a clear, patient-friendly summary of an HCM risk assessment calculation.

    **Instructions:**
    -  Present the information in a structured, easy-to-read format.
    -  Begin with a summary sentence.
    -  List the key factors that were included in the calculation.
    -  State the result and recommendation clearly.
    -  **Crucially, end with the provided disclaimer.**

    **Data Provided:**
    - **Age:** {age}
    - **Max Wall Thickness:** {thickness} mm
    - **Left Atrial Diameter:** {la_diam} mm
    - **Max LVOT Gradient:** {max_lvot} mmHg
    - **History:**
    - Family History of SCD: {fam_history_str}
    - Non-sustained VT: {nsvt_str}
    - Unexplained Syncope: {syncope_str}
    - **Result:**
    - 5-Year Risk Score: {risk_percentage:.1f}% ({risk_category})
    - Recommendation: {recommendation}

    **Output:**

    Based on the information provided, here is a summary of the 5-year risk assessment for sudden cardiac death (SCD) in Hypertrophic Cardiomyopathy (HCM).

    **Key Factors Considered:**
    *   Age: {age}
    *   Maximal wall thickness: {thickness} mm
    *   Left atrial diameter: {la_diam} mm
    *   LVOT obstruction: {max_lvot} mmHg
    *   Family history of SCD: {fam_history_str}
    *   Episodes of NSVT: {nsvt_str}
    *   History of syncope: {syncope_str}

    **Result:**
    The estimated 5-year risk of SCD is **{risk_percentage:.1f}%** ({risk_category}).

    **Recommendation:**
    Based on this risk level, the typical guidance is: **{recommendation}**.

    ---
    ***Disclaimer:** This is an estimation based on a standard clinical model and is for informational purposes only. It is NOT a diagnosis or a substitute for professional medical advice. You must discuss these results and any decisions about your health with a qualified cardiologist.*"""
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=_llm, prompt=prompt)
    result = execute_with_retry(chain, patient_data, 'basic')
    summary = result['text']
    return summary

def render_hcm_calculator():
    st.title("HCM Risk-SCD Calculator ⚕️")
    st.markdown("A tool for healthcare professionals to estimate the 5-year risk of sudden cardiac death in patients with hypertrophic cardiomyopathy, based on the **2014 ESC Guidelines**.")
    st.warning("**Disclaimer:** For use by qualified healthcare professionals only.")
    hcm_llm = setup_hcm_llm()
    
    if 'calc_report_content' not in st.session_state: st.session_state.calc_report_content = ""
    
    with st.form("hcm_risk_form"):
        st.subheader("Patient Data Input"); col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age (years)", 16, 100, 50, 1)
            thickness = st.slider("Maximal LV wall thickness (mm)", 10, 50, 15, 1)
            la_diam = st.slider("Left atrial diameter (mm)", 20, 80, 40, 1)
            max_lvot = st.slider("Maximal LVOT gradient (mmHg)", 1, 200, 30, 1)
        with col2:
            fam_history = st.radio("Family history of SCD?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
            nsvt = st.radio("Non-sustained VT?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
            syncope = st.radio("Unexplained syncope?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
        submit_button = st.form_submit_button(label='Calculate Risk & Generate Report', use_container_width=True)
    
    if submit_button:
        # HCM Risk-SCD Formula (2014 ESC Guidelines)
        prognostic_index = ((0.15939858*thickness)-(0.00294271*(thickness**2))+(0.0259082*la_diam)+(0.00446131*max_lvot)+(0.4583082*fam_history)+(0.82639195*nsvt)+(0.71650361*syncope)-(0.01799934*age))
        risk_score = (1 - (0.998 ** math.exp(prognostic_index))) * 100
        
        if risk_score < 4: risk_category, recommendation = "LOW", "ICD is generally not indicated."; st.success(f"**Risk is LOW ({risk_score:.2f}%)**. {recommendation}")
        elif 4 <= risk_score < 6: risk_category, recommendation = "INTERMEDIATE", "ICD may be considered."; st.warning(f"**Risk is INTERMEDIATE ({risk_score:.2f}%)**. {recommendation}")
        else: risk_category, recommendation = "HIGH", "ICD should be considered."; st.error(f"**Risk is HIGH ({risk_score:.2f}%)**. {recommendation}")
        
        st.divider()
        with st.spinner("AI Cardiologist is generating the clinical summary..."):
            patient_data = {"age": age, "thickness": thickness, "la_diam": la_diam, "max_lvot": max_lvot, "fam_history_str": "Yes" if fam_history == 1 else "No", "nsvt_str": "Yes" if nsvt == 1 else "No", "syncope_str": "Yes" if syncope == 1 else "No", "risk_percentage": risk_score, "risk_category": risk_category, "recommendation": recommendation}
            summary = generate_hcm_summary(patient_data, hcm_llm)
            st.subheader("AI-Generated Clinical Summary"); st.markdown(summary)
            report_header = "HCM Risk-SCD Assessment Report\n" + "="*30 + "\n\n"
            report_inputs = "**Patient Input Data:**\n" + "\n".join([f"- {key.replace('_str','').replace('_', ' ').title()}: {value}" for key, value in patient_data.items() if '_str' in key or key in ['age', 'thickness', 'la_diam', 'max_lvot']]) + "\n\n"
            report_results = f"**Assessment Results:**\n- Calculated 5-Year Risk: {risk_score:.2f}%\n- Risk Category: {risk_category}\n- Recommendation: {recommendation}\n\n"
            report_summary = f"**Clinical Summary:**\n{summary}"
            st.session_state.calc_report_content = report_header + report_inputs + report_results + report_summary
            
    if st.session_state.calc_report_content:
        st.divider()
        st.download_button(label="📥 Download Full Report", data=st.session_state.calc_report_content, file_name="HCM_Risk_SCD_Report.txt", mime="text/plain", use_container_width=True)


# --- MAIN APP NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["Symptom Analyzer 🩺", "Patient Educator 💬", "HCM Risk Calculator ⚕️"])

with tab1:
    render_symptom_analyzer()

with tab2:
    render_patient_educator()

with tab3:
    render_hcm_calculator()
