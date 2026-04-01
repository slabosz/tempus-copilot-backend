import os
import csv
import json
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# Load env vars
load_dotenv()

app = Flask(__name__)
# Enable CORS
CORS(app)

# ———— RAG Init ————
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please check your .env file.")

print("Initializing Vector Store...")

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    # ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Load and split the knowledge base
with open("data/knowledge_base.md", "r", encoding="utf-8") as f:
    kb_text = f.read()

splits = markdown_splitter.split_text(kb_text)

# Disable RAG engine for demo to increase speeds
# # Init RAG Embeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Create an in-memory Chroma vector store
# vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Init the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

print("Vector Store & LLM Initialized successfully.")

# ———— Prompt Template Init ————
try:
    with open("prompt.txt", "r", encoding="utf-8") as prompt_file:
        raw_prompt_template = prompt_file.read()
except FileNotFoundError:
    raise FileNotFoundError("Could not find prompt.txt.")

prompt_template = PromptTemplate(
    input_variables=["crm_note", "knowledge_base_context"],
    template=raw_prompt_template
)

try:
    with open("rank_prompt.txt", "r", encoding="utf-8") as ranking_file:
        raw_ranking_template = ranking_file.read()
except FileNotFoundError:
    raise FileNotFoundError("Could not find rank_prompt.txt.")

ranking_template = PromptTemplate(
    input_variables=["notes_payload", "knowledge_base_context"],
    template=raw_ranking_template
)

# ———— API Routes ————
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy", "message": "Tempus Copilot API is running."}), 200


@app.route('/api/generate-pitch', methods=['POST'])
def generate_pitch():
    """
    Main endpoint to generate the sales pitch data for a given CRM note.
    """
    data = request.get_json()
    
    if not data or 'crm_note' not in data:
        return jsonify({"error": "Missing 'crm_note' in request body."}), 400
        
    crm_note = data['crm_note']
    physician_name = data.get('physician_name', 'the doctor')
    
    try:
        # Bypassing RAG for demo
        # # Retrieve relevant chunks from ChromaDB
        # docs = retriever.invoke(crm_note)
        # context = "\n\n".join([doc.page_content for doc in docs])
        context = kb_text
        
        # Format the prompt
        formatted_prompt = prompt_template.format(
            crm_note=crm_note,
            knowledge_base_context=context
        )
        
        # Generate response from Gemini
        response = llm.invoke(formatted_prompt)
        raw_text = response.content.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
            
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        # Parse the string into a Python dictionary
        parsed_json = json.loads(raw_text.strip())
        
        return jsonify({
            "success": True,
            "physician": physician_name,
            "copilot_data": parsed_json,
            # "retrieved_context_used": [doc.metadata for doc in docs]
        }), 200

    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e} - Raw Output: {raw_text}")
        return jsonify({"error": "Failed to parse AI output into JSON."}), 500
    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/providers', methods=['GET'])
def get_providers():
    """
    Reads the doctors CSV, attaches their specific CRM note from the text files,
    and returns a ranked list based on impact score.
    """
    providers = []
    csv_path = os.path.join('data', 'market_intelligence.csv')
    notes_dir = os.path.join('data', 'crm_notes')
    rankings_path = os.path.join('data', 'processed_rankings.json')

    impact_scores = {}
    if os.path.exists(rankings_path):
        try:
            with open(rankings_path, 'r', encoding='utf-8') as f:
                rankings_data = json.load(f)
                impact_scores = {item['name']: item['impact_score'] for item in rankings_data}
        except Exception as e:
            print(f"Warning: Could not read rankings file: {e}")

    try:
        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                name = row.get('Physician Name')
                file_name = f"{name.replace(' ', '_')}.txt"
                note_path = os.path.join(notes_dir, file_name)
                
                crm_note = "No previous CRM notes available."
                if os.path.exists(note_path):
                    with open(note_path, 'r', encoding='utf-8') as note_file:
                        crm_note = note_file.read().strip()

                providers.append({
                    "id": name,
                    "name": name,
                    "hospital": row.get('Hospital / Clinic'),
                    "specialty": row.get('Specialty'),
                    "patient_population": int(row.get('Patient Population Size', 0)),
                    "crm_note": crm_note,
                    "impact_score": impact_scores.get(name, 0)
                })

        ranked_providers = sorted(providers, key=lambda x: x['impact_score'], reverse=True)

        return jsonify({"success": True, "providers": ranked_providers}), 200

    except Exception as e:
        print(f"Error fetching providers: {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/rank-providers', methods=['POST'])
def rank_providers():
    """
    Takes all the CRM notes, sends them to Gemini for sentiment/intent analysis, then
    combines that score with the patient population size to create a final impact score for each doctor.
    """
    csv_path = os.path.join('data', 'market_intelligence.csv')
    notes_dir = os.path.join('data', 'crm_notes')
    
    notes_to_analyze = []

    try:
        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                name = row['Physician Name']
                file_name = f"{name.replace(' ', '_')}.txt"
                note_path = os.path.join(notes_dir, file_name)
                
                content = "No note available."
                if os.path.exists(note_path):
                    with open(note_path, 'r') as f:
                        content = f.read().strip()
                
                notes_to_analyze.append(f"Doctor: {name}\nNote: {content}")

        # Send to Gemini for sentiment analysis
        payload = "\n---\n".join(notes_to_analyze)
        
        formatted_ranking_prompt = ranking_template.format(notes_payload=payload, knowledge_base_context=kb_text)
        
        response = llm.invoke(formatted_ranking_prompt)
        
        raw_json = response.content.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:]
        elif raw_json.startswith("```"):
            raw_json = raw_json[3:]
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3]
            
        intent_data = json.loads(raw_json.strip())
        
        # Merge with volume score and persist
        final_rankings = []
        intent_map = {item['name']: item['intent_score'] for item in intent_data}

        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            csv_file.seek(0) # Reset to top
            reader = csv.DictReader(csv_file)
            for row in reader:
                name = row['Physician Name']
                pop = int(row.get('Patient Population Size', 0))
                
                if pop <= 0:
                    volume_score = 0
                else:
                    volume_score = round(min(math.sqrt(pop) * 3.5, 60), 1)
                intent_score = intent_map.get(name, 10)
                
                final_rankings.append({
                    "name": name,
                    "impact_score": round(volume_score + intent_score, 1)
                })

        with open('data/processed_rankings.json', 'w') as f:
            json.dump(final_rankings, f)

        return jsonify({"success": True, "message": "AI-Driven rankings updated."}), 200

    except Exception as e:
        print(f"Ranking Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)