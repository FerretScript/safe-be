import os
import json
import uuid
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MyScale
from langchain_huggingface import HuggingFaceEmbeddings
import anthropic

app = Flask(__name__)
CORS(app)

# Configure your environment variables and settings here
os.environ["MYSCALE_HOST"] = "msc-5131d230.us-east-1.aws.myscale.com"
os.environ["MYSCALE_PORT"] = "443"
os.environ["MYSCALE_USERNAME"] = "danielbrmz_org_default"
os.environ["MYSCALE_PASSWORD"] = "passwd_ktandddcNhsCy4"
os.environ["ANTHROPIC_API_KEY"] = "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"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
docsearch = MyScale(embedding=embeddings, database='default')
client = anthropic.Anthropic()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/embed', methods=['POST'])
def embed():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        session_id = str(uuid.uuid4())
        filename = secure_filename(f"{session_id}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        if file.filename.endswith('.csv'):
            loader = CSVLoader(file_path)
        else:
            loader = PyPDFLoader(file_path)
        
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        
        docsearch.add_documents(split_docs)
        
        return jsonify({"message": "File processed and embedded", "session_id": session_id}), 200
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/conversation', methods=['POST'])
def conversation():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    docs = docsearch.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": f"Based on the following context, answer the question in 3 to 7 words. Also, if the user is asking to do a prediction, provide a bash command in markdown syntax. Question: {query}\n\nContext: {context}"}
        ]
    )
    
    response = message.content
    
    # Check if there's a bash command in the response
    bash_command = None
    if "```bash" in response:
        response_parts = response.split("```bash")
        response = response_parts[0].strip()
        bash_command = response_parts[1].split("```")[0].strip()
    
    return jsonify({
        "concise_response": response,
        "bash_command": bash_command
    }), 200

@app.route('/tooltip', methods=['POST'])
def tooltip():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    query = f"Create a short query to get insights about this information: {json.dumps(data)}"
    docs = docsearch.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": f"Based on the following context, create a short insight about why the information is like that in a given point or period of time. Context: {context}\n\nInformation: {json.dumps(data)}"}
        ]
    )
    
    return jsonify({"insight": message.content}), 200

@app.route('/graph', methods=['POST'])
def graph():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    docs = docsearch.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": f"Based on the following context, create a monthly graph information in the specified JSON format. The default type is 'line'. Query: {query}\n\nContext: {context}"}
        ]
    )
    
    try:
        chart_data = json.loads(message.content)
        return jsonify(chart_data), 200
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to generate valid chart data"}), 500

if __name__ == '__main__':
    app.run(debug=True)