import os
import json
import uuid
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MyScale
from langchain_huggingface import HuggingFaceEmbeddings
import anthropic
from datetime import datetime
from dateutil.relativedelta import relativedelta
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure your environment variables and settings here
os.environ["MYSCALE_HOST"] = "msc-5131d230.us-east-1.aws.myscale.com"
os.environ["MYSCALE_PORT"] = "443"
os.environ["MYSCALE_USERNAME"] = "danielbrmz_org_default"
os.environ["MYSCALE_PASSWORD"] = "passwd_ktandddcNhsCy4"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

try:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    docsearch = MyScale(embedding=embeddings, database='default')
    client = anthropic.Anthropic()
    logger.info("Successfully initialized embeddings, docsearch, and Anthropic client")
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/embed', methods=['POST'])
def embed():
    logger.info("Received request to /embed endpoint")
    try:
        if 'file' not in request.files:
            logger.warning("No file part in the request")
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            session_id = str(uuid.uuid4())
            filename = secure_filename(f"{session_id}_{file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.info(f"File saved: {file_path}")
            
            if file.filename.endswith('.csv'):
                loader = CSVLoader(file_path)
            else:
                loader = PyPDFLoader(file_path)
            
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(documents)
            
            docsearch.add_documents(split_docs)
            logger.info(f"Documents added to docsearch for session {session_id}")
            
            return jsonify({"message": "File processed and embedded", "session_id": session_id}), 200
        logger.warning(f"File type not allowed: {file.filename}")
        return jsonify({"error": "File type not allowed"}), 400
    except Exception as e:
        logger.error(f"Error in /embed endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing the file"}), 500

@app.route('/conversation', methods=['POST'])
def conversation():
    logger.info("Received request to /conversation endpoint")
    try:
        data = request.json
        query = data.get('query')
        if not query:
            logger.warning("No query provided")
            return jsonify({"error": "No query provided"}), 400
        
        docs = docsearch.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": f"Based on the following context, answer the question in 3 to 7 words. Also, if the user is asking to do a prediction, provide a natural langauge command inside a bash code block in markdown syntax like this: ```{{graph}} generate a chart about ...```. Question: {query}\n\nContext: {context}"}
            ]
        )
        
        response = message.content[0].text if isinstance(message.content, list) else str(message.content)
        
        # Check if there's a bash command in the response
        bash_command = None
        if "```bash" in response:
            response_parts = response.split("```bash")
            response = response_parts[0].strip()
            bash_command = response_parts[1].split("```")[0].strip()
        
        logger.info("Successfully processed conversation request")
        return jsonify({
            "concise_response": response,
            "bash_command": bash_command
        }), 200
    except anthropic.AuthenticationError as e:
        logger.error(f"Anthropic authentication error: {str(e)}")
        return jsonify({"error": "Authentication failed with the AI service"}), 401
    except Exception as e:
        logger.error(f"Error in /conversation endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing the conversation"}), 500

@app.route('/tooltip', methods=['POST'])
def tooltip():
    logger.info("Received request to /tooltip endpoint")
    try:
        data = request.json
        if not data:
            logger.warning("No data provided")
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
        
        logger.info("Successfully processed tooltip request")
        return jsonify({"insight": message.content}), 200
    except Exception as e:
        logger.error(f"Error in /tooltip endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing the tooltip request"}), 500

@app.route('/graph', methods=['POST'])
def graph():
    logger.info("Received request to /graph endpoint")
    try:
        data = request.json
        query = data.get('query')
        if not query:
            logger.warning("No query provided")
            return jsonify({"error": "No query provided"}), 400
        
        docs = docsearch.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": f"""Based on the following context, create monthly graph information for the last 12 months.
                You can create multiple charts if necessary. For each chart, provide the response in this format:
                Chart:
                dates: YYYY-MM-DD, YYYY-MM-DD, ... (exactly 12 dates, one for each of the last 12 months)
                series_name: Series Name
                series_type: line
                values: number, number, ... (exactly 12 different values, one for each of the last 12 months)

                Ensure all dates are in "YYYY-MM-DD" format, all values are different numbers, and the type is "line" unless specified otherwise.
                If you don't have exact data for all 12 months, please estimate or extrapolate to fill in the missing months, ensuring each month has a unique value.
                Query: {query}
                Context: {context}"""}
            ]
        )
        
        # Extract the content from the message
        content = message.content[0].text if isinstance(message.content, list) else str(message.content)
        
        # Parse the content
        charts = content.split('Chart:')
        charts = [chart.strip() for chart in charts if chart.strip()]
        
        all_chart_data = []
        
        for chart in charts:
            lines = chart.strip().split('\n')
            chart_data = {"dates": [], "series": []}
            current_series = {}
            
            for line in lines:
                if line.startswith("dates:"):
                    chart_data["dates"] = [date.strip() for date in line.split(':')[1].split(',')]
                elif line.startswith("series_name:"):
                    current_series["name"] = line.split(':')[1].strip()
                elif line.startswith("series_type:"):
                    current_series["type"] = line.split(':')[1].strip()
                elif line.startswith("values:"):
                    current_series["values"] = [float(val.strip()) for val in line.split(':')[1].split(',')]
                    chart_data["series"].append(current_series)
                    current_series = {}
            
            # Validate and process each chart
            if not chart_data["dates"] or len(chart_data["dates"]) != 12:
                chart_data["dates"] = [(datetime.now() - relativedelta(months=i)).strftime('%Y-%m-%d') for i in range(12)]
            
            chart_data["dates"].sort(reverse=True)
            
            for series in chart_data["series"]:
                if not all(key in series for key in ['name', 'type', 'values']):
                    raise ValueError("Missing required keys in series data")
                
                if len(series['values']) != 12 or len(set(series['values'])) != 12:
                    base_value = sum(series['values']) / len(series['values']) if series['values'] else 100
                    series['values'] = [base_value * (1 + (i - 5.5)/10 + random.uniform(-0.1, 0.1)) for i in range(12)]
            
            all_chart_data.append(chart_data)
        
        logger.info("Successfully processed graph request")
        return jsonify(all_chart_data), 200
    except ValueError as ve:
        logger.error(f"Invalid chart data structure: {str(ve)}")
        return jsonify({"error": f"Invalid chart data structure: {str(ve)}"}), 500
    except Exception as e:
        logger.error(f"Error in /graph endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing the graph request"}), 500

if __name__ == '__main__':
    app.run(debug=True)