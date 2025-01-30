from flask import Flask, request, render_template, jsonify, send_file, logging, Response
import os
import threading
from queue import Queue
import time
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import google.generativeai as genai
import sys
from io import StringIO
import queue

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = app.logger

# Configure Google API
os.environ['GOOGLE_API_KEY'] = ''
genai.configure(api_key='')

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = None  # Unlimited file size

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global state
conversion_status = {}
conversion_queue = Queue()


# Configure stdout capture for streaming
class StreamCapture:
    def __init__(self):
        self.queue = queue.Queue()
        self.last_message = ""

    def write(self, text):
        self.queue.put(text)
        self.last_message = text
        sys.__stdout__.write(text)  # Still write to actual stdout

    def flush(self):
        sys.__stdout__.flush()


stream_capture = StreamCapture()
sys.stdout = stream_capture


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_pdf(input_path, output_path, job_id):
    """Convert PDF to Markdown"""
    try:
        logger.info(f"Starting conversion for job {job_id}")
        conversion_status[job_id]['status'] = 'processing'

        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        from marker.config.parser import ConfigParser

        # Configuration for maximum accuracy (using your original config)
        config = {
            "output_format": "markdown",
            "force_ocr": True,
            "strip_existing_ocr": True,
            "use_llm": True,
            "torch_device": "cuda"
        }

        logger.debug(f"Using config: {config}")

        config_parser = ConfigParser(config)
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer()
        )

        # Convert document
        logger.info(f"Converting document for job {job_id}")
        rendered = converter(input_path)
        text, _, images = text_from_rendered(rendered)

        # Save output
        logger.info(f"Saving output for job {job_id}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        conversion_status[job_id]['status'] = 'completed'
        conversion_status[job_id]['output_path'] = output_path
        logger.info(f"Completed conversion for job {job_id}")

    except Exception as e:
        logger.error(f"Error in conversion for job {job_id}: {str(e)}", exc_info=True)
        conversion_status[job_id]['status'] = 'failed'
        conversion_status[job_id]['error'] = str(e)


def process_queue():
    """Background worker to process conversion queue"""
    logger.info("Starting queue processor")
    while True:
        try:
            if not conversion_queue.empty():
                job = conversion_queue.get()
                logger.info(f"Processing job: {job['job_id']}")
                convert_pdf(job['input_path'], job['output_path'], job['job_id'])
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in queue processor: {str(e)}", exc_info=True)


# Start the background worker
worker_thread = threading.Thread(target=process_queue, daemon=True)
worker_thread.start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info("Received upload request")

    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"

            # Save uploaded file
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_filename = f"{os.path.splitext(filename)[0]}_output.md"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

            logger.info(f"Saving file to {input_path}")
            file.save(input_path)

            # Initialize status tracking
            conversion_status[job_id] = {
                'status': 'queued',
                'filename': filename,
                'output_path': None
            }

            # Add to conversion queue
            conversion_queue.put({
                'input_path': input_path,
                'output_path': output_path,
                'job_id': job_id
            })

            logger.info(f"File queued for conversion: {job_id}")
            return jsonify({'job_id': job_id})

        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    logger.warning("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/status/<job_id>')
def get_status(job_id):
    logger.debug(f"Status check for job {job_id}")
    if job_id in conversion_status:
        return jsonify(conversion_status[job_id])
    return jsonify({'error': 'Job not found'}), 404


@app.route('/download/<job_id>')
def download_file(job_id):
    logger.info(f"Download request for job {job_id}")
    if job_id in conversion_status and conversion_status[job_id]['status'] == 'completed':
        output_path = conversion_status[job_id]['output_path']
        return send_file(output_path, as_attachment=True)
    return jsonify({'error': 'File not ready or job not found'}), 404


@app.route('/stream')
def stream():
    def generate():
        while True:
            try:
                # Get message from queue
                message = stream_capture.queue.get(timeout=1)
                if message:
                    yield f"data: {message}\n\n"
            except queue.Empty:
                # If no new message, send heartbeat
                yield f"data: heartbeat\n\n"
            time.sleep(0.1)

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True, port=5000)