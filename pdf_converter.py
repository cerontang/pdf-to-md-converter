import os
import google.generativeai as genai

# Set and configure the API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAAxA9opZvmVnRolGHqekQSpaqEHXwlkQM'
genai.configure(api_key='AIzaSyAAxA9opZvmVnRolGHqekQSpaqEHXwlkQM')

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser



# Configuration for maximum accuracy
config = {
    "output_format": "markdown",
    "force_ocr": True,
    "strip_existing_ocr": True,
    "use_llm": True,
    "torch_device": "cuda"
}


def convert_pdf(input_path, output_path):
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer()
    )

    # Convert document
    rendered = converter(input_path)
    text, _, images = text_from_rendered(rendered)

    # Save output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def process_directory(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process all PDF files in input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            input_path = os.path.join(input_dir, filename)
            # Extract base name without extension
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_output.md"
            output_path = os.path.join(output_dir, output_filename)

            convert_pdf(input_path, output_path)


if __name__ == "__main__":
    # Replace these paths with your actual directories
    input_directory = r"input"
    output_directory = r"output"

    process_directory(input_directory, output_directory)