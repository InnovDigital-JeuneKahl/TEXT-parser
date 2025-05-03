import os
import re
import uuid
import shutil
from typing import List, Dict, Optional, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import fitz  # PyMuPDF
from officeparserpy import parse_office
import base64
import io
from PIL import Image

app = FastAPI(title="Document Parser API", 
              description="Parse and search PDF, Office, and text files")

# Create directories if they don't exist
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("extracted_images", exist_ok=True)

# Models
class SearchRequest(BaseModel):
    file_id: str
    search_terms: List[str]
    language: str = 'en'
    context_size: int = 1  # Number of sentences for context

class DirectSearchRequest(BaseModel):
    search_terms: List[str]
    language: str = 'en'
    context_size: int = 1  # Number of sentences for context

class SearchResult(BaseModel):
    term: str
    matches: List[Dict[str, Optional[str]]]

class TranscribeResponse(BaseModel):
    text: str
    file_id: str
    metadata: Dict

# Store parsed documents in memory (in production, use a database)
parsed_documents = {}
extracted_images_info = {}

# Helper functions
def get_file_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()

def is_supported_file(filename: str) -> bool:
    extension = get_file_extension(filename)
    return extension in SUPPORTED_EXTENSIONS

# Define supported file extensions
PDF_EXTENSIONS = [".pdf"]
OFFICE_EXTENSIONS = [".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".odt", ".ods", ".odp"]
TEXT_EXTENSIONS = [".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm"]
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS + OFFICE_EXTENSIONS + TEXT_EXTENSIONS

# File parsers
def parse_pdf(file_path: str) -> Dict:
    """Parse PDF file and extract text and images"""
    doc = fitz.open(file_path)
    text = ""
    metadata = {
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "page_count": len(doc),
        "file_type": "pdf"
    }
    
    # Extract text
    for page in doc:
        text += page.get_text()
    
    # Extract images
    images_folder = f"extracted_images/{os.path.basename(file_path).split('.')[0]}"
    os.makedirs(images_folder, exist_ok=True)
    
    images_info = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Save image
            image_filename = f"page_{page_num+1}_img_{img_index+1}.png"
            image_path = os.path.join(images_folder, image_filename)
            image.save(image_path)
            
            images_info.append({
                "page": page_num + 1,
                "index": img_index,
                "path": image_path,
                "filename": image_filename
            })
    
    return {
        "text": text,
        "metadata": metadata,
        "images": images_info
    }

def parse_office_document(file_path: str) -> Dict:
    """Parse Office documents using officeparserpy"""
    with open(file_path, "rb") as f:
        content = parse_office(f.read())
    
    extension = get_file_extension(file_path)
    doc_type = extension.replace(".", "")
    
    return {
        "text": content,
        "metadata": {
            "file_type": doc_type,
            "file_path": file_path
        },
        "images": []  # officeparserpy doesn't extract images directly
    }

def parse_text_file(file_path: str) -> Dict:
    """Parse plain text files"""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    
    extension = get_file_extension(file_path)
    
    return {
        "text": content,
        "metadata": {
            "file_type": extension.replace(".", ""),
            "file_path": file_path
        },
        "images": []
    }

def find_all_sentence_contexts(
    full_text: str,
    search_terms: List[str],
    language: str = 'en',
    context_size: int = 1
) -> Dict[str, List[Dict[str, Optional[str]]]]:
    """
    Find all occurrences of search terms with surrounding context
    """
    sentence_endings = r'(?<!\w\.\w.)(?<![A-ZÀ-Ü][a-zà-ü]\.)(?<=\.|\?|\!|\…|\n)\s+'
    sentences = [s.strip() for s in re.split(sentence_endings, full_text) if s.strip()]
    
    patterns = {
        term: re.compile(rf'(?<!\w){re.escape(term)}(?!\w)', re.IGNORECASE)
        for term in search_terms
    }
    
    results = {term: [] for term in search_terms}
    
    for idx, sentence in enumerate(sentences):
        for term, pattern in patterns.items():
            if pattern.search(sentence):
                # Get context sentences
                prev_sentences = []
                next_sentences = []
                
                # Previous context
                for i in range(1, context_size + 1):
                    if idx - i >= 0:
                        prev_sentences.insert(0, sentences[idx - i])
                
                # Next context
                for i in range(1, context_size + 1):
                    if idx + i < len(sentences):
                        next_sentences.append(sentences[idx + i])
                
                context = {
                    'prev': " ".join(prev_sentences) if prev_sentences else None,
                    'match': sentence,
                    'next': " ".join(next_sentences) if next_sentences else None
                }
                results[term].append(context)
    
    return results

def process_uploaded_file(file: UploadFile) -> Dict:
    """Process an uploaded file and return the parsed data"""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not is_supported_file(file.filename):
        supported_formats = ", ".join(SUPPORTED_EXTENSIONS)
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats are: {supported_formats}"
        )
    
    # Generate a unique file ID
    file_id = str(uuid.uuid4())
    extension = get_file_extension(file.filename)
    
    # Save the uploaded file
    temp_file_path = f"temp_uploads/{file_id}{extension}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Parse the file based on its type
    try:
        if extension in PDF_EXTENSIONS:
            parsed_data = parse_pdf(temp_file_path)
        elif extension in OFFICE_EXTENSIONS:
            parsed_data = parse_office_document(temp_file_path)
        elif extension in TEXT_EXTENSIONS:
            parsed_data = parse_text_file(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Store parsed data
        parsed_documents[file_id] = {
            "text": parsed_data["text"],
            "metadata": {
                **parsed_data["metadata"],
                "original_filename": file.filename
            }
        }
        
        # Store image info if any
        if parsed_data["images"]:
            extracted_images_info[file_id] = parsed_data["images"]
        
        return {
            "file_id": file_id,
            "text": parsed_data["text"],
            "metadata": {
                **parsed_data["metadata"],
                "original_filename": file.filename
            },
            "image_count": len(parsed_data["images"])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing file: {str(e)}")

# API Routes
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_file(file: UploadFile = File(...)):
    """Upload, parse, and return the full text from a document"""
    result = process_uploaded_file(file)
    
    return {
        "text": result["text"],
        "file_id": result["file_id"],
        "metadata": result["metadata"]
    }

@app.post("/search-document", response_model=Dict[str, List[Dict[str, Optional[str]]]])
async def search_existing_document(search_request: SearchRequest):
    """Search for terms in an already uploaded document"""
    file_id = search_request.file_id
    
    if file_id not in parsed_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document_text = parsed_documents[file_id]["text"]
    
    results = find_all_sentence_contexts(
        document_text,
        search_request.search_terms,
        search_request.language,
        search_request.context_size
    )
    
    return results

@app.post("/search", response_model=Dict)
async def search_uploaded_file(
    file: UploadFile = File(...),
    search_terms: str = Form(...),
    language: str = Form("en"),
    context_size: int = Form(1)
):
    """Upload a document and immediately search for terms"""
    # Process the file
    result = process_uploaded_file(file)
    
    # Parse search terms from string (comma separated)
    terms = [term.strip() for term in search_terms.split(",")]
    
    # Perform search
    search_results = find_all_sentence_contexts(
        result["text"],
        terms,
        language,
        context_size
    )
    
    return {
        "file_id": result["file_id"],
        "metadata": result["metadata"],
        "search_results": search_results
    }

@app.get("/images/{file_id}", response_model=List[Dict])
async def get_document_images(file_id: str):
    """Get information about images extracted from a document"""
    if file_id not in extracted_images_info:
        raise HTTPException(status_code=404, detail="No images found for this document")
    
    return extracted_images_info[file_id]

@app.get("/supported-formats", response_model=Dict[str, List[str]])
async def get_supported_formats():
    """Get all supported file formats"""
    return {
        "pdf": PDF_EXTENSIONS,
        "office": OFFICE_EXTENSIONS,
        "text": TEXT_EXTENSIONS,
        "all": SUPPORTED_EXTENSIONS
    }

@app.delete("/document/{file_id}")
async def delete_document(file_id: str):
    """Delete a parsed document and its extracted images"""
    if file_id not in parsed_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from memory
    parsed_documents.pop(file_id)
    
    # Remove images if they exist
    if file_id in extracted_images_info:
        images = extracted_images_info[file_id]
        for image in images:
            if os.path.exists(image["path"]):
                os.remove(image["path"])
        
        # Remove empty directory
        images_folder = os.path.dirname(images[0]["path"])
        if os.path.exists(images_folder) and not os.listdir(images_folder):
            os.rmdir(images_folder)
        
        extracted_images_info.pop(file_id)
    
    # Remove temp file if it exists
    for ext in SUPPORTED_EXTENSIONS:
        temp_file = f"temp_uploads/{file_id}{ext}"
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return {"message": "Document deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)