"""
PDF Text Extractor Module for Resume Classification.
Extracts clean plain text from uploaded or local multi-page PDF resumes.
Handles exceptions gracefully without crashing applications.
"""
import io
import os
from typing import Union, Tuple
import pypdf

# Maximum allowed PDF size (25 MB)
MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024


def extract_text_from_pdf(pdf_source: Union[str, bytes, io.BytesIO]) -> Tuple[bool, str, str]:
    """
    Extract text from a PDF file path or file-like binary stream.
    
    Args:
        pdf_source: File path (str), binary bytes, or BytesIO stream.
        
    Returns:
        Tuple of (success: bool, extracted_text: str, error_message: str)
    """
    try:
        if isinstance(pdf_source, str):
            if not os.path.exists(pdf_source):
                return False, "", f"File not found at path: {pdf_source}"
            file_size = os.path.getsize(pdf_source)
            if file_size > MAX_PDF_SIZE_BYTES:
                return False, "", f"PDF file size ({file_size / 1024 / 1024:.1f} MB) exceeds maximum allowed 25 MB limit."
            reader = pypdf.PdfReader(pdf_source)
        elif isinstance(pdf_source, bytes):
            if len(pdf_source) > MAX_PDF_SIZE_BYTES:
                return False, "", f"PDF data size ({len(pdf_source) / 1024 / 1024:.1f} MB) exceeds maximum allowed 25 MB limit."
            stream = io.BytesIO(pdf_source)
            reader = pypdf.PdfReader(stream)
        elif hasattr(pdf_source, "read"):
            reader = pypdf.PdfReader(pdf_source)
        else:
            return False, "", "Invalid PDF source format provided."
            
        total_pages = len(reader.pages)
        if total_pages == 0:
            return False, "", "PDF contains 0 pages."
            
        page_texts = []
        for i, page in enumerate(reader.pages):
            try:
                page_str = page.extract_text()
                if page_str:
                    page_texts.append(page_str)
            except Exception as page_err:
                # Log or continue to next page if one page fails
                continue
                
        extracted_text = "\n".join(page_texts).strip()
        
        if not extracted_text:
            return False, "", "Unable to extract text from this PDF. Please upload a text-based PDF or paste the resume text."
            
        return True, extracted_text, ""
        
    except Exception as e:
        return False, "", f"Failed to parse PDF document: {str(e)}"


if __name__ == "__main__":
    test_pdf = "../data-20261005T051010Z-1-001/data/data/HR/10399912.pdf"
    if os.path.exists(test_pdf):
        success, text, err = extract_text_from_pdf(test_pdf)
        print("Success:", success)
        print("Length:", len(text))
        print("Snippet:", text[:200])
    else:
        print("Test PDF path not found for direct run.")
