"""
PDF Syllabus Extractor and Formatter

Goal:
  - Extract text from PDF syllabuses and save formatted versions as .txt files
  - Format output to match the sample NESA syllabus structure
  - Suitable for training a BPE tokenizer

Requirements:
  - pdfplumber: pip install pdfplumber
"""

import os
import re
from pathlib import Path
from typing import Optional

try:
    import pdfplumber
except ImportError:
    print("Error: pdfplumber not installed. Install it with: pip install pdfplumber")
    exit(1)

# Configuration
PDF_SOURCE_DIR = "nesa_syllabuses_PDF"
TEXT_OUTPUT_DIR = "nesa_syllabuses"

# PDF file mappings: (pdf_filename, output_filename_without_txt)
PDF_MAPPINGS = [
    ("physics.pdf", "nesa_physics"),
    ("chemistry.pdf", "nesa_chemistry"),
    ("biology.pdf", "nesa_biology"),
    ("maths_advanced.pdf", "nesa_maths_advanced"),
    ("ancient_history.pdf", "nesa_ancient_history"),
    ("modern_history.pdf", "nesa_modern_history"),
    ("legal_studies.pdf", "nesa_legal_studies"),
    ("economics.pdf", "nesa_economics"),
    ("english_advanced.pdf", "nesa_english_advanced"),
]


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file while preserving structure.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    text_content = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
                    
                # Extract table data if present
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        text_content.append("\n[TABLE]\n")
                        for row in table:
                            text_content.append(" | ".join(str(cell) if cell else "" for cell in row))
                        text_content.append("[/TABLE]\n")
                        
    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {e}")
        return ""
    
    return "\n".join(text_content)


def clean_and_format_text(text: str) -> str:
    """
    Clean and format extracted text to improve readability.
    
    Args:
        text: Raw extracted text from PDF
        
    Returns:
        Cleaned and formatted text
    """
    # Remove excessive whitespace
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Strip leading/trailing whitespace
        line = line.strip()
        
        # Skip empty lines but keep paragraph breaks
        if line or (cleaned_lines and cleaned_lines[-1] != ''):
            cleaned_lines.append(line)
    
    # Join lines and normalize spacing
    text = '\n'.join(cleaned_lines)
    
    # Remove multiple consecutive newlines (keep max 2)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    # Clean up common OCR artifacts
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Fix hyphenation at line breaks
    
    return text


def format_syllabus(text: str, subject_name: Optional[str] = None) -> str:
    """
    Format extracted text to match NESA syllabus structure.
    
    Args:
        text: Cleaned extracted text
        subject_name: Name of the subject (optional, for better formatting)
        
    Returns:
        Formatted text matching sample syllabus structure
    """
    # Basic cleanup
    formatted = clean_and_format_text(text)
    
    # Ensure consistent section headers
    # Add visual separators similar to sample files
    formatted = re.sub(
        r'^(RATIONALE|AIM|OUTCOMES|Year \d+|MODULE|\w+.*?:)',
        r'\n--- \1 ---\n',
        formatted,
        flags=re.MULTILINE
    )
    
    # Add decorative line separator if not present
    if "=" * 30 not in formatted[:500]:
        first_line_end = formatted.find('\n')
        if first_line_end > 0:
            first_line = formatted[:first_line_end]
            formatted = first_line + "\n" + "=" * 60 + "\n" + formatted[first_line_end+1:]
    
    return formatted


def extract_and_save_syllabuses() -> None:
    """
    Main function to extract PDFs and save formatted text files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)
    
    print(f"Extracting syllabuses from {PDF_SOURCE_DIR}...\n")
    
    for pdf_filename, output_base_name in PDF_MAPPINGS:
        pdf_path = os.path.join(PDF_SOURCE_DIR, pdf_filename)
        
        if not os.path.exists(pdf_path):
            print(f"⚠ Skipping: {pdf_filename} (not found)")
            continue
        
        print(f"📖 Processing: {pdf_filename}")
        
        try:
            # Extract text from PDF
            raw_text = extract_text_from_pdf(pdf_path)
            
            if not raw_text.strip():
                print(f"   ❌ No text extracted from {pdf_filename}")
                continue
            
            # Format the text
            formatted_text = format_syllabus(raw_text, output_base_name)
            
            # Save to output file
            output_path = os.path.join(TEXT_OUTPUT_DIR, f"{output_base_name}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            
            # Report statistics
            line_count = len(formatted_text.split('\n'))
            char_count = len(formatted_text)
            print(f"   ✓ Saved to {output_base_name}.txt")
            print(f"     Lines: {line_count} | Characters: {char_count}\n")
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_filename}: {e}\n")
    
    print("✅ Extraction complete!")


def extract_single_pdf(pdf_filename: str, output_filename: Optional[str] = None) -> None:
    """
    Extract a single PDF file for testing or specific needs.
    
    Args:
        pdf_filename: Name of the PDF file in source directory
        output_filename: Name for output file (without .txt extension)
    """
    pdf_path = os.path.join(PDF_SOURCE_DIR, pdf_filename)
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return
    
    if output_filename is None:
        output_filename = pdf_filename.replace('.pdf', '')
    
    os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)
    
    print(f"Extracting {pdf_filename}...")
    raw_text = extract_text_from_pdf(pdf_path)
    formatted_text = format_syllabus(raw_text)
    
    output_path = os.path.join(TEXT_OUTPUT_DIR, f"{output_filename}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_text)
    
    print(f"✓ Saved to {output_path}")
    print(f"  Lines: {len(formatted_text.split(chr(10)))} | Chars: {len(formatted_text)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        if len(sys.argv) < 3:
            print("Usage: python extractor.py --single <pdf_filename> [output_name]")
            sys.exit(1)
        pdf_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        extract_single_pdf(pdf_file, output_file)
    else:
        extract_and_save_syllabuses()

