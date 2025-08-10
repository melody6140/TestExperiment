import PyPDF2
import sys
import re

def extract_method_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            all_text = ""

            print(f"PDF has {len(pdf_reader.pages)} pages")

            # Extract text from all pages
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                all_text += page_text + "\n"

            # Look for method-related sections
            method_keywords = [
                "method", "methodology", "approach", "framework", "model", "architecture",
                "algorithm", "contrastive", "learning", "dual", "PECOLA"
            ]

            lines = all_text.split('\n')
            method_lines = []

            for i, line in enumerate(lines):
                line_lower = line.lower()
                # Check if line contains method keywords or is in a method section
                if any(keyword in line_lower for keyword in method_keywords):
                    # Include context around the line
                    start_idx = max(0, i-2)
                    end_idx = min(len(lines), i+5)
                    context = lines[start_idx:end_idx]
                    method_lines.extend([f"--- Context around line {i+1} ---"] + context + [""])

            # Also look for abstract and introduction
            abstract_start = all_text.lower().find("abstract")
            if abstract_start != -1:
                abstract_end = all_text.lower().find("introduction", abstract_start)
                if abstract_end == -1:
                    abstract_end = abstract_start + 1000
                abstract_text = all_text[abstract_start:abstract_end]
                method_lines = ["--- ABSTRACT ---", abstract_text, ""] + method_lines

            return '\n'.join(method_lines[:3000])  # Limit output size

    except Exception as e:
        return f"Error reading PDF: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_pdf_method.py <pdf_file>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    extracted_text = extract_method_from_pdf(pdf_file)
    print(extracted_text)

