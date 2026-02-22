import fitz  # PyMuPDF
import sys
import os


def extract_text_from_pdf(pdf_path, output_path):
    try:
        doc = fitz.open(pdf_path)
        
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            full_text += text + "\n\n"
        
        doc.close()
        
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(full_text)
        
        print(f"Sucess: Text Extracted from '{pdf_path}' and Saved to '{output_path}'")
        print(f"Total Pages Processed: {len(doc)}")
        
    except FileNotFoundError:
        print(f"Error: PDF File '{pdf_path}' Not Found")
    except Exception as e:
        print(f"Error Processing PDF: {str(e)}")


def main():
    if len(sys.argv) == 3:
        pdf_path = sys.argv[1]
        output_path = sys.argv[2]
    else:
        pdf_path = input("Path to PDF File: ").strip()
        output_path = input("Path for Output Text File: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' Does Not Exist")
        return
    
    if not pdf_path.lower().endswith('.pdf'):
        print("Warning: Input File Doesn't Have a .pdf Extension.")
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    extract_text_from_pdf(pdf_path, output_path)


if __name__ == "__main__":
    main()