from PyPDF2 import PdfReader
from docx import Document

def read_file(file_path):
    try:
        file_extension = file_path.split('.')[-1].lower()

        if file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return text

        elif file_extension == 'pdf':
            pdf = PdfReader(file_path)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            return text

        elif file_extension == 'docx':
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs]
            text = '\n'.join(paragraphs)
            return text

        else:
            return "Unsupported file format"

    except FileNotFoundError:
        return "File not found"
    except IsADirectoryError:
        return "Specified path is a directory"
    except Exception as e:
        return str(e)


#######################################
file_path = r'C:\Users\DELL\OneDrive\Desktop\test1.docx' # Replace with the actual path to your PDF file
file_contents = read_file(file_path)
print(type(file_contents))