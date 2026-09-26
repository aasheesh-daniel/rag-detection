# Import the PdfReader class from the pypdf library.
from pypdf import PdfReader


# Tell Python where our PDF is located.
pdf_file = "documents/MachineLearning101.pdf"


# Open the PDF.
reader = PdfReader(pdf_file)


# Find out how many pages the PDF has.
number_of_pages = len(reader.pages)

print("Number of pages:", number_of_pages)


# Create an empty string.
# We will put all the text from the PDF into this variable.
document = ""


# Go through every page of the PDF.
for page_number in range(number_of_pages):

    # Get the current page.
    page = reader.pages[page_number]

    # Extract the text from the current page.
    text = page.extract_text()

    # Add that text to our document.
    document = document + text


# Print the extracted text.
print(document)