# ============================================================
# RAG RETRIEVAL SYSTEM
# ============================================================
#
# Our program will:
#
# 1. Read a PDF
# 2. Extract the text
# 3. Split the text into chunks
# 4. Convert the chunks into embeddings
# 5. Ask the user a question
# 6. Convert the question into an embedding
# 7. Compare the question with every chunk
# 8. Find the most relevant chunks
#
# ============================================================


# ------------------------------------------------------------
# STEP 1: IMPORT LIBRARIES
# ------------------------------------------------------------

# PdfReader allows Python to read PDF files.
from pypdf import PdfReader

# SentenceTransformer converts text into embeddings.
from sentence_transformers import SentenceTransformer

# NumPy helps us perform mathematical calculations.
import numpy as np


# ------------------------------------------------------------
# STEP 2: LOAD THE EMBEDDING MODEL
# ------------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")
print()


# ------------------------------------------------------------
# STEP 3: TELL PYTHON WHERE OUR PDF IS
# ------------------------------------------------------------

pdf_file = "documents/MachineLearning101.pdf"


# ------------------------------------------------------------
# STEP 4: OPEN THE PDF
# ------------------------------------------------------------

print("Opening PDF...")

reader = PdfReader(pdf_file)

print("PDF opened!")
print()


# ------------------------------------------------------------
# STEP 5: FIND THE NUMBER OF PAGES
# ------------------------------------------------------------

number_of_pages = len(reader.pages)

print("Number of pages:", number_of_pages)
print()


# ------------------------------------------------------------
# STEP 6: EXTRACT TEXT FROM THE PDF
# ------------------------------------------------------------

print("Extracting text from PDF...")

# We start with an empty string.
document = ""


# Go through every page in the PDF.
for page_number in range(number_of_pages):

    # Get the current page.
    page = reader.pages[page_number]

    # Extract the text from this page.
    text = page.extract_text()

    # Add the page's text to our document.
    document = document + text + "\n"


print("Text extraction completed!")
print()


# ------------------------------------------------------------
# STEP 7: CONVERT THE DOCUMENT INTO WORDS
# ------------------------------------------------------------

# split() separates the text into individual words.
words = document.split()

print("Total number of words:", len(words))
print()


# ------------------------------------------------------------
# STEP 8: CREATE CHUNKS
# ------------------------------------------------------------

# Each chunk will contain approximately 100 words.
chunk_size = 100

# The next chunk will reuse 20 words from the previous chunk.
overlap = 20


# This list will store all our chunks.
chunks = []


# We start from the first word.
start = 0


# Keep creating chunks until we reach the end of the document.
while start < len(words):

    # Calculate where the current chunk should end.
    end = start + chunk_size

    # Take the words between start and end.
    chunk_words = words[start:end]

    # Join those words together into normal text.
    chunk = " ".join(chunk_words)

    # Store the chunk in our chunks list.
    chunks.append(chunk)

    # Move forward.
    #
    # We subtract the overlap because we want
    # some words from the previous chunk to appear
    # in the next chunk.
    start = end - overlap


print("Number of chunks created:", len(chunks))
print()


# ------------------------------------------------------------
# STEP 9: CREATE EMBEDDINGS FOR ALL CHUNKS
# ------------------------------------------------------------

print("Creating embeddings for chunks...")

# Convert every chunk into an embedding.
chunk_embeddings = model.encode(chunks)

print("Chunk embeddings created!")
print()


# ------------------------------------------------------------
# STEP 10: ASK THE USER A QUESTION
# ------------------------------------------------------------

query = input("Ask a question about the PDF: ")

print()


# ------------------------------------------------------------
# STEP 11: CREATE AN EMBEDDING FOR THE QUESTION
# ------------------------------------------------------------

print("Creating embedding for your question...")

query_embedding = model.encode(query)

print("Question embedding created!")
print()


# ------------------------------------------------------------
# STEP 12: CALCULATE SIMILARITY
# ------------------------------------------------------------

# We will store the similarity score and chunk number here.
scores = []


# Go through every chunk.
for i in range(len(chunks)):

    # Get the embedding of the current chunk.
    chunk_embedding = chunk_embeddings[i]

    # Calculate cosine similarity.
    similarity = np.dot(
        query_embedding,
        chunk_embedding
    ) / (
        np.linalg.norm(query_embedding)
        *
        np.linalg.norm(chunk_embedding)
    )

    # Store two things:
    #
    # 1. similarity score
    # 2. number of the chunk
    #
    scores.append((similarity, i))


# ------------------------------------------------------------
# STEP 13: SORT THE RESULTS
# ------------------------------------------------------------

# Sort the results from highest similarity
# to lowest similarity.
scores.sort(reverse=True)


# ------------------------------------------------------------
# STEP 14: SHOW THE TOP 3 CHUNKS
# ------------------------------------------------------------

print("============================================")
print("MOST RELEVANT INFORMATION FROM THE PDF")
print("============================================")


# We want to show the top 3 results.
number_of_results = 3


# Go through the first 3 results.
for i in range(number_of_results):

    # Get the similarity score.
    similarity = scores[i][0]

    # Get the chunk number.
    chunk_number = scores[i][1]

    print()
    print("RESULT", i + 1)

    print("Similarity score:", similarity)

    print()

    print("CHUNK:")
    print(chunks[chunk_number])

    print()
    print("--------------------------------------------")