# ============================================================
# RAG SYSTEM - RETRIEVAL + GENERATION + HALLUCINATION DETECTION
# ============================================================


# ------------------------------------------------------------
# STEP 1: IMPORT LIBRARIES
# ------------------------------------------------------------

# This allows Python to read PDF files.
from pypdf import PdfReader

# This converts text into embeddings.
from sentence_transformers import SentenceTransformer

# NumPy is used for cosine similarity calculations.
import numpy as np

# Transformers gives us access to the language model.
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

# This imports our hallucination detection function.
from hallucination_detection import detect_hallucination


# ------------------------------------------------------------
# STEP 2: LOAD THE EMBEDDING MODEL
# ------------------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")
print()


# ------------------------------------------------------------
# STEP 3: LOAD THE LANGUAGE MODEL
# ------------------------------------------------------------

print("Loading language model...")

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-base"
)

# Load the FLAN-T5 model.
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base"
)

print("Language model loaded!")
print()


# ------------------------------------------------------------
# STEP 4: OPEN OUR PDF
# ------------------------------------------------------------

pdf_file = "documents/MachineLearning101.pdf"

print("Opening PDF...")

reader = PdfReader(pdf_file)

print("PDF opened!")
print()


# ------------------------------------------------------------
# STEP 5: FIND NUMBER OF PAGES
# ------------------------------------------------------------

number_of_pages = len(reader.pages)

print("Number of pages:", number_of_pages)
print()


# ------------------------------------------------------------
# STEP 6: EXTRACT TEXT FROM PDF
# ------------------------------------------------------------

print("Extracting text...")

document = ""


# Go through every page.
for page_number in range(number_of_pages):

    # Get the current page.
    page = reader.pages[page_number]

    # Extract text from the page.
    text = page.extract_text()

    # Add the text to our document.
    document = document + text + "\n"


print("Text extracted!")
print()


# ------------------------------------------------------------
# STEP 7: SPLIT DOCUMENT INTO WORDS
# ------------------------------------------------------------

words = document.split()

print("Total words:", len(words))
print()


# ------------------------------------------------------------
# STEP 8: CREATE CHUNKS
# ------------------------------------------------------------

chunk_size = 100

overlap = 20

chunks = []

start = 0


while start < len(words):

    # Calculate the end position.
    end = start + chunk_size

    # Get the words for this chunk.
    chunk_words = words[start:end]

    # Join the words together.
    chunk = " ".join(chunk_words)

    # Save the chunk.
    chunks.append(chunk)

    # Move forward while keeping overlap.
    start = end - overlap


print("Number of chunks:", len(chunks))
print()


# ------------------------------------------------------------
# STEP 9: CREATE EMBEDDINGS FOR THE CHUNKS
# ------------------------------------------------------------

print("Creating chunk embeddings...")

chunk_embeddings = embedding_model.encode(
    chunks
)

print("Chunk embeddings created!")
print()


# ------------------------------------------------------------
# STEP 10: ASK THE USER A QUESTION
# ------------------------------------------------------------

query = input(
    "Ask a question about the PDF: "
)

print()


# ------------------------------------------------------------
# STEP 11: CREATE EMBEDDING FOR THE QUESTION
# ------------------------------------------------------------

query_embedding = embedding_model.encode(
    query
)


# ------------------------------------------------------------
# STEP 12: CALCULATE SIMILARITY
# ------------------------------------------------------------

scores = []


for i in range(len(chunks)):

    # Get the current chunk's embedding.
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

    # Store similarity and chunk number.
    scores.append(
        (similarity, i)
    )


# ------------------------------------------------------------
# STEP 13: SORT THE RESULTS
# ------------------------------------------------------------

scores.sort(reverse=True)


# ------------------------------------------------------------
# STEP 14: GET THE TOP 3 CHUNKS
# ------------------------------------------------------------

top_chunks = []


for i in range(3):

    # Get the chunk number.
    chunk_number = scores[i][1]

    # Get the actual chunk.
    chunk = chunks[chunk_number]

    # Add it to our top chunks list.
    top_chunks.append(chunk)


# ------------------------------------------------------------
# STEP 15: COMBINE THE TOP CHUNKS
# ------------------------------------------------------------

context = ""


for chunk in top_chunks:

    context = context + chunk + "\n"


# ------------------------------------------------------------
# STEP 16: CREATE A PROMPT FOR THE LLM
# ------------------------------------------------------------

prompt = """
Based only on the context, answer the question directly.

Context:
""" + context + """

Question:
""" + query + """

Give a complete answer in 2 sentences.

Answer:
"""


# ------------------------------------------------------------
# STEP 17: SEND THE PROMPT TO THE LLM
# ------------------------------------------------------------

print("Retrieved context:")
print("--------------------------------------------")
print(context)
print("--------------------------------------------")
print()

print("Generating answer...")
print()


# ------------------------------------------------------------
# STEP 18: GET THE GENERATED ANSWER
# ------------------------------------------------------------

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)


outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    min_new_tokens=20,
    num_beams=5,
    early_stopping=True
)


answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)


# ------------------------------------------------------------
# STEP 19: DISPLAY THE ANSWER
# ------------------------------------------------------------

print("============================================")
print("ANSWER")
print("============================================")

print(answer)

print("============================================")


# ------------------------------------------------------------
# STEP 20: CHECK ANSWER FOR HALLUCINATION
# ------------------------------------------------------------

print()
print("Checking answer for hallucination...")
print()


# Send the question, generated answer,
# and retrieved evidence to our XGBoost detector.

result = detect_hallucination(
    query,
    answer,
    top_chunks
)


# ------------------------------------------------------------
# STEP 21: DISPLAY FINAL RESULT
# ------------------------------------------------------------

print()
print("============================================")
print("FINAL RESULT")
print("============================================")


if result == 1:

    print("SUPPORTED")
    print("The answer is supported by the retrieved evidence.")

else:

    print("HALLUCINATED")
    print("The answer is not sufficiently supported by the retrieved evidence.")


print("============================================")