# ============================================================
# HALLUCINATION DETECTION - FEATURE EXTRACTION
# ============================================================

from sentence_transformers import SentenceTransformer
import numpy as np


# ------------------------------------------------------------
# STEP 1: LOAD EMBEDDING MODEL
# ------------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")
print()


# ------------------------------------------------------------
# STEP 2: OUR EXAMPLE DATA
# ------------------------------------------------------------

question = "How can machine learning help automate repetitive tasks?"

evidence = """
Machine learning systems can also automate repetitive tasks.
For example, an email service can automatically classify
thousands of messages as spam or legitimate without requiring
a person to examine every message.
"""

answer = """
An email service can automatically classify thousands of
messages as spam or legitimate without requiring a person
to examine every message.
"""


# ------------------------------------------------------------
# STEP 3: CREATE EMBEDDINGS
# ------------------------------------------------------------

print("Creating embeddings...")

question_embedding = model.encode(question)

evidence_embedding = model.encode(evidence)

answer_embedding = model.encode(answer)

print("Embeddings created!")
print()


# ------------------------------------------------------------
# STEP 4: ANSWER-EVIDENCE SIMILARITY
# ------------------------------------------------------------

answer_evidence_similarity = np.dot(
    answer_embedding,
    evidence_embedding
) / (
    np.linalg.norm(answer_embedding)
    *
    np.linalg.norm(evidence_embedding)
)


# ------------------------------------------------------------
# STEP 5: QUESTION-EVIDENCE SIMILARITY
# ------------------------------------------------------------

question_evidence_similarity = np.dot(
    question_embedding,
    evidence_embedding
) / (
    np.linalg.norm(question_embedding)
    *
    np.linalg.norm(evidence_embedding)
)


# ------------------------------------------------------------
# STEP 6: ANSWER LENGTH
# ------------------------------------------------------------

answer_words = answer.split()

answer_length = len(answer_words)


# ------------------------------------------------------------
# STEP 7: DISPLAY OUR FEATURES
# ------------------------------------------------------------

print("============================================")
print("EXTRACTED FEATURES")
print("============================================")

print("Answer-Evidence Similarity:",
      answer_evidence_similarity)

print("Question-Evidence Similarity:",
      question_evidence_similarity)

print("Answer Length:",
      answer_length)

print("============================================")