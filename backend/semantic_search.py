from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Our documents
documents = [
    "Python was created by Guido van Rossum.",
    "The Earth revolves around the Sun.",
    "The Eiffel Tower is located in Paris.",
    "Python is widely used for machine learning."
]

# User's question
query = "Who created Python?"

# Convert all documents into embeddings
document_embeddings = model.encode(documents)

# Convert the question into an embedding
query_embedding = model.encode(query)

# Compare the question with every document
scores = []

for i, document_embedding in enumerate(document_embeddings):

    similarity = np.dot(query_embedding, document_embedding) / (
        np.linalg.norm(query_embedding) *
        np.linalg.norm(document_embedding)
    )

    scores.append((similarity, documents[i]))

# Sort documents by similarity
scores.sort(reverse=True)

# Show the most relevant document
print("Most relevant document:")
print(scores[0][1])
print("Similarity:", scores[0][0])