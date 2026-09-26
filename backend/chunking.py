# This is our example document.
# Later, this text will come from a PDF.

document = """
Python is a high-level programming language.
It was created by Guido van Rossum.
Python was first released in 1991.

Python is widely used in many areas.
These include web development, data science,
machine learning, automation, and scientific computing.

Python has a large standard library.
It also has a large ecosystem of third-party packages.
"""


# We want each chunk to contain 20 words.
chunk_size = 20


# We want 5 words from the previous chunk
# to be repeated in the next chunk.
overlap = 5


# Split the document into individual words.
words = document.split()


# This list will store all of our chunks.
chunks = []


# This tells us where to start reading the words.
start = 0


# Keep creating chunks until we reach the end of the document.
while start < len(words):

    # Calculate where the current chunk should end.
    end = start + chunk_size

    # Take the words between start and end.
    chunk_words = words[start:end]

    # Join the words together to create a normal sentence/text.
    chunk = " ".join(chunk_words)

    # Store this chunk inside our chunks list.
    chunks.append(chunk)

    # Move forward.
    # We subtract the overlap so that some words
    # from the previous chunk appear in the next chunk.
    start = end - overlap


# Print all the chunks.
for i in range(len(chunks)):

    print("CHUNK", i + 1)

    print(chunks[i])

    print("--------------------")