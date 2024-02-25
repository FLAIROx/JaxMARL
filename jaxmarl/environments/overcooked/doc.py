from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# Example data (list of documents)
data = ["This is the first document",
        "This document is the second document",
        "And this is the third one",
        "Is this the first document?"]

# Tokenize the data
tokenized_data = [word_tokenize(doc.lower()) for doc in data]

# Create tagged documents (required by Doc2Vec)
tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(tokenized_data)]

# Train a Doc2Vec model
model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=100)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Infer vectors for new documents
new_doc = "This is a new document to infer"
new_vector = model.infer_vector(word_tokenize(new_doc.lower()))
print("Inferred vector for new document:", new_vector)
