import os
from word_embedding_processor import WordEmbeddingProcessor

# Get the current working directory
current_directory = os.getcwd()

# Define relative paths to the Word2Vec vector bin and phrases CSV files
word2vec_path = os.path.join(current_directory, "GoogleNews-vectors-negative300.bin.gz")
phrases_path = os.path.join(current_directory, "phrases (1).csv")

if __name__ == "__main__":
    word_embedding_processor = WordEmbeddingProcessor(word2vec_path, phrases_path)
    word_embedding_processor.process_phrases()

    while True:
        input_phrase = input("Enter a phrase (or type 'exit' to quit): ")

        if input_phrase.lower() == 'exit':
            break

        closest_match, distance = word_embedding_processor.find_closest_match(input_phrase)

        if closest_match is not None:
            print("Input Phrase:", input_phrase)
            print("Closest Match:", closest_match)
            print("Distance:", distance)
        else:
            print("No valid embeddings found for the input phrase.")
