import gensim
from gensim.models import KeyedVectors
import numpy as np
import os
import pandas as pd
import shutil

class WordEmbeddingProcessor:
    def __init__(self, word2vec_path, phrases_path):
        self.word2vec_path = word2vec_path
        self.phrases_path = phrases_path
        self.wv = None
        self.phrase_embeddings = {}

    def load_word2vec_model(self):
        if self.wv is None and self.word2vec_path:
            destination_path = "D:\GTS\Galytix-MadalaSaiVishnu\GoogleNews-vectors-negative300.bin.gz"
            if not os.path.exists(destination_path):
                shutil.copy(self.word2vec_path, destination_path)
            self.wv = KeyedVectors.load_word2vec_format(destination_path, binary=True, limit=1000000)

    def load_phrases(self):
        if os.path.exists(self.phrases_path):
            phrases_df = pd.read_csv(self.phrases_path, encoding='ISO-8859-1')
            phrases = phrases_df['Phrases']
            return phrases
        else:
            print(f"Phrases file not found. Please provide the correct file path.")
            return None

    def calculate_average_embedding(self, phrase):
        words = phrase.split()
        valid_words = [word for word in words if word in self.wv]

        if len(valid_words) > 0:
            embeddings = [self.wv[word] for word in valid_words]
            avg_embedding = np.mean(embeddings, axis=0)
            return avg_embedding
        else:
            return None

    def process_phrases(self):
        self.load_word2vec_model()
        phrases = self.load_phrases()

        if phrases is not None:
            for phrase in phrases:
                avg_embedding = self.calculate_average_embedding(phrase)
                if avg_embedding is not None:
                    self.phrase_embeddings[phrase] = avg_embedding

    def find_closest_match(self, input_phrase):
        if self.wv is not None:
            input_embedding = self.calculate_average_embedding(input_phrase)
            if input_embedding is not None:
                distances = {phrase: np.linalg.norm(input_embedding - self.phrase_embeddings[phrase]) for phrase in self.phrase_embeddings}
                closest_phrase = min(distances, key=distances.get)
                distance = distances[closest_phrase]
                return closest_phrase, distance
        else:
            print("Word2Vec model not loaded. Please call load_word2vec_model first.")
            return None