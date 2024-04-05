from typing import Any
import httpx
import pickle
import os
from tqdm import tqdm
import requests

import faiss
import numpy as np
import torch
import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                          BitsAndBytesConfig, AutoTokenizer, AutoModel)
from datasets import load_from_disk, Dataset, load_dataset
import yaml

N_BATCHES = 5 
MAX_CONTEXT = 2750
MAX_LENGTH = 4096

NUM_TITLES = 5
MAX_SEQ_LEN = 512

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', '..', 'configs', 'config.yaml')

with open('config_path', 'r') as file:
    config = yaml.safe_load(file)
    
class SentenceTransformer:
    def __init__(self, checkpoint, device="cuda:0"):
        self.device = device
        self.checkpoint = checkpoint
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device).half()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def transform(self, batch):
        tokens = self.tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQ_LEN)
        return tokens.to(self.device)  

    def get_dataloader(self, sentences, batch_size=32):
        sentences = ["Represent this sentence for searching relevant passages: " + x for x in sentences]
        dataset = Dataset.from_dict({"text": sentences})
        dataset.set_transform(self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def encode(self, sentences, show_progress_bar=False, batch_size=32):
        dataloader = self.get_dataloader(sentences, batch_size=batch_size)
        pbar = tqdm(dataloader) if show_progress_bar else dataloader

        embeddings = []
        for batch in pbar:
            with torch.no_grad():
                e = self.model(**batch).pooler_output
                e = F.normalize(e, p=2, dim=1)
                embeddings.append(e.detach().cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings

class SimpleRAG:
    """Retrieval-augmentation generation (RAG)
    """
    def __init__(self):
        self.index = None
        self.documents = None
        self.embeddings = None
        self.MODEL_PATH = config['embedding_model']['address']

    def search_faiss(self, query):
        model = SentenceTransformer(self.MODEL_PATH,  device="cuda:0")
        print("1",self.MODEL_PATH)
        query_embedding = model.encode(query, show_progress_bar=False)
        faiss_index = faiss.read_index(self.MODEL_PATH + '/faiss.index')

        search_index = faiss_index.search(np.float32(query_embedding), NUM_TITLES)[1]
        return search_index
