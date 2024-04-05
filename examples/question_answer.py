import pandas as pd
import os
import torch
from torch import nn
import numpy as np
import sys
import gc
from tqdm import tqdm
import joblib
from datasets import load_from_disk, load_dataset
import yaml

from src.rag.simplerag import SimpleRAG
from src.inference.llm_inference import LLM


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_path = config['model']['address']
data_path = config['test']['data']

df = pd.read_csv(data_path)
config = f"{model_path}/config.json"


def get_prompt(df, idx):

    question = df.iloc[idx].prompt + ' A: ' + df.iloc[idx].A + \
    ' B: ' + df.iloc[idx].B + ' C: ' + df.iloc[idx].C + \
    ' D: ' + df.iloc[idx].D + ' E: ' + df.iloc[idx].E

    prompt = f"""Choose the correct answer A, B, C, D, E. 
    Specify only the letter of the correct answer and nothing else.
    Question: {question}
    Answer: """
    return prompt


def answer_create(llm, tokenizer, prompt, device, answers_token_id, max_new_tokens):
    if config.get('RAG', False):
         system_prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\nContext:\n{context}"
         instruction = "Your task is to analyze the question and answer below. If the answer is correct, respond yes, if it is not correct respond no. As a potential aid to your answer, background context from Wikipedia articles is at your disposal, even if they might not always be relevant."
         prompt = system_prefix.format(instruction=instruction, context=row["context"])

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    output = llm.generate(input_ids=inputs["input_ids"], 
                            attention_mask=inputs["attention_mask"], 
                            max_new_tokens=max_new_tokens, 
                            return_dict_in_generate=True, 
                            output_scores=True,
                            pad_token_id=tokenizer.eos_token_id,
                         )
    first_token_probs = output.scores[0][0]
    option_scores = first_token_probs[answers_token_id].float().cpu().numpy() #ABCDE
    pred = np.array(["A", "B", "C", "D", "E"])[np.argsort(option_scores)[::-1][:3]]
    pred = ' '.join(pred)

    return pred



def main():
    submission = []
    llm_, tokenizer = LLM(model_path, config)
    answers_token_id = tokenizer.encode("A B C D E")[1:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##RAG search##
    if config.get('RAG', False):
         f = lambda row : " ".join([row["prompt"], row["A"], row["B"], row["C"], row["D"], row["E"]])
         inputs = df.apply(f, axis=1).values

         rag = SimpleRAG()
         search_index = rag.search_faiss(inputs)
         for i in range(len(df)):
             df.loc[i, "context"] = "-" + "\n-".join([dataset[int(j)]["text"] for j in search_index[i]])
         df.head(3)

    with torch.no_grad():
       for idx in tqdm(df.id):
        
          prompt = get_prompt(df, idx)
          answers = answer_create(llm_, tokenizer, prompt, device,
                                answers_token_id, max_new_tokens=1)
        
          submission.append([idx, answers])

    columns = ['id', 'prediction']
    submission = pd.DataFrame(submission, columns=columns)
    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()