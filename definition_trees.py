# To use this, you'll first have to have loaded EleutherAI/gpt-j-6B (or other model) as 'GPTmodel', along with its tokenizer as 'tokenizer' and its shape embeddings tensor as 'embeddings'
# You'll also need to specify 'save_directory'

import os
import json
import torch

from mutant_prompting import mutant_prompting
from token_utils import token_setup

class CustomEmbedding(torch.nn.Module):
    def __init__(self, original_embedding):
        super().__init__()
        self.original_embedding = original_embedding  # changed from self.embedding to self.original_embedding
        self.modifications = {}

    def add_modification(self, token_id, new_embedding):
        self.modifications[token_id] = new_embedding

    def remove_modifications(self):
        self.modifications.clear()  # This empties the dictionary of modifications

    def forward(self, input_ids=None):
        # Get the original embeddings
        original_embeddings = self.original_embedding(input_ids)

        # Apply any modifications
        for token_id, new_embedding in self.modifications.items():
            mask = (input_ids == token_id)
            original_embeddings[mask] = new_embedding

        return original_embeddings

def produce_next_token_probs(prompt, noken, GPTmodel, tokenizer, topk):

    original_wte = GPTmodel.get_input_embeddings()

    # Create a custom embedding using the original embedding layer
    custom_embedding = CustomEmbedding(original_wte)

    # Set the custom embedding as the model's new embedding layer
    GPTmodel.set_input_embeddings(custom_embedding)

    # Add the embedding modification
    custom_embedding.add_modification(62, noken)     # 62 is the index for the placeholder token "_" which will stand for the chosen embeding.

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = GPTmodel(input_ids)
        logits = outputs.logits
    last_logits = logits[0, -1, :]
    softmax_probs = torch.softmax(last_logits, dim=0)
    total_tokens = len(softmax_probs)
    prob_distribution = {tokenizer.decode([i]): softmax_probs[i].item() for i in range(total_tokens)}
    top_k_probs = dict(sorted(prob_distribution.items(), key=lambda item: item[1], reverse=True)[:topk])
    return top_k_probs

def build_def_tree(token, data, base_prompt, noken, GPTmodel, tokenizer, topk, path="", visited=None):
    if visited is None:
        visited = set()
    visited.add(path)

    current_prompt = base_prompt + path
    print(current_prompt + "      PROB: " + str(data['cumulative_prob']))
    response = produce_next_token_probs(current_prompt, noken, GPTmodel, tokenizer, topk)

    for tok, prob in response.items():
        child_path = path + tok
        if child_path in visited or prob * data['cumulative_prob'] < cutoff:
            continue
        new_child = {"token": tok, "cumulative_prob": prob * data['cumulative_prob'], "children": []}
        data['children'].append(new_child)
        build_def_tree(tok, new_child, base_prompt, noken, GPTmodel, tokenizer, topk, child_path, visited)

    return data

def count_tokens(strg, tokenizer):
    return len(tokenizer.encode(strg)) if strg else 1

def save_data(data, folder, token):
    filepath = os.path.join(folder, f'def_results_{token.strip()}.json')
    with open(filepath, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def find_cumulative_probability(tree, target_def, tokenizer):
    current_node = tree
    tok_id_list = tokenizer.encode(target_def)
    for tok_id in tok_id_list:
        token = tokenizer.decode([tok_id])
        next_node = next((child for child in current_node['children'] if child['token'] == token), None)
        if not next_node:
            return 0
        current_node = next_node
    return current_node.get('cumulative_prob', 0)

def mainfunction(data, topk, prompt, noken, GPTmodel, tokenizer):
    results_dict = {}
    tree_json = build_def_tree("", data, prompt, noken, GPTmodel, tokenizer, topk)
    results_dict["tree JSON"] = tree_json

    print("RESULTS DICT:", results_dict)
    return results_dict
