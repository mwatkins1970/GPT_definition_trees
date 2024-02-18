import os
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from definition_trees import mainfunction

# Dictionary mapping shorthand model names to their full names
model_name_dict = {
    'mistral': 'mistralai/Mistral-7B-v0.1',
    'gptj': 'EleutherAI/gpt-j-6B',
    # Add more model mappings if necessary
}


def main():
    parser = argparse.ArgumentParser(description='Run definition trees script.')
    parser.add_argument('model_name', help='Name of the model to use.')
    args = parser.parse_args()

    # Look up the full model name from the dictionary if shorthand is provided
    full_model_name = model_name_dict.get(args.model_name, args.model_name)

    # Load the model and tokenizer using Auto classes
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    model = AutoModelForCausalLM.from_pretrained(full_model_name)

    # The rest of the code from definition_trees.py should be integrated here
    # For example, you can call the mainfunction and pass the loaded model and tokenizer
    # mainfunction(data, topk, prompt, noken, model, tokenizer)

    # Additional imports from definition_trees.py

    # The following lines should be adapted from the script part of definition_trees.py
    # Set up the tokenizer and model embeddings
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    model = AutoModelForCausalLM.from_pretrained(full_model_name)
    embeddings = model.get_input_embeddings().weight

    # Set up other necessary variables (these may need to be adjusted or provided as arguments)
    save_directory = '/path/to/save/directory'  # Set the directory where the JSON trees will be saved
    topk = 5  # Top K probabilities to consider
    cutoff = 0.00001  # Probability cutoff for tree expansion
    noken = embeddings.mean(dim=0)  # Use the mean embedding as a placeholder
    prompt = "A typical definition of '_' would be '"  # Base prompt for the model

    # Prepare the initial data structure for the definition tree
    deftree_data = {"level": 0, "token": "", "cumulative_prob": 1, "children": []}

    # Run the main function from definition_trees.py
    results_dict = mainfunction(deftree_data, topk, prompt, noken, model, tokenizer)

    # Save the results
    file_path = os.path.join(save_directory, 'results.json')
    with open(file_path, 'w') as file:
        json.dump(results_dict, file, ensure_ascii=False, indent=4)


    print(f"Saved results_dict to {file_path}")
if __name__ == "__main__":
    main()
