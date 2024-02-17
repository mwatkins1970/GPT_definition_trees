import torch

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

def mutant_prompting(random_noken, GPTmodel, tokenizer, max_length):

    original_wte = GPTmodel.get_input_embeddings()

    # Create a custom embedding using the original embedding layer
    custom_embedding = CustomEmbedding(original_wte)

    # Set the custom embedding as the model's new embedding layer
    GPTmodel.set_input_embeddings(custom_embedding)

    # Add the embedding modification
    custom_embedding.add_modification(37444, random_noken)

    # Generate text with altered token
    input_prompt = f"A typical definition of ' petertodd' is"
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
    altered_output_ids = GPTmodel.generate(
    input_ids,
    attention_mask=torch.ones_like(input_ids),
    eos_token_id = 13,  # Specify the EOS token ID for "."
    max_length = max_length,
    pad_token_id=tokenizer.eos_token_id  # setting pad_token_id to eos_token_id
    )

    #output_text = tokenizer.decode(altered_output_ids[0], skip_special_tokens=True).split('. ')[0].split('\n')[0][len(input_prompt) - 1:]output_text = tokenizer.decode(altered_output_ids[0], skip_special_tokens=True).split('. ')[0].split('\n')[0][len(input_prompt) - 1:]
    output_text = tokenizer.decode(altered_output_ids[0], skip_special_tokens=True).replace('\n','')[len(input_prompt) - 1:]


    custom_embedding.remove_modifications()

    # Reset the original embeddings on the model after removing modifications
    GPTmodel.set_input_embeddings(original_wte)

    return output_text
