The main function here is definition_trees.py. This builds the JSON definition tree structure.
mutant_prompting.py and token_utils are subsidiary to this.

To rund definition_trees.py, you'll first have to have loaded EleutherAI/gpt-j-6B (or other model) as 'GPTmodel', 
along with its tokenizer as 'tokenizer' and its shape embeddings tensor as 'embeddings'

You'll also need to specify 'save_directory' for the resulting .json file to be saved to.

Once .json files have been built and saved, visualise_trees.py can be used to produce PNG tree images.
Increasing the DPI may be necessary for very large trees generated with very small cutoff, that's in this line:
    
    dot.attr(rankdir='LR', size='8,8', margin='0.2', nodesep='0.06', ranksep='5', dpi=600, bgcolor='white')      
    
