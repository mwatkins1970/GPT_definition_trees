import math
import json
import os
from graphviz import Digraph
from PIL import Image

from IPython.display import Image as IPImage, display

def add_white_background(input_path, output_path):
    with Image.open(input_path) as im:
        # Convert the image to RGBA if it's not already in that mode
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            # Create a white background image that is the same size as our image
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            # Paste the image onto the background
            bg.paste(im, (0, 0), im)
            bg.convert('RGB').save(output_path, 'PNG')
        else:
            im.convert('RGB').save(output_path, 'PNG')

# Define find_max_min_cumulative_weight outside of create_tree_diagram
def find_max_min_cumulative_weight(node, current_max=0, current_min=float('inf')):
    current_max = max(current_max, node.get('cumulative_prob', 0))
    if node.get('cumulative_prob', 1) > 0:  # Only consider non-zero probabilities
        current_min = min(current_min, node.get('cumulative_prob', 1))
    for child in node.get('children', []):
        current_max, current_min = find_max_min_cumulative_weight(child, current_max, current_min)
    return current_max, current_min

def create_tree_diagram(data, directory, name, log_base, max_thickness=33, min_thickness=1):

    max_weight, min_weight = find_max_min_cumulative_weight(data)

    def scale_edge_width(cumulative_weight, max_weight, min_weight, log_base, max_thickness=33, min_thickness=1):
        # Ensure the cumulative_weight is not less than min_weight to avoid log(0)
        cumulative_weight = max(cumulative_weight, min_weight)

        # Apply logarithmic scaling
        log_weight = math.log(cumulative_weight, log_base) - math.log(min_weight, log_base)
        log_max = math.log(max_weight, log_base) - math.log(min_weight, log_base)

        # Amplify the difference after logarithmic scaling
        amplified_weight = (log_weight / log_max) ** 2.5  #raising to a power to amplify differences

        # Scale the amplified_weight to the range of thicknesses
        scaled_weight = (amplified_weight * (max_thickness - min_thickness)) + min_thickness

        return scaled_weight


    def add_nodes_edges(dot, node, name, max_weight, min_weight, parent=None, is_root=True, depth=0):
        node_id = str(id(node))  # Unique ID for the node based on its memory address

        if parent and not is_root:
            edge_weight = scale_edge_width(node.get('cumulative_prob', 0), max_weight, min_weight, log_base)  # Pass max_weight and min_weight here
            dot.edge(parent, node_id, arrowhead='dot', arrowsize='1', color='darkblue', penwidth=str(edge_weight))

        label = node.get('token', 'ROOT') if not is_root else "'" + name + "'"
        dot.node(node_id, label=label, shape='plaintext', fontsize='36', fontname='Helvetica')

        for child in node.get('children', []):
            # Recursive call with max_weight and min_weight
            add_nodes_edges(dot, child, name, max_weight, min_weight, parent=node_id, is_root=False, depth=depth+1)


    dot = Digraph(comment='Definition Tree', format='png')
    dot.attr(rankdir='LR', size='8,8', margin='0.2', nodesep='0.06', ranksep='5', dpi=600, bgcolor='white')      # Increasing the DPI may be necessary for very large trees generated with very small cutoff

    add_nodes_edges(dot, data, name, max_weight, min_weight)  # Start recursion with the root node

    output_file_path = os.path.join(directory, 'output_tree_diagram_' + name)
    output_path = dot.render(filename=output_file_path, cleanup=True)

    return output_path  # Return the path to the PNG file

log_base = 10

directory = '/content/Drive/My Drive/DefinitionTrees'         # This should be the directory where the JSON trees were saved.
json_file = 'results.json'
json_file_path = os.path.join(directory, json_file)

# Load the JSON data
with open(json_file_path, 'r') as file:
    data = json.load(file)['tree JSON']              # Access the actual data within 'tree JSON'

# Calculate max_weight and min_weight
max_weight, min_weight = find_max_min_cumulative_weight(data)

# Now that you've got the correct part of the JSON, pass it to the function
output_path_png = create_tree_diagram(data, directory, json_file.split('.')[0], flog_base, max_weight, min_weight)

# Add white background
input_path = output_path_png
output_path_with_white_bg = os.path.splitext(output_path_png)[0].replace("_def", "_defn") + '.png'
add_white_background(input_path, output_path_with_white_bg)

# Now delete the original .png file
os.remove(output_path_png)

# Debug print to check the final output path with white background
print(f"Just saved {output_path_with_white_bg}.")

# Now display the image directly in Colab
display(IPImage(filename=output_path_with_white_bg))
