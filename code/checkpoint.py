# Code checkpoint - predicts age and sex-group for individuals from one oxford_area

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from dgl.nn import HeteroGraphConv, GraphConv

# Define the age groups and sex categories
age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
sex_categories = ['M', 'F']

# Custom aggregation function - aggregates tensors by summing them.
# Used in the HeteroGraphConv layer to combine messages from neighboring nodes.
def my_agg_func(tensors, dsttype):
    if len(tensors) == 0:
        return None
    stacked = torch.stack(tensors, dim=0)
    return torch.sum(stacked, dim=0)

# Define the GNN model using HeteroGraphConv
# 2 layers which apply GraphConv to different edge types - has_age, has_sex, self_loop
# final layer maps the embeddings to a space of size len(age_groups) + len(sex_categories)
class GNNModel(nn.Module):
    def __init__(self, num_individuals, num_age_groups, num_sex_categories, in_feats, hidden_size, out_feats):
        super(GNNModel, self).__init__()
        self.layer1 = HeteroGraphConv({
            'has_age': GraphConv(in_feats, hidden_size),
            'has_sex': GraphConv(in_feats, hidden_size),
            'self_loop': GraphConv(in_feats, hidden_size)
        }, aggregate=my_agg_func)
        self.layer2 = HeteroGraphConv({
            'has_age': GraphConv(hidden_size, out_feats),
            'has_sex': GraphConv(hidden_size, out_feats),
            'self_loop': GraphConv(hidden_size, out_feats)
        }, aggregate=my_agg_func)
        self.final_layer = nn.Linear(out_feats, len(age_groups) + len(sex_categories))

        # Learnable embeddings for all node types - individual, age, sex 
        self.individual_embeddings = nn.Parameter(torch.randn(num_individuals, in_feats))
        self.age_embeddings = nn.Parameter(torch.randn(num_age_groups, in_feats))
        self.sex_embeddings = nn.Parameter(torch.randn(num_sex_categories, in_feats))

    def forward(self, g):
        h = {
            'individual': self.individual_embeddings,
            'age': self.age_embeddings,
            'sex': self.sex_embeddings
        }
        h = self.layer1(g, h)
        h['individual'] = F.relu(h['individual'])
        h = self.layer2(g, h)
        h['individual'] = self.final_layer(h['individual'])

        # Split the output into age and sex components
        age_probs = h['individual'][:, :len(age_groups)]
        sex_probs = h['individual'][:, len(age_groups):]

        # Apply softmax separately to age and sex parts
        age_probs = F.softmax(age_probs, dim=1)
        sex_probs = F.softmax(sex_probs, dim=1)

        # Concatenate the probabilities back
        h['individual'] = torch.cat([age_probs, sex_probs], dim=1)

        return h['individual']

# Load the data from individual and cross-tables 
current_dir = os.path.dirname(os.path.abspath(__file__))
age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individual/Age_5yrs.csv'))
sex_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individual/Sex.csv'))
sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/sex_by_age_5yrs.csv'))

# Define the Oxford areas
oxford_areas = ['E02005921']

# Filter the DataFrame for the specified Oxford areas
age_df = age_df[age_df['geography code'].isin(oxford_areas)]
sex_df = sex_df[sex_df['geography code'].isin(oxford_areas)]
sex_by_age_df = sex_by_age_df[sex_by_age_df['geography code'].isin(oxford_areas)]

# Rename columns to avoid conflicts
age_df = age_df.rename(columns={'total': 'total_age'})
sex_df = sex_df.rename(columns={'total': 'total_sex'})

# Merge datasets on common keys (e.g., geography code)
merged_df = pd.merge(age_df, sex_df, on='geography code')

# Initialize the heterogeneous graph and possible relations 
hetero_graph = dgl.heterograph({
    ('individual', 'has_age', 'age'): ([], []),
    ('individual', 'has_sex', 'sex'): ([], []),
    ('individual', 'self_loop', 'individual'): ([], [])
})

# Add attribute nodes for age groups and sex
hetero_graph.add_nodes(len(age_groups), ntype='age')
hetero_graph.add_nodes(len(sex_categories), ntype='sex')

# Add individual nodes without edges
individual_count = 0
for index, row in merged_df.iterrows():
    geo_code = row['geography code']
    total_individuals = row['total_age']
    
    hetero_graph.add_nodes(total_individuals, ntype='individual')
    individual_count += total_individuals

# Add self-loops for 'individual' nodes
num_individuals = hetero_graph.num_nodes('individual')
individual_ids = list(range(num_individuals))
hetero_graph.add_edges(individual_ids, individual_ids, etype='self_loop')

num_ages = hetero_graph.num_nodes('age')
num_sexes = hetero_graph.num_nodes('sex')

individual_ids = list(range(num_individuals))
hetero_graph.add_edges(individual_ids, individual_ids, etype=('individual', 'self_loop', 'individual'))

# Function to create edges between individual nodes and age/sex nodes based on probabilistic sampling from the merged data
def create_probabilistic_edges(hetero_graph, merged_df, age_groups, sex_categories):
    individual_ids = list(range(hetero_graph.num_nodes('individual')))
    age_ids = {age: i for i, age in enumerate(age_groups)}
    sex_ids = {sex: i for i, sex in enumerate(sex_categories)}
    
    edges = {'has_age': ([], []), 'has_sex': ([], [])}
    
    for index, row in merged_df.iterrows():
        geo_code = row['geography code']
        total_individuals = row['total_age']
        
        for i in range(total_individuals):
            individual_id = individual_ids.pop(0)
            age_probs = [row.get(f"{age}", 0) / total_individuals for age in age_groups]
            sex_probs = [row.get(f"{sex}", 0) / total_individuals for sex in sex_categories]
            
            age_label = np.random.choice(len(age_groups), p=age_probs)
            sex_label = np.random.choice(len(sex_categories), p=sex_probs)
            
            edges['has_age'][0].append(individual_id)
            edges['has_age'][1].append(age_ids[age_groups[age_label]])
            edges['has_sex'][0].append(individual_id)
            edges['has_sex'][1].append(sex_ids[sex_categories[sex_label]])
    
    hetero_graph.add_edges(edges['has_age'][0], edges['has_age'][1], etype='has_age')
    hetero_graph.add_edges(edges['has_sex'][0], edges['has_sex'][1], etype='has_sex')

# Create initial random edges
create_probabilistic_edges(hetero_graph, merged_df, age_groups, sex_categories)

# Clean the cross table to remove the 'geography code' and 'total' columns
sex_by_age_df = sex_by_age_df.drop(columns=['geography code', 'total'], errors='ignore')

# Combines node embeddings into a form that can be compared with the target counts in the cross table
def aggregate(encoded_tensor, cross_table, category_dicts):
    split_sizes = [len(cat_dict) for cat_dict in category_dicts]
    if encoded_tensor.size(1) != sum(split_sizes):
        raise ValueError("Size mismatch between encoded_tensor and category_dicts")

    category_probs = torch.split(encoded_tensor, split_sizes, dim=1)
    aggregated_tensor = torch.zeros(len(cross_table.columns), device=encoded_tensor.device)

    for i, col in enumerate(cross_table.columns):
        age_sex_split = col.rsplit(' ', 1)
        if len(age_sex_split) != 2:
            raise ValueError(f"Unexpected number of category keys in {age_sex_split}")
        age, sex = age_sex_split
        expected_count = torch.ones(encoded_tensor.size(0), device=encoded_tensor.device)
        
        if age not in category_dicts[0]:
            raise ValueError(f"Age group {age} not found in category_dicts")
        if sex not in category_dicts[1]:
            raise ValueError(f"Sex category {sex} not found in category_dicts")

        age_index = list(category_dicts[0]).index(age)
        sex_index = list(category_dicts[1]).index(sex)
        
        expected_count *= category_probs[0][:, age_index]
        expected_count *= category_probs[1][:, sex_index]

        aggregated_tensor[i] = torch.sum(expected_count)
    return aggregated_tensor

# Define RMSE functions
def rmse_accuracy(computed_tensor, target_tensor):
    mse = torch.mean((target_tensor - computed_tensor) ** 2)
    rmse = torch.sqrt(mse)
    max_possible_error = torch.sqrt(torch.sum(target_tensor ** 2))
    accuracy = 1 - (rmse / max_possible_error)
    return accuracy.item()

def rmse_loss(aggregated_tensor, target_tensor):
    return torch.sqrt(torch.mean((aggregated_tensor - target_tensor) ** 2))

# Combined RMSE loss for multiple tensors
# def combined_rmse_loss(aggregated_tensor1, aggregated_tensor2, target_tensor1, target_tensor2):
#     concatenated_tensor = torch.cat([target_tensor1, target_tensor2])
#     aggregated_cat_tensor = torch.cat([aggregated_tensor1, aggregated_tensor2])
#     loss = torch.sqrt(torch.mean((aggregated_cat_tensor - concatenated_tensor) ** 2))
#     return loss

# Define the objective function
def objective_function(hetero_graph, sex_by_age_df, model):
    # Get node embeddings from the model
    h_individual = model(hetero_graph)
    
    # Aggregate the results
    aggregated_age_sex = aggregate(h_individual, sex_by_age_df, [age_groups, sex_categories])
    
    # Calculate RMSE as the objective function
    target_counts = torch.tensor(sex_by_age_df.values, dtype=torch.float32)
    loss = rmse_loss(aggregated_age_sex, target_counts)
    
    return loss

# Define and initialize the model
num_individuals = hetero_graph.num_nodes('individual')
num_age_groups = len(age_groups)
num_sex_categories = len(sex_categories)
in_feats = 10  # or the appropriate size

model = GNNModel(num_individuals, num_age_groups, num_sex_categories, in_feats, hidden_size=16, out_feats=16)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Initialize the best loss to a large value
best_loss = float('inf')

# Optimization loop
num_iterations = 2000
for iteration in range(num_iterations):
    model.train()
    optimizer.zero_grad()
    
    # Calculate the objective function
    loss = objective_function(hetero_graph, sex_by_age_df, model)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    # Check if the new graph is better
    if loss < best_loss:
        best_loss = loss
    else:
        # Randomly select an edge to modify
        etype = np.random.choice(['has_age', 'has_sex'])
        edge_id = np.random.randint(hetero_graph.num_edges(etype))
        src, dst = hetero_graph.find_edges(edge_id, etype=etype)
        
        # Remove the edge (Reconstructing the graph is more efficient than removing edges directly in DGL)
        hetero_graph = dgl.remove_edges(hetero_graph, edge_id, etype=etype)
        
        # Add a new random edge
        if etype == 'has_age':
            new_dst = np.random.choice(hetero_graph.nodes('age'))
        else:
            new_dst = np.random.choice(hetero_graph.nodes('sex'))
        
        hetero_graph.add_edges(src, new_dst, etype=etype)
        
        # Calculate the objective function
        loss = objective_function(hetero_graph, sex_by_age_df, model)
        
        # If the new graph is better, keep the change, otherwise revert
        if loss < best_loss:
            best_loss = loss
        else:
            # Revert the change by reconstructing the graph
            hetero_graph = dgl.add_edges(hetero_graph, src, dst, etype=etype)
    
    print(f"Iteration {iteration}, Loss: {loss.item()}")

print("Final graph structure:")
print(hetero_graph)


# TODO: include a comprehensive list of oxford_areas. How do we want to calculate the loss in this case? by OA, or as a sum of all areas 
# TODO: need to implement a method to adhere to constraints - each individual node has to be connected to exactly one age node and one sex node 
# TODO: atm, edge modification is a type of simulated annealing. What other methods can be used to maybe make it a bit more effective?
# TODO: lastly, increae number of attributes


# radar charts for visualisations based on age and sex 
# convergence plot 
# introduce new variables 
# create some way of importing the data 
# parameter tuning 
# performance tuning - where is the bottleneck, what other ways to improve it 