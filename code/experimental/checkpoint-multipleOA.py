# Code checkpoint - predicts age and sex-group for individuals from multiple oxford_areas (not finished yet)

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

        # Learnable embeddings for all node types
        self.individual_embeddings = nn.Parameter(torch.randn(num_individuals, in_feats))
        self.age_embeddings = nn.Parameter(torch.randn(num_age_groups, in_feats))
        self.sex_embeddings = nn.Parameter(torch.randn(num_sex_categories, in_feats))

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g):
        h = {
            'individual': self.dropout(self.individual_embeddings),
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
oxford_areas = ['E02005921', 'E02005922', 'E02005923', 'E02005924', 'E02005925']

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

# Function to create probabilistic edges
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
            
            # Calculate probabilities based on the individual's geographic code
            age_probs = [row.get(f"{age}", 0) / total_individuals for age in age_groups]
            sex_probs = [row.get(f"{sex}", 0) / total_individuals for sex in sex_categories]
            
            # Ensure probabilities sum to 1
            age_probs = np.array(age_probs)
            sex_probs = np.array(sex_probs)
            if age_probs.sum() > 0:
                age_probs /= age_probs.sum()
            if sex_probs.sum() > 0:
                sex_probs /= sex_probs.sum()
            
            # Sample age and sex based on probabilities
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
sex_by_age_df = sex_by_age_df.drop(columns=['total'], errors='ignore')

# Define the aggregate function
def aggregate(encoded_tensor, row_tensor, age_groups, sex_categories, total_individuals):
    # Create dictionaries for quick lookup
    age_dict = {age: i for i, age in enumerate(age_groups)}
    sex_dict = {sex: i for i, sex in enumerate(sex_categories)}

    # Clean row_tensor to remove non-numeric columns
    row_tensor = row_tensor.drop(labels=['geography code'], errors='ignore')
    row_tensor = row_tensor.apply(pd.to_numeric, errors='coerce')
    row_tensor = row_tensor.fillna(0)

    # Convert row_tensor values to tensor
    row_tensor_values = torch.tensor(row_tensor.values, dtype=torch.float32, device=encoded_tensor.device)

    # Create an empty tensor to hold the aggregated counts
    aggregated_tensor = torch.zeros_like(row_tensor_values)

    for i, (col_name, count) in enumerate(row_tensor.items()):
        age_sex_split = col_name.rsplit(' ', 1)
        if len(age_sex_split) != 2:
            raise ValueError(f"Unexpected column format: {col_name}")
        
        age, sex = age_sex_split
        age_index = age_dict.get(age)
        sex_index = sex_dict.get(sex)
        
        if age_index is not None and sex_index is not None:
            age_probs = F.softmax(encoded_tensor[:, :len(age_groups)], dim=1)
            sex_probs = F.softmax(encoded_tensor[:, len(age_groups):], dim=1)
            
            # Scale the probabilities to counts
            scaled_probs = age_probs[:, age_index] * sex_probs[:, sex_index] * total_individuals
            
            # Accumulate the expected counts
            aggregated_tensor[i] = torch.sum(scaled_probs)

    return aggregated_tensor

# Combined RMSE loss for multiple tensors
def combined_rmse_loss(aggregated_tensors, target_tensors):
    # print(aggregated_tensors)
    # print(target_tensors)
    concatenated_target_tensor = torch.cat(target_tensors)
    concatenated_aggregated_tensor = torch.cat(aggregated_tensors)
    loss = torch.sqrt(torch.mean((concatenated_aggregated_tensor - concatenated_target_tensor) ** 2))
    return loss

# Define the objective function
def objective_function(hetero_graph, sex_by_age_df, model):
    h_individual = model(hetero_graph)
    
    aggregated_tensors = []
    target_tensors = []
    
    # Ensure h_individual corresponds to the individuals in the same order as the cross-table
    individual_embeddings = h_individual[:hetero_graph.num_nodes('individual')]
    
    for idx, row in sex_by_age_df.iterrows():
        geo_code = row['geography code']  # Use the column name directly
        area_individuals = merged_df[merged_df['geography code'] == geo_code]
        area_encoded_tensor = individual_embeddings[:len(area_individuals)]
        total_individuals = area_individuals['total_age'].values[0]  # Ensure this gets the correct count

        # Drop 'geography code' column for aggregation
        row_without_geo = row.drop(labels='geography code')

        # Ensure the target tensor only contains numeric values
        target_counts = torch.tensor(row_without_geo.values.astype(np.float32), device=individual_embeddings.device)
        aggregated_age_sex = aggregate(area_encoded_tensor, row_without_geo, age_groups, sex_categories, total_individuals)
        
        aggregated_tensors.append(aggregated_age_sex)
        target_tensors.append(target_counts)

    loss = combined_rmse_loss(aggregated_tensors, target_tensors)
    
    return loss

# Define and initialize the model
num_individuals = hetero_graph.num_nodes('individual')
num_age_groups = len(age_groups)
num_sex_categories = len(sex_categories)
in_feats = 10  # or the appropriate size

model = GNNModel(num_individuals, num_age_groups, num_sex_categories, in_feats, hidden_size=32, out_feats=32)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adjusted learning rate

# Initialize the best loss to a large value
best_loss = float('inf')

# Optimization loop
num_iterations = 1000  # Increase the number of iterations
for iteration in range(num_iterations):
    model.train()
    optimizer.zero_grad()
    
    # Calculate the objective function
    loss = objective_function(hetero_graph, sex_by_age_df, model)
    
    # Backpropagation
    loss.backward()
    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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