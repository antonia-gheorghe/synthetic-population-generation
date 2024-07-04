# Code checkpoint - predicts age, sex-group and ethnicity for individuals from one oxford_area

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from dgl.nn import HeteroGraphConv, GraphConv
import random 
import plotly.graph_objects as go

# Define the age groups, sex categories, and ethnicity categories
age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
sex_categories = ['M', 'F']
ethnicity_categories = ['W0', 'M0', 'A0', 'B0', 'O0']

# Define the correct order of categories
column_order = [0, 1, 2]  # Sex, Age, Ethnicity

# Custom aggregation function - aggregates tensors by summing them.
def my_agg_func(tensors, dsttype):
    if len(tensors) == 0:
        return None
    stacked = torch.stack(tensors, dim=0)
    return torch.sum(stacked, dim=0)

# Define the multi-task GNN model using HeteroGraphConv
class MultiTaskGNNModel(nn.Module):
    def __init__(self, num_individuals, num_age_groups, num_sex_categories, num_ethnicity_categories, in_feats, hidden_size, out_feats):
        super(MultiTaskGNNModel, self).__init__()
        # Define common layers
        self.layer1 = HeteroGraphConv({
            'has_age': GraphConv(in_feats, hidden_size),
            'has_sex': GraphConv(in_feats, hidden_size),
            'has_ethnicity': GraphConv(in_feats, hidden_size),
            'self_loop': GraphConv(in_feats, hidden_size)
        }, aggregate='sum')
        
        self.layer2 = HeteroGraphConv({
            'has_age': GraphConv(hidden_size, hidden_size),
            'has_sex': GraphConv(hidden_size, hidden_size),
            'has_ethnicity': GraphConv(hidden_size, hidden_size),
            'self_loop': GraphConv(hidden_size, hidden_size)
        }, aggregate='sum')
        
        self.layer3 = HeteroGraphConv({
            'has_age': GraphConv(hidden_size, out_feats),
            'has_sex': GraphConv(hidden_size, out_feats),
            'has_ethnicity': GraphConv(hidden_size, out_feats),
            'self_loop': GraphConv(hidden_size, out_feats)
        }, aggregate='sum')
        
        # Define task-specific layers
        self.age_layer = nn.Linear(out_feats, num_age_groups)
        self.sex_layer = nn.Linear(out_feats, num_sex_categories)
        self.ethnicity_layer = nn.Linear(out_feats, num_ethnicity_categories)

        # Learnable embeddings
        self.individual_embeddings = nn.Parameter(torch.randn(num_individuals, in_feats))
        self.age_embeddings = nn.Parameter(torch.randn(num_age_groups, in_feats))
        self.sex_embeddings = nn.Parameter(torch.randn(num_sex_categories, in_feats))
        self.ethnicity_embeddings = nn.Parameter(torch.randn(num_ethnicity_categories, in_feats))

    def forward(self, g):
        h = {
            'individual': self.individual_embeddings,
            'age': self.age_embeddings,
            'sex': self.sex_embeddings,
            'ethnicity': self.ethnicity_embeddings
        }
        h = self.layer1(g, h)
        h['individual'] = F.relu(h['individual'])
        h = self.layer2(g, h)
        h['individual'] = F.relu(h['individual'])
        h = self.layer3(g, h)
        h['individual'] = F.relu(h['individual'])
        
        # Task-specific predictions
        age_preds = self.age_layer(h['individual'])
        sex_preds = self.sex_layer(h['individual'])
        ethnicity_preds = self.ethnicity_layer(h['individual'])
        
        # Apply softmax to each prediction set
        age_probs = F.softmax(age_preds, dim=1)
        sex_probs = F.softmax(sex_preds, dim=1)
        ethnicity_probs = F.softmax(ethnicity_preds, dim=1)
        
        # Concatenate the probabilities back
        h['individual'] = torch.cat([age_probs, sex_probs, ethnicity_probs], dim=1)
        
        return h['individual']

# Load the data from individual and cross-tables 
current_dir = os.path.dirname(os.path.abspath(__file__))
age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individual/Age_5yrs.csv'))
sex_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individual/Sex.csv'))
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individual/Ethnic.csv'))
ethnic_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/ethnic_by_sex_by_age_modified.csv'))

# Define the Oxford areas
oxford_areas = ['E02005921']

# Filter the DataFrame for the specified Oxford areas
age_df = age_df[age_df['geography code'].isin(oxford_areas)]
sex_df = sex_df[sex_df['geography code'].isin(oxford_areas)]
ethnicity_df = ethnicity_df[ethnicity_df['geography code'].isin(oxford_areas)]
ethnic_by_sex_by_age_df = ethnic_by_sex_by_age_df[ethnic_by_sex_by_age_df['geography code'].isin(oxford_areas)]

# Rename columns to avoid conflicts
age_df = age_df.rename(columns={'total': 'total_age'})
sex_df = sex_df.rename(columns={'total': 'total_sex'})
ethnicity_df = ethnicity_df.rename(columns={'total': 'total_ethnicity'})

# Merge datasets on common keys (e.g., geography code)
merged_df = pd.merge(pd.merge(age_df, sex_df, on='geography code'), ethnicity_df, on='geography code')

# Initialize the heterogeneous graph and possible relations 
hetero_graph = dgl.heterograph({
    ('individual', 'has_age', 'age'): ([], []),
    ('individual', 'has_sex', 'sex'): ([], []),
    ('individual', 'has_ethnicity', 'ethnicity'): ([], []),
    ('individual', 'self_loop', 'individual'): ([], [])
})

# Add attribute nodes for age groups, sex, and ethnicity
hetero_graph.add_nodes(len(age_groups), ntype='age')
hetero_graph.add_nodes(len(sex_categories), ntype='sex')
hetero_graph.add_nodes(len(ethnicity_categories), ntype='ethnicity')

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

# Function to create exact number of edges between individual nodes and age/sex/ethnicity nodes based on the merged data
def create_exact_edges(hetero_graph, merged_df, age_groups, sex_categories, ethnicity_categories):
    individual_ids = list(range(hetero_graph.num_nodes('individual')))
    age_ids = {age: i for i, age in enumerate(age_groups)}
    sex_ids = {sex: i for i, sex in enumerate(sex_categories)}
    ethnicity_ids = {ethnicity: i for i, ethnicity in enumerate(ethnicity_categories)}
    
    edges = {'has_age': ([], []), 'has_sex': ([], []), 'has_ethnicity': ([], [])}
    
    for index, row in merged_df.iterrows():
        geo_code = row['geography code']
        total_individuals = row['total_age']
        
        # Collect age, sex, and ethnicity counts from the dataframe
        age_counts = {age: int(row.get(age, 0)) for age in age_groups}
        sex_counts = {sex: int(row.get(sex, 0)) for sex in sex_categories}
        ethnicity_counts = {ethnicity: int(row.get(ethnicity, 0)) for ethnicity in ethnicity_categories}
        
        # Create a list of age labels based on counts
        age_labels = []
        for age, count in age_counts.items():
            age_labels.extend([age_ids[age]] * count)
        random.shuffle(age_labels)
        
        # Create a list of sex labels based on counts
        sex_labels = []
        for sex, count in sex_counts.items():
            sex_labels.extend([sex_ids[sex]] * count)
        random.shuffle(sex_labels)
        
        # Create a list of ethnicity labels based on counts
        ethnicity_labels = []
        for ethnicity, count in ethnicity_counts.items():
            ethnicity_labels.extend([ethnicity_ids[ethnicity]] * count)
        random.shuffle(ethnicity_labels)
        
        # Assign age edges
        for i in range(total_individuals):
            edges['has_age'][0].append(individual_ids[i])
            edges['has_age'][1].append(age_labels[i])
        
        # Assign sex edges
        for i in range(total_individuals):
            edges['has_sex'][0].append(individual_ids[i])
            edges['has_sex'][1].append(sex_labels[i])
        
        # Assign ethnicity edges
        for i in range(total_individuals):
            edges['has_ethnicity'][0].append(individual_ids[i])
            edges['has_ethnicity'][1].append(ethnicity_labels[i])
    
    hetero_graph.add_edges(edges['has_age'][0], edges['has_age'][1], etype='has_age')
    hetero_graph.add_edges(edges['has_sex'][0], edges['has_sex'][1], etype='has_sex')
    hetero_graph.add_edges(edges['has_ethnicity'][0], edges['has_ethnicity'][1], etype='has_ethnicity')

# Use the new function
create_exact_edges(hetero_graph, merged_df, age_groups, sex_categories, ethnicity_categories)

# Clean the cross table to remove the 'geography code' and 'total' columns
ethnic_by_sex_by_age_df = ethnic_by_sex_by_age_df.drop(columns=['geography code', 'total'], errors='ignore')

# Updated aggregate function
def aggregate(encoded_tensor, cross_table, category_dicts, column_order):
    split_sizes = [len(cat_dict) for cat_dict in category_dicts]
    if encoded_tensor.size(1) != sum(split_sizes):
        raise ValueError("Size mismatch between encoded_tensor and category_dicts")

    category_probs = torch.split(encoded_tensor, split_sizes, dim=1)
    aggregated_tensor = torch.zeros(len(cross_table.columns), device=encoded_tensor.device)

    for i, col in enumerate(cross_table.columns):
        categories = col.split()
        if len(categories) != len(column_order):
            raise ValueError(f"Unexpected number of category keys in {categories}")

        expected_count = torch.ones(encoded_tensor.size(0), device=encoded_tensor.device)

        for cat_index, category in enumerate(categories):
            category_dict_index = column_order[cat_index]

            if category not in category_dicts[category_dict_index]:
                raise ValueError(f"Category {category} not found in category_dicts[{category_dict_index}]")

            cat_pos = list(category_dicts[category_dict_index]).index(category)
            expected_count *= category_probs[category_dict_index][:, cat_pos]

        aggregated_tensor[i] = torch.sum(expected_count)

    return aggregated_tensor

def rmse_loss(aggregated_tensor, target_tensor):
    return torch.sqrt(torch.mean((aggregated_tensor - target_tensor) ** 2))

# Define the objective function
def objective_function(hetero_graph, ethnic_by_sex_by_age_df, model):
    # Get node embeddings from the model
    h_individual = model(hetero_graph)
    
    # Specify the order of categories in the columns of the cross-table
    column_order = [0, 1, 2]  # Sex, Age, Ethnicity

    # Aggregate the results for ethnic_by_sex_by_age
    aggregated_ethnic_sex_age = aggregate(h_individual, ethnic_by_sex_by_age_df, [sex_categories, age_groups, ethnicity_categories], column_order)
    
    # Calculate RMSE as the objective function
    target_counts_ethnic_sex_age = torch.tensor(ethnic_by_sex_by_age_df.values, dtype=torch.float32)
    loss_ethnic_sex_age = rmse_loss(aggregated_ethnic_sex_age, target_counts_ethnic_sex_age)
    
    return loss_ethnic_sex_age

# Combined function to count the number of edges for either age groups or sex categories
def count_edges(graph, etype, categories):
    edges = graph.edges(etype=etype)
    counts = torch.zeros(len(categories))
    
    for dst in edges[1]:
        counts[dst] += 1
    
    counts_dict = {categories[i]: int(counts[i].item()) for i in range(len(categories))}
    
    return counts_dict

# Define and initialize the model
num_individuals = hetero_graph.num_nodes('individual')
num_age_groups = len(age_groups)
num_sex_categories = len(sex_categories)
num_ethnicity_categories = len(ethnicity_categories)
in_feats = 32  # or the appropriate size
hidden_size = 64
out_feats = 64

model = MultiTaskGNNModel(num_individuals, num_age_groups, num_sex_categories, num_ethnicity_categories, in_feats, hidden_size, out_feats)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize the best loss to a large value
best_loss = float('inf')

# List to store loss values
loss_values = []

# Optimization loop
num_iterations = 200
for iteration in range(num_iterations):
    model.train()
    optimizer.zero_grad()
    
    # Calculate the objective function
    loss = objective_function(hetero_graph, ethnic_by_sex_by_age_df, model)
    
    # Backpropagation
    loss.backward()
    optimizer.step()

    # Store the loss value
    loss_values.append(loss.item())
    
    # Check if the new graph is better
    original_loss = loss.item()  # Store the original loss

    # Randomly choose an edge type
    etype = np.random.choice(['has_age', 'has_sex', 'has_ethnicity'])
    
    # Get all individual nodes and their neighbors for the chosen edge type
    src_nodes, dst_nodes = hetero_graph.edges(etype=etype)
    
    # Find two random individual nodes with different age or sex for the chosen edge type
    while True:
        idx1, idx2 = np.random.choice(len(src_nodes), 2, replace=False)
        node1, node2 = src_nodes[idx1], src_nodes[idx2]
        
        if dst_nodes[idx1] != dst_nodes[idx2]:  # Ensure different age or sex based on edge type
            break

    # Get the edge IDs to remove
    edge_id1 = hetero_graph.edge_ids(node1, dst_nodes[idx1], etype=etype)
    edge_id2 = hetero_graph.edge_ids(node2, dst_nodes[idx2], etype=etype)

    # Swap the edges for the chosen edge type
    hetero_graph = dgl.remove_edges(hetero_graph, [edge_id1, edge_id2], etype=etype)
    hetero_graph.add_edges([node1, node2], [dst_nodes[idx2], dst_nodes[idx1]], etype=etype)

    # Calculate the new loss after the modification
    new_loss = objective_function(hetero_graph, ethnic_by_sex_by_age_df, model)
    
    if new_loss < original_loss:  # Compare new loss with the original loss
        best_loss = new_loss
    else:
        # Revert the change by swapping the edges back
        edge_id1_new = hetero_graph.edge_ids(node1, dst_nodes[idx2], etype=etype)
        edge_id2_new = hetero_graph.edge_ids(node2, dst_nodes[idx1], etype=etype)
        hetero_graph = dgl.remove_edges(hetero_graph, [edge_id1_new, edge_id2_new], etype=etype)
        hetero_graph.add_edges([node1, node2], [dst_nodes[idx1], dst_nodes[idx2]], etype=etype)
    
    # if iteration % 20 == 0:
        # age_counts = count_edges(hetero_graph, 'has_age', age_groups)
        # sex_counts = count_edges(hetero_graph, 'has_sex', sex_categories)
        # ethnicity_counts = count_edges(hetero_graph, 'has_ethnicity', ethnicity_categories)
        # print(f"Age counts: {age_counts}")
        # print(f"Sex counts: {sex_counts}")
        # print(f"Ethnicity counts: {ethnicity_counts}")
    print(f"Iteration {iteration}, Loss: {loss.item()}")

print("Final graph structure:")
print(hetero_graph)

# Updated function to count the actual number of people in each sex-age-ethnicity category from the cross table
def count_actual_population(ethnic_by_sex_by_age_df, age_groups, sex_categories, ethnicity_categories):
    actual_counts = {(sex, age, ethnicity): 0 for age in age_groups for sex in sex_categories for ethnicity in ethnicity_categories}
    
    for col in ethnic_by_sex_by_age_df.columns:
        categories = col.split()
        if len(categories) != 3:
            continue
        sex, age, ethnicity = categories
        if sex in sex_categories and age in age_groups and ethnicity in ethnicity_categories:
            actual_counts[(sex, age, ethnicity)] = ethnic_by_sex_by_age_df[col].sum()
    
    return actual_counts

# Updated function to count the number of people in each sex-age-ethnicity category from the generated population
def count_generated_population(hetero_graph, age_groups, sex_categories, ethnicity_categories):
    age_edges = hetero_graph.edges(etype='has_age')
    sex_edges = hetero_graph.edges(etype='has_sex')
    ethnicity_edges = hetero_graph.edges(etype='has_ethnicity')
    
    generated_counts = {(sex, age, ethnicity): 0 for age in age_groups for sex in sex_categories for ethnicity in ethnicity_categories}
    
    for i in range(len(age_edges[0])):
        individual = age_edges[0][i]
        age = age_groups[age_edges[1][i].item()]
        sex = sex_categories[sex_edges[1][individual].item()]
        ethnicity = ethnicity_categories[ethnicity_edges[1][individual].item()]
        generated_counts[(sex, age, ethnicity)] += 1
    
    return generated_counts

def create_radar_chart(actual_counts, generated_counts, age_groups, sex_categories, ethnicity_categories):
    categories = [f"{sex} {age} {ethnicity}" for sex in sex_categories for age in age_groups for ethnicity in ethnicity_categories]
    
    actual_values = [actual_counts[(sex, age, ethnicity)] for sex in sex_categories for age in age_groups for ethnicity in ethnicity_categories]
    generated_values = [generated_counts[(sex, age, ethnicity)] for sex in sex_categories for age in age_groups for ethnicity in ethnicity_categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=actual_values,
        theta=categories,
        fill='toself',
        name='Actual'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=generated_values,
        theta=categories,
        fill='toself',
        name='Generated'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(actual_values), max(generated_values))]
            )),
        showlegend=True
    )
    
    fig.show()

# Count the actual and generated populations
actual_counts = count_actual_population(ethnic_by_sex_by_age_df, age_groups, sex_categories, ethnicity_categories)
generated_counts = count_generated_population(hetero_graph, age_groups, sex_categories, ethnicity_categories)

# Create the radar chart
create_radar_chart(actual_counts, generated_counts, age_groups, sex_categories, ethnicity_categories)