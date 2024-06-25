import pandas as pd
import numpy as np
import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import os

# Load the data
age_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/preprocessed-data/individual/Age_5yrs.csv'))
sex_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/preprocessed-data/individual/Sex.csv'))
sex_by_age_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/preprocessed-data/crosstables/sex_by_age_5yrs.csv'))

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

# Initialize the heterogeneous graph
hetero_graph = dgl.heterograph({
    ('individual', 'has_age', 'age'): ([], []),
    ('individual', 'has_sex', 'sex'): ([], []),
})

# Define attribute nodes (age groups, sex)
age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
sex_categories = ['M', 'F']

# Add attribute nodes for age groups and sex
for age in age_groups:
    hetero_graph.add_nodes(1, ntype='age')
for sex in sex_categories:
    hetero_graph.add_nodes(1, ntype='sex')

# Add individual nodes without edges
individual_count = 0
for index, row in merged_df.iterrows():
    geo_code = row['geography code']
    total_individuals = row['total_age']
    
    for i in range(total_individuals):
        hetero_graph.add_nodes(1, ntype='individual')
        individual_count += 1

# Function to create probabilistic edges
def create_probabilistic_edges(hetero_graph, merged_df, age_groups, sex_categories):
    individual_ids = list(range(hetero_graph.num_nodes('individual')))
    age_ids = {f"age_{age}": hetero_graph.nodes('age')[i].item() for i, age in enumerate(age_groups)}
    sex_ids = {f"sex_{sex}": hetero_graph.nodes('sex')[i].item() for i, sex in enumerate(sex_categories)}
    
    edges = {'has_age': ([], []), 'has_sex': ([], [])}
    
    for index, row in merged_df.iterrows():
        geo_code = row['geography code']
        total_individuals = row['total_age']
        
        for i in range(total_individuals):
            individual_id = individual_ids.pop(0)
            age_probs = [row[f"{age}"] / total_individuals for age in age_groups]
            sex_probs = [row[f"{sex}"] / total_individuals for sex in sex_categories]
            
            age_label = np.random.choice(age_groups, p=age_probs)
            sex_label = np.random.choice(sex_categories, p=sex_probs)
            
            edges['has_age'][0].append(individual_id)
            edges['has_age'][1].append(age_ids[f"age_{age_label}"])
            edges['has_sex'][0].append(individual_id)
            edges['has_sex'][1].append(sex_ids[f"sex_{sex_label}"])
    
    hetero_graph.add_edges(edges['has_age'][0], edges['has_age'][1], etype='has_age')
    hetero_graph.add_edges(edges['has_sex'][0], edges['has_sex'][1], etype='has_sex')

create_probabilistic_edges(hetero_graph, merged_df, age_groups, sex_categories)


# Print graph structure and node features
print("Graph structure:")
print(hetero_graph)

# Print node and edge features
print("Node types and their features:")
for ntype in hetero_graph.ntypes:
    print(f"Node type '{ntype}':")
    print(hetero_graph.nodes[ntype].data)

print("Edge types and their features:")
for etype in hetero_graph.etypes:
    print(f"Edge type '{etype}':")
    print(hetero_graph.edges[etype].data)
    

class HeteroGNN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(HeteroGNN, self).__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            'has_age': dglnn.GraphConv(in_feats, h_feats),
            'has_sex': dglnn.GraphConv(in_feats, h_feats)
        }, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            'has_age': dglnn.GraphConv(h_feats, out_feats),
            'has_sex': dglnn.GraphConv(h_feats, out_feats)
        }, aggregate='sum')

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        print("After conv1:", h)  # Debug: print intermediate result
        h = {k: torch.relu(v) for k, v in h.items()}
        print("After ReLU:", h)  # Debug: print intermediate result

        # Ensure we only pass required node types to conv2
        h_conv2_inputs = {k: v for k, v in h.items() if k in ['age', 'sex']}
        print("Inputs to conv2:", h_conv2_inputs)  # Debug: check inputs to conv2
        
        h = self.conv2(g, h_conv2_inputs)
        print("After conv2:", h)  # Debug: print final output
        return h

# Edge prediction function
def edge_prediction(g, h_dict, etype):
    with g.local_scope():
        for ntype in h_dict:
            g.nodes[ntype].data['h'] = h_dict[ntype]
        g.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'), etype=etype)
        return g.edata['score'][etype]


# Initialize model, optimizer, and loss function
model = HeteroGNN(in_feats=10, h_feats=16, out_feats=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

# Prepare dummy features and labels for training
features = {
    'individual': torch.randn((hetero_graph.num_nodes('individual'), 10)),
    'age': torch.randn((hetero_graph.num_nodes('age'), 10)),
    'sex': torch.randn((hetero_graph.num_nodes('sex'), 10))
}

# Create labels for edge prediction
age_labels = torch.zeros(hetero_graph.num_edges(('individual', 'has_age', 'age')))
sex_labels = torch.zeros(hetero_graph.num_edges(('individual', 'has_sex', 'sex')))
labels = torch.cat([age_labels, sex_labels])

# Example training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    h_dict = model(hetero_graph, features)  # Get node embeddings
    print("Node embeddings shapes:", {k: v.shape for k, v in h_dict.items()})  # Debug: print shapes
    
    age_logits = edge_prediction(hetero_graph, h_dict, 'has_age')
    sex_logits = edge_prediction(hetero_graph, h_dict, 'has_sex')
    logits_combined = torch.cat([age_logits, sex_logits])
    
    loss = loss_fn(logits_combined, labels)  # Calculate the loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")


import matplotlib.pyplot as plt

def generate_synthetic_individuals(model, hetero_graph, num_individuals, age_groups, sex_categories):
    model.eval()
    synthetic_individuals = []

    with torch.no_grad():
        for _ in range(num_individuals):
            new_node_id = hetero_graph.number_of_nodes(ntype='individual')
            hetero_graph.add_nodes(1, ntype='individual')

            h_dict = model(hetero_graph, features)
            predicted_attributes = torch.sigmoid(h_dict['individual'][new_node_id]).numpy()
            
            age_probabilities = predicted_attributes[:len(age_groups)]
            sex_probabilities = predicted_attributes[len(age_groups):]
            
            age = np.random.choice(age_groups, p=age_probabilities)
            sex = np.random.choice(sex_categories, p=sex_probabilities)
            
            synthetic_individuals.append({
                'individual_id': new_node_id,
                'age': age,
                'sex': sex
            })
            
            # Add edges based on predicted attributes
            age_node = hetero_graph.nodes('age')[age_groups.index(age)]
            sex_node = hetero_graph.nodes('sex')[sex_categories.index(sex)]
            hetero_graph.add_edges(new_node_id, age_node, etype='has_age')
            hetero_graph.add_edges(new_node_id, sex_node, etype='has_sex')

    return synthetic_individuals

# Generate synthetic individuals
synthetic_population = generate_synthetic_individuals(model, hetero_graph, 1000, age_groups, sex_categories)

def plot_distribution(real_data, synthetic_data, attribute):
    plt.figure(figsize=(10, 5))
    plt.hist(real_data, bins=20, alpha=0.5, label='Real')
    plt.hist(synthetic_data, bins=20, alpha=0.5, label='Synthetic')
    plt.xlabel(attribute)
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title(f'Distribution of {attribute}')
    plt.show()

# Example: Compare age distribution
real_age_distribution = age_df.values.flatten().tolist()[2:]  # Skip 'geography code' and 'total'
synthetic_age_distribution = [ind['age'] for ind in synthetic_population]
plot_distribution(real_age_distribution, synthetic_age_distribution, 'Age')

# Example: Compare sex distribution
real_sex_distribution = sex_df.values.flatten().tolist()[2:]  # Skip 'geography code' and 'total'
synthetic_sex_distribution = [ind['sex'] for ind in synthetic_population]
plot_distribution(real_sex_distribution, synthetic_sex_distribution, 'Sex')