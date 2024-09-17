import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GraphNorm
import random
import json

# Set display options permanently
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

substring_mapping = {
    'SP-Elder': '1PE',
    'SP-Adult': '1PA',
    'OF-Elder': '1FE',
    'OF-Married-0C': '1FM-0C',
    'OF-Married-2C': '1FM-2C',
    'OF-Married-ND': '1FM-nA',
    'OF-Cohabiting-0C': '1FC-0C',
    'OF-Cohabiting-2C': '1FC-2C',
    'OF-Cohabiting-ND': '1FC-nA',
    'OF-Lone-2C': '1FL-2C',
    'OF-Lone-ND': '1FL-nA',
    'OH-2C': '1H-2C',
    'OH-Student': '1H-nS',
    'OH-Elder': '1H-nE',
    'OH-Adult': '1H-nA',
}

# Set print options to display all elements of the tensor
torch.set_printoptions(edgeitems=torch.inf)

def get_target_tensors(cross_table, hh_categories, hh_map, feature_categories, feature_map):
    y_hh = torch.zeros(num_households, dtype=torch.long)
    y_feature = torch.zeros(num_households, dtype=torch.long)
    
    # Populate target tensors based on the cross-table and categories
    household_idx = 0

    for _, row in cross_table.iterrows():
        for hh in hh_categories:
            for feature in feature_categories:
                col_name = f'{hh} {feature}'
                count = int(row.get(col_name, -1))
                if count == -1: 
                    print(col_name)
                for _ in range(count):
                    if household_idx < num_households:
                        y_hh[household_idx] = hh_map.get(hh, -1)
                        y_feature[household_idx] = feature_map.get(feature, -1)
                        household_idx += 1

    return y_hh, y_feature

# Load the data from individual tables
current_dir = os.path.dirname(os.path.abspath(__file__))
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Ethnic.csv'))
religion_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Religion.csv'))
hhcomp_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/HH_compositions_old.csv'))
hhcomp_by_ethnicity_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/HH_composition_by_ethnicity_new.csv'))
hhcomp_by_religion_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/HH_composition_by_religion_new.csv'))

# Define the Oxford areas
oxford_areas = ['E02005924']

ethnicity_categories = ['W0', 'M0', 'A0', 'B0', 'O0']
religion_categories = ['C','B','H','J','M','S','OR','NR','NS']

# Filter the DataFrame for the specified Oxford areas
ethnicity_df = ethnicity_df[ethnicity_df['geography code'].isin(oxford_areas)]
religion_df = religion_df[religion_df['geography code'].isin(oxford_areas)]
hhcomp_df = hhcomp_df[hhcomp_df['geography code'].isin(oxford_areas)]
hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df[hhcomp_by_ethnicity_df['geography code'].isin(oxford_areas)]
hhcomp_by_religion_df = hhcomp_by_religion_df[hhcomp_by_religion_df['geography code'].isin(oxford_areas)]

hhcomp_df['1FM-2C'] = hhcomp_df['1FM-1C'] + hhcomp_df['1FM-nC']
hhcomp_df['1FC-2C'] = hhcomp_df['1FC-1C'] + hhcomp_df['1FC-nC']
hhcomp_df['1FL-2C'] = hhcomp_df['1FL-1C'] + hhcomp_df['1FL-nC']
hhcomp_df['1H-2C'] = hhcomp_df['1H-1C'] + hhcomp_df['1H-nC']
hhcomp_df.drop(columns=['1FM-1C', '1FM-nC', '1FC-1C', '1FC-nC', '1FL-1C', '1FL-nC', '1H-1C', '1H-nC', 'total', 'geography code'], inplace=True)
hhcomp_df = hhcomp_df.drop(['1FM', '1FC', '1FL'], axis=1)
hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-2C', '1FM-nA','1FC-0C','1FC-2C','1FC-nA','1FL-nA','1FL-2C','1H-nS','1H-nE','1H-nA', '1H-2C']

filtered_columns = [col for col in hhcomp_by_ethnicity_df.columns if any(substring in col for substring in ['geography code', 'total', 'W0', 'M0', 'A0', 'B0', 'O0'])]
hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df[filtered_columns]

for col in hhcomp_by_ethnicity_df.columns:
    for old_substring, new_substring in substring_mapping.items():
        if old_substring in col:
            new_col = col.replace(old_substring, new_substring)
            hhcomp_by_ethnicity_df.rename(columns={col: new_col}, inplace=True)
            break

for col in hhcomp_by_religion_df.columns:
    for old_substring, new_substring in substring_mapping.items():
        if old_substring in col:
            new_col = col.replace(old_substring, new_substring)
            hhcomp_by_religion_df.rename(columns={col: new_col}, inplace=True)
            break

filtered_columns = [col for col in hhcomp_by_ethnicity_df.columns if not any(substring in col for substring in ['OF-Married', 'OF-Cohabiting', 'OF-LoneParent'])]
hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df[filtered_columns]

filtered_columns = [col for col in hhcomp_by_religion_df.columns if not any(substring in col for substring in ['OF-Married', 'OF-Cohabiting', 'OF-LoneParent'])]
hhcomp_by_religion_df = hhcomp_by_religion_df[filtered_columns]

hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df.drop(columns = ['total', 'geography code'])
hhcomp_by_religion_df = hhcomp_by_religion_df.drop(columns = ['total', 'geography code'])

mapping_dict = {
    '1PE O': '1PE OR',
    '1PE N': '1PE NR',
    '1PA O': '1PA OR',
    '1PA N': '1PA NR',
    '1FE O': '1FE OR',
    '1FE N': '1FE NR',
    '1FM-0C O': '1FM-0C OR',
    '1FM-0C N': '1FM-0C NR',
    '1FM-2C O': '1FM-2C OR',
    '1FM-2C N': '1FM-2C NR',
    '1FM-nA O': '1FM-nA OR',
    '1FM-nA N': '1FM-nA NR',
    '1FC-0C O': '1FC-0C OR',
    '1FC-0C N': '1FC-0C NR',
    '1FC-2C O': '1FC-2C OR',
    '1FC-2C N': '1FC-2C NR',
    '1FC-nA O': '1FC-nA OR',
    '1FC-nA N': '1FC-nA NR',
    '1FL-nA O': '1FL-nA OR',
    '1FL-nA N': '1FL-nA NR',
    '1FL-2C O': '1FL-2C OR',
    '1FL-2C N': '1FL-2C NR',
    '1H-nS O': '1H-nS OR',
    '1H-nS N': '1H-nS NR',
    '1H-nE O': '1H-nE OR',
    '1H-nE N': '1H-nE NR',
    '1H-nA O': '1H-nA OR',
    '1H-nA N': '1H-nA NR',
    '1H-2C O': '1H-2C OR',
    '1H-2C N': '1H-2C NR'
}
hhcomp_by_religion_df.rename(columns=mapping_dict, inplace=True)

# Encode the categories to indices
ethnicity_map = {category: i for i, category in enumerate(ethnicity_categories)}
religion_map = {category: i for i, category in enumerate(religion_categories)}
hh_map = {category: i for i, category in enumerate(hh_compositions)}

# Total number of households from the total column
num_households = 4852

# Create households nodes with unique IDs
households_nodes = torch.arange(num_households).view(num_households, 1)

# Create nodes for ethnicity categories
ethnicity_nodes = torch.tensor([[ethnicity_map[ethnicity]] for ethnicity in ethnicity_categories], dtype=torch.float)

# Create nodes for religion categories
religion_nodes = torch.tensor([[religion_map[religion]] for religion in religion_categories], dtype=torch.float)

# Combine all nodes into a single tensor
node_features = torch.cat([households_nodes, ethnicity_nodes, religion_nodes], dim=0)

def generate_edge_index(num_households):
    edge_index = []
    num_ethnicities = len(ethnicity_map)
    num_religions = len(religion_map)

    ethnicity_start_idx = num_households
    religion_start_idx = ethnicity_start_idx + num_ethnicities

    for i in range(num_households):
        # Randomly select an ethnicity and religion
        ethnicity_category = random.choice(range(ethnicity_start_idx, ethnicity_start_idx + num_ethnicities))
        religion_category = random.choice(range(religion_start_idx, religion_start_idx + num_religions))
        
        # Append edges for the selected categories
        edge_index.append([i, ethnicity_category])
        edge_index.append([i, religion_category])

    # Convert edge_index to a tensor and transpose for PyTorch Geometric
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

# Generate edge index using the new function
edge_index = generate_edge_index(num_households)

# Create the data object for PyTorch Geometric
data = Data(x=node_features, edge_index=edge_index)

class EnhancedGNNModelHousehold(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, mlp_hidden_dim, out_channels_hh, out_channels_ethnicity, out_channels_religion):
        super(EnhancedGNNModelHousehold, self).__init__()
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        
        # Graph normalization layers
        self.graph_norm1 = GraphNorm(hidden_channels)
        self.graph_norm2 = GraphNorm(hidden_channels)
        self.graph_norm3 = GraphNorm(hidden_channels)
        self.graph_norm4 = GraphNorm(hidden_channels)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(0.5)
        
        # MLP layers for each classification target
        self.mlp_hh = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_hh)
        )
        
        self.mlp_ethnicity = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_ethnicity)
        )
        
        self.mlp_religion = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_religion)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Pass through GraphSAGE layers with GraphNorm
        x = self.conv1(x, edge_index)
        x = self.graph_norm1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.graph_norm2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.graph_norm3(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        x = self.conv4(x, edge_index)
        x = self.graph_norm4(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        # Pass the node embeddings through the MLPs for final attribute predictions
        hh_out = self.mlp_hh(x[:num_households])
        ethnicity_out = self.mlp_ethnicity(x[:num_households])
        religion_out = self.mlp_religion(x[:num_households])
        
        return hh_out, ethnicity_out, religion_out

targets = []
targets.append(
    (
        ('hhcomp', 'religion'), 
        get_target_tensors(hhcomp_by_religion_df, hh_compositions, hh_map, religion_categories, religion_map)
    )
)
targets.append(
    (
        ('hhcomp', 'ethnicity'), 
        get_target_tensors(hhcomp_by_ethnicity_df, hh_compositions, hh_map, ethnicity_categories, ethnicity_map)
    )
)

# Initialize model, optimizer, and loss functions
model = EnhancedGNNModelHousehold(
    in_channels=node_features.size(1), 
    hidden_channels=512, 
    mlp_hidden_dim=256,
    out_channels_hh=len(hh_compositions), 
    out_channels_ethnicity=len(ethnicity_categories), 
    out_channels_religion=len(religion_categories)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Custom loss function
def custom_loss_function(hh_out, feature_out, y_hh, y_feature):
    loss_hh = F.cross_entropy(hh_out, y_hh) / 2
    loss_feature = F.cross_entropy(feature_out, y_feature)
    total_loss = loss_hh + loss_feature
    return total_loss

# Save loss and accuracy data as dictionaries
loss_data = {}
accuracy_data = {}

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    if epoch % 100 == 0: 
        print(epoch)
    model.train()
    optimizer.zero_grad()

    # Forward pass
    hh_out, ethnicity_out, religion_out = model(data)

    out = {}
    out['hhcomp'] = hh_out[:num_households]  # Only take household nodes' outputs
    out['ethnicity'] = ethnicity_out[:num_households]
    out['religion'] = religion_out[:num_households]

    # Calculate loss
    loss = 0
    for i in range(len(targets)):
        loss += custom_loss_function(
            out[targets[i][0][0]], out[targets[i][0][1]],
            targets[i][1][0], targets[i][1][1]
        )
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Store loss and accuracy data for each epoch
    loss_data[epoch] = loss.item()
    accuracy_data[epoch] = {}

    with torch.no_grad():
        for i in range(len(targets)):
            accuracy_data[epoch][f'{targets[i][0][0]}, {targets[i][0][1]}'] = {}

            for j in range(2):
                pred = out[targets[i][0][j]].argmax(dim=1)
                accuracy = (pred == targets[i][1][j]).sum().item() / num_households
                accuracy_data[epoch][f'{targets[i][0][0]}, {targets[i][0][1]}'][f'{targets[i][0][j]}'] = accuracy

            # Calculate net accuracy (when both predictions are correct)
            pred_1 = out[targets[i][0][0]].argmax(dim=1)
            pred_2 = out[targets[i][0][1]].argmax(dim=1)
            net_accuracy = ((pred_1 == targets[i][1][0]) & (pred_2 == targets[i][1][1])).sum().item() / num_households
            accuracy_data[epoch][f'{targets[i][0][0]}, {targets[i][0][1]}']['net'] = net_accuracy

# Define paths for saving loss and accuracy data
loss_data_path = os.path.join(current_dir, '../results/household/loss_data.json')
accuracy_data_path = os.path.join(current_dir, '../results/household/accuracy_data.json')

# Save loss data
with open(loss_data_path, 'w') as f:
    json.dump(loss_data, f)

# Save accuracy data
with open(accuracy_data_path, 'w') as f:
    json.dump(accuracy_data, f)

print(f"Loss and accuracy data saved to {loss_data_path} and {accuracy_data_path}")
# Get the final predictions after training
# model.eval()  # Set model to evaluation mode
with torch.no_grad():
    hh_out, ethnicity_out, religion_out = model(data)
    hh_pred = hh_out[:num_households].argmax(dim=1)
    ethnicity_pred = ethnicity_out[:num_households].argmax(dim=1)
    religion_pred = religion_out[:num_households].argmax(dim=1)


# Combine individual features into a single tensor
household_nodes = torch.cat([
    hh_pred.unsqueeze(1), 
    ethnicity_pred.unsqueeze(1), 
    religion_pred.unsqueeze(1),
], dim=1).float()

# Save the tensor to the file
torch.save(household_nodes, os.path.join(current_dir, '../results/household/household_nodes.pt'))

print(f"Person nodes tensor")

# Calculate observed counts for household composition, ethnicity, and religion
hh_ethnicity_counts = {}
for hh in hh_compositions:
    for ethnicity in ethnicity_categories:
        key = f"{hh}-{ethnicity}"
        count = int(hhcomp_by_ethnicity_df[f"{hh} {ethnicity}"].sum())
        hh_ethnicity_counts[key] = count

hh_religion_counts = {}
for hh in hh_compositions:
    for religion in religion_categories:
        key = f"{hh}-{religion}"
        count = int(hhcomp_by_religion_df[f"{hh} {religion}"].sum())
        hh_religion_counts[key] = count

# Calculate predicted counts for household composition, ethnicity, and religion
predicted_counts = {}
for hh in hh_compositions:
    for ethnicity in ethnicity_categories:
        for religion in religion_categories:
            key = f"{hh}-{ethnicity}-{religion}"
            predicted_counts[key] = 0

for i in range(num_households):
    hh = hh_compositions[hh_pred[i]]
    ethnicity = ethnicity_categories[ethnicity_pred[i]]
    religion = religion_categories[religion_pred[i]]
    key = f"{hh}-{ethnicity}-{religion}"
    predicted_counts[key] += 1

# Prepare the data for saving
counts_data = {
    'hh_ethnicity_counts': hh_ethnicity_counts,
    'hh_religion_counts': hh_religion_counts,
    'hh_ethnicity_religion_counts': predicted_counts
}

# Define the output path for counts data
counts_data_path = os.path.join(current_dir, '../results/household/counts.json')

# Save counts data to JSON
with open(counts_data_path, 'w') as f:
    json.dump(counts_data, f)

print(f"Counts data saved to {counts_data_path}")