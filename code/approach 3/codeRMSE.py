import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch.nn import MSELoss

# Load the data from individual tables
current_dir = os.path.dirname(os.path.abspath(__file__))
age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Age_5yrs.csv'))
sex_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Sex.csv'))
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Ethnic.csv'))
ethnic_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/ethnic_by_sex_by_age_modified.csv'))

# Define the Oxford areas
oxford_areas = ['E02005921']

# Filter the DataFrame for the specified Oxford areas
age_df = age_df[age_df['geography code'].isin(oxford_areas)]
sex_df = sex_df[sex_df['geography code'].isin(oxford_areas)]
ethnicity_df = ethnicity_df[ethnicity_df['geography code'].isin(oxford_areas)]
ethnic_by_sex_by_age_df = ethnic_by_sex_by_age_df[ethnic_by_sex_by_age_df['geography code'].isin(oxford_areas)]

# Define the age groups, sex categories, and ethnicity categories
age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
sex_categories = ['M', 'F']
ethnicity_categories = ['W0', 'M0', 'A0', 'B0', 'O0']

# Encode the categories to indices
age_map = {category: i for i, category in enumerate(age_groups)}
sex_map = {category: i for i, category in enumerate(sex_categories)}
ethnicity_map = {category: i for i, category in enumerate(ethnicity_categories)}

# Total number of persons from the total column
num_persons = int(age_df['total'].sum())

# Create person nodes with unique IDs
person_nodes = torch.arange(num_persons).view(num_persons, 1)

# Create nodes for age categories
age_nodes = torch.tensor([[age_map[age]] for age in age_groups], dtype=torch.float)

# Create nodes for sex categories
sex_nodes = torch.tensor([[sex_map[sex]] for sex in sex_categories], dtype=torch.float)

# Create nodes for ethnicity categories
ethnicity_nodes = torch.tensor([[ethnicity_map[ethnicity]] for ethnicity in ethnicity_categories], dtype=torch.float)

# Combine all nodes into a single tensor
node_features = torch.cat([person_nodes, age_nodes, sex_nodes, ethnicity_nodes], dim=0)

# Edge index construction
edge_index = []

# Connect each person to age nodes based on the age_df
person_idx = 0
for _, row in age_df.iterrows():
    for age in age_groups:
        count = int(row[age])
        for _ in range(count):
            if person_idx < num_persons:
                age_id = age_map[age] + num_persons
                edge_index.append([person_idx, age_id])
                person_idx += 1

# Reset person index and connect each person to sex nodes based on the sex_df
person_idx = 0
for _, row in sex_df.iterrows():
    for sex in sex_categories:
        count = int(row[sex])
        for _ in range(count):
            if person_idx < num_persons:
                sex_id = sex_map[sex] + num_persons + len(age_groups)
                edge_index.append([person_idx, sex_id])
                person_idx += 1

# Reset person index and connect each person to ethnicity nodes based on the ethnicity_df
person_idx = 0
for _, row in ethnicity_df.iterrows():
    for ethnicity in ethnicity_categories:
        count = int(row[ethnicity])
        for _ in range(count):
            if person_idx < num_persons:
                ethnicity_id = ethnicity_map[ethnicity] + num_persons + len(age_groups) + len(sex_categories)
                edge_index.append([person_idx, ethnicity_id])
                person_idx += 1

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Create the data object for PyTorch Geometric
data = Data(x=node_features, edge_index=edge_index)

# Initialize target tensors
y_age = torch.zeros(num_persons, dtype=torch.float)  # Change to float for regression
y_sex = torch.zeros(num_persons, dtype=torch.float)
y_ethnicity = torch.zeros(num_persons, dtype=torch.float)

# Populate target tensors based on the cross table
person_idx = 0
for _, row in ethnic_by_sex_by_age_df.iterrows():
    for age in age_groups:
        for sex in sex_categories:
            for ethnicity in ethnicity_categories:
                col_name = f'{sex} {age} {ethnicity}'
                count = int(row.get(col_name, 0))
                for _ in range(count):
                    if person_idx < num_persons:
                        y_age[person_idx] = age_map.get(age, -1)
                        y_sex[person_idx] = sex_map.get(sex, -1)
                        y_ethnicity[person_idx] = ethnicity_map.get(ethnicity, -1)
                        person_idx += 1

# Define the enhanced GNN model using GraphSAGE layers
class EnhancedGNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels_age, out_channels_sex, out_channels_ethnicity):
        super(EnhancedGNNModel, self).__init__()
        # Define the GraphSAGE layers for the model
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4_age = SAGEConv(hidden_channels, out_channels_age)
        self.conv4_sex = SAGEConv(hidden_channels, out_channels_sex)
        self.conv4_ethnicity = SAGEConv(hidden_channels, out_channels_ethnicity)

    def forward(self, data):
        # Perform forward pass through the network
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        # Separate outputs for age, sex, and ethnicity
        age_out = self.conv4_age(x, edge_index)
        sex_out = self.conv4_sex(x, edge_index)
        ethnicity_out = self.conv4_ethnicity(x, edge_index)
        return age_out, sex_out, ethnicity_out

# Initialize model, optimizer, and loss functions
model = EnhancedGNNModel(in_channels=node_features.size(1), hidden_channels=64, out_channels_age=1, out_channels_sex=1, out_channels_ethnicity=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn_age = MSELoss()
loss_fn_sex = MSELoss()
loss_fn_ethnicity = MSELoss()

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear gradients

    # Forward pass
    age_out, sex_out, ethnicity_out = model(data)
    age_out = age_out[:num_persons].squeeze()  # Only take person nodes' outputs and squeeze for regression
    sex_out = sex_out[:num_persons].squeeze()
    ethnicity_out = ethnicity_out[:num_persons].squeeze()

    # Calculate loss for age, sex, and ethnicity
    loss_age = loss_fn_age(age_out, y_age)
    loss_sex = loss_fn_sex(sex_out, y_sex)
    loss_ethnicity = loss_fn_ethnicity(ethnicity_out, y_ethnicity)

    # Total loss
    loss = loss_age + loss_sex + loss_ethnicity

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Calculate RMSE
    with torch.no_grad():
        rmse_age = torch.sqrt(loss_age).item()
        rmse_sex = torch.sqrt(loss_sex).item()
        rmse_ethnicity = torch.sqrt(loss_ethnicity).item()

        # Convert outputs to class indices for accuracy calculation
        age_pred = age_out.round().clamp(0, len(age_groups) - 1).long()  # Clamp to ensure valid indices
        sex_pred = sex_out.round().clamp(0, len(sex_categories) - 1).long()
        ethnicity_pred = ethnicity_out.round().clamp(0, len(ethnicity_categories) - 1).long()

        # Calculate accuracies for age, sex, and ethnicity
        age_accuracy = (age_pred == y_age.long()).sum().item() / num_persons
        sex_accuracy = (sex_pred == y_sex.long()).sum().item() / num_persons
        ethnicity_accuracy = (ethnicity_pred == y_ethnicity.long()).sum().item() / num_persons

        # Calculate net accuracy (all predictions correct)
        net_accuracy = ((age_pred == y_age.long()) & (sex_pred == y_sex.long()) & (ethnicity_pred == y_ethnicity.long())).sum().item() / num_persons

    # Print metrics every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Total Loss: {loss.item():.4f}, RMSE Age: {rmse_age:.4f}, RMSE Sex: {rmse_sex:.4f}, RMSE Ethnicity: {rmse_ethnicity:.4f}, Age Accuracy: {age_accuracy:.4f}, Sex Accuracy: {sex_accuracy:.4f}, Ethnicity Accuracy: {ethnicity_accuracy:.4f}, Net Accuracy: {net_accuracy:.4f}')

# Get the final predictions after training
# model.eval()  # Set model to evaluation mode
with torch.no_grad():
    age_out, sex_out, ethnicity_out = model(data)
    age_pred = age_out[:num_persons].squeeze().round().clamp(0, len(age_groups) - 1).long()
    sex_pred = sex_out[:num_persons].squeeze().round().clamp(0, len(sex_categories) - 1).long()
    ethnicity_pred = ethnicity_out[:num_persons].squeeze().round().clamp(0, len(ethnicity_categories) - 1).long()