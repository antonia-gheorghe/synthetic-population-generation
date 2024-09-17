import os
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm

# show full length tensor
torch.set_printoptions(profile="full")

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


df = ethnic_by_sex_by_age_df
#drop the geography code and total columns
df = df.drop(columns=['geography code', 'total'])
# iterate and convert to dictionary
observed_counts = {}
for index, row in df.iterrows():
    for col in df.columns:
        observed_counts[col] = row[col]


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

def generate_edge_index(num_persons):
    edge_index = []
    age_start_idx = num_persons
    sex_start_idx = age_start_idx + len(age_groups)
    ethnicity_start_idx = sex_start_idx + len(sex_categories)

    for i in range(num_persons):
        age_category = random.choice(range(age_start_idx, sex_start_idx))
        sex_category = random.choice(range(sex_start_idx, ethnicity_start_idx))
        ethnicity_category = random.choice(range(ethnicity_start_idx, ethnicity_start_idx + len(ethnicity_categories)))
        # print(age_category)
        edge_index.append([i, age_category])
        edge_index.append([i, sex_category])
        edge_index.append([i, ethnicity_category])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

edge_index = generate_edge_index(num_persons)

# Create the data object for PyTorch Geometric
data = Data(x=node_features, edge_index=edge_index)

# Initialize target tensors
y_age = torch.zeros(num_persons, dtype=torch.long)
y_sex = torch.zeros(num_persons, dtype=torch.long)
y_ethnicity = torch.zeros(num_persons, dtype=torch.long)

# person_idx = 0
# for key, count in observed_counts.items():
#     sex, age, ethnicity = key.split(' ')
#     for _ in range(count):
#         y_age[person_idx] = age_map[age]
#         y_sex[person_idx] = sex_map[sex]
#         y_ethnicity[person_idx] = ethnicity_map[ethnicity]
#         person_idx += 1


# Initialize target tensors
# y_age_2 = torch.zeros(num_persons, dtype=torch.long)
# y_sex_2 = torch.zeros(num_persons, dtype=torch.long)
# y_ethnicity_2 = torch.zeros(num_persons, dtype=torch.long)

# Populate target tensors based on the cross table
person_idx = 0
for _, row in ethnic_by_sex_by_age_df.iterrows():
    for sex in sex_categories:
        for age in age_groups:
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
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.batch_norm2 = BatchNorm(hidden_channels)
        self.batch_norm3 = BatchNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(0.5)
        self.conv4_age = SAGEConv(hidden_channels, out_channels_age)
        self.conv4_sex = SAGEConv(hidden_channels, out_channels_sex)
        self.conv4_ethnicity = SAGEConv(hidden_channels, out_channels_ethnicity)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.dropout(x)
        age_out = self.conv4_age(x, edge_index)
        sex_out = self.conv4_sex(x, edge_index)
        ethnicity_out = self.conv4_ethnicity(x, edge_index)
        return age_out, sex_out, ethnicity_out


# Custom loss function
def custom_loss_function(age_out, sex_out, ethnicity_out, y_age, y_sex, y_ethnicity):
    age_pred = age_out.argmax(dim=1)
    sex_pred = sex_out.argmax(dim=1)
    ethnicity_pred = ethnicity_out.argmax(dim=1)
    loss_age = F.cross_entropy(age_out, y_age)
    loss_sex = F.cross_entropy(sex_out, y_sex)
    loss_ethnicity = F.cross_entropy(ethnicity_out, y_ethnicity)
    total_loss = loss_age + loss_sex + loss_ethnicity
    return total_loss

# Initialize model, optimizer, and loss functions
model = EnhancedGNNModel(in_channels=node_features.size(1), hidden_channels=256, out_channels_age=21, out_channels_sex=2, out_channels_ethnicity=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Training loop
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear gradients

    # Forward pass
    age_out, sex_out, ethnicity_out = model(data)
    age_out = age_out[:num_persons]  # Only take person nodes' outputs
    sex_out = sex_out[:num_persons]
    ethnicity_out = ethnicity_out[:num_persons]

    # print(sex_out)

    # Calculate loss
    loss = custom_loss_function(age_out, sex_out, ethnicity_out, y_age, y_sex, y_ethnicity)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    # scheduler.step()

    # Calculate accuracy
    with torch.no_grad():
        age_pred = age_out.argmax(dim=1)
        sex_pred = sex_out.argmax(dim=1)
        ethnicity_pred = ethnicity_out.argmax(dim=1)

        # Calculate accuracies for age, sex, and ethnicity
        age_accuracy = (age_pred == y_age).sum().item() / num_persons
        sex_accuracy = (sex_pred == y_sex).sum().item() / num_persons
        ethnicity_accuracy = (ethnicity_pred == y_ethnicity).sum().item() / num_persons

        # Calculate net accuracy (all predictions correct)
        net_accuracy = ((age_pred == y_age) & (sex_pred == y_sex) & (ethnicity_pred == y_ethnicity)).sum().item() / num_persons

    # Print metrics every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}, Age Accuracy: {age_accuracy:.4f}, Sex Accuracy: {sex_accuracy:.4f}, Ethnicity Accuracy: {ethnicity_accuracy:.4f}, Net Accuracy: {net_accuracy:.4f}')

# Get the final predictions after training
# model.eval()  # Set model to evaluation mode
with torch.no_grad():
    age_out, sex_out, ethnicity_out = model(data)
    age_pred = age_out[:num_persons].argmax(dim=1)
    sex_pred = sex_out[:num_persons].argmax(dim=1)
    ethnicity_pred = ethnicity_out[:num_persons].argmax(dim=1)

# Calculate observed counts
observed_counts = {}
for age in age_groups:
    for sex in sex_categories:
        for ethnicity in ethnicity_categories:
            key = f"{age}-{sex}-{ethnicity}"
            count = int(ethnic_by_sex_by_age_df[f"{sex} {age} {ethnicity}"].sum())
            observed_counts[key] = count

# Calculate predicted counts
predicted_counts = {key: 0 for key in observed_counts.keys()}
for i in range(num_persons):
    age = age_groups[age_pred[i]]
    sex = sex_categories[sex_pred[i]]
    ethnicity = ethnicity_categories[ethnicity_pred[i]]
    key = f"{age}-{sex}-{ethnicity}"
    if key in predicted_counts:
        predicted_counts[key] += 1

# Create a DataFrame for comparison
comparison_df = pd.DataFrame({
    "Combination": list(observed_counts.keys()),
    "Observed": list(observed_counts.values()),
    "Predicted": [predicted_counts[key] for key in observed_counts.keys()]
})

# Save the comparison DataFrame to a CSV file
output_path = os.path.join(current_dir, 'comparison_results.csv')
comparison_df.to_csv(output_path, index=False)

# Print the comparison DataFrame
print("Comparison results saved to:", output_path)
print(comparison_df)