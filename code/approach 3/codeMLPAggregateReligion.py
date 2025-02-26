import json
import os
import time
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GraphNorm
import random

# Device selection: Use MPS (Metal Performance Shaders) for Mac M1 GPU, or fallback to CPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
print(device)

def get_target_tensors(cross_table, feature_1_categories, feature_1_map, feature_2_categories, feature_2_map, feature_3_categories, feature_3_map):
    y_feature_1 = torch.zeros(num_persons, dtype=torch.long, device=device)
    y_feature_2 = torch.zeros(num_persons, dtype=torch.long, device=device)
    y_feature_3 = torch.zeros(num_persons, dtype=torch.long, device=device)
    
    # Populate target tensors based on the cross table and feature categories
    person_idx = 0
    for _, row in cross_table.iterrows():
        for feature_1 in feature_1_categories:
            for feature_2 in feature_2_categories:
                for feature_3 in feature_3_categories:
                    col_name = f'{feature_1} {feature_2} {feature_3}'
                    count = int(row.get(col_name, 0))
                    for _ in range(count):
                        if person_idx < num_persons:
                            y_feature_1[person_idx] = feature_1_map.get(feature_1, -1)
                            y_feature_2[person_idx] = feature_2_map.get(feature_2, -1)
                            y_feature_3[person_idx] = feature_3_map.get(feature_3, -1)
                            person_idx += 1

    return (y_feature_1, y_feature_2, y_feature_3)


# Load the data from individual tables
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Age_5yrs.csv'))
sex_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Sex.csv'))
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Ethnic.csv'))
religion_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Religion.csv'))
marital_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Marital.csv'))
ethnic_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/ethnic_by_sex_by_age_modified.csv'))
religion_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/religion_by_sex_by_age.csv'))
marital_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/marital_by_sex_by_age.csv'))

# Define the Oxford areas
oxford_areas = ['E02005924']

# Filter the DataFrame for the specified Oxford areas
age_df = age_df[age_df['geography code'].isin(oxford_areas)]
sex_df = sex_df[sex_df['geography code'].isin(oxford_areas)]
ethnicity_df = ethnicity_df[ethnicity_df['geography code'].isin(oxford_areas)]
religion_df = religion_df[religion_df['geography code'].isin(oxford_areas)]
marital_df = marital_df[marital_df['geography code'].isin(oxford_areas)]
ethnic_by_sex_by_age_df = ethnic_by_sex_by_age_df[ethnic_by_sex_by_age_df['geography code'].isin(oxford_areas)]
religion_by_sex_by_age_df = religion_by_sex_by_age_df[religion_by_sex_by_age_df['geography code'].isin(oxford_areas)]
marital_by_sex_by_age_df = marital_by_sex_by_age_df[marital_by_sex_by_age_df['geography code'].isin(oxford_areas)]


# Define the new columns for religion by summing the appropriate religion columns
christian = ['C']  # Christian group
islam = ['M']  # Islam group
other_religions = ['B', 'H', 'J', 'S', 'OR', 'NR', 'NS']  # Other Religions group

# Aggregating religion categories into the defined groups
religion_df['Christian'] = religion_df[christian].sum(axis=1)
religion_df['Islam'] = religion_df[islam].sum(axis=1)
religion_df['Other'] = religion_df[other_religions].sum(axis=1)

# Drop the original religion columns if no longer needed
religion_df = religion_df.drop(columns=christian + islam + other_religions)

# Define the age groups, sex categories, and ethnicity categories
age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
sex_categories = ['M', 'F']
ethnicity_categories = ['W0', 'M0', 'A0', 'B0', 'O0']
religion_categories = ['Christian','Islam','Other']
marital_categories = ['Single','Married','Partner','Separated','Divorced','Widowed']

def aggregate_religion_groups(df, christian, islam, other_religions, sex_categories, age_categories):
    df_res = pd.DataFrame()

    # Iterate through sex categories and religion groups
    for sex in sex_categories:
        for age in age_categories:
            # Christian group
            christian_cols = [col for col in df.columns if any(col == f'{sex} {age} {rel}' for rel in christian)]
            df_res[f'{sex} {age} Christian'] = df[christian_cols].sum(axis=1)

            # Islam group
            islam_cols = [col for col in df.columns if any(col == f'{sex} {age} {rel}' for rel in islam)]
            df_res[f'{sex} {age} Islam'] = df[islam_cols].sum(axis=1)

            # Other Religions group
            other_cols = [col for col in df.columns if any(col == f'{sex} {age} {rel}' for rel in other_religions)]
            df_res[f'{sex} {age} Other'] = df[other_cols].sum(axis=1)

    return df_res

religion_by_sex_by_age_df = aggregate_religion_groups(religion_by_sex_by_age_df,christian, islam, other_religions, sex_categories, age_groups)


print(religion_by_sex_by_age_df.sum(axis=1))


# Encode the categories to indices
age_map = {category: i for i, category in enumerate(age_groups)}
sex_map = {category: i for i, category in enumerate(sex_categories)}
ethnicity_map = {category: i for i, category in enumerate(ethnicity_categories)}
religion_map = {category: i for i, category in enumerate(religion_categories)}
marital_map = {category: i for i, category in enumerate(marital_categories)}

# Total number of persons from the total column
num_persons = int(age_df['total'].sum())

# Create person nodes with unique IDs
person_nodes = torch.arange(num_persons).view(num_persons, 1).to(device)

# Create nodes for age categories
age_nodes = torch.tensor([[age_map[age]] for age in age_groups], dtype=torch.float).to(device)

# Create nodes for sex categories
sex_nodes = torch.tensor([[sex_map[sex]] for sex in sex_categories], dtype=torch.float).to(device)

# Create nodes for ethnicity categories
ethnicity_nodes = torch.tensor([[ethnicity_map[ethnicity]] for ethnicity in ethnicity_categories], dtype=torch.float).to(device)

# Create nodes for religion categories
religion_nodes = torch.tensor([[religion_map[religion]] for religion in religion_categories], dtype=torch.float).to(device)

# Create nodes for marital categories
marital_nodes = torch.tensor([[marital_map[marital]] for marital in marital_categories], dtype=torch.float).to(device)

# Combine all nodes into a single tensor
node_features = torch.cat([person_nodes, age_nodes, sex_nodes, ethnicity_nodes, religion_nodes, marital_nodes], dim=0).to(device)

# Calculate the distribution for age categories
age_probabilities = age_df.drop(columns = ["geography code", "total"]) / num_persons
sex_probabilities = sex_df.drop(columns = ["geography code", "total"]) / num_persons
ethnicity_probabilities = ethnicity_df.drop(columns = ["geography code", "total"]) / num_persons
religion_probabilities = religion_df.drop(columns = ["geography code", "total"]) / num_persons
marital_probabilities = marital_df.drop(columns = ["geography code", "total"]) / num_persons

# New function to generate edge index
def generate_edge_index(num_persons):
    edge_index = []
    age_start_idx = num_persons
    sex_start_idx = age_start_idx + len(age_groups)
    ethnicity_start_idx = sex_start_idx + len(sex_categories)
    religion_start_idx = ethnicity_start_idx + len(ethnicity_categories)
    marital_start_idx = religion_start_idx + len(religion_categories)

    # Convert the probability series to a list of probabilities for sampling
    age_prob_list = age_probabilities.values.tolist()[0]
    sex_prob_list = sex_probabilities.values.tolist()[0]
    ethnicity_prob_list = ethnicity_probabilities.values.tolist()[0]
    religion_prob_list = religion_probabilities.values.tolist()[0]
    marital_prob_list = marital_probabilities.values.tolist()[0]

    for i in range(num_persons):
        # Sample the categories using weighted random sampling
        age_category = random.choices(range(age_start_idx, sex_start_idx), weights=age_prob_list, k=1)[0]
        sex_category = random.choices(range(sex_start_idx, ethnicity_start_idx), weights=sex_prob_list, k=1)[0]
        ethnicity_category = random.choices(range(ethnicity_start_idx, religion_start_idx), weights=ethnicity_prob_list, k=1)[0]
        religion_category = random.choices(range(religion_start_idx, marital_start_idx), weights=religion_prob_list, k=1)[0]
        marital_category = random.choices(range(marital_start_idx, marital_start_idx + len(marital_categories)), weights=marital_prob_list, k=1)[0]
        
        # Append edges for each category
        edge_index.append([i, age_category])
        edge_index.append([i, sex_category])
        edge_index.append([i, ethnicity_category])
        edge_index.append([i, religion_category])
        edge_index.append([i, marital_category])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    return edge_index

# Generate edge index using the new function
edge_index = generate_edge_index(num_persons)

# Create the data object for PyTorch Geometric
data = Data(x=node_features, edge_index=edge_index).to(device)

# Get target tensors
targets = []
targets.append(
    (
        ('sex', 'age', 'ethnicity'), 
        get_target_tensors(ethnic_by_sex_by_age_df, sex_categories, sex_map, age_groups, age_map, ethnicity_categories, ethnicity_map)
    )
)
targets.append(
    (
        ('sex', 'age', 'marital'), 
        get_target_tensors(marital_by_sex_by_age_df, sex_categories, sex_map, age_groups, age_map, marital_categories, marital_map)
    )
)
targets.append(
    (
        ('sex', 'age', 'religion'), 
        get_target_tensors(religion_by_sex_by_age_df, sex_categories, sex_map, age_groups, age_map, religion_categories, religion_map)
    )
)

# Define the enhanced GNN model using GraphSAGE layers
class EnhancedGNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels_age, out_channels_sex, out_channels_ethnicity, out_channels_religion, out_channels_marital):
        super(EnhancedGNNModel, self).__init__()
        # Define the GraphSAGE layers for the model
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.batch_norm1 = GraphNorm(hidden_channels)
        self.batch_norm2 = GraphNorm(hidden_channels)
        self.batch_norm3 = GraphNorm(hidden_channels)
        self.batch_norm4 = GraphNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(0.5)
        
        # Add an MLP (two fully connected layers with non-linearity) for each output
        mlp_hidden_size = 256  # Set this based on your preference
        self.mlp_age = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_size, out_channels_age)
        )
        self.mlp_sex = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_size, out_channels_sex)
        )
        self.mlp_ethnicity = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_size, out_channels_ethnicity)
        )
        self.mlp_religion = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_size, out_channels_religion)
        )
        self.mlp_marital = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_size, out_channels_marital)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = self.batch_norm4(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        # Pass through MLPs for each prediction task
        age_out = self.mlp_age(x[:num_persons])
        sex_out = self.mlp_sex(x[:num_persons])
        ethnicity_out = self.mlp_ethnicity(x[:num_persons])
        religion_out = self.mlp_religion(x[:num_persons])
        marital_out = self.mlp_marital(x[:num_persons])
        
        return age_out, sex_out, ethnicity_out, religion_out, marital_out

# Custom loss function
def custom_loss_function(first_out, second_out, third_out, y_first, y_second, y_third):
    first_pred = first_out.argmax(dim=1)
    second_pred = second_out.argmax(dim=1)
    third_pred = third_out.argmax(dim=1)
    loss_first = F.cross_entropy(first_out, y_first) /3
    loss_second = F.cross_entropy(second_out, y_second)/3
    loss_third = F.cross_entropy(third_out, y_third)
    total_loss = loss_first + loss_second + loss_third
    return total_loss


# # Custom KL divergence loss function with log_softmax
# def custom_kl_loss_function(first_out, second_out, third_out, y_first, y_second, y_third):
#     eps = 1e-9  # Small constant to avoid log(0)
#     first_log_prob = F.log_softmax(first_out + eps, dim=1)
#     second_log_prob = F.log_softmax(second_out + eps, dim=1)
#     third_log_prob = F.log_softmax(third_out + eps, dim=1)
    
#     # Normalize the targets with one-hot encoding
#     y_first_onehot = F.one_hot(y_first, num_classes=first_out.size(1)).float()
#     y_second_onehot = F.one_hot(y_second, num_classes=second_out.size(1)).float()
#     y_third_onehot = F.one_hot(y_third, num_classes=third_out.size(1)).float()
    
#     # Compute KL divergence loss for each prediction task
#     loss_first = F.kl_div(first_log_prob, y_first_onehot, reduction='batchmean') /3 
#     loss_second = F.kl_div(second_log_prob, y_second_onehot, reduction='batchmean') /3 
#     loss_third = F.kl_div(third_log_prob, y_third_onehot, reduction='batchmean')
    
#     # Total loss is the sum of the individual losses
#     total_loss = loss_first + loss_second + loss_third
#     return total_loss


# Initialize model, optimizer, and loss functions
model = EnhancedGNNModel(in_channels=node_features.size(1), hidden_channels=256, out_channels_age=21, out_channels_sex=2, out_channels_ethnicity=5, out_channels_religion=9, out_channels_marital=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
# weight_decay=5e-4

# Save numbers
loss_data = {}
acc_data = {}
time_data = {}

start = time.time()
# Training loop
num_epochs = 3000
freq = 100
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear gradients

    # Forward pass
    age_out, sex_out, ethnicity_out, religion_out, marital_out = model(data)

    out = {}
    out['age'] = age_out[:num_persons]  # Only take person nodes' outputs
    out['sex'] = sex_out[:num_persons]
    out['ethnicity'] = ethnicity_out[:num_persons]
    out['religion'] = religion_out[:num_persons]
    out['marital'] = marital_out[:num_persons]

    
    loss = 0
    for i in range(len(targets)):
        loss += custom_loss_function(
            out[targets[i][0][0]], out[targets[i][0][1]], out[targets[i][0][2]],
            targets[i][1][0], targets[i][1][1], targets[i][1][2]
        )

    # Backward pass and optimization
    loss.backward()
    # Clip gradients (use a threshold, e.g., 1.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    time_data[epoch] = time.time()
    loss_data[epoch] = loss.item()
    if epoch % freq == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    acc_data[epoch] = {}
    with torch.no_grad():
        for i in range(len(targets)):
            acc_data[epoch][f'{targets[i][0][0]}, {targets[i][0][1]}, {targets[i][0][2]}'] = {}
            if epoch % freq == 0:
                print(f'Accuracy for {targets[i][0][0]}, {targets[i][0][1]}, {targets[i][0][2]}:')

            for j in range(3):
                pred = out[targets[i][0][j]].argmax(dim=1)
                accuracy = (pred == targets[i][1][j]).sum().item() / num_persons

                acc_data[epoch][f'{targets[i][0][0]}, {targets[i][0][1]}, {targets[i][0][2]}'][f'{targets[i][0][j]}'] = accuracy
                if epoch % freq == 0:
                    print(f'    Accuracy for {targets[i][0][j]}: {accuracy:.4f}')
            
            pred_1 = out[targets[i][0][0]].argmax(dim=1)
            pred_2 = out[targets[i][0][1]].argmax(dim=1)
            pred_3 = out[targets[i][0][2]].argmax(dim=1)
            net_accuracy = ((pred_1 == targets[i][1][0]) & (pred_2 == targets[i][1][1]) & (pred_3 == targets[i][1][2])).sum().item() / num_persons
            
            acc_data[epoch][f'{targets[i][0][0]}, {targets[i][0][1]}, {targets[i][0][2]}']['net'] = net_accuracy
            if epoch % freq == 0:
                print(f'    Net accuracy: {net_accuracy:.4f}')
        if epoch % freq == 0:
            print('-------------------------------------')

end = time.time()

print("time:", end-start)
# save loss and accuracy data as json
loss_data_path = os.path.join(current_dir, '../results/individual/loss_data.json')
acc_data_path = os.path.join(current_dir, '../results/individual/acc_data.json')
time_data_path = os.path.join(current_dir, '../results/individual/time_data.json')

with open(loss_data_path, 'w') as f:
    json.dump(loss_data, f)

with open(acc_data_path, 'w') as f:
    json.dump(acc_data, f)

with open(time_data_path, 'w') as f:
    json.dump(time_data, f)

with torch.no_grad():
    age_out, sex_out, ethnicity_out, religion_out, marital_out = model(data)
    age_pred = age_out[:num_persons].argmax(dim=1)
    sex_pred = sex_out[:num_persons].argmax(dim=1)
    ethnicity_pred = ethnicity_out[:num_persons].argmax(dim=1)
    religion_pred = religion_out[:num_persons].argmax(dim=1)
    marital_pred = marital_out[:num_persons].argmax(dim=1)

print("YOOO")
#------------------------------------------------------------
#------------------------------------------------------------
out = {}
out['age'] = age_out[:num_persons]  # Only take person nodes' outputs
out['sex'] = sex_out[:num_persons]
out['ethnicity'] = ethnicity_out[:num_persons]
out['religion'] = religion_out[:num_persons]
out['marital'] = marital_out[:num_persons]

with torch.no_grad():
    for i in range(len(targets)):
        print(f'Accuracy for {targets[i][0][0]}, {targets[i][0][1]}, {targets[i][0][2]}:')

        for j in range(3):
            pred = out[targets[i][0][j]].argmax(dim=1)
            accuracy = (pred == targets[i][1][j]).sum().item() / num_persons
            print(f'    Accuracy for {targets[i][0][j]}: {accuracy:.4f}')
        
        pred_1 = out[targets[i][0][0]].argmax(dim=1)
        pred_2 = out[targets[i][0][1]].argmax(dim=1)
        pred_3 = out[targets[i][0][2]].argmax(dim=1)
        net_accuracy = ((pred_1 == targets[i][1][0]) & (pred_2 == targets[i][1][1]) & (pred_3 == targets[i][1][2])).sum().item() / num_persons
        
        print(f'    Net accuracy: {net_accuracy:.4f}')

    print('-------------------------------------')
#------------------------------------------------------------
#------------------------------------------------------------

# Calculate observed counts
age_sex_ethnicity_counts = {}
for age in age_groups:
    for sex in sex_categories:
        for ethnicity in ethnicity_categories:
            key = f"{age}-{sex}-{ethnicity}"
            count = int(ethnic_by_sex_by_age_df[f"{sex} {age} {ethnicity}"].sum())
            age_sex_ethnicity_counts[key] = count

age_sex_religion_counts = {}
for age in age_groups:
    for sex in sex_categories:
        for religion in religion_categories:
            key = f"{age}-{sex}-{religion}"
            count = int(religion_by_sex_by_age_df[f"{sex} {age} {religion}"].sum())
            age_sex_religion_counts[key] = count


age_sex_marital_counts = {}
for age in age_groups:
    for sex in sex_categories:
        for marital in marital_categories:
            key = f"{age}-{sex}-{marital}"
            count = int(marital_by_sex_by_age_df[f"{sex} {age} {marital}"].sum())
            age_sex_marital_counts[key] = count

# Calculate predicted counts
predicted_counts = {}
for age in age_groups:
    for sex in sex_categories:
        for ethnicity in ethnicity_categories:
            for religion in religion_categories:
                for marital in marital_categories:
                    key = f"{age}-{sex}-{ethnicity}-{religion}-{marital}"
                    predicted_counts[key] = 0

for i in range(num_persons):
    age = age_groups[age_pred[i]]
    sex = sex_categories[sex_pred[i]]
    ethnicity = ethnicity_categories[ethnicity_pred[i]]
    religion = religion_categories[religion_pred[i]]
    marital = marital_categories[marital_pred[i]]
    key = f"{age}-{sex}-{ethnicity}-{religion}-{marital}"
    predicted_counts[key] += 1

counts_data = {}

counts_data['age_sex_ethnicity_counts'] = age_sex_ethnicity_counts
counts_data['age_sex_religion_counts'] = age_sex_religion_counts
counts_data['age_sex_marital_counts'] = age_sex_marital_counts
counts_data['age_sex_ethnicity_religion_marital_counts'] = predicted_counts

# Save the comparison DataFrame to a json file
output_path = os.path.join(current_dir, '../results/individual/counts.json')
with open(output_path, 'w') as f:
    json.dump(counts_data, f)

# Define the file path where you want to save the tensor
file_path = current_dir + "/person_nodes.pt"

# The predictions are logits; convert them to categorical labels
age_pred = age_out.argmax(dim=1)
sex_pred = sex_out.argmax(dim=1)
ethnicity_pred = ethnicity_out.argmax(dim=1)
religion_pred = religion_out.argmax(dim=1)
marital_pred = marital_out.argmax(dim=1)

# Combine individual features into a single tensor
individual_nodes = torch.cat([
    age_pred.unsqueeze(1), 
    sex_pred.unsqueeze(1), 
    ethnicity_pred.unsqueeze(1), 
    religion_pred.unsqueeze(1),
    marital_pred.unsqueeze(1)
], dim=1).float()

# Save the tensor to the file
torch.save(individual_nodes, file_path)

print(f"Person nodes tensor saved to {file_path}")
