import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
from torch.nn import CrossEntropyLoss
import random

# Set display options permanently
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Set print options to display all elements of the tensor
torch.set_printoptions(edgeitems=torch.inf)

# Device selection: Use MPS (Metal Performance Shaders) for Mac M1 GPU, or fallback to CPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'

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
age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Age_5yrs.csv'))
sex_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Sex.csv'))
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Ethnic.csv'))
religion_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Religion.csv'))
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
ethnic_by_sex_by_age_df = ethnic_by_sex_by_age_df[ethnic_by_sex_by_age_df['geography code'].isin(oxford_areas)]
religion_by_sex_by_age_df = religion_by_sex_by_age_df[religion_by_sex_by_age_df['geography code'].isin(oxford_areas)]
marital_by_sex_by_age_df = marital_by_sex_by_age_df[marital_by_sex_by_age_df['geography code'].isin(oxford_areas)]

# ages = []
# for name in marital_by_sex_by_age_df.columns.to_list():
#     if name != 'geography code' and name != 'total':
#         x = name.split()
#         age = x[1]
#         ages.append(age)

# print(set(ages))

# Define the new columns for age by summing the appropriate age columns
child_ages = ['0_4', '5_7', '8_9', '10_14', '15', '16_17']
adult_ages = ['18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59']
elderly_ages = ['60_64', '65_69', '70_74', '75_79', '80_84', '85+']
age_df['child'] = age_df[child_ages].sum(axis=1)
age_df['adult'] = age_df[adult_ages].sum(axis=1)
age_df['elderly'] = age_df[elderly_ages].sum(axis=1)

# Drop the original age columns if no longer needed
age_df = age_df.drop(columns=['0_4', '5_7', '8_9', '10_14', '15', '16_17',
                              '18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59',
                              '60_64', '65_69', '70_74', '75_79', '80_84', '85+'])

# Define the age groups, sex categories, and ethnicity categories
# age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
age_groups = ['child', 'adult', 'elderly']
sex_categories = ['M', 'F']
ethnicity_categories = ['W0', 'M0', 'A0', 'B0', 'O0']
religion_categories = ['C','B','H','J','M','S','OR','NR','NS']
marital_categories = ['Single','Married','Partner','Separated','Divorced','Widowed']

def aggregate_age_groups(df, child_ages, adult_ages, elderly_ages, sex_categories, attribute_categories):
    df_res = pd.DataFrame()
    for sex in sex_categories:
        for attribute in attribute_categories:
            # prefix = f'{sex} '
            child_cols = [col for col in df.columns if any(col == f'{sex} {age} {attribute}' for age in child_ages)]
            adult_cols = [col for col in df.columns if any(col == f'{sex} {age} {attribute}' for age in adult_ages)]
            elderly_cols = [col for col in df.columns if any(col == f'{sex} {age} {attribute}' for age in elderly_ages)]
            df_res[f'{sex} child {attribute}'] = df[child_cols].sum(axis=1)
            df_res[f'{sex} adult {attribute}'] = df[adult_cols].sum(axis=1)
            df_res[f'{sex} elderly {attribute}'] = df[elderly_cols].sum(axis=1)
            
    return df_res

ethnic_by_sex_by_age_df = aggregate_age_groups(ethnic_by_sex_by_age_df, child_ages, adult_ages, elderly_ages, sex_categories, ethnicity_categories)
religion_by_sex_by_age_df = aggregate_age_groups(religion_by_sex_by_age_df, child_ages, adult_ages, elderly_ages, sex_categories, religion_categories)
marital_by_sex_by_age_df = aggregate_age_groups(marital_by_sex_by_age_df, child_ages, adult_ages, elderly_ages, sex_categories, marital_categories)

print(ethnic_by_sex_by_age_df.sum(axis=1))
print(religion_by_sex_by_age_df.sum(axis=1))
print(marital_by_sex_by_age_df.sum(axis=1))


print(ethnic_by_sex_by_age_df)


# print("Ethnicity cross table sum: ", sum([115,0,0,1,0,77,0,0,0,0,65,1,0,0,0,143,0,0,0,0,23,1,0,0,0,47,1,0,0,0,40,0,0,0,0,97,0,0,0,0,87,0,0,0,0,70,1,0,0,0,117,1,1,0,0,199,0,0,0,0,232,0,0,0,0,220,0,1,0,0,199,0,0,0,0,234,1,0,0,0,183,0,0,0,0,130,0,0,0,0,115,0,0,0,0,70,0,0,0,0,46,0,0,0,0,99,0,0,0,0,79,0,0,0,1,55,0,0,0,0,135,1,0,1,0,23,0,0,0,0,46,0,0,0,0,31,1,0,0,0,74,0,0,0,1,75,0,0,0,0,98,0,0,0,0,142,0,0,2,0,198,1,0,0,0,227,0,1,2,0,233,0,0,0,0,200,0,0,0,0,230,1,0,0,0,192,0,0,0,0,126,1,0,0,0,108,0,0,0,0,87,1,0,0,0,113,0,1,0,0]))
# print("Religion cross table sum: ", sum([88,0,0,0,0,0,0,21,17,64,0,0,2,0,0,0,16,3,36,0,0,0,0,0,0,26,4,102,0,0,1,0,0,0,32,14,18,0,0,0,0,0,0,6,1,29,0,0,0,0,0,0,17,6,22,0,0,0,0,0,0,14,4,58,1,0,0,0,0,1,35,7,41,1,0,0,0,0,0,42,7,48,0,0,0,1,0,0,24,8,76,3,0,0,1,0,0,41,7,111,0,0,1,0,0,1,73,18,153,0,0,0,0,0,2,72,13,156,2,0,1,1,0,0,56,16,135,0,0,0,0,0,3,48,17,160,1,0,1,0,0,0,47,34,143,0,1,0,0,0,1,29,15,103,0,0,1,0,0,0,25,6,100,0,0,0,0,0,1,14,6,60,0,0,0,0,0,0,4,7,39,0,0,0,0,0,0,4,5,59,0,0,0,1,0,0,23,20,61,0,1,0,1,0,0,13,7,40,0,0,1,0,0,0,13,3,89,0,0,0,1,0,0,32,15,17,0,0,0,0,0,0,6,2,30,0,0,0,0,0,0,11,5,23,0,0,0,1,0,0,7,2,48,0,0,0,1,0,0,24,7,48,0,0,0,0,0,0,26,7,70,1,0,1,0,0,0,30,5,105,4,0,2,0,0,1,37,10,148,4,1,1,1,0,0,44,10,173,1,0,1,0,0,1,47,20,182,1,0,1,0,0,1,40,20,166,1,1,0,0,0,2,29,13,181,1,0,1,0,0,1,31,24,162,0,0,0,0,0,0,17,15,111,0,0,0,0,0,0,13,7,94,0,0,0,0,0,0,9,7,84,0,0,0,0,0,0,3,6,104,0,0,0,0,0,0,3,9]))
# print("Marital cross table sum: ", sum([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,52,0,0,0,0,0,46,0,0,0,0,0,40,0,0,0,0,0,33,0,0,0,0,0,101,0,0,0,1,0,77,3,0,0,0,0,77,13,1,0,0,0,65,16,0,0,0,0,50,28,1,1,0,1,44,58,0,1,4,0,39,79,3,1,5,1,31,113,0,9,6,0,38,140,1,6,19,0,27,145,1,10,21,5,30,171,2,9,28,0,25,175,3,4,32,4,23,174,1,4,27,3,14,178,2,8,38,5,17,155,1,5,23,2,7,163,0,3,27,12,9,190,2,5,33,4,11,190,0,5,21,12,16,146,0,2,15,10,11,138,0,6,16,23,6,114,0,0,12,3,1,88,0,1,8,33,5,93,2,1,5,15,0,64,0,0,8,38,2,53,1,1,1,13,4,25,0,0,3,61,1,25,1,0,1,20,4,24,0,0,1,87]))

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

# New function to generate edge index
def generate_edge_index(num_persons):
    edge_index = []
    age_start_idx = num_persons
    sex_start_idx = age_start_idx + len(age_groups)
    ethnicity_start_idx = sex_start_idx + len(sex_categories)
    religion_start_idx = ethnicity_start_idx + len(ethnicity_categories)
    marital_start_idx = religion_start_idx + len(religion_categories)

    for i in range(num_persons):
        age_category = random.choice(range(age_start_idx, sex_start_idx))
        sex_category = random.choice(range(sex_start_idx, ethnicity_start_idx))
        ethnicity_category = random.choice(range(ethnicity_start_idx, ethnicity_start_idx + len(ethnicity_categories)))
        religion_category = random.choice(range(religion_start_idx, religion_start_idx + len(religion_categories)))
        marital_category = random.choice(range(marital_start_idx, marital_start_idx + len(marital_categories)))
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

# print(targets[0][1][1].tolist().count(0))
# print(targets[0][1][1].tolist().count(1))
# print(targets[0][1][1].tolist().count(2))
# print()
# print(targets[1][1][1].tolist().count(0))
# print(targets[1][1][1].tolist().count(1))
# print(targets[1][1][1].tolist().count(2))
# print()
# print(targets[2][1][1].tolist().count(0))
# print(targets[2][1][1].tolist().count(1))
# print(targets[2][1][1].tolist().count(2))
# print()

# Define the enhanced GNN model using GraphSAGE layers
class EnhancedGNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels_age, out_channels_sex, out_channels_ethnicity, out_channels_religion, out_channels_marital):
        super(EnhancedGNNModel, self).__init__()
        # Define the GraphSAGE layers for the model
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.batch_norm2 = BatchNorm(hidden_channels)
        self.batch_norm3 = BatchNorm(hidden_channels)
        self.batch_norm4 = BatchNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(0.5)
        self.conv5_age = SAGEConv(hidden_channels, out_channels_age)
        self.conv5_sex = SAGEConv(hidden_channels, out_channels_sex)
        self.conv5_ethnicity = SAGEConv(hidden_channels, out_channels_ethnicity)
        self.conv5_religion = SAGEConv(hidden_channels, out_channels_religion)
        self.conv5_marital = SAGEConv(hidden_channels, out_channels_marital)

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
        x = self.conv4(x, edge_index)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.dropout(x)
        age_out = self.conv5_age(x, edge_index)
        sex_out = self.conv5_sex(x, edge_index)
        ethnicity_out = self.conv5_ethnicity(x, edge_index)
        religion_out = self.conv5_religion(x, edge_index)
        marital_out = self.conv5_marital(x, edge_index)
        return age_out, sex_out, ethnicity_out, religion_out, marital_out

# Custom loss function
def custom_loss_function(first_out, second_out, third_out, y_first, y_second, y_third):
    first_pred = first_out.argmax(dim=1)
    second_pred = second_out.argmax(dim=1)
    third_pred = third_out.argmax(dim=1)
    loss_first = F.cross_entropy(first_out, y_first)
    loss_second = F.cross_entropy(second_out, y_second)
    loss_third = F.cross_entropy(third_out, y_third)
    total_loss = loss_first + loss_second + loss_third
    return total_loss

# Initialize model, optimizer, and loss functions
model = EnhancedGNNModel(in_channels=node_features.size(1), hidden_channels=512, out_channels_age=21, out_channels_sex=2, out_channels_ethnicity=5, out_channels_religion=9, out_channels_marital=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# weight_decay=5e-4


# Training loop
num_epochs = 3000
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
    optimizer.step()

    # Print metrics every 100 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

        with torch.no_grad():
            for i in range(len(targets)):
                print(f'Accuracy for {targets[i][0][0]}, {targets[i][0][1]}, {targets[i][0][2]}:')

                for j in range(3):
                    pred = out[targets[i][0][j]].argmax(dim=1)
                    accuracy = (pred == targets[i][1][j]).sum().item() / num_persons
                    print(f'    Accuracy for {targets[i][0][j]}: {accuracy:.4f}')
                    # if j ==1:
                    #     print(pred.tolist().count(0),  targets[i][1][j].tolist().count(0))
                    #     print()
                    #     print(pred.tolist().count(1),  targets[i][1][j].tolist().count(1))
                    #     print()
                    #     print(pred.tolist().count(2),  targets[i][1][j].tolist().count(2))
                    #     print()
                
                pred_1 = out[targets[i][0][0]].argmax(dim=1)
                pred_2 = out[targets[i][0][1]].argmax(dim=1)
                pred_3 = out[targets[i][0][2]].argmax(dim=1)
                net_accuracy = ((pred_1 == targets[i][1][0]) & (pred_2 == targets[i][1][1]) & (pred_3 == targets[i][1][2])).sum().item() / num_persons
                print(f'    Net accuracy: {net_accuracy:.4f}')
            print('-------------------------------------')

# Get the final predictions after training
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    age_out, sex_out, ethnicity_out, religion_out, marital_out = model(data)
    age_pred = age_out[:num_persons].argmax(dim=1)
    sex_pred = sex_out[:num_persons].argmax(dim=1)
    ethnicity_pred = ethnicity_out[:num_persons].argmax(dim=1)
    religion_pred = religion_out[:num_persons].argmax(dim=1)
    marital_pred = marital_out[:num_persons].argmax(dim=1)

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
for i in range(num_persons):
    age = age_groups[age_pred[i]]
    sex = sex_categories[sex_pred[i]]
    ethnicity = ethnicity_categories[ethnicity_pred[i]]
    religion = religion_categories[religion_pred[i]]
    marital = marital_categories[marital_pred[i]]
    key = f"{age}-{sex}-{ethnicity}-{religion}-{marital}"
    if key in predicted_counts:
        predicted_counts[key] += 1
    else:
        predicted_counts[key] = 1

# Create a DataFrame for comparison
# print(age_sex_ethnicity_counts.keys())
# print(age_sex_religion_counts.keys())
# print(age_sex_marital_counts.keys())
# print(predicted_counts.keys())


# comparison_df = pd.DataFrame({
#     "age_sex_ethnicity_counts": list(age_sex_ethnicity_counts.keys()),
#     "age_sex_religion_counts": list(age_sex_religion_counts.keys()),
#     "age_sex_marital_counts": list(age_sex_marital_counts.keys()),
#     "age_sex_ethnicity_religion_marital_counts": list(predicted_counts.keys()),
# })

# # Save the comparison DataFrame to a CSV file
# output_path = os.path.join(current_dir, 'comparison_results.csv')
# comparison_df.to_csv(output_path, index=False)

# # Print the comparison DataFrame
# print("Comparison results saved to:", output_path)
# print(comparison_df)


# After training the individual GNN model
with torch.no_grad():
    age_out, sex_out, ethnicity_out, religion_out, marital_out = model(data)

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

# Define the file path where you want to save the tensor
file_path = current_dir + "/person_nodes.pt"

# Save the tensor to the file
torch.save(individual_nodes, file_path)

print(f"Person nodes tensor saved to {file_path}")
