import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import os
import random
import pandas as pd 

# Set print options to display all elements of the tensor
torch.set_printoptions(edgeitems=torch.inf)

# Step 1: Load the tensors and household size data
current_dir = os.path.dirname(os.path.abspath(__file__))
persons_file_path = os.path.join(current_dir, "person_nodes.pt")
households_file_path = os.path.join(current_dir, "household_nodes.pt")
hh_size_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/HH_size.csv'))

# Define the Oxford areas
oxford_areas = ['E02005924']
hh_size_df = hh_size_df[hh_size_df['geography code'].isin(oxford_areas)]

# Load the tensors from the files
person_nodes = torch.load(persons_file_path)  # Example size: (num_persons x 5)
household_nodes = torch.load(households_file_path)  # Example size: (num_households x 3)

# Define the household composition categories and mapping
hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-2C', '1FM-nA','1FC-0C','1FC-2C','1FC-nA','1FL-nA','1FL-2C','1H-nS','1H-nE','1H-nA', '1H-2C']
hh_map = {category: i for i, category in enumerate(hh_compositions)}
reverse_hh_map = {v: k for k, v in hh_map.items()}  # Reverse mapping to decode

# Extract the household composition predictions
hh_pred = household_nodes[:, 0].long() 

# Flattening size and weight lists
values_size_org = [k for k in hh_size_df.columns if k not in ['geography code', 'total']]
weights_size_org = hh_size_df.iloc[0, 2:].tolist()  # Assuming first row, and skipping the first two columns

household_size_dist = {k: v for k, v in zip(hh_size_df.columns[2:], hh_size_df.iloc[0, 2:]) if k != '1'}
values_size, weights_size = zip(*household_size_dist.items())

household_size_dist_na = {k: v for k, v in zip(hh_size_df.columns[2:], hh_size_df.iloc[0, 2:]) if k not in ['1', '2']}
values_size_na, weights_size_na = zip(*household_size_dist_na.items())

# Define the size assignment function based on household composition
fixed_hh = {"1PE": 1, "1PA": 1, "1FM-0C": 2, "1FC-0C": 2}
three_or_more_hh = {'1FM-2C', '1FM-nA', '1FC-2C', '1FC-nA'}
two_or_more_hh = {'1FL-2C', '1FL-nA', '1H-2C'}

def fit_household_size(composition):
    if composition in fixed_hh:
        return fixed_hh[composition]
    elif composition in three_or_more_hh:
        return int(random.choices(values_size_na, weights=weights_size_na)[0].replace('8+', '8'))
    elif composition in two_or_more_hh:
        return int(random.choices(values_size, weights=weights_size)[0].replace('8+', '8'))
    else:
        return int(random.choices(values_size_org, weights=weights_size_org)[0].replace('8+', '8'))

# Assign sizes to each household based on its composition
household_sizes = torch.tensor([fit_household_size(reverse_hh_map[hh_pred[i].item()]) for i in range(len(hh_pred))], dtype=torch.long)
print("Done assigning household sizes")

# Step 2: Define the GNN model
class HouseholdAssignmentGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_households):
        super(HouseholdAssignmentGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_households)

    def forward(self, x, edge_index):
        # GCN layers to process person nodes
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        # Fully connected layer to output logits for each household
        out = self.fc(x)
        return out  # Output shape: (num_persons, num_households)

# Define Gumbel-Softmax
def gumbel_softmax(logits, tau=1.0, hard=False):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = logits + gumbel_noise
    y = F.softmax(y / tau, dim=-1)

    if hard:
        # Straight-through trick: take the index of the max value, but keep the gradient.
        y_hard = torch.zeros_like(logits).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y

# Step 3: Create the graph
num_persons = person_nodes.size(0)
print(num_persons)
num_households = household_sizes.size(0)
print(num_households)

# Define the columns for religion and ethnicity 
religion_col_persons, religion_col_households = 2, 2
ethnicity_col_persons, ethnicity_col_households = 3, 1

# Check if the edge_index already exists
edge_index_file_path = os.path.join(current_dir, "edge_index.pt")
if os.path.exists(edge_index_file_path):
    # Load the saved edge index
    edge_index = torch.load(edge_index_file_path)
    print(f"Loaded edge index from {edge_index_file_path}")
else:
    # Step 3: Create the graph with edges only between people with the same religion and ethnicity
    edge_index = [[], []]  # Placeholder for edges
    cnt = 0 
    for i in range(num_persons):
        if i % 10 == 0:
            print(i)
        for j in range(i + 1, num_persons):  # Avoid duplicate edges by starting at i + 1
            # Check if both persons have the same religion and ethnicity
            if person_nodes[i, religion_col_persons] == person_nodes[j, religion_col_persons] and person_nodes[i, ethnicity_col_persons] == person_nodes[j, ethnicity_col_persons]:
                edge_index[0].append(i)
                edge_index[1].append(j)
                # Since it's an undirected graph, add both directions
                edge_index[0].append(j)
                edge_index[1].append(i)
                cnt += 1
    print(f"Generated {cnt} edges")

    # Convert the edge index list into a tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Save the edge_index to a file for future use
    torch.save(edge_index, edge_index_file_path)
    print(f"Edge index saved to {edge_index_file_path}")

# Step 4: Initialize the GNN model
in_channels = person_nodes.size(1)  # Assuming 5 characteristics per person
hidden_channels = 32
model = HouseholdAssignmentGNN(in_channels, hidden_channels, num_households)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Step 5: Define the compute_loss function
def compute_loss(assignments, household_sizes, person_nodes, household_nodes, penalty_weight=1.0):
    # Calculate household size mismatch loss (MSE)
    household_counts = assignments.sum(dim=0)  # Sum the soft assignments across households
    size_loss = F.mse_loss(household_counts.float(), household_sizes.float())  # MSE loss

    # Calculate penalty for mismatches in ethnicity and religion
    mismatch_penalty = 0.0
    for person_idx, household_idx in enumerate(torch.argmax(assignments, dim=1)):
        person_religion = person_nodes[person_idx, religion_col_persons]
        person_ethnicity = person_nodes[person_idx, ethnicity_col_persons]
        household_religion = household_nodes[household_idx, religion_col_households]
        household_ethnicity = household_nodes[household_idx, ethnicity_col_households]
        
        # Penalize if religion or ethnicity doesn't match
        if person_religion != household_religion or person_ethnicity != household_ethnicity:
            mismatch_penalty += 1.0
    
    # Scale the penalty by the penalty weight
    total_loss = size_loss + penalty_weight * mismatch_penalty
    return total_loss

# Step 6: Training loop
epochs = 100
tau = 0.05
penalty_weight = 0.1  # Weight of the mismatch penalty in the loss function

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(person_nodes, edge_index)  # Shape: (num_persons, num_households)
    
    # Apply Gumbel-Softmax to get differentiable assignments
    assignments = gumbel_softmax(logits, tau=tau, hard=False)  # Shape: (num_persons, num_households)
    
    # Calculate the loss using the household sizes as targets
    loss = compute_loss(assignments, household_sizes, person_nodes, household_nodes, penalty_weight)
    loss.backward()
    optimizer.step()
    
    # Print the loss for each epoch
    print(f'Epoch {epoch}, Loss: {loss.item()}')

# Step 7: Final assignments after training
final_assignments = torch.argmax(assignments, dim=1)  # Get final discrete assignments
# print(final_assignments)

def calculate_individual_compliance_accuracy(assignments, person_nodes, household_nodes):
    # Define the columns for religion and ethnicity in persons and households
    religion_col_persons, religion_col_households = 2, 2
    ethnicity_col_persons, ethnicity_col_households = 3, 1

    total_people = assignments.size(0)
    
    correct_religion_assignments = 0
    correct_ethnicity_assignments = 0

    # Loop over each person and their assigned household
    for person_idx, household_idx in enumerate(assignments):
        household_idx = household_idx.item()  # Get the household assignment for the person

        # Get the person's religion and ethnicity
        person_religion = person_nodes[person_idx, religion_col_persons]
        person_ethnicity = person_nodes[person_idx, ethnicity_col_persons]

        # Get the household's religion and ethnicity
        household_religion = household_nodes[household_idx, religion_col_households]
        household_ethnicity = household_nodes[household_idx, ethnicity_col_households]

        # Check if the person's religion matches the household's religion
        if person_religion == household_religion:
            correct_religion_assignments += 1

        # Check if the person's ethnicity matches the household's ethnicity
        if person_ethnicity == household_ethnicity:
            correct_ethnicity_assignments += 1

    # Calculate individual compliance accuracy for religion and ethnicity
    religion_compliance = correct_religion_assignments / total_people
    ethnicity_compliance = correct_ethnicity_assignments / total_people

    # Print the results for visual feedback
    print(f"Religion compliance accuracy (per person): {religion_compliance * 100:.2f}%")
    print(f"Ethnicity compliance accuracy (per person): {ethnicity_compliance * 100:.2f}%")

    return religion_compliance, ethnicity_compliance

# Assuming `final_assignments` contains the assignments after training
religion_compliance, ethnicity_compliance = calculate_individual_compliance_accuracy(
    final_assignments,       # The household assignments predicted by the model
    person_nodes,            # The tensor containing person features (including religion and ethnicity)
    household_nodes          # The tensor containing household features (including religion and ethnicity)
)