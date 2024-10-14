import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        return x

# Define input, hidden, and output dimensions
input_dim = 768
hidden_dim = 512
output_dim = 768

train_jsons_dir = '/path/to/train_features'
test_jsons_dir = '/path/to/test_features'

# Instantiate the model
model = MLP(input_dim, hidden_dim, output_dim).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
num_epochs, batch_size = 10, 10000
for epoch in range(num_epochs):
    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    for json_name in os.listdir(train_jsons_dir):
        # a list where each element shows one bbox, maybe multiple elements correspond to the same image
        curr_json = json.load(open(os.path.join(train_jsons_dir, json_name)))
        for idx_batch in range(0, len(curr_json), batch_size):
            x = torch.tensor([x['keypoints'] for x in curr_json[idx_batch: min([len(curr_json), idx_batch + batch_size])]]).to(device)
            y = torch.tensor([x['keypoints'] for x in curr_json[idx_batch: min([len(curr_json), idx_batch + batch_size])]]).to(device)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Loss: {loss.item():.4f}')


    # Save the model weights
    model_save_path = "reconstruction_weights.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")

    # Load the model weights
    loaded_model = MLP(input_dim, hidden_dim, output_dim).to(device)
    loaded_model.load_state_dict(torch.load(model_save_path))
    loaded_model.eval()
    print("Model weights loaded successfully.")