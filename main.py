import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from dataset import TimeSeriesDataset
from model import LSTMModel

# Load your dataset
data = pd.read_excel('../docs/feat.xlsx', index_col=0).to_numpy()


# Split the data into training, validation, and test sets
train_data, test_data = train_test_split(data, test_size=0.1, shuffle=False)
train_data, val_data = train_test_split(train_data, test_size=0.1, shuffle=False)
test_data_cp = test_data.copy()
# Initialize the scaler and fit it on the training data
scaler = MinMaxScaler()
scaler.fit(train_data)

# Apply the scaler to all datasets
train_data = scaler.transform(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

# test_data_inv = scaler.inverse_transform(test_data)
# print(test_data_cp[-3:])
# print(test_data[-3:])
# print(test_data_inv[-3:])

train_dataset = TimeSeriesDataset(train_data)
val_dataset = TimeSeriesDataset(val_data)
test_dataset = TimeSeriesDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model parameters
input_dim = train_dataset.features_num
hidden_dim = 128
num_layers = 3
output_dim = 1
num_epochs = 30

# Initialize model, loss function, and optimizer
model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to evaluate the model on the validation set
def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs, labels
            outputs = model(inputs).squeeze()  # Squeeze the outputs to match the labels shape
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Training loop with validation
best_val_loss = float('inf')
best_model_path = 'best_lstm_model811.pth'
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        inputs, labels = inputs, labels
        
        outputs = model(inputs).squeeze()  # Squeeze the outputs to match the labels shape
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validate the model
    val_loss = evaluate_model(model, val_loader, criterion)
    val_losses.append(val_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}')

    # Save the model if it has the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'Saving model with validation loss: {best_val_loss:.8f}')

# Load the best model for further use
model.load_state_dict(torch.load(best_model_path))

# Plot training and validation losses
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

# Test the model and plot predictions vs. actuals
model.eval()
actuals = []
predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs, labels
        outputs = model(inputs).squeeze()  # Squeeze the outputs to match the labels shape
        actuals.append(labels.item())
        predictions.append(outputs.item())

rmse_norm = np.sqrt(mean_squared_error(actuals, predictions))
print(f'Test RMSE norm: {rmse_norm:.8f}')
# Convert lists to numpy arrays
actuals = np.array(actuals).reshape(-1, 1)
predictions = np.array(predictions).reshape(-1, 1)

# Create empty arrays with the same shape as the original data
actuals_full = np.zeros((actuals.shape[0], train_data.shape[1]))
predictions_full = np.zeros((predictions.shape[0], train_data.shape[1]))

# Insert the actual and predicted values into the last column
actuals_full[:, test_dataset.label_col] = actuals.flatten()
predictions_full[:, test_dataset.label_col] = predictions.flatten()

# Inverse transform the full arrays
actuals_inverse = scaler.inverse_transform(actuals_full)[:, test_dataset.label_col]
# actuals_inverse1 = test_data_cp[-len(actuals_inverse):,test_dataset.label_col]
predictions_inverse = scaler.inverse_transform(predictions_full)[:, test_dataset.label_col]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actuals_inverse, predictions_inverse))
error1 = actuals_inverse - predictions_inverse
# print("error1",error1[:20])
print(f'Test RMSE: {rmse:.8f}')

# Plot actual vs. predicted values
plt.figure()
plt.plot(actuals_inverse, label='Actual')
plt.plot(predictions_inverse, label='Prediction')
plt.xlabel('Time Step')
plt.ylabel('Click Count')
plt.title('Actual vs. Predicted Click Counts')
plt.legend()
plt.show()