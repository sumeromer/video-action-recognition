import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from train import ClipsDataset  # Import your dataset class
from cnn_model import ViViT
from thop import profile  # For FLOPs and parameter profiling
from torchvision import transforms
import time

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
log_file_name = './logs/SLAM-Vivit_Cls/test_output_val_acc_08513.log'
logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(asctime)s - %(message)s')

# Helper functions
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}", ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig("./logs/SLAM-Vivit_Cls/test_wave_confusion_matrix.jpg")
    plt.show()

def calculate_gflops_and_parameters(model, input_size):
    model.eval()  # Set the model to evaluation mode
    dummy_input = torch.randn(1, *input_size[1:], device=device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    gflops = flops / 1e9
    return gflops, params

def calculate_inference_speed(model, data_loader):
    model.eval()
    total_time = 0
    num_samples = 0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            start_time = time.time()
            _ = model(x)
            total_time += time.time() - start_time
            num_samples += 1
    avg_time_per_sample = total_time / num_samples
    return avg_time_per_sample

def test(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

    # Precision, Recall, F1-Score
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

if __name__ == "__main__":
    # Test dataset and loader setup
    test_data_path = "./data/SLAM" # "/mnt/data/LaparoClipsP1"
    test_csv_file = "./data/SLAM/test.csv" # "/mnt/data/LaparoClipsP1/test.csv"

    # Define the transform (same as validation in training)
    test_transform = transforms.Compose([transforms.ToTensor()])

    # Initialize dataset and loader
    test_dataset = ClipsDataset(test_data_path, test_csv_file, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model setup
    #model_path = "/root/SLAM-Vivit_Cls/weights/best_model_wave_11Cls_768_BS4_FR16_val_acc_0.8564.pkl"
    model_path = "./logs/SLAM-Vivit_Cls/weights/best_model_768_BS2_FR16_val_acc_0.8513.pkl"
    
    class_names = [
          "UseClipper", "HookCut", "PanoView", "Suction", "AbdominalEntry", "Needle", "LocPanoView"
    ]

    height = width = 768
    img_size = height  # Since height = width
    time_size = 16
    model = ViViT(height, 16, 7, time_size) # ViViT(height, 16, 11, time_size)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Compute GFLOPs and parameters
    input_size = next(iter(test_loader))[0].shape
    gflops, params = calculate_gflops_and_parameters(model, input_size)
    logging.info(f"GFLOPs: {gflops}, Parameters: {params}")
    print(f"GFLOPs: {gflops}, Parameters: {params}")

    # Compute inference speed
    inference_speed = calculate_inference_speed(model, test_loader)
    logging.info(f"Inference Speed (s/sample): {inference_speed:.4f}")
    print(f"Inference Speed (s/sample): {inference_speed:.4f}")

    # Run testing
    test(model, test_loader, class_names)