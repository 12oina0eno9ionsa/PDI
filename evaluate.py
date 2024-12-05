import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from eca_cnn import ECACNN
from utils import load_data
from collections import Counter

def evaluate_predictions(model, test_loader, device, class_names=['high', 'low', 'mid']):
    model.eval()
    all_preds = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for inputs, paths in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    probs_array = np.array(all_probs)
    confidence_scores = np.max(probs_array, axis=1)
    
    print("\nConfidence Statistics:")
    print(f"Mean confidence: {confidence_scores.mean():.3f}")
    print(f"Min confidence: {confidence_scores.min():.3f}")
    print(f"Max confidence: {confidence_scores.max():.3f}")

    # Distribution of predictions
    pred_dist = Counter(all_preds)
    print("\nPrediction Distribution:")
    for idx, count in pred_dist.items():
        print(f"{class_names[idx]}: {count}")

    return {
        'predictions': all_preds,
        'probabilities': all_probs,
        'paths': all_paths
    }

def evaluate_model(data_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECACNN().to(device)
    model.load_state_dict(torch.load("eca_cnn_model.pth"))
    model.eval()

    val_loader = load_data(data_dir, batch_size)

    results = evaluate_predictions(model, val_loader, device)
    return results

if __name__ == "__main__":
    data_dir = "annotatted/SAMPLES/dataset_4_12/data"
    evaluate_model(data_dir)