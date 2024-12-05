import os
import torch
import torch.nn.functional as F
from eca_cnn import ECACNN
from utils import load_test_data
from evaluate import evaluate_predictions

def test_mixed(data_dir, batch_size=32):
    abs_path = os.path.abspath(data_dir)
    print(f"Absolute path: {abs_path}")
    
    if not os.path.exists(abs_path):
        print(f"Directory does not exist: {abs_path}")
        return
    
    files = os.listdir(abs_path)
    image_files = [f for f in files if f.upper().endswith('.PNG')]
    print(f"Image files found: {len(image_files)}")
    
    if not image_files:
        print("No image files found!")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ECACNN().to(device)
    model.load_state_dict(torch.load("eca_cnn_model.pth"))
    model.eval()
    
    test_loader = load_test_data(abs_path, batch_size)
    print(f"Test loader size: {len(test_loader)}")
    
    class_names = ['high', 'low', 'mid']
    
    results = evaluate_predictions(model, test_loader, device, class_names)
    
    for img_path, pred, prob in zip(image_files, results['predictions'], results['probabilities']):
        confidence = prob[pred]
        print(f"\nImage: {img_path}")
        print(f"Predicted: {class_names[pred]} (confidence: {confidence:.3f})")
        print("Class probabilities:")
        for cls, p in zip(class_names, prob):
            print(f"  {cls}: {p:.3f}")

if __name__ == "__main__":
    data_dir = "annotatted/SAMPLES/dataset_4_12/data/high"
    test_mixed(data_dir)
