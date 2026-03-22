"""
PHASE II - STEP 4: Extract Embeddings (Enhanced Version)
=========================================================
Features:
- Extracts embeddings from trained model
- Saves predictions and ground truth
- Memory-efficient chunked processing
- MPS-optimized for Mac
- Progress tracking with checkpoints
- Comprehensive metadata saving
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import json
import time
from pathlib import Path

# Import from your model
from step2_model_s import Phase2Model

# ============================================================
# CONFIG - Using paths from your training
# ============================================================
CSV_PATH = "/Users/sowmyaalamuri/Desktop/Capstone_project/Phase2/phase2_dataset_enhanced.csv"
MODEL_PATH = "/Users/sowmyaalamuri/Desktop/Capstone_project/Phase2/training_output_enhanced/best_model_s.pth"
OUTPUT_DIR = "/Users/sowmyaalamuri/Desktop/Capstone_project/Phase2/embeddings_output_enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# M1/MPS Optimized Settings
EMBED_DIM = 128
BATCH_SIZE = 64  # Reduced for stability
IMG_SIZE = 160
CHUNK_SIZE = 2000  # Process 2000 images at a time
NUM_WORKERS = 0  # Must be 0 for Mac

# ============================================================
# DEVICE
# ============================================================
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ Using Apple MPS")
        torch.set_default_dtype(torch.float32)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA")
    else:
        device = torch.device("cpu")
        print(f"⚠ Using CPU")
    return device

device = get_device()

# ============================================================
# DATASET (Memory Efficient - No Caching)
# ============================================================
class EmbeddingDataset(Dataset):
    """Dataset for embedding extraction - loads images on-the-fly"""
    
    def __init__(self, df, start_idx=0, end_idx=None):
        # Only include tumor slices for embedding extraction
        df = df[df["tumor_present"] == 1].reset_index(drop=True)
        
        if end_idx is None:
            end_idx = len(df)
        
        self.df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"  Dataset slice: {len(self.df)} images (indices {start_idx}-{end_idx})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image on-the-fly
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load {row['image_path']}, using black image")
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        
        img_tensor = self.transform(img)
        
        return {
            "image": img_tensor,
            "idx": idx,
            "image_path": row["image_path"],
            "retrieval_label": int(row.get("retrieval_label", 0)),
            "grade_label": int(row.get("grade_label", 0)),
            "severity_label": int(row.get("severity_bin", row.get("severity_label", 0))),
            "size_label": int(row.get("size_label", 0)),
            "location_label": int(row.get("location_label", 0)),
            "tumor_label": int(row["tumor_present"]),
            "sample_weight": float(row.get("sample_weight", 1.0)),
            "clinical_relevance": float(row.get("clinical_relevance", 0.5)),
        }


# ============================================================
# MODEL LOADING
# ============================================================
def load_model(model_path, device):
    """Load trained model"""
    print(f"\n📦 Loading model from: {os.path.basename(model_path)}")
    
    model = Phase2Model(embedding_dim=EMBED_DIM, pretrained=False).to(device)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"  ✓ Loaded checkpoint from epoch {epoch}")
            else:
                model.load_state_dict(checkpoint)
                print("  ✓ Model loaded successfully!")
        else:
            model.load_state_dict(checkpoint)
            print("  ✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        print("  Trying with strict=False...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
            print("  ✓ Model loaded with strict=False")
        except Exception as e2:
            print(f"  ❌ Still failed: {e2}")
            return None
    
    model.eval()
    return model


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================
def extract_chunk(model, loader, device):
    """Extract embeddings for one chunk"""
    
    embeddings_list = []
    paths_list = []
    
    # Predictions and ground truth
    preds = {
        "grade": [], "severity": [], "size": [], "location": [], "tumor": []
    }
    gt = {
        "retrieval_label": [], 
        "grade_label": [], 
        "severity_label": [], 
        "size_label": [], 
        "location_label": [],
        "tumor_label": [],
        "clinical_relevance": [],
        "sample_weight": []
    }
    
    # Confidence scores
    confidences = {
        "grade": [], "severity": [], "size": [], "location": [], "tumor": []
    }

    with torch.no_grad():
        pbar = tqdm(loader, desc="    Extracting", leave=False)
        for batch in pbar:
            # Move to device
            images = batch["image"].to(device, dtype=torch.float32)
            
            # Forward pass
            outputs = model(images)
            
            # Store embeddings
            embeddings_list.append(outputs["embedding"].cpu().numpy())
            
            # Store predictions with confidence
            for task in ["grade", "severity", "size", "location", "tumor"]:
                logits = outputs[f"{task}_logits"]
                probs = torch.softmax(logits, dim=1)
                preds[task].extend(logits.argmax(1).cpu().numpy())
                confidences[task].extend(probs.max(1)[0].cpu().numpy())
            
            # Store ground truth
            for key in gt:
                if key in batch:
                    gt[key].extend(batch[key].cpu().numpy())
            
            paths_list.extend(batch["image_path"])
            
            # Update progress bar
            pbar.set_postfix({
                'batch': f"{len(embeddings_list[-1])} imgs"
            })
    
    # Stack embeddings
    if embeddings_list:
        embeddings = np.vstack(embeddings_list)
    else:
        embeddings = np.array([])
    
    return embeddings, paths_list, preds, gt, confidences


# ============================================================
# CHECKPOINT HANDLING
# ============================================================
def save_checkpoint(chunk_idx, last_processed, output_dir):
    """Save extraction checkpoint"""
    checkpoint = {
        "chunk_idx": chunk_idx,
        "last_processed": last_processed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    checkpoint_path = os.path.join(output_dir, "extraction_checkpoint.json")
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    return checkpoint_path


def load_checkpoint(output_dir):
    """Load extraction checkpoint if exists"""
    checkpoint_path = os.path.join(output_dir, "extraction_checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        return checkpoint
    return None


# ============================================================
# SAVE RESULTS
# ============================================================
def save_results(all_embeddings, all_paths, all_preds, all_gt, all_confidences, output_dir):
    """Save all extracted data"""
    
    print("\n💾 Saving results...")
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, all_embeddings)
    print(f"  ✓ Embeddings saved: {embeddings_path} ({all_embeddings.shape})")
    
    # Create metadata DataFrame
    grade_map = {0: "LGG", 1: "HGG"}
    severity_map = {0: "low", 1: "medium", 2: "high"}
    size_map = {0: "small", 1: "medium", 2: "large"}
    location_map = {0: "left", 1: "right", 2: "bilateral"}
    tumor_map = {0: "absent", 1: "present"}
    
    metadata = pd.DataFrame({
        "image_path": all_paths,
        "retrieval_label": all_gt["retrieval_label"],
        "true_grade": [grade_map[x] for x in all_gt["grade_label"]],
        "true_grade_code": all_gt["grade_label"],
        "true_severity": [severity_map[x] for x in all_gt["severity_label"]],
        "true_severity_code": all_gt["severity_label"],
        "true_size": [size_map[x] for x in all_gt["size_label"]],
        "true_size_code": all_gt["size_label"],
        "true_location": [location_map[x] for x in all_gt["location_label"]],
        "true_location_code": all_gt["location_label"],
        "true_tumor": [tumor_map[x] for x in all_gt["tumor_label"]],
        "true_tumor_code": all_gt["tumor_label"],
        "pred_grade": [grade_map[x] for x in all_preds["grade"]],
        "pred_grade_code": all_preds["grade"],
        "pred_severity": [severity_map[x] for x in all_preds["severity"]],
        "pred_severity_code": all_preds["severity"],
        "pred_size": [size_map[x] for x in all_preds["size"]],
        "pred_size_code": all_preds["size"],
        "pred_location": [location_map[x] for x in all_preds["location"]],
        "pred_location_code": all_preds["location"],
        "pred_tumor": [tumor_map[x] for x in all_preds["tumor"]],
        "pred_tumor_code": all_preds["tumor"],
        "conf_grade": all_confidences["grade"],
        "conf_severity": all_confidences["severity"],
        "conf_size": all_confidences["size"],
        "conf_location": all_confidences["location"],
        "conf_tumor": all_confidences["tumor"],
        "clinical_relevance": all_gt.get("clinical_relevance", [0.5] * len(all_paths)),
        "sample_weight": all_gt.get("sample_weight", [1.0] * len(all_paths)),
    })
    
    # Add correctness flags
    metadata["grade_correct"] = (metadata["true_grade_code"] == metadata["pred_grade_code"]).astype(int)
    metadata["severity_correct"] = (metadata["true_severity_code"] == metadata["pred_severity_code"]).astype(int)
    metadata["size_correct"] = (metadata["true_size_code"] == metadata["pred_size_code"]).astype(int)
    metadata["location_correct"] = (metadata["true_location_code"] == metadata["pred_location_code"]).astype(int)
    metadata["tumor_correct"] = (metadata["true_tumor_code"] == metadata["pred_tumor_code"]).astype(int)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata.to_csv(metadata_path, index=False)
    print(f"  ✓ Metadata saved: {metadata_path} ({len(metadata)} rows)")
    
    return metadata


# ============================================================
# COMPUTE ACCURACIES
# ============================================================
def print_accuracies(metadata):
    """Print classification accuracies"""
    
    print("\n" + "="*70)
    print("  CLASSIFICATION ACCURACIES")
    print("="*70)
    
    # Overall accuracies
    grade_acc = metadata["grade_correct"].mean()
    severity_acc = metadata["severity_correct"].mean()
    size_acc = metadata["size_correct"].mean()
    location_acc = metadata["location_correct"].mean()
    tumor_acc = metadata["tumor_correct"].mean()
    
    print(f"\n  Overall Accuracies:")
    print(f"    Grade    : {grade_acc*100:.2f}%")
    print(f"    Severity : {severity_acc*100:.2f}%")
    print(f"    Size     : {size_acc*100:.2f}%")
    print(f"    Location : {location_acc*100:.2f}%")
    print(f"    Tumor    : {tumor_acc*100:.2f}%")
    
    # Average accuracy
    avg_acc = (grade_acc + severity_acc + size_acc + location_acc) / 4 * 100
    print(f"\n  Average Classification Accuracy: {avg_acc:.2f}%")
    
    # Per-class accuracies
    print(f"\n  Per-Class Accuracies:")
    
    # Grade
    print(f"    Grade LGG: {metadata[metadata['true_grade_code']==0]['grade_correct'].mean()*100:.2f}%")
    print(f"    Grade HGG: {metadata[metadata['true_grade_code']==1]['grade_correct'].mean()*100:.2f}%")
    
    # Severity
    for level in [0, 1, 2]:
        subset = metadata[metadata['true_severity_code']==level]
        if len(subset) > 0:
            sev_name = ["Low", "Medium", "High"][level]
            print(f"    Severity {sev_name}: {subset['severity_correct'].mean()*100:.2f}%")
    
    print("="*70)
    
    return {
        "grade": float(grade_acc),
        "severity": float(severity_acc),
        "size": float(size_acc),
        "location": float(location_acc),
        "tumor": float(tumor_acc),
        "average": float(avg_acc / 100)
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*70)
    print("  PHASE II - STEP 4: Extract Embeddings (Enhanced)")
    print("="*70)
    
    print(f"\n📊 Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Chunk size: {CHUNK_SIZE} images")
    print(f"  Output dir: {OUTPUT_DIR}")
    
    # Load data
    print(f"\n📂 Loading dataset...")
    if not os.path.exists(CSV_PATH):
        print(f"  ❌ Dataset not found: {CSV_PATH}")
        return
    
    df_full = pd.read_csv(CSV_PATH)
    
    # Only extract embeddings for tumor slices
    df_full = df_full[df_full["tumor_present"] == 1].reset_index(drop=True)
    total_images = len(df_full)
    
    print(f"\n  Total tumor slices: {total_images:,}")
    print(f"  Total volumes: {df_full['volume_id'].nunique()}")
    
    # Load model
    model = load_model(MODEL_PATH, device)
    if model is None:
        return
    
    # Check for checkpoint
    checkpoint = load_checkpoint(OUTPUT_DIR)
    start_idx = 0
    if checkpoint:
        print(f"\n📋 Found checkpoint from {checkpoint.get('timestamp', 'unknown')}")
        resume = input("  Resume from checkpoint? (y/n): ").strip().lower()
        if resume == 'y':
            start_idx = checkpoint.get("last_processed", 0)
            print(f"  Resuming from image {start_idx}")
    
    # Initialize storage
    all_embeddings = []
    all_paths = []
    all_preds = {"grade": [], "severity": [], "size": [], "location": [], "tumor": []}
    all_gt = {
        "retrieval_label": [], "grade_label": [], "severity_label": [],
        "size_label": [], "location_label": [], "tumor_label": [],
        "clinical_relevance": [], "sample_weight": []
    }
    all_confidences = {"grade": [], "severity": [], "size": [], "location": [], "tumor": []}
    
    # Process in chunks
    num_chunks = (total_images - start_idx + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"\n🔄 Processing {num_chunks} chunks...")
    
    try:
        for chunk_idx in range(num_chunks):
            chunk_start = start_idx + chunk_idx * CHUNK_SIZE
            chunk_end = min(chunk_start + CHUNK_SIZE, total_images)
            
            print(f"\n{'='*70}")
            print(f"  Chunk {chunk_idx + 1}/{num_chunks}: Images {chunk_start:,}-{chunk_end:,}")
            print(f"{'='*70}")
            
            # Create dataset and loader for this chunk
            dataset = EmbeddingDataset(df_full, start_idx=chunk_start, end_idx=chunk_end)
            loader = DataLoader(
                dataset, 
                batch_size=BATCH_SIZE,
                shuffle=False, 
                num_workers=NUM_WORKERS
            )
            
            # Extract embeddings
            embeddings, paths, preds, gt, confidences = extract_chunk(model, loader, device)
            
            # Accumulate results
            if len(embeddings) > 0:
                all_embeddings.append(embeddings)
                all_paths.extend(paths)
                
                for key in all_preds:
                    all_preds[key].extend(preds[key])
                for key in all_gt:
                    if key in gt:
                        all_gt[key].extend(gt[key])
                for key in all_confidences:
                    all_confidences[key].extend(confidences[key])
            
            # Save checkpoint
            save_checkpoint(chunk_idx, chunk_end, OUTPUT_DIR)
            print(f"\n  ✓ Chunk {chunk_idx + 1} complete - Checkpoint saved")
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Interrupted! Progress saved. Run again to resume.")
        return
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Progress saved. You can resume from checkpoint.")
        return
    
    # Concatenate all embeddings
    print("\n" + "="*70)
    print("  Finalizing results...")
    print("="*70)
    
    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
        print(f"\n  Final embeddings shape: {all_embeddings.shape}")
    else:
        print("  ⚠ No embeddings extracted!")
        return
    
    # Save results
    metadata = save_results(
        all_embeddings, all_paths, all_preds, all_gt, all_confidences, OUTPUT_DIR
    )
    
    # Print accuracies
    accuracies = print_accuracies(metadata)
    
    # Save accuracies to JSON
    acc_path = os.path.join(OUTPUT_DIR, "accuracies.json")
    with open(acc_path, "w") as f:
        json.dump(accuracies, f, indent=2)
    print(f"\n💾 Accuracies saved: {acc_path}")
    
    # Clean up checkpoint
    checkpoint_path = os.path.join(OUTPUT_DIR, "extraction_checkpoint.json")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("  ✓ Checkpoint removed")
    
    # Summary
    print("\n" + "="*70)
    print("  ✅ STEP 4 COMPLETE!")
    print("="*70)
    print(f"\n  Outputs saved to: {OUTPUT_DIR}")
    print(f"\n  Files created:")
    print(f"    - embeddings.npy     : {all_embeddings.shape} embedding vectors")
    print(f"    - metadata.csv       : {len(metadata)} rows with predictions")
    print(f"    - accuracies.json    : Classification performance")
    print(f"\n  Next step: Step 5 - FAISS Indexing")
    print("="*70)


if __name__ == "__main__":
    main()