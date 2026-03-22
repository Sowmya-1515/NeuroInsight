"""
PHASE II - STEP 5: FAISS Indexing with Enhanced Features
=========================================================
Features:
- Builds both Flat (exact) and IVF-PQ (approximate) indexes
- Optimized for MPS/Mac
- Automatic parameter tuning
- Comprehensive index metadata
- Sanity checking
- Label distribution analysis
"""

import os
import json
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import time
from pathlib import Path

# ============================================================
# CONFIG - Using paths from your setup
# ============================================================
EMBEDDINGS_DIR = "/Users/sowmyaalamuri/Desktop/Capstone_project/Phase2/embeddings_output_enhanced"
OUTPUT_DIR = "/Users/sowmyaalamuri/Desktop/Capstone_project/Phase2/faiss_index_enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FAISS Configuration
EMBED_DIM = 128
NPROBE = 10  # Number of clusters to probe during search
USE_GPU = False  # FAISS doesn't support MPS, CPU is fine

# ============================================================
# LOAD EMBEDDINGS AND METADATA
# ============================================================
def load_data(embeddings_dir):
    """Load embeddings and metadata"""
    print("\n[1/5] Loading embeddings and metadata...")
    
    embeddings_path = os.path.join(embeddings_dir, "embeddings.npy")
    metadata_path = os.path.join(embeddings_dir, "metadata.csv")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    # Load embeddings
    embeddings = np.load(embeddings_path).astype('float32')
    print(f"  ✓ Embeddings : {embeddings.shape} (dtype={embeddings.dtype})")
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    print(f"  ✓ Metadata   : {metadata.shape}")
    
    # Verify alignment
    assert len(embeddings) == len(metadata), \
        f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) length mismatch!"
    
    return embeddings, metadata


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================
def normalize_embeddings(embeddings):
    """L2 normalize embeddings for cosine similarity"""
    print("\n[2/5] L2-normalizing embeddings...")
    
    norms_before = np.linalg.norm(embeddings, axis=1)
    print(f"  Mean norm before: {norms_before.mean():.4f}")
    
    # Normalize
    faiss.normalize_L2(embeddings)
    
    norms_after = np.linalg.norm(embeddings, axis=1)
    print(f"  Mean norm after : {norms_after.mean():.4f} (should be ~1.0)")
    
    return embeddings


# ============================================================
# AUTO-TUNE IVF PARAMETERS
# ============================================================
def auto_tune_ivf_parameters(n_samples):
    """Automatically tune IVF parameters based on dataset size"""
    
    # Rule of thumb: nlist = 4*sqrt(n_samples) to 8*sqrt(n_samples)
    sqrt_n = int(np.sqrt(n_samples))
    nlist = min(max(4 * sqrt_n, 50), 500)  # Between 50 and 500
    
    # For M=16, each sub-vector is 128/16 = 8 dimensions
    # 8 bits per sub-vector gives 256 centroids per sub-space
    M = 16  # Number of sub-vectors
    nbits = 8  # Bits per sub-vector
    
    print(f"\n  Auto-tuned parameters for {n_samples} samples:")
    print(f"    nlist (clusters) : {nlist}")
    print(f"    M (sub-vectors)  : {M}")
    print(f"    nbits            : {nbits}")
    print(f"    nprobe           : {NPROBE}")
    
    return nlist, M, nbits


# ============================================================
# BUILD FLAT INDEX (Exact Search)
# ============================================================
def build_flat_index(embeddings):
    """Build exact search index (L2)"""
    print("\n[3/5] Building Flat L2 index (exact search)...")
    
    index_flat = faiss.IndexFlatL2(EMBED_DIM)
    index_flat.add(embeddings)
    
    print(f"  ✓ Flat index contains {index_flat.ntotal} vectors")
    
    return index_flat


# ============================================================
# BUILD IVF-PQ INDEX (Approximate Search)
# ============================================================
def build_ivfpq_index(embeddings, nlist, M, nbits):
    """Build IVF-PQ index for fast approximate search"""
    print(f"\n[4/5] Building IVF-PQ index...")
    print(f"  Clusters (nlist) : {nlist}")
    print(f"  Sub-vectors (M)  : {M}")
    print(f"  Bits per sub     : {nbits}")
    print(f"  Probe at query   : {NPROBE}")
    
    # Calculate compression ratio
    raw_size = embeddings.dtype.itemsize * EMBED_DIM  # bytes per vector
    compressed_size = M * nbits // 8  # bytes per vector after PQ
    compression_ratio = raw_size / compressed_size
    print(f"  Compression ratio: ~{compression_ratio:.1f}x vs raw float32")
    
    # Create quantizer and index
    quantizer = faiss.IndexFlatL2(EMBED_DIM)
    index_ivfpq = faiss.IndexIVFPQ(quantizer, EMBED_DIM, nlist, M, nbits)
    
    # Set number of probes at query time
    index_ivfpq.nprobe = NPROBE
    
    # Train the index
    print(f"\n  Training on {len(embeddings)} vectors...")
    start_time = time.time()
    
    # FAISS training can use multiple threads
    faiss.omp_set_num_threads(os.cpu_count())
    
    index_ivfpq.train(embeddings)
    train_time = time.time() - start_time
    print(f"  ✓ Training done in {train_time:.1f}s")
    
    # Add vectors
    print(f"  Adding vectors to index...")
    start_time = time.time()
    index_ivfpq.add(embeddings)
    add_time = time.time() - start_time
    print(f"  ✓ Added {index_ivfpq.ntotal} vectors in {add_time:.1f}s")
    
    return index_ivfpq


# ============================================================
# SANITY CHECK
# ============================================================
def sanity_check(embeddings, metadata, index_flat, index_ivfpq, n_queries=5):
    """Run sanity check by comparing flat vs ivfpq results"""
    print("\n[5/5] Sanity check — querying both indexes...")
    
    # Randomly select query indices
    np.random.seed(42)
    query_indices = np.random.choice(len(embeddings), min(n_queries, len(embeddings)), replace=False)
    
    results = []
    
    for q_idx in query_indices:
        query_emb = embeddings[q_idx:q_idx+1]  # Keep as 2D
        query_row = metadata.iloc[q_idx]
        
        print(f"\n  Query image: {os.path.basename(query_row['image_path'])}")
        print(f"  True: grade={query_row['true_grade']} | "
              f"severity={query_row['true_severity']} | "
              f"size={query_row['true_size']} | "
              f"location={query_row['true_location']}")
        
        # Search flat index (exact)
        k = 10
        D_flat, I_flat = index_flat.search(query_emb, k)
        
        # Search IVF-PQ index (approximate)
        D_ivfpq, I_ivfpq = index_ivfpq.search(query_emb, k)
        
        # Compare results
        print(f"\n  Top-{k} Flat (exact) results:")
        for i, (dist, idx) in enumerate(zip(D_flat[0], I_flat[0])):
            if idx < 0 or idx >= len(metadata):
                continue
            row = metadata.iloc[idx]
            match = "✓" if (row['true_grade'] == query_row['true_grade'] and 
                           row['true_severity'] == query_row['true_severity']) else " "
            print(f"    [{i+1}] {match} grade={row['true_grade']:<4} "
                  f"sev={row['true_severity']:<6} "
                  f"size={row['true_size']:<6} "
                  f"loc={row['true_location']:<8} "
                  f"dist={dist:.4f}")
        
        print(f"\n  Top-{k} IVF-PQ (approx) results:")
        for i, (dist, idx) in enumerate(zip(D_ivfpq[0], I_ivfpq[0])):
            if idx < 0 or idx >= len(metadata):
                continue
            row = metadata.iloc[idx]
            match = "✓" if (row['true_grade'] == query_row['true_grade'] and 
                           row['true_severity'] == query_row['true_severity']) else " "
            print(f"    [{i+1}] {match} grade={row['true_grade']:<4} "
                  f"sev={row['true_severity']:<6} "
                  f"size={row['true_size']:<6} "
                  f"loc={row['true_location']:<8} "
                  f"dist={dist:.4f}")
        
        # Calculate overlap
        flat_set = set(I_flat[0][:5])
        ivfpq_set = set(I_ivfpq[0][:5])
        overlap = len(flat_set.intersection(ivfpq_set))
        print(f"\n  Overlap in top-5: {overlap}/5 ({overlap*20}%)")
        
        results.append({
            "query_idx": int(q_idx),
            "query_path": query_row["image_path"],
            "true_grade": query_row["true_grade"],
            "true_severity": query_row["true_severity"],
            "true_size": query_row["true_size"],
            "true_location": query_row["true_location"],
            "flat_top5": [int(i) for i in I_flat[0][:5]],
            "ivfpq_top5": [int(i) for i in I_ivfpq[0][:5]],
            "overlap_5": overlap
        })
    
    return results


# ============================================================
# LABEL DISTRIBUTION ANALYSIS
# ============================================================
def analyze_label_distribution(metadata):
    """Analyze distribution of labels in the index"""
    print("\n📊 Label distribution in index:")
    
    distributions = {}
    
    for column in ['true_grade', 'true_severity', 'true_size', 'true_location']:
        if column in metadata.columns:
            counts = metadata[column].value_counts()
            distributions[column] = counts.to_dict()
            
            print(f"\n  {column}:")
            total = len(metadata)
            for value, count in counts.items():
                percentage = count / total * 100
                print(f"    {value:<12}: {count:6d} ({percentage:5.1f}%)")
    
    return distributions


# ============================================================
# SAVE INDEXES AND METADATA
# ============================================================
def save_indexes(index_flat, index_ivfpq, metadata, sanity_results, output_dir):
    """Save FAISS indexes and associated metadata"""
    print("\n💾 Saving indexes...")
    
    # Save flat index
    flat_path = os.path.join(output_dir, "faiss_flat.index")
    faiss.write_index(index_flat, flat_path)
    print(f"  ✓ Flat index saved: {flat_path}")
    
    # Save IVF-PQ index
    ivfpq_path = os.path.join(output_dir, "faiss_ivfpq.index")
    faiss.write_index(index_ivfpq, ivfpq_path)
    print(f"  ✓ IVF-PQ index saved: {ivfpq_path}")
    
    # Save index metadata
    metadata_info = {
        "num_vectors": index_flat.ntotal,
        "embedding_dim": EMBED_DIM,
        "index_type": {
            "flat": "IndexFlatL2",
            "ivfpq": f"IndexIVFPQ (nlist={index_ivfpq.nlist}, M={index_ivfpq.pq.M}, nbits={index_ivfpq.pq.nbits})"
        },
        "nprobe": NPROBE,
        "sanity_checks": sanity_results,
        "creation_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_path = os.path.join(output_dir, "index_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_info, f, indent=2, default=str)
    print(f"  ✓ Index metadata saved: {metadata_path}")
    
    # Save label distribution
    distributions = analyze_label_distribution(metadata)
    dist_path = os.path.join(output_dir, "label_distribution.json")
    with open(dist_path, "w") as f:
        json.dump(distributions, f, indent=2)
    print(f"  ✓ Label distribution saved: {dist_path}")
    
    # Also save as CSV for easy viewing
    dist_rows = []
    for column, counts in distributions.items():
        for value, count in counts.items():
            dist_rows.append({
                "attribute": column.replace("true_", ""),
                "value": value,
                "count": count,
                "percentage": count / len(metadata) * 100
            })
    
    dist_df = pd.DataFrame(dist_rows)
    dist_csv_path = os.path.join(output_dir, "label_distribution.csv")
    dist_df.to_csv(dist_csv_path, index=False)
    print(f"  ✓ Label distribution CSV: {dist_csv_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*70)
    print("  PHASE II - STEP 5: FAISS Indexing (Enhanced)")
    print("="*70)
    
    try:
        # Load data
        embeddings, metadata = load_data(EMBEDDINGS_DIR)
        
        # Normalize embeddings
        embeddings = normalize_embeddings(embeddings)
        
        # Auto-tune IVF parameters
        nlist, M, nbits = auto_tune_ivf_parameters(len(embeddings))
        
        # Build flat index
        index_flat = build_flat_index(embeddings)
        
        # Build IVF-PQ index
        index_ivfpq = build_ivfpq_index(embeddings, nlist, M, nbits)
        
        # Run sanity check
        sanity_results = sanity_check(embeddings, metadata, index_flat, index_ivfpq)
        
        # Save everything
        save_indexes(index_flat, index_ivfpq, metadata, sanity_results, OUTPUT_DIR)
        
        # Print summary
        print("\n" + "="*70)
        print("  ✅ STEP 5 COMPLETE!")
        print("="*70)
        print(f"\n  Outputs saved to: {OUTPUT_DIR}")
        print(f"\n  Files created:")
        print(f"    - faiss_flat.index     : Exact search index")
        print(f"    - faiss_ivfpq.index    : Fast approximate search index")
        print(f"    - index_metadata.json  : Index configuration")
        print(f"    - label_distribution.* : Class distributions")
        print(f"\n  Index stats:")
        print(f"    Vectors indexed : {index_flat.ntotal:,}")
        print(f"    Embedding dim   : {EMBED_DIM}")
        print(f"    IVF clusters    : {nlist}")
        print(f"    IVF probes      : {NPROBE}")
        print(f"\n  Next step: Step 6 - Multi-Modal Retrieval!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()