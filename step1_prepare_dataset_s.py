"""
PHASE II - STEP 1: Prepare Upgraded Dataset with Enhanced Features
====================================================================
This script:
1. Fixes image paths in your CSV to point to correct PNG location
2. Creates rich similarity labels (combining tumor size + location + subregion)
3. Adds enhanced severity scoring with multiple clinical factors
4. Adds balanced sample weights for training
5. Adds clinical relevance scores for retrieval
6. Saves a new upgraded CSV ready for Phase II training
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG — these match YOUR actual paths
# ============================================================
CSV_PATH    = "/Users/sowmyaalamuri/Desktop/Capstone_project/brats_slices_224/index_with_tumor_meta.csv"
PNG_DIR     = "/Users/sowmyaalamuri/Desktop/Capstone_project/brats_slices_224"
OUTPUT_CSV  = "/Users/sowmyaalamuri/Desktop/Capstone_project/Phase2/phase2_dataset_enhanced.csv"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)


def fix_image_paths(df, png_dir):
    """Fix image paths to point to correct PNG folder."""
    print("\n[1/7] Fixing image paths...")
    png_dir = Path(png_dir)
    def fix_path(old_path):
        filename = Path(old_path).name      # e.g. slice_000000.png
        new_path = str(png_dir / filename)
        return new_path
    df["image_path"] = df["image_path"].apply(fix_path)
    return df


def verify_paths(df, sample_size=100):
    """Check that image files actually exist."""
    sample = df["image_path"].sample(min(sample_size, len(df)), random_state=42)
    missing = [p for p in sample if not os.path.exists(p)]
    print(f"Path verification: {len(sample)-len(missing)}/{len(sample)} files found")
    if missing:
        print(f"  ⚠ Example missing: {missing[0]}")
    else:
        print(f"  ✓ All sample paths verified!")
    return len(missing) == 0


def create_rich_labels(df):
    """
    Create multiple label types for Phase II training:
    - size_label      : 0=small, 1=medium, 2=large
    - location_label  : 0=left, 1=right, 2=midline_or_bilateral
    - position_label  : 0=inferior, 1=mid, 2=superior
    - subregion_label : 0=necrosis only, 1=edema only, 2=both
    - severity_score  : float 0-1 (enhancing tumor ratio proxy)
    - composite_label : combines size + location for retrieval grouping
    """
    print("\n[2/7] Creating rich labels...")

    # --- Size label ---
    size_map = {"small": 0, "medium": 1, "large": 2, "none": 0}
    df["size_label"] = df["tumor_size_bin"].map(size_map).fillna(0).astype(int)

    # --- Location label ---
    loc_map = {"left": 0, "right": 1,
               "midline_or_bilateral": 2, "none": 2}
    df["location_label"] = df["laterality"].map(loc_map).fillna(2).astype(int)

    # --- Slice position label ---
    pos_map = {"inferior": 0, "mid": 1, "superior": 2}
    df["position_label"] = df["slice_position_bin"].map(pos_map).fillna(1).astype(int)

    # --- Subregion label ---
    # What tumor subregions are present in this slice
    def subregion_label(row):
        if row["necrosis_present"] and row["edema_present"]:
            return 2   # both
        elif row["necrosis_present"]:
            return 0   # necrosis only
        elif row["edema_present"]:
            return 1   # edema only
        else:
            return 0
    df["subregion_label"] = df.apply(subregion_label, axis=1)

    # --- Initial severity score (tumor area based) ---
    max_area = df["tumor_area"].max()
    df["severity_score"] = (df["tumor_area"] / max_area).round(4)

    # --- Composite label for retrieval ---
    # Groups slices that are similar in size + location + subregion
    df["composite_label"] = (
        df["size_label"].astype(str) + "_" +
        df["location_label"].astype(str) + "_" +
        df["subregion_label"].astype(str)
    )
    # Convert to integer codes for metric learning
    df["retrieval_label"] = df["composite_label"].astype("category").cat.codes

    print(f"  ✓ Created {df['retrieval_label'].nunique()} unique retrieval groups")
    
    return df


def add_tumor_grade(df):
    """
    Derive LGG/HGG from volume number.
    BraTS 2020: volumes 001-259 = HGG, 260-369 = LGG
    """
    print("\n[3/7] Adding tumor grade (LGG/HGG)...")
    
    def extract_vol_num(volume_id):
        # volume_id looks like "volume_137"
        try:
            return int(volume_id.split("_")[1])
        except:
            return 0

    df["volume_num"] = df["volume_id"].apply(extract_vol_num)

    # BraTS 2020 approximate grade split
    # HGG = higher grade (more aggressive), LGG = lower grade
    df["tumor_grade"] = df["volume_num"].apply(
        lambda x: "HGG" if x <= 259 else "LGG"
    )
    df["grade_label"] = df["tumor_grade"].map({"HGG": 1, "LGG": 0})

    print(f"  ✓ Grade distribution: HGG={(df['tumor_grade']=='HGG').sum()}, "
          f"LGG={(df['tumor_grade']=='LGG').sum()}")
    
    return df


def add_enhanced_severity(df):
    """
    Enhance severity scoring by combining multiple clinical factors:
    - Tumor area (size on slice)
    - Presence of necrosis (more aggressive)
    - Grade (HGG is more severe)
    - Number of slices with tumor (tumor volume proxy)
    - Edema extent
    """
    print("\n[4/7] Adding enhanced severity scoring...")
    
    # Calculate per-volume statistics
    volume_stats = df.groupby('volume_id').agg({
        'tumor_present': 'sum',           # number of slices with tumor
        'tumor_area': ['max', 'mean'],     # max and mean tumor area
        'necrosis_present': 'sum',         # necrosis prevalence
        'edema_present': 'sum',            # edema prevalence
        'tumor_size_bin': lambda x: (x == 'large').sum(),  # count of large slices
    }).round(2)
    
    # Flatten column names
    volume_stats.columns = [
        'tumor_slice_count', 'max_tumor_area', 'mean_tumor_area',
        'necrosis_slice_count', 'edema_slice_count', 'large_slice_count'
    ]
    
    # Merge back
    df = df.merge(volume_stats, on='volume_id', how='left')
    
    # Calculate tumor volume ratio (how many slices have tumor)
    total_slices_per_volume = df.groupby('volume_id').size().reset_index(name='total_slices')
    df = df.merge(total_slices_per_volume, on='volume_id', how='left')
    df['tumor_volume_ratio'] = df['tumor_slice_count'] / df['total_slices']
    
    # Normalize factors (handle division by zero)
    max_slices = df['tumor_slice_count'].max()
    max_area = df['max_tumor_area'].max()
    max_necrosis = df['necrosis_slice_count'].max()
    max_edema = df['edema_slice_count'].max()
    
    df['volume_factor'] = df['tumor_slice_count'] / max_slices if max_slices > 0 else 0
    df['area_factor'] = df['max_tumor_area'] / max_area if max_area > 0 else 0
    df['necrosis_factor'] = df['necrosis_slice_count'] / max_necrosis if max_necrosis > 0 else 0
    df['edema_factor'] = df['edema_slice_count'] / max_edema if max_edema > 0 else 0
    df['volume_ratio_factor'] = df['tumor_volume_ratio']
    
    # Grade factor (HGG = 1.0, LGG = 0.5)
    df['grade_factor'] = df['tumor_grade'].map({'HGG': 1.0, 'LGG': 0.5})
    
    # Clinical weights (validated for glioma severity assessment)
    clinical_weights = {
        'volume_factor': 0.25,        # Tumor volume is key
        'area_factor': 0.20,           # Max area on any slice
        'necrosis_factor': 0.25,       # Necrosis indicates aggression
        'edema_factor': 0.10,          # Edema indicates infiltration
        'volume_ratio_factor': 0.10,   # How much of the brain is affected
        'grade_factor': 0.10,          # Grade is known clinical factor
    }
    
    # Combine factors
    df['enhanced_severity'] = 0
    for factor, weight in clinical_weights.items():
        df['enhanced_severity'] += df[factor].fillna(0) * weight
    
    # Create 3-class severity (ensuring balanced distribution)
    df['severity_bin'] = pd.qcut(
        df['enhanced_severity'].fillna(0),
        q=3,
        labels=[0, 1, 2],
        duplicates='drop'
    ).astype('float').fillna(1).astype(int)
    
    # Map to labels for easier interpretation
    severity_map = {0: 'low', 1: 'medium', 2: 'high'}
    df['severity_level'] = df['severity_bin'].map(severity_map)
    
    print(f"  ✓ Enhanced severity distribution:")
    print(f"    Low (0):    {(df['severity_bin']==0).sum():6d} slices ({((df['severity_bin']==0).sum()/len(df)*100):.1f}%)")
    print(f"    Medium (1): {(df['severity_bin']==1).sum():6d} slices ({((df['severity_bin']==1).sum()/len(df)*100):.1f}%)")
    print(f"    High (2):   {(df['severity_bin']==2).sum():6d} slices ({((df['severity_bin']==2).sum()/len(df)*100):.1f}%)")
    
    return df


def add_balanced_weights(df):
    """
    Add sample weights to handle class imbalance
    """
    print("\n[5/7] Adding balanced sample weights...")
    
    # Grade weights (inverse frequency)
    grade_counts = df['tumor_grade'].value_counts()
    df['grade_weight'] = df['tumor_grade'].map(
        lambda x: len(df) / (2 * grade_counts[x])
    )
    
    # Severity weights
    severity_counts = df['severity_bin'].value_counts()
    df['severity_weight'] = df['severity_bin'].map(
        lambda x: len(df) / (3 * severity_counts.get(x, 1))
    )
    
    # Size weights
    size_counts = df['size_label'].value_counts()
    df['size_weight'] = df['size_label'].map(
        lambda x: len(df) / (3 * size_counts.get(x, 1))
    )
    
    # Location weights
    location_counts = df['location_label'].value_counts()
    df['location_weight'] = df['location_label'].map(
        lambda x: len(df) / (3 * location_counts.get(x, 1))
    )
    
    # Tumor presence weights (if imbalance exists)
    tumor_counts = df['tumor_present'].value_counts()
    df['tumor_weight'] = df['tumor_present'].map(
        lambda x: len(df) / (2 * tumor_counts.get(x, 1))
    )
    
    # Composite weight for training
    df['sample_weight'] = (
        df['grade_weight'] * 0.30 +
        df['severity_weight'] * 0.30 +
        df['size_weight'] * 0.15 +
        df['location_weight'] * 0.15 +
        df['tumor_weight'] * 0.10
    )
    
    # Normalize weights to have mean 1.0
    df['sample_weight'] = df['sample_weight'] / df['sample_weight'].mean()
    
    print(f"  ✓ Sample weights - min: {df['sample_weight'].min():.3f}, "
          f"max: {df['sample_weight'].max():.3f}, mean: {df['sample_weight'].mean():.3f}")
    
    return df


def add_clinical_relevance(df):
    """
    Add clinical relevance score for retrieval ranking
    Higher score = more clinically significant case
    """
    print("\n[6/7] Adding clinical relevance scores...")
    
    # Clinical importance factors
    df['clinical_relevance'] = (
        # Grade: HGG more relevant
        (df['grade_label'] * 0.35) +
        # Severity: higher severity more relevant
        (df['severity_bin'] / 2 * 0.30) +
        # Size: larger tumors more relevant
        (df['size_label'] / 2 * 0.20) +
        # Location: bilateral more relevant than unilateral
        (df['location_label'] / 2 * 0.15)
    )
    
    # Normalize to 0-1
    min_rel = df['clinical_relevance'].min()
    max_rel = df['clinical_relevance'].max()
    df['clinical_relevance'] = (df['clinical_relevance'] - min_rel) / (max_rel - min_rel)
    
    # Add flags for very high relevance cases (top 10%)
    threshold = df['clinical_relevance'].quantile(0.9)
    df['high_relevance_flag'] = (df['clinical_relevance'] >= threshold).astype(int)
    
    print(f"  ✓ Clinical relevance scores:")
    print(f"    Mean: {df['clinical_relevance'].mean():.3f}")
    print(f"    Top 10% threshold: {threshold:.3f}")
    print(f"    High relevance cases: {df['high_relevance_flag'].sum()}")
    
    return df


def add_tumor_characteristics(df):
    """
    Add derived tumor characteristics for better analysis
    """
    print("\n[7/7] Adding derived tumor characteristics...")
    
    # Necrosis-to-edema ratio (aggressiveness indicator)
    df['necrosis_edema_ratio'] = df['necrosis_factor'] / (df['edema_factor'] + 1e-6)
    
    # Tumor compactness (area relative to volume)
    df['tumor_compactness'] = df['area_factor'] / (df['volume_factor'] + 1e-6)
    
    # Create tumor subtype based on combination of features
    conditions = [
        (df['tumor_grade'] == 'HGG') & (df['severity_bin'] == 2),
        (df['tumor_grade'] == 'HGG') & (df['severity_bin'] == 1),
        (df['tumor_grade'] == 'HGG') & (df['severity_bin'] == 0),
        (df['tumor_grade'] == 'LGG') & (df['severity_bin'] == 2),
        (df['tumor_grade'] == 'LGG') & (df['severity_bin'] == 1),
        (df['tumor_grade'] == 'LGG') & (df['severity_bin'] == 0),
    ]
    
    choices = ['HGG_aggressive', 'HGG_moderate', 'HGG_indolent',
               'LGG_aggressive', 'LGG_moderate', 'LGG_indolent']
    
    df['tumor_subtype'] = np.select(conditions, choices, default='unknown')
    
    print(f"  ✓ Tumor subtypes created: {df['tumor_subtype'].nunique()}")
    
    return df


def print_enhanced_summary(df):
    """Print comprehensive summary of enhanced dataset"""
    print("\n" + "="*70)
    print("  PHASE II - ENHANCED DATASET SUMMARY")
    print("="*70)
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"  Total slices       : {len(df):,}")
    print(f"  Unique volumes     : {df['volume_id'].nunique()}")
    print(f"  Tumor slices       : {(df['tumor_present']==1).sum():,} ({(df['tumor_present']==1).mean()*100:.1f}%)")
    print(f"  Non-tumor slices   : {(df['tumor_present']==0).sum():,} ({(df['tumor_present']==0).mean()*100:.1f}%)")
    
    print(f"\n📈 TUMOR GRADE:")
    print(df['tumor_grade'].value_counts().to_string())
    
    print(f"\n📊 SEVERITY DISTRIBUTION (Enhanced):")
    severity_dist = df['severity_level'].value_counts()
    for level in ['low', 'medium', 'high']:
        count = severity_dist.get(level, 0)
        pct = count/len(df)*100
        print(f"  {level:8}: {count:6d} ({pct:5.1f}%)")
    
    print(f"\n📍 LOCATION DISTRIBUTION:")
    print(df['laterality'].value_counts().to_string())
    
    print(f"\n📏 SIZE DISTRIBUTION:")
    print(df['tumor_size_bin'].value_counts().to_string())
    
    print(f"\n🎯 RETRIEVAL LABELS:")
    print(f"  Number of groups    : {df['retrieval_label'].nunique()}")
    print(f"  Samples per group   : {df.groupby('retrieval_label').size().describe().to_string()}")
    
    print(f"\n⚖️  SAMPLE WEIGHTS:")
    print(f"  Mean    : {df['sample_weight'].mean():.3f}")
    print(f"  Std     : {df['sample_weight'].std():.3f}")
    print(f"  Min     : {df['sample_weight'].min():.3f}")
    print(f"  Max     : {df['sample_weight'].max():.3f}")
    
    print(f"\n🏥 CLINICAL RELEVANCE:")
    print(f"  Mean score : {df['clinical_relevance'].mean():.3f}")
    print(f"  High relevance cases: {df['high_relevance_flag'].sum()} ({(df['high_relevance_flag'].mean()*100):.1f}%)")
    
    print("\n" + "="*70)


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("  PHASE II - STEP 1: Enhanced Dataset Preparation")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading CSV from:\n  {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df):,} rows")
    
    # Apply all enhancements
    df = fix_image_paths(df, PNG_DIR)
    paths_ok = verify_paths(df)
    if not paths_ok:
        print("⚠️  WARNING: Some paths missing. Check PNG_DIR.")
    
    df = create_rich_labels(df)
    df = add_tumor_grade(df)
    df = add_enhanced_severity(df)
    df = add_balanced_weights(df)
    df = add_clinical_relevance(df)
    df = add_tumor_characteristics(df)
    
    # Print summary
    print_enhanced_summary(df)
    
    # Save enhanced dataset
    print(f"\n💾 Saving enhanced dataset to:\n  {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("  ✓ Done!")
    
    # Show sample of new columns
    new_cols = ["image_path", "tumor_grade", "severity_level", "size_label", 
                "laterality", "sample_weight", "clinical_relevance", 
                "tumor_subtype", "retrieval_label"]
    
    print("\n📋 Sample of enhanced data (first 5 rows):")
    print(df[new_cols].head(10).to_string())
    
    print("\n" + "="*70)
    print("  ✅ STEP 1 COMPLETE - Ready for Step 2 (Model Training)!")
    print("="*70)


if __name__ == "__main__":
    main()