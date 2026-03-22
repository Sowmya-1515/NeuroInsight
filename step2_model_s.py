"""
PHASE II - STEP 2: Enhanced Multi-Task Model (Optimized for Mac MPS)
======================================================================
Features:
- Multi-task learning with 5 heads (tumor, grade, severity, size, location)
- Focal Loss for handling class imbalance (especially for severity)
- Uncertainty-based task weighting (learned during training)
- Monte Carlo Dropout support for uncertainty estimation
- Sample weights for balanced training
- Mask fix: classification heads only train on tumor slices
- Boosted weights for size and severity heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
from pytorch_metric_learning import losses

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = 160
EMBED_DIM = 128

# ============================================================
# TRANSFORMS
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# ENHANCED DATASET WITH SAMPLE WEIGHTS
# ============================================================
class Phase2Dataset(Dataset):
    def __init__(self, df, transform=None, filter_no_tumor=False, 
                 use_sample_weights=True):
        """
        Enhanced dataset with support for sample weights
        
        Args:
            df: DataFrame with image paths and labels
            transform: Image transforms
            filter_no_tumor: If True, only return tumor slices
            use_sample_weights: If True, return sample weights for balanced training
        """
        if filter_no_tumor:
            self.df = df[df["tumor_present"] == 1].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)
            
        print(f"  Dataset size: {len(self.df)} slices")
        print(f"  Tumor: {(self.df['tumor_present']==1).sum()}  No-tumor: {(self.df['tumor_present']==0).sum()}")

        self.transform = transform or val_transform
        self.use_sample_weights = use_sample_weights

        # Cache images in RAM for faster training
        print("  Caching images into RAM (parallel loading)...")
        self.cache = {}
        failed = 0

        def load_one(idx):
            path = self.df.iloc[idx]["image_path"]
            try:
                img = Image.open(path).convert("RGB")
                return idx, img.copy(), False
            except Exception as e:
                print(f"  ⚠ Failed to load {path}: {e}")
                return idx, Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0)), True

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for idx, img, fail in tqdm(
                executor.map(load_one, range(len(self.df))),
                total=len(self.df),
                desc="  Loading images",
                leave=False
            ):
                self.cache[idx] = img
                if fail:
                    failed += 1

        print(f"  Cached {len(self.cache) - failed} images successfully "
              f"({failed} failed → black placeholder)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.transform(self.cache[idx])
        
        # Base dictionary with all labels
        item = {
            "image": img,
            "tumor_label": int(row["tumor_present"]),
            "retrieval_label": int(row.get("retrieval_label", 0)),
            "grade_label": int(row.get("grade_label", 0)),
            "severity_label": int(row.get("severity_bin", row.get("severity_label", 0))),
            "size_label": int(row.get("size_label", 0)),
            "location_label": int(row.get("location_label", 0)),
            "image_path": row["image_path"],
        }
        
        # Add sample weight if available
        if self.use_sample_weights and "sample_weight" in row:
            item["sample_weight"] = float(row["sample_weight"])
        else:
            item["sample_weight"] = 1.0
            
        # Add clinical relevance if available (for evaluation)
        if "clinical_relevance" in row:
            item["clinical_relevance"] = float(row["clinical_relevance"])
            
        return item


def create_weighted_sampler(dataset):
    """
    Create a weighted sampler for balanced training
    """
    print("\n  Creating weighted sampler for balanced training...")
    
    # Get sample weights
    if hasattr(dataset, 'df') and 'sample_weight' in dataset.df.columns:
        weights = dataset.df['sample_weight'].values
        print(f"  Using sample weights from dataset")
    else:
        # Fallback: create weights based on tumor presence
        print(f"  No sample weights found, using tumor-based weights")
        tumor_labels = [dataset[i]['tumor_label'] for i in range(len(dataset))]
        class_counts = np.bincount(tumor_labels)
        weights = 1.0 / class_counts[tumor_labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


# ============================================================
# FOCAL LOSS FOR CLASS IMBALANCE
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    gamma=2.0 works well for most cases
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Focal loss modulation
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================
# ENHANCED MULTI-TASK MODEL
# ============================================================
class Phase2Model(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True, dropout_rate=0.3):
        """
        Enhanced multi-task model with Monte Carlo Dropout support
        
        Args:
            embedding_dim: Dimension of retrieval embeddings
            pretrained: Use ImageNet pretrained weights
            dropout_rate: Dropout rate for all heads
        """
        super().__init__()

        # Backbone: ResNet50
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048
        self.dropout_rate = dropout_rate

        # Projection head for retrieval embeddings
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, embedding_dim)
        )

        # Grade head (LGG vs HGG)
        self.grade_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 2)
        )
        
        # Severity head (low, medium, high)
        self.severity_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 3)
        )
        
        # Size head (small, medium, large)
        self.size_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 3)
        )
        
        # Location head (left, right, bilateral)
        self.location_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 3)
        )
        
        # Tumor presence head (0=no tumor, 1=tumor)
        self.tumor_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 2)
        )
        
        # Tumor confidence head (for uncertainty estimation)
        self.tumor_confidence_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Store dropout state for MC Dropout
        self.dropout_layers = []
        self._store_dropout_layers()

    def _store_dropout_layers(self):
        """Store references to all dropout layers for MC Dropout"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                self.dropout_layers.append(module)

    def forward(self, x, mc_dropout=False):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
            mc_dropout: If True, enable dropout during inference (for uncertainty)
        """
        # Save original training state
        if mc_dropout:
            original_states = [layer.training for layer in self.dropout_layers]
            for layer in self.dropout_layers:
                layer.train(True)
        
        # Extract features
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        
        # Generate embeddings
        embedding = F.normalize(self.projection_head(features), dim=1)
        
        # Generate predictions
        output = {
            "embedding": embedding,
            "grade_logits": self.grade_head(features),
            "severity_logits": self.severity_head(features),
            "size_logits": self.size_head(features),
            "location_logits": self.location_head(features),
            "tumor_logits": self.tumor_head(features),
            "tumor_confidence": self.tumor_confidence_head(features).squeeze(),
            "features": features,  # Raw features for analysis
        }
        
        # Restore original training state
        if mc_dropout:
            for layer, state in zip(self.dropout_layers, original_states):
                layer.train(state)
        
        return output

    def get_embedding(self, x):
        """Get normalized embedding for retrieval"""
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return F.normalize(self.projection_head(features), dim=1)

    def freeze_backbone(self):
        """Freeze encoder backbone"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("  Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze encoder backbone"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("  Backbone unfrozen")
        
    def get_trainable_params(self):
        """Get trainable parameters count"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# ENHANCED MULTI-TASK LOSS WITH UNCERTAINTY WEIGHTING
# ============================================================
class EnhancedMultiTaskLoss(nn.Module):
    """
    Enhanced Multi-Task Loss with:
    - Focal loss for severity (handles class imbalance)
    - Sample weights for balanced training
    - Adaptive loss weighting based on uncertainty
    - Masked losses for tumor slices only
    """
    def __init__(self, grade_weight=None, severity_weight=None,
                 size_weight=None, location_weight=None, tumor_weight=None,
                 device='cpu'):
        super().__init__()
        
        # Metric learning losses
        self.supcon_loss = losses.SupConLoss(temperature=0.07)
        self.ntxent_loss = losses.NTXentLoss(temperature=0.07)
        
        # Classification losses
        self.ce_grade = nn.CrossEntropyLoss(weight=grade_weight)
        self.ce_size = nn.CrossEntropyLoss(weight=size_weight)
        self.ce_location = nn.CrossEntropyLoss(weight=location_weight)
        self.ce_tumor = nn.CrossEntropyLoss(weight=tumor_weight)
        
        # Focal loss for severity (handles imbalance better)
        self.focal_severity = FocalLoss(
            alpha=severity_weight,
            gamma=2.0  # Focus on hard examples
        )
        
        # Learnable task weights (uncertainty weighting)
        # Based on "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al. 2018)
        self.log_vars = nn.ParameterDict({
            'supcon': nn.Parameter(torch.zeros(1, device=device)),
            'ntxent': nn.Parameter(torch.zeros(1, device=device)),
            'grade': nn.Parameter(torch.zeros(1, device=device)),
            'severity': nn.Parameter(torch.zeros(1, device=device)),
            'size': nn.Parameter(torch.zeros(1, device=device)),
            'location': nn.Parameter(torch.zeros(1, device=device)),
            'tumor': nn.Parameter(torch.zeros(1, device=device)),
        })
        
        # Store device
        self.device = device
        
    def forward(self, outputs, batch):
        """
        Forward pass with uncertainty weighting
        
        Args:
            outputs: Model outputs dictionary
            batch: Batch dictionary with labels
        """
        emb = outputs["embedding"]
        rlbl = batch["retrieval_label"]
        dev = outputs["tumor_logits"].device
        
        # Get sample weights if available
        sample_weights = batch.get("sample_weight", None)
        
        # Metric learning losses
        supcon = self.supcon_loss(emb, rlbl)
        ntxent = self.ntxent_loss(emb, rlbl)
        
        # Tumor loss - ALL slices
        tumor_ce = self.ce_tumor(outputs["tumor_logits"], batch["tumor_label"])
        
        # Apply sample weights to tumor loss if available
        if sample_weights is not None:
            # Weighted cross entropy for tumor
            tumor_ce = (tumor_ce * sample_weights).mean()
        
        # Classification losses - ONLY tumor slices
        mask = batch["tumor_label"] == 1
        
        if mask.sum() > 0:
            # Grade loss
            grade_ce = self.ce_grade(
                outputs["grade_logits"][mask], 
                batch["grade_label"][mask]
            )
            
            # Severity loss (using focal loss)
            severity_ce = self.focal_severity(
                outputs["severity_logits"][mask], 
                batch["severity_label"][mask]
            )
            
            # Size loss
            size_ce = self.ce_size(
                outputs["size_logits"][mask], 
                batch["size_label"][mask]
            )
            
            # Location loss
            location_ce = self.ce_location(
                outputs["location_logits"][mask], 
                batch["location_label"][mask]
            )
            
            # Apply sample weights if available
            if sample_weights is not None:
                sw = sample_weights[mask]
                grade_ce = (grade_ce * sw).mean()
                severity_ce = (severity_ce * sw).mean()
                size_ce = (size_ce * sw).mean()
                location_ce = (location_ce * sw).mean()
        else:
            grade_ce = torch.tensor(0.0, device=dev)
            severity_ce = torch.tensor(0.0, device=dev)
            size_ce = torch.tensor(0.0, device=dev)
            location_ce = torch.tensor(0.0, device=dev)
        
        # Uncertainty weighting (Kendall et al. 2018)
        # Loss = sum( 1/(2*sigma^2) * loss_i + log(sigma) )
        precision = {
            'supcon': torch.exp(-self.log_vars['supcon']),
            'ntxent': torch.exp(-self.log_vars['ntxent']),
            'grade': torch.exp(-self.log_vars['grade']),
            'severity': torch.exp(-self.log_vars['severity']),
            'size': torch.exp(-self.log_vars['size']),
            'location': torch.exp(-self.log_vars['location']),
            'tumor': torch.exp(-self.log_vars['tumor']),
        }
        
        # Weighted total loss
        total = (
            precision['supcon'] * supcon + self.log_vars['supcon'] +
            precision['ntxent'] * ntxent + self.log_vars['ntxent'] +
            precision['tumor'] * tumor_ce + self.log_vars['tumor'] +
            precision['grade'] * grade_ce + self.log_vars['grade'] +
            precision['severity'] * severity_ce + self.log_vars['severity'] +
            precision['size'] * size_ce + self.log_vars['size'] +
            precision['location'] * location_ce + self.log_vars['location']
        )
        
        return {
            "total": total,
            "supcon": supcon.item(),
            "ntxent": ntxent.item(),
            "grade": grade_ce.item(),
            "severity": severity_ce.item(),
            "size": size_ce.item(),
            "location": location_ce.item(),
            "tumor": tumor_ce.item(),
            "task_weights": {k: v.item() for k, v in precision.items()},
        }


# ============================================================
# MODEL UTILITIES
# ============================================================
def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_class_weights(df, device='cpu'):
    """
    Calculate class weights for balanced loss
    
    Args:
        df: DataFrame with labels
        device: Torch device
    """
    weights = {}
    
    # Tumor weights
    tumor_counts = df["tumor_present"].value_counts().to_dict()
    weights['tumor'] = torch.tensor([
        len(df) / (2 * tumor_counts.get(0, 1)),
        len(df) / (2 * tumor_counts.get(1, 1))
    ], dtype=torch.float32).to(device)
    
    # Only for tumor slices
    tumor_df = df[df["tumor_present"] == 1]
    
    if len(tumor_df) > 0:
        # Grade weights
        grade_counts = tumor_df["grade_label"].value_counts().to_dict()
        weights['grade'] = torch.tensor([
            len(tumor_df) / (2 * grade_counts.get(i, 1)) for i in range(2)
        ], dtype=torch.float32).to(device)
        
        # Severity weights
        sev_counts = tumor_df["severity_bin"].value_counts().to_dict()
        weights['severity'] = torch.tensor([
            len(tumor_df) / (3 * sev_counts.get(i, 1)) for i in range(3)
        ], dtype=torch.float32).to(device)
        
        # Size weights
        size_counts = tumor_df["size_label"].value_counts().to_dict()
        weights['size'] = torch.tensor([
            len(tumor_df) / (3 * size_counts.get(i, 1)) for i in range(3)
        ], dtype=torch.float32).to(device)
        
        # Location weights
        loc_counts = tumor_df["location_label"].value_counts().to_dict()
        weights['location'] = torch.tensor([
            len(tumor_df) / (3 * loc_counts.get(i, 1)) for i in range(3)
        ], dtype=torch.float32).to(device)
    else:
        # Default weights if no tumor slices
        weights['grade'] = torch.ones(2, dtype=torch.float32).to(device)
        weights['severity'] = torch.ones(3, dtype=torch.float32).to(device)
        weights['size'] = torch.ones(3, dtype=torch.float32).to(device)
        weights['location'] = torch.ones(3, dtype=torch.float32).to(device)
    
    return weights


def get_optimizer(model, lr, frozen=True):
    """
    Get optimizer with different learning rates for different heads
    
    Args:
        model: The model
        lr: Base learning rate
        frozen: Whether backbone is frozen
    """
    if frozen:
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-5
        )
    else:
        return torch.optim.Adam([
            {"params": model.encoder.parameters(), "lr": lr * 0.1},
            {"params": model.projection_head.parameters(), "lr": lr},
            {"params": model.grade_head.parameters(), "lr": lr},
            {"params": model.severity_head.parameters(), "lr": lr * 2.0},  # Boosted
            {"params": model.size_head.parameters(), "lr": lr * 2.0},      # Boosted
            {"params": model.location_head.parameters(), "lr": lr},
            {"params": model.tumor_head.parameters(), "lr": lr},
            {"params": model.tumor_confidence_head.parameters(), "lr": lr},
        ], weight_decay=1e-5)


# ============================================================
# MODEL INITIALIZATION FUNCTION
# ============================================================
def create_model(embedding_dim=128, pretrained=True, device='cpu'):
    """
    Create and initialize the model
    
    Args:
        embedding_dim: Dimension of embeddings
        pretrained: Use pretrained weights
        device: Torch device
    """
    print("\n" + "="*60)
    print("  Creating Enhanced Multi-Task Model")
    print("="*60)
    
    model = Phase2Model(embedding_dim=embedding_dim, pretrained=pretrained).to(device)
    
    print(f"\nModel Architecture:")
    print(f"  Backbone: ResNet50 (pretrained={pretrained})")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Dropout rate: {model.dropout_rate}")
    print(f"  Trainable parameters: {count_parameters(model):,}")
    
    print(f"\nOutput Heads:")
    print(f"  - Grade (2 classes: LGG/HGG)")
    print(f"  - Severity (3 classes: low/medium/high)")
    print(f"  - Size (3 classes: small/medium/large)")
    print(f"  - Location (3 classes: left/right/bilateral)")
    print(f"  - Tumor presence (2 classes)")
    print(f"  - Tumor confidence (regression)")
    print(f"  - Retrieval embedding ({embedding_dim}D normalized)")
    
    return model


# ============================================================
# EXPORTS
# ============================================================
__all__ = [
    'Phase2Dataset',
    'Phase2Model',
    'EnhancedMultiTaskLoss',
    'FocalLoss',
    'create_weighted_sampler',
    'get_class_weights',
    'get_optimizer',
    'create_model',
    'train_transform',
    'val_transform',
]