import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
import yaml
from collections import Counter
import logging
from datetime import datetime
from model import MyModel
import platform
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from torch.utils.tensorboard import SummaryWriter


def get_device():
    """Get the device to be used for training.

    Returns:
        torch.device: The device to be used (CPU or GPU).  
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        return torch.device("mps")  # Metal backend for Mac
    else:
        return torch.device("cpu")


class FruitDataset(Dataset):
    """Fruit dataset used for training.

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, data_dir, config):
        self.data_dir = data_dir # ‼️‼️‼️‼️ directory with your data ‼️‼️‼️‼️
        self.images = []
        self.masks = []
        self.metadata = []
        for folder in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, folder)):
                folder_path = os.path.join(data_dir, folder)
                self.images.extend([os.path.join(folder_path, "Images", img) for img in os.listdir(os.path.join(folder_path, "Images"))])
                self.masks.extend([os.path.join(folder_path, "Mask", msk) for msk in os.listdir(os.path.join(folder_path, "Mask"))])
                self.metadata.extend([folder] * len(os.listdir(os.path.join(folder_path, "Images"))))
        unique_fruits = sorted(list(set(self.metadata)))
        name_to_label = {name: idx for idx, name in enumerate(unique_fruits)}
        self.labels = [name_to_label[name] for name in self.metadata]
        self.config = config  # loaded yaml config
        self.loaded_images = {}  # ‼️‼️‼️‼️ To store loaded images if caching is enabled ‼️‼️‼️‼️

    def __len__(self):  # ‼️‼️‼️‼️
        """Return the total number of samples in the dataset.
        """
        return len(self.images) # ‼️‼️‼️‼️ TODO ‼️‼️‼️‼️

    def __getitem__(self, idx):  # ‼️‼️‼️‼️
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, mask, metadata, label, idx)
            image (torch.Tensor): The image tensor.
            mask (torch.Tensor): The mask tensor.
            metadata (dict): The metadata for the sample (fruit name).
            label (torch.Tensor): The label tensor (fruit name but numbered)
            idx (int): The index of the sample.
        """

        # ‼️‼️‼️‼️ TODO ‼️‼️‼️‼️
        if self.config['cache_images'] and idx in self.loaded_images:
            image = self.loaded_images[idx]['image']
            mask = self.loaded_images[idx]['mask']
        else:
            image_path = self.images[idx]
            mask_path = self.masks[idx]

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            target_size = tuple(self.config.get('image_size', [512, 512]))

            if image.size != target_size:
                image = image.resize(target_size)

            if mask.size != target_size:
                mask = mask.resize(target_size)

            image = np.array(image).astype(np.float32) / 255.0

            mask = np.array(mask).astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)

            if len(mask.shape) == 2:
                mask = mask[..., np.newaxis]

            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)
            if self.config['cache_images']:
                self.loaded_images[idx] = {'image': image, 'mask': mask}

        metadata = self.metadata[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, mask, metadata, label, idx


def setup_experiment_dir(config):
    """
    Create experiment directory with timestamp and setup logging
    Returns the experiment directory path and tensorboard writer
    """
    # Create timestamp for the experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"experiment_{timestamp}"

    # Create experiment directory
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "tensorboard"), exist_ok=True)

    # Copy config file to experiment directory
    config_save_path = os.path.join(exp_dir, "config.yml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Setup TensorBoard writer
    tb_dir = os.path.join(exp_dir, "tensorboard")
    writer = SummaryWriter(tb_dir)

    # Setup logging
    log_file = os.path.join(exp_dir, "logs", "experiment.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Experiment started: {exp_name}")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Config saved to: {config_save_path}")
    logger.info(f"TensorBoard logs: {tb_dir}")

    return exp_dir, logger, writer


def print_fruit_distribution(train_idx, val_idx, labels, logger):
    """Print the distribution of fruit classes in the training and validation sets.

    Args:
        train_idx (list): Indices of the training samples.
        val_idx (list): Indices of the validation samples.
        labels (list): List of labels for each sample.
        logger (logging.Logger): Logger instance for logging.
    """
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    logger.info("Training set distribution:")
    for fruit, count in sorted(train_counts.items()):
        logger.info(f"  {fruit}: {count} samples")

    logger.info("Validation set distribution:")
    for fruit, count in sorted(val_counts.items()):
        logger.info(f"  {fruit}: {count} samples")


def compute_seg_metrics(pred, target, class_idx):
    """Compute segmentation metrics for a specific class.

    Args:
        pred (torch.Tensor): Predicted segmentation map.
        target (torch.Tensor): Ground truth segmentation map.
        class_idx (int): Class index to compute metrics for.

    Returns:
        tuple: (precision, recall, iou)
    """
    pred = pred == class_idx
    target = target == class_idx
    TP = (pred & target).sum()
    FP = (pred & ~target).sum()
    FN = (~pred & target).sum()

    iou = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    return precision, recall, iou


def compute_embedding_bank(dataset, model):
    """
    Compute embeddings for all samples in the dataset and store them in an embedding bank.
    Returns: embedding_bank (tensor), labels (list)
    """
    device = get_device()
    model.eval()

    # Pre-allocate tensors for efficiency
    dataset_size = 1969
    embedding_dim = None  # Will be determined from first batch
    all_embeddings = None
    all_labels = torch.zeros(dataset_size, dtype=torch.long, device=device)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    labels_found = set()
    with torch.no_grad():
        for batch_idx, (images, _, metadata, labels, idxs) in enumerate(tqdm(dataloader, desc="Computing embeddings")):
            images = images.to(device)
            batch_embeddings = model.get_embedding(images)

            # Initialize embedding tensor on first batch
            if all_embeddings is None:
                embedding_dim = batch_embeddings.shape[1]
                all_embeddings = torch.full(
                    (dataset_size, embedding_dim), float('nan'), device=device)

            # Convert indices to tensor if needed and move to device
            if not torch.is_tensor(idxs):
                idxs = torch.tensor(idxs, device=device)
            else:
                idxs = idxs.to(device)

            # Move labels to device for assignment
            labels = labels.to(device)
            labels_found.update(labels.cpu().numpy())

            # Vectorized assignment - no loops!
            all_embeddings[idxs] = batch_embeddings
            all_labels[idxs] = labels

    # Convert labels to list for compatibility
    all_labels = all_labels.tolist()
    different_labels = set(all_labels)
    return all_embeddings, all_labels


def create_triplets_batch(anchor_embeddings, anchor_labels, anchor_idxs,
                          embedding_bank, all_labels, hard_percentage=0.2):
    """
    Create triplets from a batch of anchors using percentage-based hard mining.

    Args:
        anchor_embeddings: Batch of anchor embeddings [batch_size, embedding_dim]
        anchor_labels: Labels for anchors [batch_size]
        anchor_idxs: Indices of anchor samples in dataset [batch_size]
        embedding_bank: All embeddings in the dataset [dataset_size, embedding_dim]
        all_labels: All labels in the dataset [dataset_size]
        hard_percentage: Percentage of hardest positives and negatives to sample from

    Returns:
        triplets: List of (anchor_idx, positive_idx, negative_idx) tuples
    """
    batch_size = anchor_embeddings.size(0)
    triplets = []

    # Create label-to-indices mapping for efficient lookup
    label_to_indices = {}
    for idx, label in enumerate(all_labels):
        label_to_indices.setdefault(label, []).append(idx)

    for i in range(batch_size):
        anchor_idx = anchor_idxs[i].item()
        anchor_embedding = anchor_embeddings[i].unsqueeze(
            0)  # [1, embedding_dim]
        anchor_label = anchor_labels[i].item()

        # Positive and negative candidates
        positive_candidates = [
            idx for idx in label_to_indices[anchor_label] if idx != anchor_idx]
        negative_candidates = [idx for lbl, idxs in label_to_indices.items(
        ) if lbl != anchor_label for idx in idxs]

        if len(positive_candidates) == 0 or len(negative_candidates) == 0:
            continue

        # filter nan - get valid mask for entire embedding bank
        valid_mask = ~torch.isnan(embedding_bank).any(dim=1)

        # Convert candidates to tensors for vectorized operations
        pos_candidates_tensor = torch.tensor(
            positive_candidates, device=embedding_bank.device)
        neg_candidates_tensor = torch.tensor(
            negative_candidates, device=embedding_bank.device)

        # Filter using vectorized operations instead of loops
        valid_pos_mask = valid_mask[pos_candidates_tensor]
        valid_neg_mask = valid_mask[neg_candidates_tensor]

        valid_positive_candidates = pos_candidates_tensor[valid_pos_mask]
        valid_negative_candidates = neg_candidates_tensor[valid_neg_mask]

        if len(valid_positive_candidates) == 0 or len(valid_negative_candidates) == 0:
            continue

        # Compute similarities using only valid candidates
        pos_embeds = embedding_bank[valid_positive_candidates]
        neg_embeds = embedding_bank[valid_negative_candidates]

        pos_sims = torch.nn.functional.cosine_similarity(
            anchor_embedding, pos_embeds, dim=1)
        neg_sims = torch.nn.functional.cosine_similarity(
            anchor_embedding, neg_embeds, dim=1)

        # Select same count based on smaller pool of valid candidates
        pos_count = max(
            1, int(len(valid_positive_candidates) * hard_percentage))
        neg_count = max(
            1, int(len(valid_negative_candidates) * hard_percentage))
        num_to_sample = min(pos_count, neg_count)

        # Hard positive: least similar (harder to distinguish)
        _, hard_pos_indices = torch.topk(
            pos_sims, num_to_sample, largest=False)
        positive_idx = valid_positive_candidates[hard_pos_indices]

        # Hard negative: most similar (harder to separate)
        _, hard_neg_indices = torch.topk(neg_sims, num_to_sample, largest=True)
        negative_idx = valid_negative_candidates[hard_neg_indices]

        for j in range(num_to_sample):
            # Append triplet (anchor_idx, positive_idx, negative_idx)
            triplets.append(
                (anchor_embeddings[i], embedding_bank[positive_idx[j]], embedding_bank[negative_idx[j]]))
    return triplets


def compute_triplet_loss(triplets, margin=1.0):  # ‼️‼️‼️‼️
    """Compute triplet loss for a batch of triplets.

    Args:
        triplets (list): List of triplets (anchor, positive, negative).
        margin (float, optional): Margin for the triplet loss. Defaults to 1.0.

    Returns:
        torch.Tensor: Computed triplet loss.
    """

    # ‼️‼️‼️‼️ TODO ‼️‼️‼️‼️
    anchor_embeddings = torch.nn.functional.normalize(
        torch.stack([t[0] for t in triplets]), dim=1)
    positive_embeddings = torch.nn.functional.normalize(
        torch.stack([t[1] for t in triplets]), dim=1)
    negative_embeddings = torch.nn.functional.normalize(
        torch.stack([t[2] for t in triplets]), dim=1)
    
    pos_dist = 1 - (anchor_embeddings * positive_embeddings).sum(dim=1)
    neg_dist = 1 - (anchor_embeddings * negative_embeddings).sum(dim=1)

    loss = torch.relu(pos_dist - neg_dist + margin).mean()

    return loss


def compute_roc_metrics(model, dataset, target_fpr=0.05, embedding_bank=None, all_labels=None):
    """
    Vectorized ROC computation for embedding-based similarity - only uses valid (non-NaN) embeddings.

    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        target_fpr: Target false positive rate (default: 0.05)
        embedding_bank: Pre-computed embedding bank [N, D] (optional, will compute if None)
        all_labels: Corresponding labels [N] (optional, will compute if None)

    Returns:
        roc_auc: Area under ROC curve
        tpr_at_fpr: True positive rate at target FPR
        fpr_values: Array of FPR values
        tpr_values: Array of TPR values
        thresholds: Array of threshold values
    """
    device = get_device()

    # Convert to tensor and remove rows with NaNs
    embedding_bank = embedding_bank.to(device)
    # rint random embedding
    all_labels = torch.tensor(all_labels, device=device)

    valid_mask = ~torch.isnan(embedding_bank).any(dim=1)
    embedding_bank = embedding_bank[valid_mask]
    all_labels = all_labels[valid_mask]

    # Normalize for cosine similarity
    E = torch.nn.functional.normalize(embedding_bank, dim=1)  # (N_valid, D)

    # Cosine similarity matrix
    S = E @ E.T  # (N_valid, N_valid)

    # Label comparison matrix
    same_class = (all_labels[:, None] == all_labels[None, :])

    # Mask: upper triangle without diagonal (to avoid self-pairs and duplicates)
    N = E.shape[0]
    mask = torch.triu(torch.ones(N, N, device=device,
                      dtype=torch.bool), diagonal=1)

    # Select valid similarity scores and labels
    y_true = same_class[mask].cpu().numpy().astype(int)
    y_scores = S[mask].cpu().numpy()

    # Compute ROC and AUC
    fpr_values, tpr_values, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr_values, tpr_values)

    # Interpolate TPR at given FPR
    tpr_at_fpr = float(np.interp(target_fpr, fpr_values, tpr_values)) if len(
        fpr_values) > 1 else 0.0

    return roc_auc, tpr_at_fpr, fpr_values, tpr_values, thresholds


def run_fold(train_loader, val_loader, config, fold_num, exp_dir, logger, model, writer):
    """Train and validate for a single fold.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        config (dict): Configuration dictionary.
        fold_num (int): Current fold number.
        exp_dir (str): Experiment directory path.
        logger (logging.Logger): Logger instance.
        model (nn.Module): Model to train and evaluate.

    Returns:
        float: Best validation score achieved during training.
    """
    logger.info(f"Training fold {fold_num}...")
    best_val_score = 0.0
    device = get_device()
    species_criterion = torch.nn.CrossEntropyLoss()  # ‼️‼️‼️‼️ Define your criterion ‼️‼️‼️‼️
    mask_criterion = torch.nn.BCEWithLogitsLoss() # ‼️‼️‼️‼️ Define your criterion ‼️‼️‼️‼️
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])  # ‼️‼️‼️‼️ Define your optimizer ‼️‼️‼️‼️
    fold_dir = os.path.join(exp_dir, "models", f"fold_{fold_num}")
    os.makedirs(fold_dir, exist_ok=True)

    # Compute initial embedding bank once before training
    logger.info("Computing initial embedding bank...")
    # ‼️‼️‼️‼️ Compute embedding bank before training for the training set ‼️‼️‼️‼️
    embedding_bank, all_labels = compute_embedding_bank(train_loader.dataset, model)
    # ‼️‼️‼️‼️ Compute embedding bank for validation set ‼️‼️‼️‼️
    embedding_bank_val, all_labels_val = compute_embedding_bank(val_loader.dataset, model)

    for epoch in range(config['num_epochs']):
        trn_loss = 0.0
        train_correct = 0
        train_total = 0
        # ‼️‼️‼️‼️ SET MODEL TO TRAIN MODE ‼️‼️‼️‼️
        model.train()

        # Training loop with tqdm progress bar
        train_pbar = tqdm(train_loader, desc=f"Fold {fold_num} - Epoch {epoch + 1}/{config['num_epochs']} [Train]",
                          leave=False, unit="batch")

        for batch_idx, (images, masks, metadata, labels, idxs) in enumerate(train_pbar):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            # ‼️‼️‼️‼️ ZERO GRAD! ‼️‼️‼️‼️
            optimizer.zero_grad()

            outputs, masks_pred, embeddings = model(images) # ‼️‼️‼️‼️ make predictions ‼️‼️‼️‼️

            # update embedding bank with current batch
            all_embeddings = embeddings.detach()
            if not torch.is_tensor(idxs):
                idxs = torch.tensor(idxs, device=images.device)
            else:
                idxs = idxs.to(images.device)
            embedding_bank[idxs] = all_embeddings

            # Classification loss
            classification_loss = species_criterion(outputs, labels) # ‼️‼️‼️‼️ compute species loss ‼️‼️‼️‼️

            # Segmentation loss - ensure both tensors are float32 and same shape
            segmentation_loss = mask_criterion(masks_pred, masks) # ‼️‼️‼️‼️ compute mask loss ‼️‼️‼️‼️

            # triplet loss
            triplets = create_triplets_batch(embeddings, labels, idxs, embedding_bank, all_labels) # ‼️‼️‼️‼️ create triplets ‼️‼️‼️‼️

            if len(triplets) > 0:
                triplet_loss = compute_triplet_loss(triplets, margin=config['triplet_margin']) # ‼️‼️‼️‼️ compute triplet loss ‼️‼️‼️‼️
            else:
                triplet_loss = torch.tensor(0.0, device=images.device)

            # Combine losses with weights
            loss = (classification_loss * config['species_loss_weight'] + 
                    segmentation_loss * config['segmentation_loss_weight'] + 
                    triplet_loss * config['triplet_loss_weight'])  # ‼️‼️‼️‼️ compute combined loss ‼️‼️‼️‼️

            # Check for invalid loss values
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected: {loss.item()}")
                continue

            # ‼️‼️‼️‼️ BACKPROPAGATE AND MAKE OPTIMIZER STEP ‼️‼️‼️‼️

            batch_loss = loss.item()
            trn_loss += batch_loss

            loss.backward()
            optimizer.step()

            # Calculate training accuracy (add this section for TensorBoard logging)
            # ‼️‼️‼️‼️ Students: calculate predictions from outputs and compare with labels ‼️‼️‼️‼️
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            # Update progress bar with current batch loss and average loss
            avg_loss = trn_loss / (batch_idx + 1)
            train_pbar.set_postfix({
                'Batch Loss': f'{batch_loss:.4f}',
                'Avg Loss': f'{avg_loss:.4f}',
                'Class Loss': f'{classification_loss.item():.4f}',
                'Seg Loss': f'{segmentation_loss.item():.4f}',
                'Triplet Loss': f'{triplet_loss.item():.4f}'
            })

        trn_loss /= len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0

        # Log training metrics to TensorBoard
        global_step = epoch * len(train_loader) + len(train_loader)
        writer.add_scalar(
            f'Loss/Train_Total_Fold{fold_num}', trn_loss, global_step)
        writer.add_scalar(
            f'Accuracy/Train_Fold{fold_num}', train_accuracy, global_step)

        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}, "
                    f"Train Loss: {trn_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        if (epoch + 1) % config['validate_every'] == 0:
            # validate the model
            # ‼️‼️‼️‼️ SET MODEL TO VALIDATION MODE ‼️‼️‼️‼️
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            # Validation loop with tqdm progress bar
            val_pbar = tqdm(val_loader, desc=f"Fold {fold_num} - Epoch {epoch + 1}/{config['num_epochs']} [Val]",
                            leave=False, unit="batch")
            precision1, recall1, iou1 = [], [], []
            precision2, recall2, iou2 = [], [], []

            with torch.no_grad():
                for batch_idx, (images, masks, metadata, labels, idxs) in enumerate(val_pbar):
                    images = images.to(device)
                    masks = masks.to(device)
                    labels = labels.to(device)

                    outputs, masks_pred, embeddings = model(images)

                    # Update embedding bank with current batch
                    all_embeddings = embeddings.detach()
                    if not torch.is_tensor(idxs):
                        idxs = torch.tensor(idxs, device=images.device)
                    else:
                        idxs = idxs.to(images.device)
                    embedding_bank_val[idxs] = all_embeddings

                    # Classification loss
                    classification_loss = species_criterion(outputs, labels) # ‼️‼️‼️‼️ compute species loss ‼️‼️‼️‼️

                    # Segmentation loss
                    segmentation_loss = mask_criterion(masks_pred, masks) # ‼️‼️‼️‼️ compute mask loss ‼️‼️‼️‼️
                    # triplet loss using updated embedding bank
                    triplets = create_triplets_batch(all_embeddings, labels, idxs, embedding_bank_val, all_labels_val) # ‼️‼️‼️‼️ create triplets ‼️‼️‼️‼️

                    if len(triplets) > 0:
                        triplet_loss = compute_triplet_loss(triplets, config['triplet_margin']) # ‼️‼️‼️‼️ compute triplet loss ‼️‼️‼️‼️
                    else:
                        triplet_loss = torch.tensor(0.0, device=images.device)

                    # Combined loss
                    loss = (classification_loss * config['species_loss_weight'] + 
                            segmentation_loss * config['segmentation_loss_weight'] + 
                            triplet_loss * config['triplet_loss_weight'])  # ‼️‼️‼️‼️ compute combined loss ‼️‼️‼️‼️

                    batch_loss = loss.item()
                    val_loss += batch_loss

                    predictions = outputs.argmax(dim=1)
                    batch_correct = (predictions == labels).sum().item()
                    correct += batch_correct
                    total += labels.size(0)

                    # Update progress bar with current batch loss and accuracy
                    avg_val_loss = val_loss / (batch_idx + 1)
                    current_acc = correct / total if total > 0 else 0

                    # Convert predictions to binary for IoU calculation
                    masks_pred_binary = torch.sigmoid(masks_pred) > 0.5
                    p1, r1, i1 = compute_seg_metrics(
                        masks_pred_binary.float(), masks, class_idx=1)
                    p2, r2, i2 = compute_seg_metrics(
                        masks_pred_binary.float(), masks, class_idx=0)
                    precision1.append(p1.item())
                    recall1.append(r1.item())
                    iou1.append(i1.item())
                    precision2.append(p2.item())
                    recall2.append(r2.item())
                    iou2.append(i2.item())

                    val_pbar.set_postfix({
                        'Batch Loss': f'{batch_loss:.4f}',
                        'Avg Loss': f'{avg_val_loss:.4f}',
                        'Accuracy': f'{current_acc:.4f}',
                        'Class Loss': f'{classification_loss.item():.4f}',
                        'Seg Loss': f'{segmentation_loss.item():.4f}',
                    })

            # Calculate average validation loss and metrics
            roc, tpr_at_fpr, fpr_values, tpr_values, thresholds = compute_roc_metrics(
                model, val_loader.dataset, target_fpr=config.get(
                    'target_fpr', 0.05),
                embedding_bank=embedding_bank_val, all_labels=all_labels_val)

            val_loss /= len(val_loader)
            val_score = correct / total

            # Log all validation metrics to TensorBoard
            global_step = epoch * len(train_loader) + len(train_loader)
            writer.add_scalar(
                f'Loss/Val_Total_Fold{fold_num}', val_loss, global_step)
            writer.add_scalar(
                f'Accuracy/Val_Fold{fold_num}', val_score, global_step)

            # IoU metrics
            writer.add_scalar(
                f'IoU/Val_Class1_Fold{fold_num}', np.mean(iou1), global_step)
            writer.add_scalar(
                f'IoU/Val_Class0_Fold{fold_num}', np.mean(iou2), global_step)

            # Precision metrics
            writer.add_scalar(
                f'Precision/Val_Class1_Fold{fold_num}', np.mean(precision1), global_step)
            writer.add_scalar(
                f'Precision/Val_Class0_Fold{fold_num}', np.mean(precision2), global_step)

            # Recall metrics
            writer.add_scalar(
                f'Recall/Val_Class1_Fold{fold_num}', np.mean(recall1), global_step)
            writer.add_scalar(
                f'Recall/Val_Class0_Fold{fold_num}', np.mean(recall2), global_step)

            # ROC metrics
            writer.add_scalar(f'ROC/AUC_Fold{fold_num}', roc, global_step)
            writer.add_scalar(
                f'ROC/TPR_at_{config.get("target_fpr", 0.05)}_FPR_Fold{fold_num}', tpr_at_fpr, global_step)

            logger.info(f"Validation Loss: {val_loss:.4f}, "
                        f"Validation Accuracy: {val_score:.4f}, "
                        f"Validation IoU: {np.mean(iou1):.4f} (class 1), "
                        f"{np.mean(iou2):.4f} (class 0), "
                        f"Validation Precision: {np.mean(precision1):.4f} (class 1), "
                        f"{np.mean(precision2):.4f} (class 0), "
                        f"Validation Recall: {np.mean(recall1):.4f} (class 1), "
                        f"{np.mean(recall2):.4f} (class 0), "
                        f"ROC AUC: {roc:.4f}, "
                        f"TPR at {config.get('target_fpr', 0.05)} FPR: {tpr_at_fpr:.4f}")
            if val_score > best_val_score:
                best_val_score = val_score
                # Save the best model
                model_path = os.path.join(fold_dir, "best_model.pth")
                torch.save(model.state_dict(), model_path)

    return best_val_score


def main():
    config_path = "confs/config.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Setup experiment directory and logging
    exp_dir, logger, writer = setup_experiment_dir(config)

    # Log configuration
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    dataset = FruitDataset(data_dir=config['data_dir'], config=config)
    labels = [dataset.metadata[i] for i in range(len(dataset))]

    fold_results = []

    # Count samples per fruit type
    fruit_counts = Counter(labels)
    logger.info(f"Total samples in dataset: {len(dataset)}")
    logger.info("Fruit distribution in dataset:")
    for fruit, count in sorted(fruit_counts.items()):
        logger.info(f"  {fruit}: {count} samples")

    # Handle single fold as 80-20 split vs multi-fold cross-validation
    if config['num_folds'] == 1:
        logger.info("Using 80-20 train-validation split (single fold)")

        train_idx, val_idx = train_test_split(
            range(len(dataset)),
            test_size=0.2,
            stratify=labels,
            random_state=config['random_seed']
        )
        splits = [(train_idx, val_idx)]
    else:
        skf = StratifiedKFold(n_splits=config['num_folds'],
                              shuffle=True, random_state=config['random_seed'])
        logger.info(
            f"Starting {config['num_folds']}-fold stratified cross-validation")
        splits = list(skf.split(range(len(dataset)), labels))

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"\n{'='*50}")
        if config['num_folds'] == 1:
            logger.info(f"Single fold (80-20 split)")
        else:
            logger.info(f"Fold {fold + 1}/{config['num_folds']}")
        logger.info(
            f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

        # Check distribution in this fold
        print_fruit_distribution(train_idx, val_idx, labels, logger)

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create data loaders for train and validation
        train_loader = DataLoader(
            train_subset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(
            val_subset, batch_size=config['batch_size'], shuffle=False)

        # Training loop for current fold
        input_dim = 786432
        model = MyModel(output_size=config['output_dim'])
        model.to(torch.device(get_device()))
        fold_result = run_fold(
            train_loader, val_loader, config, fold + 1, exp_dir, logger, model, writer)

        fold_results.append(fold_result)

    # Print overall results
    logger.info("\n" + "="*50)
    if config['num_folds'] == 1:
        logger.info("Single fold (80-20 split) Results:")
        logger.info(f"Validation Score: {fold_results[0]:.4f}")
    else:
        logger.info(
            f"{config['num_folds']}-Fold Stratified Cross-Validation Results:")
        for i, result in enumerate(fold_results):
            logger.info(f"Fold {i+1}: {result:.4f}")

    # Calculate and print average performance
    if fold_results and isinstance(fold_results[0], (int, float)):
        if config['num_folds'] == 1:
            logger.info(f"Final Validation Score: {fold_results[0]:.4f}")
        else:
            avg_result = np.mean(fold_results)
            std_result = np.std(fold_results)
            logger.info(f"\nAverage: {avg_result:.4f} ± {std_result:.4f}")

        # Save final results
        results_file = os.path.join(exp_dir, "results", "final_results.txt")
        with open(results_file, 'w') as f:
            if config['num_folds'] == 1:
                f.write("Single fold (80-20 split) Results:\n")
                f.write(f"Validation Score: {fold_results[0]:.4f}\n")
            else:
                f.write(
                    f"{config['num_folds']}-Fold Stratified Cross-Validation Results:\n")
                for i, result in enumerate(fold_results):
                    f.write(f"Fold {i+1}: {result:.4f}\n")
                f.write(f"\nAverage: {avg_result:.4f} ± {std_result:.4f}\n")

        logger.info(f"Results saved to: {results_file}")

    logger.info(f"Experiment completed. All files saved in: {exp_dir}")

    # Close TensorBoard writer
    writer.close()
    logger.info(
        f"TensorBoard logs saved to: {os.path.join(exp_dir, 'tensorboard')}")


if __name__ == "__main__":
    main()
