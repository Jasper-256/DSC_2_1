import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import ToPILImage
import os
from pathlib import Path
from tqdm import tqdm
from IPython.display import display
import torchmetrics

# Hyperparameters
N_FRAMES = 3
EPOCHS = 40
IMAGES_TO_SAVE = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
SLIDING_WINDOW_STEP = 3
TEST_EPOCH_FREQ = 4  # Test on test set every 4 epochs

def get_img_dict(img_dir):
    """Get dictionary of image files organized by type"""
    img_files = [x for x in img_dir.iterdir() if x.name.endswith('.png') or x.name.endswith('.tiff')]
    img_files.sort()

    img_dict = {}
    for img_file in img_files:
        img_type = img_file.name.split('_')[0]
        if img_type not in img_dict:
            img_dict[img_type] = []
        img_dict[img_type].append(img_file)
    return img_dict


def get_sample_dict(sample_dir):
    """Get complete dictionary structure for a sample directory"""
    camera_dirs = [x for x in sample_dir.iterdir() if 'camera' in x.name]
    camera_dirs.sort()
    
    sample_dict = {}

    for cam_dir in camera_dirs:
        cam_dict = {}
        cam_dict['scene'] = get_img_dict(cam_dir)

        obj_dirs = [x for x in cam_dir.iterdir() if 'obj_' in x.name]
        obj_dirs.sort()
        
        for obj_dir in obj_dirs:
            cam_dict[obj_dir.name] = get_img_dict(obj_dir)

        sample_dict[cam_dir.name] = cam_dict

    return sample_dict


class MultiSceneDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None, verbose=False, save_plots=False, plot_save_dir='./plots', n_frames=N_FRAMES):
        """
        Dataset for loading RGB images with modal and amodal masks from multiple scene folders.
        Modified for Task 2.1 to handle video sequences.

        Args:
            data_root (str): Root directory containing train and test folders
            split (str): Either 'train' or 'test'
            transform (callable, optional): Transform to apply to RGB images
            verbose (bool): If True, plot images during iteration
            save_plots (bool): If True, save plots instead of displaying them
            plot_save_dir (str): Directory to save plots when save_plots=True
            n_frames (int): Number of consecutive frames to load as a sequence
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.verbose = verbose
        self.save_plots = save_plots
        self.plot_save_dir = plot_save_dir
        self.n_frames = n_frames
        self.data_samples = []

        # Create plot directory if saving plots
        if self.save_plots:
            os.makedirs(self.plot_save_dir, exist_ok=True)

        # Get the split directory (train or test)
        split_dir = self.data_root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")

        print(f"Loading {split} data from {split_dir}")
        
        # Get all scene directories in the split
        scene_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        scene_dirs.sort()
        
        print(f"Found {len(scene_dirs)} scene directories in {split} split")

        # Track filtering statistics
        total_sequences_checked = 0
        filtered_sequences = 0

        # Process each scene
        for scene_dir in scene_dirs:
            print(f"Processing scene: {scene_dir.name}")
            sample_dict = get_sample_dict(scene_dir)
            
            # For each camera in the scene
            for cam_name, cam_data in sample_dict.items():
                # For each object in the camera
                obj_names = [name for name in cam_data.keys() if name.startswith('obj_')]
                
                for obj_name in obj_names:
                    obj_id = int(obj_name.split('_')[1])
                    
                    # For video sequences: we need at least 2*n_frames+1 consecutive frames
                    # where n_frames before + center frame + n_frames after
                    num_frames = len(cam_data['scene']['rgba'])
                    total_frames_needed = 2 * self.n_frames + 1
                    
                    # Create sequences centered around target frames
                    # Center frame can range from n_frames to num_frames-n_frames-1
                    min_center_frame = self.n_frames
                    max_center_frame = num_frames - self.n_frames - 1
                    
                    for center_frame in range(min_center_frame, max_center_frame + 1, SLIDING_WINDOW_STEP):
                        if center_frame - self.n_frames < 0 or center_frame + self.n_frames >= num_frames:
                            break
                            
                        try:
                            total_sequences_checked += 1
                            
                            # Collect paths for this sequence
                            sequence_data = {
                                'rgb_paths': [],
                                'modal_paths': [],
                                'amodal_paths': [],
                                'obj_id': obj_id,
                                'scene_name': scene_dir.name,
                                'camera_name': cam_name,
                                'object_name': obj_name,
                                'center_frame': center_frame
                            }
                            
                            # Collect paths for all frames in sequence (n_frames before + center + n_frames after)
                            start_frame = center_frame - self.n_frames
                            end_frame = center_frame + self.n_frames + 1
                            for frame_idx in range(start_frame, end_frame):
                                rgb_path = cam_data['scene']['rgba'][frame_idx]
                                modal_path = cam_data['scene']['segmentation'][frame_idx]
                                amodal_path = cam_data[obj_name]['segmentation'][frame_idx]
                                
                                sequence_data['rgb_paths'].append(str(rgb_path))
                                sequence_data['modal_paths'].append(str(modal_path))
                                sequence_data['amodal_paths'].append(str(amodal_path))
                            
                            # Check if center frame has valid modal mask (not all black)
                            center_modal_path = sequence_data['modal_paths'][self.n_frames]  # Center frame is at index n_frames
                            if self._is_valid_center_frame(center_modal_path, obj_id):
                                self.data_samples.append(sequence_data)
                            else:
                                filtered_sequences += 1
                                
                        except (IndexError, KeyError) as e:
                            # Skip if files are missing
                            continue

        print(f"Total video sequences checked: {total_sequences_checked}")
        print(f"Filtered out sequences (all-black modal masks): {filtered_sequences}")
        print(f"Valid video sequences in {split} split: {len(self.data_samples)}")

    def _is_valid_center_frame(self, modal_path, obj_id):
        """
        Check if the center frame has a non-black modal mask.
        
        Args:
            modal_path (str): Path to modal mask file for center frame
            obj_id (int): Object ID to check for in the modal mask
            
        Returns:
            bool: True if center frame has a non-black modal mask, False otherwise
        """
        try:
            # Load modal mask
            modal_mask_img = Image.open(modal_path)
            modal_mask_np = np.array(modal_mask_img)
            
            # Create binary mask for this object
            modal_mask_binary = np.where(modal_mask_np == obj_id, 1, 0)
            
            # Check if mask has any positive pixels
            return np.sum(modal_mask_binary) > 0
                
        except Exception as e:
            # If we can't load the mask, consider it invalid
            return False

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample_info = self.data_samples[idx]

        # Load sequence of RGB images (2*n_frames+1 total)
        rgb_sequences = []
        modal_sequences = []
        # Load amodal masks for all frames (for visualization), but only use center for training
        amodal_sequences = []
        total_frames = 2 * self.n_frames + 1

        # Get target size from transform or use original size
        target_size = None
        if self.transform:
            # Extract target size from transforms
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    if isinstance(t.size, int):
                        target_size = (t.size, t.size)
                    elif isinstance(t.size, (list, tuple)) and len(t.size) == 2:
                        target_size = (int(t.size[0]), int(t.size[1]))
                    break

        # Load all frames in the sequence (2*n_frames+1 total frames)
        for seq_idx in range(total_frames):
            # Load RGB image
            rgb_image = Image.open(sample_info['rgb_paths'][seq_idx]).convert('RGB')
            
            if target_size is None:
                target_size = rgb_image.size[::-1]  # (H, W)
            
            # Ensure target_size is a valid tuple of ints
            if not isinstance(target_size, tuple) or len(target_size) != 2:
                target_size = rgb_image.size[::-1]  # Fall back to original size
            target_size = (int(target_size[0]), int(target_size[1]))

            # Load and process modal mask for all frames
            modal_mask_img = Image.open(sample_info['modal_paths'][seq_idx])
            modal_mask_np = np.array(modal_mask_img)
            modal_mask_binary = np.where(modal_mask_np == sample_info['obj_id'], 1, 0)

            # Resize modal mask to match image size
            modal_mask_pil = Image.fromarray(modal_mask_binary.astype(np.uint8))
            modal_mask_resized = modal_mask_pil.resize((target_size[1], target_size[0]), Image.Resampling.NEAREST)
            modal_mask = torch.from_numpy(np.array(modal_mask_resized)).float()

            # Load and process amodal mask for all frames (for visualization)
            amodal_mask_img = Image.open(sample_info['amodal_paths'][seq_idx])
            amodal_mask_np = np.array(amodal_mask_img) / 255.0  # Convert 0,255 to 0,1

            # Resize amodal mask to match image size
            amodal_mask_pil = Image.fromarray((amodal_mask_np * 255).astype(np.uint8))
            amodal_mask_resized = amodal_mask_pil.resize((target_size[1], target_size[0]), Image.Resampling.NEAREST)
            amodal_mask = torch.from_numpy(np.array(amodal_mask_resized) / 255.0).float()

            # Apply transforms to RGB image
            if self.transform:
                rgb_image = self.transform(rgb_image)
            else:
                rgb_image = transforms.ToTensor()(rgb_image)

            rgb_sequences.append(rgb_image)
            modal_sequences.append(modal_mask)
            amodal_sequences.append(amodal_mask)

        # Stack sequences: (T, C, H, W) where T is temporal dimension
        rgb_sequence = torch.stack(rgb_sequences, dim=0)  # (2*N_FRAMES+1, 3, H, W)
        modal_sequence = torch.stack(modal_sequences, dim=0)  # (2*N_FRAMES+1, H, W)
        amodal_sequence_full = torch.stack(amodal_sequences, dim=0)  # (2*N_FRAMES+1, H, W)
        
        # Extract only center frame amodal mask for training (but return full sequence for visualization)
        center_amodal_mask = amodal_sequence_full[self.n_frames:self.n_frames+1]  # (1, H, W)
        
        # Ensure target_size is valid
        if target_size is None:
            target_size = (256, 256)  # Default fallback size

        # Verbose plotting (only plot the center frame of sequence for now)
        if self.verbose:
            if self.save_plots:
                self.save_sample_plot(rgb_sequence[self.n_frames], modal_sequence[self.n_frames], amodal_sequence_full[self.n_frames], idx)
            else:
                self.plot_sample(rgb_sequence[self.n_frames], modal_sequence[self.n_frames], amodal_sequence_full[self.n_frames], idx)

        # Return: full RGB and modal sequences, center frame amodal for training, full amodal sequence for visualization
        return rgb_sequence, modal_sequence, center_amodal_mask, amodal_sequence_full

    def save_sample_plot(self, rgb_image, modal_mask, amodal_mask, idx):
        """Save RGB image, modal mask, and amodal mask in subplots to file"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot RGB image
        if rgb_image.shape[0] == 3:  # If tensor is CxHxW
            rgb_np = rgb_image.permute(1, 2, 0).numpy()
        else:
            rgb_np = rgb_image.numpy()

        axes[0].imshow(rgb_np)
        axes[0].set_title(f'RGB Image (Sample {idx})')
        axes[0].axis('off')

        # Plot modal mask
        axes[1].imshow(modal_mask.numpy(), cmap='gray')
        axes[1].set_title(f'Modal Mask (Obj ID: {self.data_samples[idx]["obj_id"]})')
        axes[1].axis('off')

        # Plot amodal mask
        axes[2].imshow(amodal_mask.numpy(), cmap='gray')
        axes[2].set_title('Amodal Mask')
        axes[2].axis('off')

        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(self.plot_save_dir, f'sample_{idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close to prevent memory leaks
        print(f"Plot saved: {save_path}")

    def plot_sample(self, rgb_image, modal_mask, amodal_mask, idx):
        """Plot RGB image, modal mask, and amodal mask in subplots"""
        matplotlib.use('Agg')  # Use non-interactive backend temporarily

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot RGB image
        if rgb_image.shape[0] == 3:  # If tensor is CxHxW
            rgb_np = rgb_image.permute(1, 2, 0).numpy()
        else:
            rgb_np = rgb_image.numpy()

        axes[0].imshow(rgb_np)
        axes[0].set_title(f'RGB Image (Sample {idx})')
        axes[0].axis('off')

        # Plot modal mask
        axes[1].imshow(modal_mask.numpy(), cmap='gray')
        axes[1].set_title(f'Modal Mask (Obj ID: {self.data_samples[idx]["obj_id"]})')
        axes[1].axis('off')

        # Plot amodal mask
        axes[2].imshow(amodal_mask.numpy(), cmap='gray')
        axes[2].set_title('Amodal Mask')
        axes[2].axis('off')

        plt.tight_layout()

        # Force display in Jupyter
        display(fig)
        plt.close(fig)  # Close to prevent memory leaks


def get_transforms(image_size):
    """
    Get basic transformations without normalization

    Args:
        image_size (int): Target image size for resizing

    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])


# Set random seed for reproducible results
torch.manual_seed(42)
np.random.seed(42)

# Create datasets for train and test
print("Creating video datasets...")
data_root = './'  # Current directory containing train and test folders
transform = get_transforms(image_size=256)

# Create datasets first - now using video sequences
train_dataset = MultiSceneDataset(
    data_root=data_root,
    split='train',
    transform=transform,
    verbose=False,
    save_plots=False,
    plot_save_dir='./plots',
    n_frames=N_FRAMES
)

test_dataset = MultiSceneDataset(
    data_root=data_root,
    split='test',
    transform=transform,
    verbose=False,
    save_plots=False,
    plot_save_dir='./plots',
    n_frames=N_FRAMES
)

# Create dataloaders
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # Set to 0 for debugging
    pin_memory=torch.cuda.is_available()
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(f"Train batches: {len(train_dataloader)}")
print(f"Test batches: {len(test_dataloader)}")
print("-" * 50)

# Test each dataloader
print("\nTesting Train dataloader:")
for batch_idx, (images, modal_masks, amodal_masks, amodal_masks_full) in enumerate(train_dataloader):
    print(f"  Batch {batch_idx}:")
    print(f"    Images shape: {images.shape}")
    print(f"    Modal masks shape: {modal_masks.shape}")
    print(f"    Amodal masks (center) shape: {amodal_masks.shape}")
    print(f"    Amodal masks (full) shape: {amodal_masks_full.shape}")
    print(f"    Modal mask unique values: {torch.unique(modal_masks)}")
    print(f"    Amodal mask (center) unique values: {torch.unique(amodal_masks)}")
    print(f"    Amodal mask (full) unique values: {torch.unique(amodal_masks_full)}")
    print(f"    Images dtype: {images.dtype}")
    print(f"    Modal masks dtype: {modal_masks.dtype}")
    print(f"    Amodal masks dtype: {amodal_masks.dtype}")
    # Only test first batch
    break

print("\nTesting Test dataloader:")
for batch_idx, (images, modal_masks, amodal_masks, amodal_masks_full) in enumerate(test_dataloader):
    print(f"  Batch {batch_idx}:")
    print(f"    Images shape: {images.shape}")
    print(f"    Modal masks shape: {modal_masks.shape}")
    print(f"    Amodal masks (center) shape: {amodal_masks.shape}")
    print(f"    Amodal masks (full) shape: {amodal_masks_full.shape}")
    print(f"    Modal mask unique values: {torch.unique(modal_masks)}")
    print(f"    Amodal mask (center) unique values: {torch.unique(amodal_masks)}")
    print(f"    Amodal mask (full) unique values: {torch.unique(amodal_masks_full)}")
    print(f"    Images dtype: {images.dtype}")
    print(f"    Modal masks dtype: {modal_masks.dtype}")
    print(f"    Amodal masks dtype: {amodal_masks.dtype}")
    # Only test first batch
    break

print("-" * 50)
print("Dataloader test completed!")


# Lightweight 3D U-Net for Task 2.1: Video-based Modal Mask -> Amodal Mask (Center Frame Prediction)
class Lightweight3DUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, n_frames=N_FRAMES):
        super(Lightweight3DUNet, self).__init__()
        self.n_frames = n_frames

        # Much smaller channel sizes for lightweight model
        # Encoder (downsampling path) - using 3D convolutions
        self.enc1 = self.conv3d_block(in_channels, 16)
        self.enc2 = self.conv3d_block(16, 32)
        self.enc3 = self.conv3d_block(32, 64)

        # Bottleneck (smaller)
        self.bottleneck = self.conv3d_block(64, 128)

        # Decoder (upsampling path) - using 3D transpose convolutions
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = self.conv3d_block(128, 64)  # 64 + 64 from skip connection

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = self.conv3d_block(64, 32)   # 32 + 32 from skip connection

        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = self.conv3d_block(32, 16)   # 16 + 16 from skip connection

        # Final layer
        self.final = nn.Conv3d(16, out_channels, kernel_size=1)

        # Max pooling (only spatial dimensions, preserve temporal)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def conv3d_block(self, in_channels, out_channels):
        """Single 3D convolution with ReLU and batch norm"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input shape: (B, C, T, H, W) where T is temporal dimension (2*n_frames+1)
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Final output for all frames
        output = self.final(dec1)  # (B, 1, T, H, W)
        
        # Extract only the center frame (index n_frames in temporal dimension)
        center_frame_output = output[:, :, self.n_frames:self.n_frames+1, :, :]  # (B, 1, 1, H, W)
        
        return center_frame_output

# Test the 3D model
print("Testing Lightweight 3D U-Net model...")
test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_3d = Lightweight3DUNet(in_channels=4, out_channels=1, n_frames=N_FRAMES).to(test_device)
total_frames = 2 * N_FRAMES + 1
test_input_3d = torch.randn(1, 4, total_frames, 256, 256).to(test_device)  # Batch size 1, 4 channels (RGB + modal mask), total_frames temporal, 256x256
test_output_3d = model_3d(test_input_3d)
print(f"Input shape: {test_input_3d.shape}")
print(f"Output shape: {test_output_3d.shape}")
print(f"Total frames processed: {total_frames} (2*{N_FRAMES}+1)")
print(f"Predicting only center frame (index {N_FRAMES})")

# Count parameters
total_params_3d = sum(p.numel() for p in model_3d.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params_3d:,}")
print("3D model created successfully for video processing!")


# Training utilities - Using TorchMetrics
def create_metrics(device):
    """Create TorchMetrics objects for evaluation"""
    return {
        'accuracy': torchmetrics.Accuracy(task='binary', threshold=0.5).to(device),
        'iou': torchmetrics.JaccardIndex(task='binary', threshold=0.5).to(device),
        'miou': torchmetrics.JaccardIndex(task='binary', threshold=0.5, average='macro').to(device)
    }

def calculate_metrics(predictions, targets, metrics_dict):
    """Calculate all metrics using TorchMetrics"""
    # Apply sigmoid to predictions to get probabilities
    pred_probs = torch.sigmoid(predictions)
    
    # Calculate metrics
    accuracy = metrics_dict['accuracy'](pred_probs, targets.int())
    iou = metrics_dict['iou'](pred_probs, targets.int())
    miou = metrics_dict['miou'](pred_probs, targets.int())
    
    return accuracy.item(), iou.item(), miou.item()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch with video sequences"""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    total_miou = 0.0
    num_batches = 0
    
    # Create metrics for this epoch
    metrics = create_metrics(device)

    for rgb_sequences, modal_sequences, amodal_sequences, amodal_sequences_full in tqdm(dataloader, desc="Training"):
        # Move data to device
        # rgb_sequences: (B, 2*N_FRAMES+1, C, H, W) -> need to transpose to (B, C, 2*N_FRAMES+1, H, W)
        # modal_sequences: (B, 2*N_FRAMES+1, H, W) -> need to add channel dim and transpose to (B, 1, 2*N_FRAMES+1, H, W)  
        # amodal_sequences: (B, 1, H, W) -> need to add temporal dim to (B, 1, 1, H, W)
        
        rgb_sequences = rgb_sequences.permute(0, 2, 1, 3, 4).to(device)  # (B, C, 2*N_FRAMES+1, H, W)
        modal_sequences = modal_sequences.unsqueeze(2).permute(0, 2, 1, 3, 4).to(device)  # (B, 1, 2*N_FRAMES+1, H, W)
        amodal_sequences = amodal_sequences.unsqueeze(2).to(device)  # (B, 1, 1, H, W)

        # Combine RGB and modal mask as input (4 channels total: 3 RGB + 1 modal)
        inputs = torch.cat([rgb_sequences, modal_sequences], dim=1)  # (B, 4, 2*N_FRAMES+1, H, W)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)  # (B, 1, 1, H, W) - only center frame
        loss = criterion(outputs, amodal_sequences)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metrics using TorchMetrics
        accuracy, iou, miou = calculate_metrics(outputs, amodal_sequences, metrics)
        total_loss += loss.item()
        total_accuracy += accuracy
        total_iou += iou
        total_miou += miou
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_iou = total_iou / num_batches
    avg_miou = total_miou / num_batches

    return avg_loss, avg_accuracy, avg_iou, avg_miou

def test_epoch(model, dataloader, criterion, device):
    """Test the model for one epoch with video sequences"""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    total_miou = 0.0
    num_batches = 0
    
    # Create metrics for this epoch
    metrics = create_metrics(device)

    with torch.no_grad():
        for rgb_sequences, modal_sequences, amodal_sequences, amodal_sequences_full in tqdm(dataloader, desc="Testing"):
            # Move data to device and rearrange dimensions
            rgb_sequences = rgb_sequences.permute(0, 2, 1, 3, 4).to(device)  # (B, C, 2*N_FRAMES+1, H, W)
            modal_sequences = modal_sequences.unsqueeze(2).permute(0, 2, 1, 3, 4).to(device)  # (B, 1, 2*N_FRAMES+1, H, W)
            amodal_sequences = amodal_sequences.unsqueeze(2).to(device)  # (B, 1, 1, H, W)

            # Combine RGB and modal mask as input
            inputs = torch.cat([rgb_sequences, modal_sequences], dim=1)  # (B, 4, 2*N_FRAMES+1, H, W)

            # Forward pass
            outputs = model(inputs)  # (B, 1, 1, H, W) - only center frame
            loss = criterion(outputs, amodal_sequences)

            # Calculate metrics using TorchMetrics
            accuracy, iou, miou = calculate_metrics(outputs, amodal_sequences, metrics)
            total_loss += loss.item()
            total_accuracy += accuracy
            total_iou += iou
            total_miou += miou
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_iou = total_iou / num_batches
    avg_miou = total_miou / num_batches

    return avg_loss, avg_accuracy, avg_iou, avg_miou


print("Training utilities loaded successfully!")


# Training setup and execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model, loss, and optimizer
model = Lightweight3DUNet(in_channels=4, out_channels=1, n_frames=N_FRAMES).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training parameters
num_epochs = EPOCHS
best_train_iou = 0.0
best_test_iou = 0.0
train_losses = []
train_accuracies = []
train_ious = []
test_losses = []
test_accuracies = []
test_ious = []

print(f"Starting training for {num_epochs} epochs with lightweight model...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 30)

    # Training phase
    train_loss, train_acc, train_iou, train_miou = train_epoch(model, train_dataloader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    train_ious.append(train_iou)

    print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, IoU: {train_iou:.4f}, mIoU: {train_miou:.4f}")

    # Test evaluation every TEST_EPOCH_FREQ epochs
    if (epoch + 1) % TEST_EPOCH_FREQ == 0:
        print(f"\n{'='*50}")
        print(f"TEST EVALUATION - EPOCH {epoch+1}")
        print(f"{'='*50}")
        
        test_loss, test_acc, test_iou, test_miou = test_epoch(model, test_dataloader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        test_ious.append(test_iou)
        
        print(f"Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, IoU: {test_iou:.4f}, mIoU: {test_miou:.4f}")
        
        # Performance comparison table
        print(f"\n{'Metric':<12} {'Train':<10} {'Test':<10} {'Difference':<12}")
        print("-" * 50)
        print(f"{'Loss':<12} {train_loss:<10.4f} {test_loss:<10.4f} {abs(train_loss - test_loss):<12.4f}")
        print(f"{'Accuracy':<12} {train_acc:<10.4f} {test_acc:<10.4f} {abs(train_acc - test_acc):<12.4f}")
        print(f"{'IoU':<12} {train_iou:<10.4f} {test_iou:<10.4f} {abs(train_iou - test_iou):<12.4f}")
        print(f"{'mIoU':<12} {train_miou:<10.4f} {test_miou:<10.4f} {abs(train_miou - test_miou):<12.4f}")
        
        # Track best test performance
        if test_iou > best_test_iou:
            best_test_iou = test_iou
            torch.save(model.state_dict(), 'best_test_lightweight_3dunet_task2_1.pth')
            print(f"\n🎯 NEW BEST TEST IoU: {best_test_iou:.4f} - Model saved as 'best_test_lightweight_3dunet_task2_1.pth'!")
        
        print(f"{'='*50}")

    # Save best model based on training IoU
    if train_iou > best_train_iou:
        best_train_iou = train_iou
        torch.save(model.state_dict(), 'best_lightweight_3dunet_task2_1.pth')
        print(f"New best training IoU: {best_train_iou:.4f} - 3D Model saved!")



print("\nTraining completed!")
print(f"Best training IoU: {best_train_iou:.4f}")
if best_test_iou > 0:
    print(f"Best test IoU (during training): {best_test_iou:.4f}")
else:
    print("No test evaluation performed during training")

# Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loss
axes[0].plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
if len(test_losses) > 0:
    test_epochs = list(range(TEST_EPOCH_FREQ, len(train_losses) + 1, TEST_EPOCH_FREQ))
    axes[0].plot(test_epochs, test_losses, label='Test Loss', marker='s', color='red')
axes[0].set_title('Training & Test Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy
axes[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', marker='o')
if len(test_accuracies) > 0:
    test_epochs = list(range(TEST_EPOCH_FREQ, len(train_accuracies) + 1, TEST_EPOCH_FREQ))
    axes[1].plot(test_epochs, test_accuracies, label='Test Accuracy', marker='s', color='red')
axes[1].set_title('Training & Test Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

# IoU
axes[2].plot(range(1, len(train_ious) + 1), train_ious, label='Train IoU', marker='o')
if len(test_ious) > 0:
    test_epochs = list(range(TEST_EPOCH_FREQ, len(train_ious) + 1, TEST_EPOCH_FREQ))
    axes[2].plot(test_epochs, test_ious, label='Test IoU', marker='s', color='red')
axes[2].set_title('Training & Test IoU')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('IoU')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()


# Final testing phase
print("=" * 50)
print("FINAL TESTING PHASE")
print("=" * 50)

# Load the best model
model.load_state_dict(torch.load('best_lightweight_3dunet_task2_1.pth'))
print("Loaded best 3D model from training")

# Final test on test set
final_test_loss, final_test_acc, final_test_iou, final_test_miou = test_epoch(model, test_dataloader, criterion, device)

print(f"\nFinal Test Results:")
print(f"Test Loss: {final_test_loss:.4f}")
print(f"Test Accuracy: {final_test_acc:.4f}")
print(f"Test IoU: {final_test_iou:.4f}")
print(f"Test mIoU: {final_test_miou:.4f}")

# Create a comparison table
print(f"\nPerformance Summary:")
print(f"{'Split':<15} {'Loss':<10} {'Accuracy':<10} {'IoU':<10} {'mIoU':<10}")
print("-" * 65)
print(f"{'Final Train':<15} {train_losses[-1]:<10.4f} {train_accuracies[-1]:<10.4f} {train_ious[-1]:<10.4f} {train_miou:<10.4f}")
print(f"{'Final Test':<15} {final_test_loss:<10.4f} {final_test_acc:<10.4f} {final_test_iou:<10.4f} {final_test_miou:<10.4f}")

# Visualize predictions on test set - save multiple different video sequence examples  
total_frames = 2 * N_FRAMES + 1
center_frame_idx = N_FRAMES
print(f"\nSaving multiple organized video sequence visualizations:")
print(f"  - Layout: {total_frames} rows (frames) x 5 columns (RGB | Modal | GT Amodal | Pred Continuous | Pred Binary)")
print(f"  - Ground truth amodal mask shown ONLY for center frame (frame {center_frame_idx+1}/{total_frames})")
print(f"  - Predictions shown only on center frame (frame {center_frame_idx+1}/{total_frames})")
print(f"  - Processing {N_FRAMES} frames before + 1 center + {N_FRAMES} frames after = {total_frames} total frames")
print(f"  - Saving to test_video_predictions/ ...")

def save_multiple_predictions(model, dataloader, device, num_images=20, threshold=0.5, save_dir="test_predictions"):
    """Save multiple individual prediction images with organized temporal layout - OPTIMIZED VERSION"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    saved_count = 0
    total_frames = 2 * N_FRAMES + 1
    center_frame_idx = N_FRAMES
    
    # Create progress bar for saving predictions
    pbar = tqdm(total=num_images, desc="Saving video predictions", unit="images")
    
    with torch.no_grad():
        for batch_idx, (rgb_sequences, modal_sequences, amodal_sequences, amodal_sequences_full) in enumerate(dataloader):
            if saved_count >= num_images:
                break
                
            batch_size = rgb_sequences.shape[0]
            
            # Rearrange dimensions and move to device
            rgb_sequences = rgb_sequences.permute(0, 2, 1, 3, 4).to(device)  # (B, C, 2*N_FRAMES+1, H, W)
            modal_sequences = modal_sequences.unsqueeze(2).permute(0, 2, 1, 3, 4).to(device)  # (B, 1, 2*N_FRAMES+1, H, W)
            amodal_sequences = amodal_sequences.unsqueeze(2).to(device)  # (B, 1, 1, H, W) - center frame only
            amodal_sequences_full = amodal_sequences_full.permute(0, 1, 2, 3).to(device)  # (B, 2*N_FRAMES+1, H, W) - all frames
            
            # Combine RGB and modal mask as input
            inputs = torch.cat([rgb_sequences, modal_sequences], dim=1)
            
            # Get predictions
            outputs = model(inputs)  # (B, 1, 1, H, W) - only center frame
            predictions = torch.sigmoid(outputs)
            predictions_binary = (predictions > threshold).float()
            
            rgb_vis = rgb_sequences.permute(0, 2, 1, 3, 4).cpu()  # (B, 2*N_FRAMES+1, C, H, W)
            modal_vis = modal_sequences.squeeze(1).permute(0, 1, 2, 3).cpu()  # (B, 2*N_FRAMES+1, H, W)
            amodal_vis_full = amodal_sequences_full.cpu()  # (B, 2*N_FRAMES+1, H, W) - all ground truth frames
            pred_center = predictions.squeeze(1).squeeze(1).cpu()  # (B, H, W) - prediction for center frame only
            pred_binary_center = predictions_binary.squeeze(1).squeeze(1).cpu()  # (B, H, W)
            
            # Save each sample in the batch as a separate image
            for i in range(batch_size):
                if saved_count >= num_images:
                    break
                
                fig, axes = plt.subplots(total_frames, 5, figsize=(15, 3*total_frames))
                if total_frames == 1:
                    axes = axes.reshape(1, -1)
                
                for frame_idx in range(total_frames):
                    # Column 0: RGB Image
                    img_np = rgb_vis[i, frame_idx].permute(1, 2, 0).numpy()
                    axes[frame_idx, 0].imshow(img_np)
                    axes[frame_idx, 0].set_title(f'RGB {frame_idx+1}', fontsize=10)  # Smaller font
                    axes[frame_idx, 0].axis('off')
                    
                    # Column 1: Modal Mask
                    modal = modal_vis[i, frame_idx].numpy()
                    axes[frame_idx, 1].imshow(modal, cmap='gray')
                    axes[frame_idx, 1].set_title(f'Modal Mask {frame_idx+1}', fontsize=10)
                    axes[frame_idx, 1].axis('off')
                    
                    # Column 2: Ground Truth Amodal (only for center frame)
                    if frame_idx == center_frame_idx:  # Center frame
                        amodal_gt = amodal_vis_full[i, frame_idx].numpy()
                        axes[frame_idx, 2].imshow(amodal_gt, cmap='gray')
                        axes[frame_idx, 2].set_title('GT Amodal Mask', fontsize=10)
                        axes[frame_idx, 2].axis('off')
                    else:
                        # Empty column for non-center frames
                        axes[frame_idx, 2].axis('off')
                        axes[frame_idx, 2].set_title('', fontsize=10)
                    
                    # Column 3 & 4: Predictions (only on center frame)
                    if frame_idx == center_frame_idx:  # Center frame
                        # Predicted amodal mask (continuous)
                        amodal_pred = pred_center[i].numpy()
                        axes[frame_idx, 3].imshow(amodal_pred, cmap='gray')
                        axes[frame_idx, 3].set_title('Predicted Raw', fontsize=10)
                        axes[frame_idx, 3].axis('off')
                        
                        # Predicted amodal mask (binary)
                        amodal_pred_binary = pred_binary_center[i].numpy()
                        axes[frame_idx, 4].imshow(amodal_pred_binary, cmap='gray')
                        axes[frame_idx, 4].set_title(f'Predicted Binary', fontsize=10)
                        axes[frame_idx, 4].axis('off')
                    else:
                        # Empty columns for non-center frames
                        axes[frame_idx, 3].axis('off')
                        axes[frame_idx, 3].set_title('', fontsize=10)
                        axes[frame_idx, 4].axis('off')
                        axes[frame_idx, 4].set_title('', fontsize=10)
                
                fig.suptitle(f'Task 2.1 Sample {saved_count+1} - Amodal Mask Prediction', fontsize=12)
                plt.tight_layout(rect=(0, 0, 1, 0.97))  # Leave space for title at top
                
                save_path = os.path.join(save_dir, f'video_sequence_{saved_count:03d}.png')
                plt.savefig(save_path, dpi=150)
                plt.close(fig)
                
                saved_count += 1
                pbar.update(1)
    
    pbar.close()
    print(f"Saved {saved_count} optimized video sequence visualizations to {save_dir}/")

# Save prediction examples
save_multiple_predictions(
    model,
    test_dataloader,
    device,
    num_images=IMAGES_TO_SAVE,
    save_dir="test_video_predictions"
)

print("\n" + "=" * 50)
print("TASK 2.1 COMPLETED SUCCESSFULLY!")
print("=" * 50)
print(f"✓ Model Architecture: Lightweight 3D U-Net (Center Frame Prediction)")
print(f"✓ Input: RGB Sequences ({2*N_FRAMES+1} frames, 3 channels) + Modal Mask Sequences ({2*N_FRAMES+1} frames, 1 channel) = 4 channels")
print(f"✓ Output: Amodal Mask for Center Frame ONLY (1 frame, 1 channel)")
print(f"✓ Data Structure: Video sequences of {2*N_FRAMES+1} frames ({N_FRAMES} before + 1 center + {N_FRAMES} after)")
print(f"✓ Temporal Processing: 3D Convolutions for spatiotemporal feature learning")
print(f"✓ Prediction Target: Only center frame (frame {N_FRAMES+1} out of {2*N_FRAMES+1})")
print(f"✓ Best Training IoU: {best_train_iou:.4f}")
if best_test_iou > 0:
    print(f"✓ Best Test IoU (during training): {best_test_iou:.4f}")
print(f"✓ Final Test Accuracy: {final_test_acc:.4f}")
print(f"✓ Final Test IoU: {final_test_iou:.4f}")
print(f"✓ Final Test mIoU: {final_test_miou:.4f}")
print(f"✓ Test Evaluation Frequency: Every {TEST_EPOCH_FREQ} epochs")
print(f"✓ Total Training Sequences: {len(train_dataset)}")
print(f"✓ Total Test Sequences: {len(test_dataset)}")
print(f"✓ Context Frames: {N_FRAMES} before + {N_FRAMES} after = {2*N_FRAMES} context frames")
print("=" * 50)
