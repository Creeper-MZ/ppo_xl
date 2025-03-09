import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Import the aesthetic model definition from your code
from pixiv_reward_model import PixivRewardModel

class AestheticRewardModel:
    """Wrapper for the Pixiv aesthetic scoring model to use as a reward function"""
    
    def __init__(self, model_path="output/final_model.pth", device=None):
        """
        Initialize the aesthetic reward model
        
        Args:
            model_path: Path to the pretrained model weights
            device: Device to run the model on (defaults to CUDA if available)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Loading aesthetic reward model from {model_path}")
        
        # Create model instance
        self.model = PixivRewardModel(pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle DataParallel state dict if necessary
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        
        # Move to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.img_size = 384  # Match the model's expected input size
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Aesthetic reward model loaded successfully")
    
    def preprocess_image(self, image):
        """
        Preprocess an image for the model
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Preprocessed tensor ready for the model
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image or path to an image")
            
        img_tensor = self.transform(image)
        return img_tensor.unsqueeze(0)  # Add batch dimension
    
    def get_score(self, image):
        """
        Get the raw aesthetic score for an image
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Aesthetic score (0-1000 range)
        """
        img_tensor = self.preprocess_image(image)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            score = outputs['bookmark_prediction'].item()
            
        # Scale the score to the 0-1000 range
        score = min(max(score, 0), 1000)
        
        return score
    
    def get_reward(self, image):
        """
        Calculate the reward value based on the aesthetic score
        Designed to peak for scores between 200-500
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Reward value (higher is better)
        """
        score = self.get_score(image)
        
        # Reward function that peaks between 200-500
        if score < 200:
            # Linear increase from 0 to 1 as score goes from 0 to 200
            reward = score / 200.0
        elif 200 <= score <= 500:
            # Maximum reward (1.0) for scores in the target range
            reward = 1.0
        else:
            # Exponential decay for scores above 500
            reward = np.exp(-0.002 * (score - 500))
        
        return reward, score
    
    def batch_rewards(self, images):
        """
        Get rewards for a batch of images
        
        Args:
            images: List of PIL Images or paths
            
        Returns:
            Tuple of (rewards, scores)
        """
        rewards = []
        scores = []
        
        for image in images:
            reward, score = self.get_reward(image)
            rewards.append(reward)
            scores.append(score)
            
        return rewards, scores