import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import Block
from collections import OrderedDict

class PixivRewardModel(nn.Module):
    """Pixiv bookmark prediction model"""
    def __init__(self, 
                 model_name="eva02_large_patch14_clip_224.merged2b",
                 img_size=384,
                 pretrained=False,
                 dropout=0.1):
        super().__init__()
        
        # 1. Base vision model (EVA-CLIP)
        if pretrained:
            print(f"Loading pretrained vision encoder: {model_name}")
            self.vision_encoder = timm.create_model(
                model_name, 
                pretrained=True,
                num_classes=0,
                img_size=img_size
            )
        else:
            print(f"Initializing vision encoder: {model_name} (without pretrained weights)")
            self.vision_encoder = timm.create_model(
                model_name, 
                pretrained=False,
                num_classes=0,
                img_size=img_size
            )
        
        # Get hidden dimension
        if hasattr(self.vision_encoder, 'embed_dim'):
            self.hidden_dim = self.vision_encoder.embed_dim
        else:
            # Fallback
            self.hidden_dim = 1024
            print(f"Could not find embed_dim, using default {self.hidden_dim}")
        
        print(f"Vision feature dimension: {self.hidden_dim}")
        
        # 2. Feature enhancement layers
        try:
            # Try to check Block class parameters
            import inspect
            block_params = inspect.signature(Block.__init__).parameters
            block_param_names = list(block_params.keys())
            
            # Print parameter names for debugging
            print(f"Block class parameter names: {block_param_names}")
            
            # Dynamically create Block based on parameter names
            block_kwargs = {
                'dim': self.hidden_dim,
                'num_heads': 16,
                'mlp_ratio': 4.0,
                'qkv_bias': True,
            }
            
            # Add dropout parameters (based on available parameter names)
            if 'drop' in block_param_names:
                block_kwargs['drop'] = dropout
            elif 'drop_rate' in block_param_names:
                block_kwargs['drop_rate'] = dropout
            elif 'dropout_rate' in block_param_names:
                block_kwargs['dropout_rate'] = dropout
            
            # Add attention dropout parameters (based on available parameter names)
            if 'attn_drop' in block_param_names:
                block_kwargs['attn_drop'] = dropout
            elif 'attn_drop_rate' in block_param_names:
                block_kwargs['attn_drop_rate'] = dropout
            elif 'attention_dropout_rate' in block_param_names:
                block_kwargs['attention_dropout_rate'] = dropout
            
            print(f"Using Block parameters: {block_kwargs}")
            
            self.feature_enhancer = nn.ModuleList([
                Block(**block_kwargs) for _ in range(2)
            ])
        except Exception as e:
            print(f"Error creating Block: {e}")
            print("Using alternative feature enhancement layers...")
            
            # Use standard Transformer encoder layer as fallback
            self.feature_enhancer = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=16,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True
                ) for _ in range(2)
            ])
        
        # 3. Attention pooling
        self.attn_pool = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 4, 1)
        )
        
        # 4. Dual-stream feature processing
        # Global style features
        self.global_processor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        # Local detail features
        self.local_processor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        # 5. Feature fusion layer
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 6. Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # 7. Calibration layer
        self.calibration = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Extract visual features
        with torch.no_grad():  # Freeze the base vision encoder
            features = self.vision_encoder.forward_features(x)
        
        # 2. Separate class token and patch tokens
        if features.dim() == 3:  # If sequence is returned
            cls_token = features[:, 0:1]  # Keep dimension [B, 1, D]
            patch_tokens = features[:, 1:]
        else:  # If pooled features are returned
            cls_token = features.unsqueeze(1)  # Add sequence dimension
            # Try to get patch tokens
            try:
                patch_tokens = self.vision_encoder.forward_features(x, return_tokens=True)[:, 1:]
            except:
                # If patch tokens can't be retrieved, use fallback
                patch_tokens = features.unsqueeze(1).repeat(1, 16, 1)  # Create 16 identical tokens
        
        # 3. Feature enhancement - apply to all tokens
        all_tokens = torch.cat([cls_token, patch_tokens], dim=1)
        for block in self.feature_enhancer:
            all_tokens = block(all_tokens)
        
        cls_token = all_tokens[:, 0:1]
        patch_tokens = all_tokens[:, 1:]
        
        # 4. Attention pooling - intelligently aggregate patch tokens
        attn_weights = self.attn_pool(patch_tokens)
        attn_weights = F.softmax(attn_weights, dim=1)
        attended_features = (patch_tokens * attn_weights).sum(dim=1)
        
        # 5. Dual-stream feature processing
        global_features = self.global_processor(cls_token.squeeze(1))
        local_features = self.local_processor(attended_features)
        
        # Combine features
        combined_features = torch.cat([global_features, local_features], dim=1)
        fused_features = self.fusion(combined_features)
        
        # 6. Predict bookmark count
        prediction = self.prediction_head(fused_features)
        
        # 7. Calibrate prediction
        calibrated = self.calibration(prediction)
        
        # Ensure prediction is positive
        final_prediction = F.softplus(calibrated)
        
        return {
            'bookmark_prediction': final_prediction.squeeze(),
            'raw_prediction': prediction.squeeze(),
            'global_features': global_features,
            'local_features': local_features
        }