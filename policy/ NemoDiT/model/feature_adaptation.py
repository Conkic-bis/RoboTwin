import torch
import torch.nn as nn
import math

class FeatureAdapter(nn.Module):
    """
    Adapts vision features to DiT's expected dimension.

    This module projects vision features from the backbone's output dimension
    to the hidden dimension expected by the DiT model.

    Args:
        vision_feature_dim: Input dimension from vision backbone
        dit_hidden_size: Output dimension for DiT (token_size or hidden_size)
        use_layernorm: Whether to apply layer normalization
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        vision_feature_dim: int,
        dit_hidden_size: int,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vision_feature_dim = vision_feature_dim
        self.dit_hidden_size = dit_hidden_size

        # Projection layer
        self.projection = nn.Linear(vision_feature_dim, dit_hidden_size)

        # Optional layer normalization
        self.layernorm = nn.LayerNorm(dit_hidden_size) if use_layernorm else nn.Identity()

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize projection weights."""
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.constant_(self.projection.bias, 0)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to DiT dimension.

        Args:
            vision_features: (batch_size, num_patches, vision_feature_dim)

        Returns:
            adapted_features: (batch_size, num_patches, dit_hidden_size)
        """
        # Project features
        adapted_features = self.projection(vision_features)

        # Apply layer normalization
        adapted_features = self.layernorm(adapted_features)

        # Apply dropout
        adapted_features = self.dropout(adapted_features)

        return adapted_features


class MLPAdapter(nn.Module):
    """
    Multi-layer perceptron adapter for more expressive feature transformation.

    Args:
        vision_feature_dim: Input dimension from vision backbone
        dit_hidden_size: Output dimension for DiT
        hidden_size: Intermediate hidden dimension (default: None, uses average)
        num_layers: Number of MLP layers (default: 2)
        use_layernorm: Whether to apply layer normalization
        dropout: Dropout rate
    """

    def __init__(
        self,
        vision_feature_dim: int,
        dit_hidden_size: int,
        hidden_size: int = None,
        num_layers: int = 2,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vision_feature_dim = vision_feature_dim
        self.dit_hidden_size = dit_hidden_size
        self.num_layers = num_layers

        # Default hidden size is the average of input and output
        if hidden_size is None:
            hidden_size = (vision_feature_dim + dit_hidden_size) // 2

        # Build MLP layers
        layers = []
        input_dim = vision_feature_dim

        for i in range(num_layers):
            output_dim = dit_hidden_size if i == num_layers - 1 else hidden_size

            layers.append(nn.Linear(input_dim, output_dim))

            # Add activation and normalization for all but last layer
            if i < num_layers - 1:
                layers.append(nn.GELU())
                if use_layernorm:
                    layers.append(nn.LayerNorm(output_dim))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            input_dim = output_dim

        self.mlp = nn.Sequential(*layers)

        # Final layer normalization
        self.final_norm = nn.LayerNorm(dit_hidden_size) if use_layernorm else nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize MLP weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Transform vision features through MLP.

        Args:
            vision_features: (batch_size, num_patches, vision_feature_dim)

        Returns:
            adapted_features: (batch_size, num_patches, dit_hidden_size)
        """
        adapted_features = self.mlp(vision_features)
        adapted_features = self.final_norm(adapted_features)

        return adapted_features


class AttentionPoolingAdapter(nn.Module):
    """
    Adapter with attention-based pooling to aggregate spatial features.

    This reduces the number of tokens while preserving important information.

    Args:
        vision_feature_dim: Input dimension from vision backbone
        dit_hidden_size: Output dimension for DiT
        num_queries: Number of output queries/tokens (default: 1)
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        vision_feature_dim: int,
        dit_hidden_size: int,
        num_queries: int = 1,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vision_feature_dim = vision_feature_dim
        self.dit_hidden_size = dit_hidden_size
        self.num_queries = num_queries
        self.num_heads = num_heads

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, dit_hidden_size))

        # Project vision features to dit_hidden_size before attention
        self.input_projection = nn.Linear(vision_feature_dim, dit_hidden_size)

        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dit_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.norm = nn.LayerNorm(dit_hidden_size)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.constant_(self.input_projection.bias, 0)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Pool vision features using attention.

        Args:
            vision_features: (batch_size, num_patches, vision_feature_dim)

        Returns:
            adapted_features: (batch_size, num_queries, dit_hidden_size)
        """
        batch_size = vision_features.shape[0]

        # Project input features
        key_value = self.input_projection(vision_features)  # (B, num_patches, dit_hidden_size)

        # Expand query tokens for batch
        query = self.query_tokens.expand(batch_size, -1, -1)  # (B, num_queries, dit_hidden_size)

        # Apply cross-attention: queries attend to vision features
        adapted_features, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
        )

        # Layer normalization
        adapted_features = self.norm(adapted_features)

        return adapted_features


# Factory function for creating adapters
def create_feature_adapter(
    adapter_type: str,
    vision_feature_dim: int,
    dit_hidden_size: int,
    **kwargs
):
    """
    Factory function to create feature adapters.

    Args:
        adapter_type: Type of adapter ('linear', 'mlp', 'attention_pooling')
        vision_feature_dim: Input dimension from vision backbone
        dit_hidden_size: Output dimension for DiT
        **kwargs: Additional arguments for specific adapter types
            - linear: use_layernorm, dropout
            - mlp: hidden_size, num_layers, use_layernorm, dropout
            - attention_pooling: num_queries, num_heads, dropout

    Returns:
        adapter: Feature adapter module
    """
    # Define valid kwargs for each adapter type
    linear_kwargs = {'use_layernorm', 'dropout'}
    mlp_kwargs = {'hidden_size', 'num_layers', 'use_layernorm', 'dropout'}
    attention_kwargs = {'num_queries', 'num_heads', 'dropout'}

    if adapter_type == 'linear':
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in linear_kwargs}
        return FeatureAdapter(vision_feature_dim, dit_hidden_size, **filtered_kwargs)
    elif adapter_type == 'mlp':
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in mlp_kwargs}
        return MLPAdapter(vision_feature_dim, dit_hidden_size, **filtered_kwargs)
    elif adapter_type == 'attention_pooling':
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in attention_kwargs}
        return AttentionPoolingAdapter(vision_feature_dim, dit_hidden_size, **filtered_kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Feature Adapters...")

    batch_size = 2
    num_patches = 196  # 14x14 for ViT or flattened ResNet features
    vision_dim = 768  # ViT-B dimension
    dit_dim = 1152  # DiT hidden size

    dummy_features = torch.randn(batch_size, num_patches, vision_dim)

    # Test Linear Adapter
    print("\n1. Testing Linear Adapter...")
    linear_adapter = FeatureAdapter(vision_dim, dit_dim)
    output = linear_adapter(dummy_features)
    print(f"   Input shape: {dummy_features.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, num_patches, dit_dim)

    # Test MLP Adapter
    print("\n2. Testing MLP Adapter...")
    mlp_adapter = MLPAdapter(vision_dim, dit_dim, num_layers=2)
    output = mlp_adapter(dummy_features)
    print(f"   Input shape: {dummy_features.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, num_patches, dit_dim)

    # Test Attention Pooling Adapter
    print("\n3. Testing Attention Pooling Adapter...")
    attn_adapter = AttentionPoolingAdapter(vision_dim, dit_dim, num_queries=1)
    output = attn_adapter(dummy_features)
    print(f"   Input shape: {dummy_features.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, 1, dit_dim)

    # Test Factory Function
    print("\n4. Testing Factory Function...")
    adapter = create_feature_adapter('mlp', vision_dim, dit_dim, num_layers=3)
    output = adapter(dummy_features)
    print(f"   Created MLP adapter with 3 layers")
    print(f"   Output shape: {output.shape}")

    print("\nAll adapter tests passed!")
