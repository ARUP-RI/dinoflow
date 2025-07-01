import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    A standard two-layer feed-forward network as used in Transformer architectures.
    It consists of two linear layers with a ReLU activation in between.
    """
    def __init__(self, dim_model: int, dim_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IndAttnBlock(nn.Module):
    """
    Implementation of the Induced Attention Block as described in the paper[cite: 56, 71].
    This block is based on the Set Transformer architecture [cite: 52] and is designed to
    handle sets of inputs in a permutation-invariant manner with linear complexity[cite: 52, 67, 72].

    It follows the structure from equation (4):
    IndAttnBlock(X) = TransfBlock(X, TransfBlock(I, X))
    where I are the learnable inducing points.
    """
    def __init__(self, dim_model: int, num_heads: int, num_ind_points: int):
        super().__init__()
        dim_ff = dim_model * 4 # A common choice for the feed-forward dimension

        self.inducing_points = nn.Parameter(torch.randn(1, num_ind_points, dim_model))
        
        # This corresponds to the inner TransfBlock(I, X)
        self.mha1 = nn.MultiheadAttention(dim_model, num_heads, batch_first=True)
        self.ffn1 = FeedForward(dim_model, dim_ff)
        self.ln1 = nn.LayerNorm(dim_model)
        self.ln2 = nn.LayerNorm(dim_model)
        
        # This corresponds to the outer TransfBlock(X, H_temp)
        self.mha2 = nn.MultiheadAttention(dim_model, num_heads, batch_first=True)
        self.ffn2 = FeedForward(dim_model, dim_ff)
        self.ln3 = nn.LayerNorm(dim_model)
        self.ln4 = nn.LayerNorm(dim_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The paper processes samples with a batch size of 1.
        # We expand the inducing points to match the batch size of the input.
        batch_size = x.size(0)
        inducing_points = self.inducing_points.expand(batch_size, -1, -1)

        # Inner Transformer Block: TransfBlock(I, X)
        # 1. Attention between inducing points (Q) and input (K,V)
        attn_out1, _ = self.mha1(query=inducing_points, key=x, value=x)
        # 2. First residual connection and LayerNorm (AttnBlock)
        h = self.ln1(inducing_points + attn_out1)
        # 3. Feed-forward, second residual, and LayerNorm (TransfBlock)
        h_temp = self.ln2(h + self.ffn1(h)) # These are the latent features

        # Outer Transformer Block: TransfBlock(X, H_temp)
        # 1. Attention between input (Q) and latent features (K,V)
        attn_out2, _ = self.mha2(query=x, key=h_temp, value=h_temp)
        # 2. First residual connection and LayerNorm (AttnBlock)
        z = self.ln3(x + attn_out2)
        # 3. Feed-forward, second residual, and LayerNorm (TransfBlock)
        output = self.ln4(z + self.ffn2(z))

        return output


class FlowTransformer(nn.Module):
    """
    The main model for automated identification of cell populations in flow
    cytometry data, as described in Wödlinger et al., 2022.

    The architecture consists of an input embedding layer (a necessary assumption),
    a sequence of Induced Attention Blocks (IndAttnBlock), and a final linear
    layer for classification.

    Hyperparameters are set according to the paper:
    - Number of induced points: 16
    - Latent embedding dimension: 32
    - Number of attention heads: 4
    - Number of IndAttnBlock layers: 3
    """
    def __init__(self, input_dim: int, num_classes: int = 1):
        super().__init__()
        
        # --- Model Hyperparameters from the paper  ---
        latent_dim = 32
        num_heads = 4
        num_ind_points = 16
        num_blocks = 3

        # --- Model Layers ---
        # 1. Input Embedding Layer (Necessary Assumption)
        # This projects the m-dimensional input features to the latent dimension d.
        self.embedding = nn.Linear(input_dim, latent_dim)

        # 2. Sequence of three Induced Attention Blocks 
        self.blocks = nn.Sequential(
            *[IndAttnBlock(latent_dim, num_heads, num_ind_points) for _ in range(num_blocks)]
        )

        # 3. Final row-wise linear layer for classification 
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes a single sample of FCM data in a single forward pass.
        The paper uses a batch size of 1 during training[cite: 104].
        Input shape should be (B, N, m) where B is batch size (e.g., 1),
        N is the number of cells, and m is the number of markers.

        Args:
            x (torch.Tensor): The input event matrix.

        Returns:
            torch.Tensor: The binary classification label for every input cell.
        """
        # No positional embedding is used 
        embedded_x = self.embedding(x)
        
        # Pass through the stack of IndAttnBlocks
        processed_x = self.blocks(embedded_x)
        
        # Get final classification logits
        # The sigmoid function would be applied with the loss function (BCEWithLogitsLoss)
        # for numerical stability.
        logits = self.classifier(processed_x)
        
        return logits


if __name__ == '__main__':
    # --- Example Usage ---
    # Create a dummy FCM data sample.
    # A single sample is a matrix E of shape (N, m), where N is the number of cells
    # and m is the number of markers[cite: 58].
    
    num_cells = 50000  # N can range from 10^5 to 10^6 [cite: 37, 58]
    num_markers = 10   # m is typically 10-20 [cite: 58]
    batch_size = 1     # Training was done with a batch size of 1 [cite: 104]

    # Dummy input tensor (one sample)
    dummy_fcm_sample = torch.randn(batch_size, num_cells, num_markers)

    print(f"Input sample shape: {dummy_fcm_sample.shape}")
    print("-" * 30)

    # Instantiate the model
    # The input_dim `m` must be specified.
    model = FlowTransformer(input_dim=num_markers)
    
    # The paper states the model has only 27,657 parameters.
    # Let's verify our implementation.
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Instantiated FlowTransformer model.")
    print(f"Total number of trainable parameters: {num_params}")
    print("-" * 30)
    
    # Perform a forward pass
    print("Performing a forward pass...")
    with torch.no_grad():
        output_logits = model(dummy_fcm_sample)
    
    print("Forward pass successful.")
    # The output is a prediction for each cell [cite: 55]
    print(f"Output shape: {output_logits.shape}")