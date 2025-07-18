import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepCyTof(nn.Module):

    def __init__(self, num_features, pool_height, input_channels=1, output_scale_factor=1.0):
        """
        A pytorch re-implementation of the Keras DeepCyTof model (see https://github.com/hzc363/DeepLearningCyTOF))

        Args:
            input_channels (int): Number of input channels (e.g., 1 if grayscale or single feature map).
                                  Corresponds to the channels dimension of the input shape.
            num_features (int): Number of features in the input data width dimension.
                                Equivalent to x_train.shape[2] in the Keras code.
            pool_height (int): The height dimension over which to average pool.
                               Equivalent to x_train.shape[1] in the Keras code.
        """
        super(DeepCyTof, self).__init__()
        self.model_conf = {
            'input_channels': input_channels,
            'num_features': num_features,
            'pool_height': pool_height,
            'output_scale_factor': output_scale_factor
        }
        self.output_scale_factor = output_scale_factor
        # --- First Convolution Block ---
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=3,
                               kernel_size=(1, num_features),
                               padding='valid') # Keras default padding is 'valid'
        self.bn1 = nn.BatchNorm2d(num_features=3) # Batch norm after conv, before activation

        # --- Second Convolution Block ---
        # Input channels to conv2 is output channels of conv1
        self.conv2 = nn.Conv2d(in_channels=3,
                               out_channels=3,
                               kernel_size=(1, 1),
                               padding='valid')
        self.bn2 = nn.BatchNorm2d(num_features=3)

        self.pool = nn.AvgPool2d(kernel_size=(pool_height, 1),
                                 stride=(pool_height, 1),
                                 padding=(0, 0))
        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(in_features=3, out_features=3)
        self.bn3 = nn.BatchNorm1d(num_features=3) 
        self.dense2 = nn.Linear(in_features=3, out_features=1)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, a=-0.05, b=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=3)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, a=-0.01, b=0.01)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape
                              (batch_size, input_channels, pool_height, num_features).
        """
        # Insert a channel dimension at index 1
        x = x.unsqueeze(1) # Uncomment if input is 2D and needs a channel dimension
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.pool(x)
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.dense2(x)
        x = x * self.output_scale_factor

        return x
    

if __name__=="__main__":
    num_features = 13
    num_cells = 100
    x = torch.randn(32, 1, num_cells, num_features) # Example input tensor
    model = DeepCyTof(num_features, pool_height=num_cells)

    # Print parameters for each component
    print("Parameters by component:")
    print(f"conv1: {sum(p.numel() for p in model.conv1.parameters())}")
    print(f"bn1: {sum(p.numel() for p in model.bn1.parameters())}")
    print(f"conv2: {sum(p.numel() for p in model.conv2.parameters())}")
    print(f"bn2: {sum(p.numel() for p in model.bn2.parameters())}")
    print(f"pool: {sum(p.numel() for p in model.pool.parameters())}")
    print(f"dense1: {sum(p.numel() for p in model.dense1.parameters())}")
    print(f"bn3: {sum(p.numel() for p in model.bn3.parameters())}")
    print(f"dense2: {sum(p.numel() for p in model.dense2.parameters())}")

    # Count total number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")
    output = model(x)
    print(output.shape)