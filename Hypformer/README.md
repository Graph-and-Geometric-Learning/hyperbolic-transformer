# Simplified Hypformer Code

This folder contains simplified code for Hypformer, designed to be easily adaptable for various research applications in GNN, Text, Image processing, and more.

## Overview

The Hypformer implementation includes two types of attention mechanisms:
1. Full attention (softmax-based)
2. Linear attention (kernel-based)

## Prerequisites

Before running the code, ensure you have the required dependencies installed, particularly `geoopt` for hyperbolic operations.

## Installation

1. Install geoopt:
   ```bash
   pip install geoopt
   ```

2. Install other dependencies (if any):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run a simple demonstration of Hypformer:

1. Execute the main script:
   ```bash
   python main.py
   ```

2. Expected output:
   ```
   Input shape: torch.Size([10, 16])
   Output shape: torch.Size([10, 5])
   ```

## Code Structure

- `main.py`: Contains a simple example showcasing the usage of Hypformer.
- `hypformer.py`: The core implementation of the Hypformer model.

## Customization

To adapt Hypformer for your specific research needs:

1. Open `hypformer.py`
2. Modify the attention mechanisms or model architecture as required
3. Adjust the input/output dimensions in `main.py` to match your data

## Example

Here's a basic example of how to use Hypformer in your code:

```python
from hypformer import Hypformer
import torch

# Initialize Hypformer
model = Hypformer(input_dim=16, hidden_dim=32, output_dim=5, num_layers=2)

# Create sample input
x = torch.randn(10, 16)

# Forward pass
output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

## Contributing

We welcome contributions to improve the Hypformer implementation. Please feel free to submit pull requests or open issues for any bugs or feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file in the parent directory for details.

## Contact

For any questions or concerns about this simplified implementation, please open an issue in the repository or contact the first author.