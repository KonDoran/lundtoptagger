"""
Quantum Circuit Layers for Graph Neural Networks

This module contains quantum circuit implementations adapted for use in PyTorch GNNs.
Extracted and generalized from the QGNN water molecule prediction project.
"""

try:
    import pennylane as qml
except ImportError as e:
    raise ImportError(
        "PennyLane is required for quantum layers. Install with: pip install pennylane"
    ) from e

import torch
import torch.nn as nn
import numpy as np


# ===== QUANTUM CIRCUIT HELPER FUNCTIONS =====

def encode_input_positions(inputs, qubits, n_features_per_qubit=2):
    """
    Encode input features into quantum states using rotation gates.
    
    This is the quantum analog of classical feature embedding. Each qubit receives
    a subset of input features encoded as rotation angles.
    
    Args:
        inputs: Input features tensor of shape (n_qubits * n_features_per_qubit,) or (batch_size, n_qubits * n_features_per_qubit)
        qubits: List of qubit indices
        n_features_per_qubit: Number of features to encode per qubit (default: 2)
    
    Note:
        Features are encoded using RX and RY gates. For more features per qubit,
        additional rotation gates (RZ) can be added.
    """
    # Check if inputs is a batch (2D) or single sample (1D)
    is_batch = len(inputs.shape) > 1
    
    for i, qubit in enumerate(qubits):
        # Extract features for this qubit
        start_idx = i * n_features_per_qubit
        end_idx = start_idx + n_features_per_qubit
        
        feature_dim = inputs.shape[-1]
        
        if start_idx < feature_dim:
            # Apply RX rotation
            val = inputs[:, start_idx] if is_batch else inputs[start_idx]
            qml.RX(val, wires=qubit)
            
        if start_idx + 1 < feature_dim and n_features_per_qubit > 1:
            # Apply RY rotation
            val = inputs[:, start_idx + 1] if is_batch else inputs[start_idx + 1]
            qml.RY(val, wires=qubit)


def apply_interaction_gates(interactions, qubits, layer_idx=0):
    """
    Apply entangling gates between qubits based on interaction strengths.
    
    This creates quantum entanglement between qubits, analogous to message
    passing in classical GNNs. The interaction strength can be based on
    distances, edge weights, or other graph properties.
    
    Args:
        interactions: Interaction strengths (can be scalar or array)
        qubits: List of qubit indices
        layer_idx: Current layer index (for gate variation)
    
    Note:
        Uses IsingXX gates for entanglement. Other options include:
        - IsingYY, IsingZZ for different coupling types
        - CNOT for discrete entanglement
        - CRX, CRY, CRZ for parameterized entanglement
    """
    n_qubits = len(qubits)
    
    # Apply entangling gates between adjacent qubits
    for i in range(n_qubits - 1):
        # Simple ring connectivity
        interaction_strength = 0.1  # Default small coupling
        
        if isinstance(interactions, (torch.Tensor, np.ndarray)):
            if len(interactions) > i:
                interaction_strength = float(interactions[i])
        elif isinstance(interactions, (int, float)):
            interaction_strength = float(interactions)
        
        # Apply Ising XX gate (creates entanglement)
        qml.IsingXX(interaction_strength, wires=[qubits[i], qubits[i + 1]])


def apply_param_gates(param_offset, quantum_params, qubits):
    """
    Apply parameterized rotation gates (learnable parameters).
    
    These are the trainable quantum parameters, analogous to weights in
    classical neural networks. They are optimized via backpropagation.
    
    Args:
        param_offset: Starting index in quantum_params for this layer
        quantum_params: Tensor of all learnable quantum parameters
        qubits: List of qubit indices
    
    Note:
        Each qubit gets 2 learnable parameters (RX and RY rotations).
        Total parameters per layer = n_qubits * 2
    """
    params_per_qubit = 2  # RX and RY rotations
    
    for i, qubit in enumerate(qubits):
        idx = param_offset + i * params_per_qubit
        
        if idx + 1 < len(quantum_params):
            # Learnable RX rotation
            qml.RX(quantum_params[idx], wires=qubit)
            # Learnable RY rotation
            qml.RY(quantum_params[idx + 1], wires=qubit)


def measure_quantum_state(qubits):
    """
    Measure quantum state and return expectation values.
    
    This extracts classical information from the quantum state.
    Uses Pauli-Z measurements which return values in [-1, 1].
    
    Args:
        qubits: List of qubit indices
    
    Returns:
        List of measurement expectation values (one per qubit)
    
    Note:
        Alternative measurement strategies:
        - PauliX, PauliY for different observables
        - Multiple measurements per qubit for richer features
        - Hadamard basis measurements
    """
    measurements = []
    for qubit in qubits:
        measurements.append(qml.expval(qml.PauliZ(qubit)))
    return measurements


# ===== QUANTUM EDGE CONVOLUTION LAYER =====

class QuantumEdgeConv(nn.Module):
    """
    Quantum EdgeConv layer for graph neural networks.
    
    This layer processes edge features through quantum circuits, providing
    a quantum analog to classical edge convolution operations. Each edge's
    features are encoded into a quantum state, processed through parameterized
    quantum gates, and measured to produce classical output features.
    
    Architecture:
        Classical Preprocessing → Quantum Circuit → Classical Postprocessing
        
    The quantum circuit acts as a highly expressive feature transformation
    with exponentially large Hilbert space (2^n_qubits states).
    
    Args:
        in_channels: Input feature dimension per node
        out_channels: Output feature dimension per node
        n_qubits: Number of qubits in quantum circuit (4-8 recommended)
        n_layers: Number of quantum circuit layers (1-3 recommended)
        aggr: Aggregation method for edge-to-node ('add', 'mean', 'max')
    
    Attributes:
        quantum_params: Learnable quantum gate parameters
        pre_quantum: Classical preprocessing MLP
        post_quantum: Classical postprocessing MLP
        qnode: PennyLane quantum node (circuit)
    
    Note:
        Computational complexity: O(edges x 2^n_qubits)
        Start with small n_qubits (4-6) for feasibility.
    """
    
    def __init__(self, in_channels, out_channels, n_qubits=4, n_layers=1, aggr='add'):
        super(QuantumEdgeConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.aggr = aggr
        
        # Quantum parameters: n_layers × n_qubits × 2 (RX + RY per qubit per layer)
        n_params = n_layers * n_qubits * 2
        self.quantum_params = nn.Parameter(torch.randn(n_params) * 0.1)
        
        # Classical preprocessing: Map input features to quantum input dimension
        # EdgeConv uses [x_i || x_j] (concatenation), so input is 2 × in_channels
        self.pre_quantum = nn.Sequential(
            nn.Linear(in_channels * 2, n_qubits * 2),
            nn.Tanh()  # Normalize to [-1, 1] for stable quantum encoding
        )
        
        # Classical postprocessing: Map quantum measurements to output dimension
        self.post_quantum = nn.Sequential(
            nn.Linear(n_qubits, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        # Define quantum device and circuit
        # Try to use GPU-accelerated device if available
        try:
            self.dev = qml.device('lightning.gpu', wires=n_qubits)
            diff_method = 'adjoint'
        except Exception:
            try:
                self.dev = qml.device('lightning.qubit', wires=n_qubits)
                diff_method = 'adjoint'
            except Exception:
                self.dev = qml.device('default.qubit', wires=n_qubits)
                diff_method = 'backprop'
        
        print(f"Using quantum device: {self.dev.name}")
        
        self.qnode = qml.QNode(
            self._quantum_circuit, 
            self.dev, 
            interface='torch', 
            diff_method=diff_method
        )
    
    def _quantum_circuit(self, inputs):
        """
        The core quantum circuit implementation.
        
        Circuit structure:
        1. Input encoding: Map classical features to quantum states
        2. Parameterized layers: Apply learnable transformations
           - Interaction gates (entanglement)
           - Parameterized rotations (trainable)
        3. Measurement: Extract classical features
        
        Args:
            inputs: Preprocessed features of shape (n_qubits × 2,) or (batch, n_qubits × 2)
        
        Returns:
            Measurement results of shape (n_qubits,) or (batch, n_qubits)
        """
        qubits = list(range(self.n_qubits))
        
        # Step 1: Encode input features into quantum state
        encode_input_positions(inputs, qubits, n_features_per_qubit=2)
        
        # Step 2: Apply parameterized quantum layers
        for layer in range(self.n_layers):
            param_offset = layer * self.n_qubits * 2
            
            # Apply entangling gates (creates quantum correlations)
            interactions = 0.1  # Simple uniform coupling
            apply_interaction_gates(interactions, qubits, layer)
            
            # Apply trainable rotations
            apply_param_gates(param_offset, self.quantum_params, qubits)
        
        # Step 3: Measure quantum state to get classical outputs
        return measure_quantum_state(qubits)
    
    def forward(self, x, edge_index):
        """
        Forward pass compatible with PyTorch Geometric.
        
        Processing pipeline:
        1. Construct edge features [x_i || x_j] for each edge
        2. Preprocess: Classical MLP to quantum input dimension
        3. Quantum: Process through quantum circuit
        4. Postprocess: Classical MLP to output dimension
        5. Aggregate: Edge features → node features
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
        
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        row, col = edge_index
        
        # Step 1: Construct edge features [x_i || x_j] for each edge
        edge_features = torch.cat([x[row], x[col]], dim=-1)  # [num_edges, in_channels × 2]
        
        # Step 2: Preprocess for quantum circuit
        quantum_input = self.pre_quantum(edge_features)  # [num_edges, n_qubits × 2]
        
        # Step 3: Process through quantum circuit (batch processing)
        # We pass the entire batch of edge features to the QNode at once
        # PennyLane handles broadcasting/batching
        q_out = self.qnode(quantum_input)
        
        # Stack measurements into tensor if needed
        # QNode output with batching is usually a tuple of tensors (one per measurement)
        # where each tensor has shape (batch_size,)
        if isinstance(q_out, (list, tuple)):
            quantum_output = torch.stack(q_out, dim=-1) # [num_edges, n_qubits]
        else:
            quantum_output = q_out
            if len(quantum_output.shape) == 1:
                quantum_output = quantum_output.unsqueeze(-1)
        
        # Ensure correct shape and type
        quantum_output = quantum_output.float()
        
        # Step 4: Postprocess quantum output
        edge_output = self.post_quantum(quantum_output)  # [num_edges, out_channels]
        
        # Step 5: Aggregate edge features back to nodes
        num_nodes = x.size(0)
        node_output = torch.zeros(num_nodes, self.out_channels, 
                                   device=x.device, dtype=x.dtype)
        
        if self.aggr == 'add':
            node_output.index_add_(0, row, edge_output)
        elif self.aggr == 'mean':
            node_output.index_add_(0, row, edge_output)
            # Count edges per node for averaging
            count = torch.zeros(num_nodes, 1, device=x.device)
            count.index_add_(0, row, torch.ones_like(edge_output[:, :1]))
            node_output = node_output / count.clamp(min=1)
        elif self.aggr == 'max':
            # Max aggregation (requires scatter_max from torch_geometric)
            from torch_geometric.utils import scatter
            node_output = scatter(edge_output, row, dim=0, dim_size=num_nodes, reduce='max')
        
        return node_output


# ===== CONVENIENCE FUNCTION FOR TESTING =====

def test_quantum_layer():
    """
    Test function to verify QuantumEdgeConv functionality.
    
    Creates a small random graph and processes it through the quantum layer.
    Useful for debugging and understanding layer behavior.
    """
    print("Testing QuantumEdgeConv layer...")
    
    # Create dummy data
    num_nodes = 5
    num_edges = 8
    in_channels = 3
    out_channels = 16
    n_qubits = 4
    
    # Random node features
    x = torch.randn(num_nodes, in_channels)
    
    # Random edge connectivity
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create quantum layer
    layer = QuantumEdgeConv(
        in_channels=in_channels,
        out_channels=out_channels,
        n_qubits=n_qubits,
        n_layers=2,
        aggr='add'
    )
    
    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Quantum parameters: {layer.quantum_params.shape[0]}")
    
    # Forward pass
    output = layer(x, edge_index)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: [{num_nodes}, {out_channels}]")
    
    # Check gradient flow
    loss = output.sum()
    loss.backward()
    
    print(f"Gradient on quantum params: {layer.quantum_params.grad is not None}")
    print("Test passed!")


if __name__ == "__main__":
    test_quantum_layer()
