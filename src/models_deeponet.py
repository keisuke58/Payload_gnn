import torch
import torch.nn as nn
import torch.nn.functional as F

class BranchNet(nn.Module):
    """Branch Network: Encodes the input function (defect configuration).

    Input: Function values evaluated at m fixed sensor locations.
           For H3 Fairing: stress/wave readings at sensor positions.
    Output: p-dimensional coefficient vector.
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, u_sensors):
        # u_sensors: (batch, m) where m = number of sensor locations
        return self.net(u_sensors)


class TrunkNet(nn.Module):
    """Trunk Network: Encodes the query coordinates.

    Input: Spatial coordinate y = (x, y, z) or (theta, z) on fairing surface.
    Output: p-dimensional basis function evaluation.
    """
    def __init__(self, coord_dim=2, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.Tanh(),  # Tanh works well for coordinate encoding
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, y):
        # y: (N, coord_dim) where N = number of query points
        return self.net(y)


class DeepONet(nn.Module):
    """Deep Operator Network.

    Learns the operator G: input_function → output_function.
    For H3 Fairing SHM:
        Input function:  Defect parameter field (stiffness distribution on surface)
        Output function: Wave field / stress field response
    """
    def __init__(self, sensor_dim, coord_dim=2, hidden_dim=128, basis_dim=64):
        super().__init__()
        self.branch = BranchNet(sensor_dim, hidden_dim, basis_dim)
        self.trunk = TrunkNet(coord_dim, hidden_dim, basis_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors, y_coords):
        """
        Args:
            u_sensors: (batch, m) - Input function at sensor locations
            y_coords:  (batch, N, coord_dim) or (N, coord_dim) - Query coordinates

        Returns:
            output: (batch, N) - Predicted field values at query points
        """
        branch_out = self.branch(u_sensors)  # (batch, p)

        if y_coords.dim() == 2:
            # Same query points for all batch items
            trunk_out = self.trunk(y_coords)  # (N, p)
            # Output: (batch, N) via einsum
            output = torch.einsum('bp,np->bn', branch_out, trunk_out) + self.bias
        else:
            # Different query points per batch item
            batch_size = y_coords.size(0)
            N = y_coords.size(1)
            trunk_out = self.trunk(y_coords.reshape(-1, y_coords.size(-1)))  # (batch*N, p)
            trunk_out = trunk_out.view(batch_size, N, -1)  # (batch, N, p)
            output = torch.einsum('bp,bnp->bn', branch_out, trunk_out) + self.bias

        return output
