# Circle Attention
Forecasting network traffic by learning interpretable spatial relationships from intersecting circles.

## Code

```python
import enum
from torchtyping import TensorType
import torch


class OverlapEnum(enum.IntEnum):
    IOA = 1
    IOU = 2
    IOA_OTHER = 3
    IOA_MIN = 4
    IOA_MAX = 5


N_sectors: int = 1490
N_batch: int = 1024
N_neighbors: int = 16


class CircleAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        # Circle parameters
        log_radius: TensorType[N_sectors, float],
        log_length: TensorType[N_sectors, float],
        azimuth: TensorType[N_sectors, float],
        # Tower coordinates (equal for sectors of the same tower)
        x_tower: TensorType[N_sectors, float],
        y_tower: TensorType[N_sectors, float],
        # Flags for fixed/learnable
        learn_radius: bool = True,
        learn_length: bool = True,
        learn_azimuth: bool = True,
        # Enum for different areas to be used as denominator of the intersection score
        overlap_type: OverlapEnum = OverlapEnum.IOA,
        # Masking flags
        mask_inside: bool = True,
        mask_outside: bool = False,
        # Epsilon values
        eps_rl: float = 1e-16,  # epsilon for radius and length
        eps_d: float = 1e-32,  # epsilon for distance
        eps_h: float = 1e-5,  # epsilon for triangle height
    ):
        super().__init__()

        self.log_radius = torch.nn.Parameter(log_radius, requires_grad=learn_radius)
        self.log_length = torch.nn.Parameter(log_length, requires_grad=learn_length)
        self.azimuth = torch.nn.Parameter(azimuth, requires_grad=learn_azimuth)

        self.x_tower = torch.nn.Parameter(x_tower, requires_grad=False)
        self.y_tower = torch.nn.Parameter(y_tower, requires_grad=False)

        self.overlap_type = overlap_type

        self.mask_inside = mask_inside
        self.mask_outside = mask_outside

        self.eps_rl = eps_rl
        self.eps_d = eps_d
        self.eps_h = eps_h

    def _select(  # a[i]
        self,
        a: TensorType[N_sectors, float],
        i: TensorType[N_batch, N_neighbors, int],
    ) -> TensorType[N_batch, N_neighbors, float]:
        # a is 1-d tensor of shape N_sectors
        # i is 2-d tensor of shape N_batch x N_neighbors, containing valid indices for a
        # return 2-d tensor with shape of i containing the components a[i]
        B = i.size(0)  # batch size
        N = i.size(1)  # neighborhood size
        ii = i.ravel()
        return torch.index_select(a, 0, ii).view(B, N)

    def forward(
        self,
        idx: TensorType[N_batch, N_neighbors, int],
    ) -> TensorType[N_batch, N_neighbors, float]:
        # idx contains indicies corresponding to sectors.
        # Indicies in row idx[j] are the neighbors of idx[j,0]

        # Get the relevant coordinates and parameters
        x_tower = self._select(self.x_tower, idx)
        y_tower = self._select(self.y_tower, idx)
        #
        log_radius = self._select(self.log_radius, idx)
        log_length = self._select(self.log_length, idx)
        azimuth = self._select(self.azimuth, idx)

        # Compute radius and length from their log-representation
        # Add a small epsilon to ensure that they are never zero
        radius = log_radius.exp() + self.eps_rl
        length = log_length.exp() + self.eps_rl

        # Compute the coordinates of the circles
        # Azimuth is clockwise angle from north (y-axis).
        x_circle = x_tower + length * azimuth.sin()
        y_circle = y_tower + length * azimuth.cos()

        # Compute the squared radius and the area of each circle
        radius2 = radius.square()
        area = radius2 * torch.pi

        # Create some nicer names
        x_self = x_circle[:, :1]
        y_self = y_circle[:, :1]
        area_self = area[:, :1]
        radius_self = radius[:, :1]
        radius2_self = radius2[:, :1]
        #
        x_other = x_circle
        y_other = y_circle
        area_other = area
        radius_other = radius
        radius2_other = radius2

        # Find the distance between the centers of the circles
        dx = x_other - x_self
        dy = y_other - y_self
        dist2 = dx.square() + dy.square()
        dist = (dist2 + self.eps_d).sqrt()

        # Height of the isoscles triangle made by the two points on the line where the circles intersect and the center of the circle
        radius2_diff = radius2_self - radius2_other
        height_self = 0.5 * (dist2 + radius2_diff) / dist
        height_other = 0.5 * (dist2 - radius2_diff) / dist

        # Note: if the circles intersect, and then:
        # height_self + height_other == dist

        # Note: the height might be negative.
        # This means that the line of intersection is "behind" the center of the circle.

        # We make sure that abs(height) < radius - eps
        # This avoids issues with gradients that arise when we approach the ends of the valid domains of acos and sqrt
        height_self = height_self.clamp(-radius_self + self.eps_h, radius_self - self.eps_h)
        height_other = height_other.clamp(-radius_other + self.eps_h, radius_other - self.eps_h)

        # Angle made by the two points on the line where the circles intersect and the center of the circle
        # ( acos maps a number from [-1, +1] to [0, 2pi] )
        angle_self = torch.acos(height_self / radius_self)
        angle_other = torch.acos(height_other / radius_other)

        # Multiply angle by radius squared to get the area of the sector
        sector_area_self = radius2_self * angle_self
        sector_area_other = radius2_other * angle_other

        # Use Pythagoras to find the base of the isoscles triangle (divided by 2)
        half_base_self = torch.sqrt(radius2_self - height_self.square())
        half_base_other = torch.sqrt(radius2_other - height_other.square())

        # Find triangle area (no need to divide by 2)
        triangle_area_self = height_self * half_base_self
        triangle_area_other = height_other * half_base_other

        # Note: the area might be negative in the case height is negative.
        # All of the formulas still work because in this case we need to add the area of the triangle, instead of subtracting it.

        # The area of each circular segment is the area of the sector minus the (signed) area of the triangle
        segment_area_self = sector_area_self - triangle_area_self
        segment_area_other = sector_area_other - triangle_area_other

        # Intersection is the sum of the segments
        intersection = segment_area_self + segment_area_other

        # Optional masking steps:

        # Sets the intersection of circle inside other circles to its own area.
        # The computed areas should alrady be correct, but we found this to improve performance,
        # possibly due to numeric issues that pop up during backprop.
        if self.mask_inside:
            inside_other = dist + radius_self <= radius_other
            intersection = intersection.masked_fill(inside_other, 0.0)
            intersection = intersection + inside_other * area_self

            other_inside = dist + radius_other <= radius_self
            intersection = intersection.masked_fill(other_inside, 0.0)
            intersection = intersection + other_inside * area_other

        # Sets the intersection of circles that do not intersect to 0.
        # (Even though the computed areas should alrady be 0 in this case.)
        # We found this to have little to no effect.
        if self.mask_outside:
            not_overlapping = dist >= radius + radius_self
            intersection = intersection.masked_fill(not_overlapping, 0.0)

        # The intersection scores can be defined in many different ways.
        # We found IoA (i.e. intersection over the current circle's area) to work well.
        if self.overlap_type is OverlapEnum.IOA:
            denom = area_self
        elif self.overlap_type is OverlapEnum.IOU:
            denom = area_self + area_other - intersection
        elif self.overlap_type is OverlapEnum.IOA_OTHER:
            denom = area_other
        elif self.overlap_type is OverlapEnum.IOA_MIN:
            denom = torch.minimum(area_self, area_other)
        elif self.overlap_type is OverlapEnum.IOA_MAX:
            denom = torch.maximum(area_self, area_other)
        else:
            raise ValueError

        # Compute intersection scores
        ioa = intersection / denom

        return ioa
```
