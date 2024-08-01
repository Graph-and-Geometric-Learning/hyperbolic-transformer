import torch
import torch.nn as nn
from typing import Tuple, Optional
import geoopt
from geoopt import Manifold
from geoopt import Lorentz as LorentzOri
from geoopt.utils import size2shape
import manifolds.lmath as math
from manifolds.utils import acosh


def arcosh(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse hyperbolic cosine (arcosh) of the input tensor.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The arcosh of the input tensor.
    """
    dtype = x.dtype
    z = torch.sqrt(torch.clamp_min(x.pow(2) - 1.0, 1e-7))
    return torch.log(x + z).to(dtype)


class Lorentz(LorentzOri):
    def __init__(self, k=1.0, learnable=False):
        """
        Initialize a Lorentz manifold with curvature k.

        Parameters:
            k (float): Curvature of the manifold.
            learnable (bool): If True, k is learnable. Default is False.
        """
        super().__init__(k, learnable)

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1) -> Tuple[bool, Optional[str]]:
        """
        Check if a point lies on the manifold.

        Parameters:
            x (torch.Tensor): Point to check.
            atol (float): Absolute tolerance.
            rtol (float): Relative tolerance.
            dim (int): Dimension to check.

        Returns:
            Tuple[bool, Optional[str]]: A boolean indicating if the point is on the manifold, and an optional reason string.
        """
        dn = x.size(dim) - 1
        x = x ** 2
        quad_form = -x.narrow(dim, 0, 1) + x.narrow(dim, 1, dn).sum(dim=dim, keepdim=True)
        ok = torch.allclose(quad_form, -self.k, atol=atol, rtol=rtol)
        reason = None if ok else f"'x' minkowski quadratic form is not equal to {-self.k.item()}"
        return ok, reason

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1) -> Tuple[
        bool, Optional[str]]:
        """
        Check if a vector lies on the tangent space at a point.

        Parameters:
            x (torch.Tensor): Point on the manifold.
            u (torch.Tensor): Vector to check.
            atol (float): Absolute tolerance.
            rtol (float): Relative tolerance.
            dim (int): Dimension to check.

        Returns:
            Tuple[bool, Optional[str]]: A boolean indicating if the vector is on the tangent space, and an optional reason string.
        """
        inner_ = math.inner(u, x, dim=dim)
        ok = torch.allclose(inner_, torch.zeros(1), atol=atol, rtol=rtol)
        reason = None if ok else "Minkowski inner product is not equal to zero"
        return ok, reason

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        """
        Compute the distance between two points on the manifold.

        Parameters:
            x (torch.Tensor): First point.
            y (torch.Tensor): Second point.
            keepdim (bool): If True, retains the last dimension.
            dim (int): Dimension to compute distance.

        Returns:
            torch.Tensor: Distance between x and y.
        """
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim)

    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        """
        Compute the distance from the origin to a point on the manifold.

        Parameters:
            x (torch.Tensor): Point on the manifold.
            keepdim (bool): If True, retains the last dimension.
            dim (int): Dimension to compute distance.

        Returns:
            torch.Tensor: Distance from the origin to x.
        """
        return math.dist0(x, k=self.k, dim=dim, keepdim=keepdim)

    def cdist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise distance between points in x and y.

        Parameters:
            x (torch.Tensor): First set of points.
            y (torch.Tensor): Second set of points.

        Returns:
            torch.Tensor: Pairwise distances between points in x and y.
        """
        return math.cdist(x, y, k=self.k)

    def lorentz_to_klein(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a point from Lorentz to Klein coordinates.

        Parameters:
            x (torch.Tensor): Point in Lorentz coordinates.

        Returns:
            torch.Tensor: Point in Klein coordinates.
        """
        dim = x.shape[-1] - 1
        return acosh(x.narrow(-1, 1, dim) / x.narrow(-1, 0, 1))

    def klein_to_lorentz(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a point from Klein to Lorentz coordinates.

        Parameters:
            x (torch.Tensor): Point in Klein coordinates.

        Returns:
            torch.Tensor: Point in Lorentz coordinates.
        """
        norm = (x * x).sum(dim=-1, keepdim=True)
        size = x.shape[:-1] + (1,)
        return torch.cat([x.new_ones(size), x], dim=-1) / torch.clamp_min(torch.sqrt(1 - norm), 1e-7)

    def lorentz_to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a point from Lorentz to Poincare coordinates.

        Parameters:
            x (torch.Tensor): Point in Lorentz coordinates.

        Returns:
            torch.Tensor: Point in Poincare coordinates.
        """
        return math.lorentz_to_poincare(x, self.k)

    def norm(self, u: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        """
        Compute the norm of a tangent vector.

        Parameters:
            u (torch.Tensor): Tangent vector.
            keepdim (bool): If True, retains the last dimension.
            dim (int): Dimension to compute the norm.

        Returns:
            torch.Tensor: Norm of u.
        """
        return math.norm(u, keepdim=keepdim, dim=dim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Convert Euclidean gradient to Riemannian gradient.

        Parameters:
            x (torch.Tensor): Point on the manifold.
            u (torch.Tensor): Euclidean gradient.
            dim (int): Dimension to compute the gradient.

        Returns:
            torch.Tensor: Riemannian gradient.
        """
        return math.egrad2rgrad(x, u, k=self.k, dim=dim)

    def projx(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Project a point onto the manifold.

        Parameters:
            x (torch.Tensor): Point to project.
            dim (int): Dimension to project.

        Returns:
            torch.Tensor: Projected point.
        """
        return math.project(x, k=self.k, dim=dim)

    def proju(self, x: torch.Tensor, v: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Project a tangent vector onto the tangent space at a point.

        Parameters:
            x (torch.Tensor): Point on the manifold.
            v (torch.Tensor): Tangent vector to project.
            dim (int): Dimension to project.

        Returns:
            torch.Tensor: Projected tangent vector.
        """
        v = math.project_u(x, v, k=self.k, dim=dim)
        return v

    def proju0(self, v: torch.Tensor) -> torch.Tensor:
        """
        Project a tangent vector onto the tangent space at the origin.

        Parameters:
            v (torch.Tensor): Tangent vector to project.

        Returns:
            torch.Tensor: Projected tangent vector.
        """
        return math.project_u0(v)

    def expmap(self, x: torch.Tensor, u: torch.Tensor, *, norm_tan=True, project=True, dim=-1) -> torch.Tensor:
        """
        Perform the exponential map to move from a point in the tangent space to the manifold.

        Parameters:
            x (torch.Tensor): Point on the manifold.
            u (torch.Tensor): Tangent vector.
            norm_tan (bool): If True, normalize the tangent vector. Default is True.
            project (bool): If True, project the result back onto the manifold. Default is True.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Point on the manifold.
        """
        if norm_tan:
            u = self.proju(x, u, dim=dim)
        res = math.expmap(x, u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def expmap0(self, u: torch.Tensor, *, project=True, dim=-1) -> torch.Tensor:
        """
        Perform the exponential map from the origin.

        Parameters:
            u (torch.Tensor): Tangent vector.
            project (bool): If True, project the result back onto the manifold. Default is True.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Point on the manifold.
        """
        res = math.expmap0(u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Perform the logarithmic map to move from a point on the manifold to the tangent space.

        Parameters:
            x (torch.Tensor): Point on the manifold.
            y (torch.Tensor): Point on the manifold.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Tangent vector.
        """
        return math.logmap(x, y, k=self.k, dim=dim)

    def logmap0(self, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Perform the logarithmic map from the origin.

        Parameters:
            y (torch.Tensor): Point on the manifold.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Tangent vector.
        """
        return math.logmap0(y, k=self.k, dim=dim)

    def logmap0back(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Perform the inverse logarithmic map to move from the tangent space to the manifold.

        Parameters:
            x (torch.Tensor): Tangent vector.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Point on the manifold.
        """
        return math.logmap0back(x, k=self.k, dim=dim)

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: Optional[torch.Tensor] = None, *, keepdim=False,
              dim=-1) -> torch.Tensor:
        """
        Compute the inner product of two tangent vectors at a point.

        Parameters:
            x (torch.Tensor): Point on the manifold.
            u (torch.Tensor): First tangent vector.
            v (torch.Tensor, optional): Second tangent vector. If None, uses u.
            keepdim (bool): If True, retains the last dimension. Default is False.
            dim (int): Dimension to compute the inner product.

        Returns:
            torch.Tensor: Inner product.
        """
        if v is None:
            v = u
        return math.inner(u, v, dim=dim, keepdim=keepdim)

    def inner0(self, v: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        """
        Compute the inner product of a tangent vector at the origin.

        Parameters:
            v (torch.Tensor): Tangent vector.
            keepdim (bool): If True, retains the last dimension. Default is False.
            dim (int): Dimension to compute the inner product.

        Returns:
            torch.Tensor: Inner product.
        """
        return math.inner0(v, k=self.k, dim=dim, keepdim=keepdim)

    def cinner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-inner product of two points.

        Parameters:
            x (torch.Tensor): First point.
            y (torch.Tensor): Second point.

        Returns:
            torch.Tensor: Cross-inner product.
        """
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Perform parallel transport of a tangent vector.

        Parameters:
            x (torch.Tensor): Starting point on the manifold.
            y (torch.Tensor): Ending point on the manifold.
            v (torch.Tensor): Tangent vector to transport.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Transported tangent vector.
        """
        return math.parallel_transport(x, y, v, k=self.k, dim=dim)

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Perform parallel transport from the origin.

        Parameters:
            y (torch.Tensor): Ending point on the manifold.
            u (torch.Tensor): Tangent vector to transport.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Transported tangent vector.
        """
        return math.parallel_transport0(y, u, k=self.k, dim=dim)

    def transp0back(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Perform inverse parallel transport to the origin.

        Parameters:
            x (torch.Tensor): Starting point on the manifold.
            u (torch.Tensor): Tangent vector to transport.
            dim (int): Dimension to perform the operation.

        Returns:
            torch.Tensor: Transported tangent vector.
        """
        return math.parallel_transport0back(x, u, k=self.k, dim=dim)

    def transp_follow_expmap(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1,
                             project=True) -> torch.Tensor:
        """
        Perform parallel transport following an exponential map.

        Parameters:
            x (torch.Tensor): Starting point on the manifold.
            u (torch.Tensor): Tangent vector for exponential map.
            v (torch.Tensor): Tangent vector to transport.
            dim (int): Dimension to perform the operation.
            project (bool): If True, project the result back onto the manifold. Default is True.

        Returns:
            torch.Tensor: Transported tangent vector.
        """
        y = self.expmap(x, u, dim=dim, project=project)
        return self.transp(x, y, v, dim=dim)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Perform Mobius addition of two points.

        Parameters:
            x (torch.Tensor): First point.
            y (torch.Tensor): Second point.

        Returns:
            torch.Tensor: Result of Mobius addition.
        """
        v = self.logmap0(y)
        v = self.transp0(x, v)
        return self.expmap(x, v)

    def geodesic_unit(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor, *, dim=-1, project=True) -> torch.Tensor:
        """
        Compute a point on a geodesic given a time parameter.

        Parameters:
            t (torch.Tensor): Time parameter.
            x (torch.Tensor): Starting point on the manifold.
            u (torch.Tensor): Tangent vector.
            dim (int): Dimension to perform the operation.
            project (bool): If True, project the result back onto the manifold. Default is True.

        Returns:
            torch.Tensor: Point on the geodesic.
        """
        res = math.geodesic_unit(t, x, u, k=self.k)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def random_normal(self, *size, mean=0, std=1, dtype=None, device=None) -> geoopt.ManifoldTensor:
        """
        Create a random point on the manifold, induced by a normal distribution on the tangent space of zero.

        Parameters:
            size: Desired shape.
            mean (float or torch.Tensor): Mean value for the normal distribution.
            std (float or torch.Tensor): Standard deviation value for the normal distribution.
            dtype (torch.dtype): Target dtype for the sample. Should match manifold dtype if not None.
            device (torch.device): Target device for the sample. Should match manifold device if not None.

        Returns:
            geoopt.ManifoldTensor: Random points on the hyperboloid.
        """
        self._assert_check_shape(size2shape(*size), "x")
        if device is not None and device != self.k.device:
            raise ValueError("`device` does not match the projector `device`, set the `device` argument to None")
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError("`dtype` does not match the projector `dtype`, set the `dtype` argument to None")
        tens = torch.randn(*size, device=self.k.device, dtype=self.k.dtype) * std + mean
        tens /= tens.norm(dim=-1, keepdim=True)
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)

    def origin(self, *size, dtype=None, device=None, seed=42) -> geoopt.ManifoldTensor:
        """
        Create a zero point origin on the manifold.

        Parameters:
            size: Desired shape.
            dtype (torch.dtype): Desired dtype.
            device (torch.device): Desired device.
            seed (int): Ignored.

        Returns:
            geoopt.ManifoldTensor: Zero point on the manifold.
        """
        if dtype is None:
            dtype = self.k.dtype
        if device is None:
            device = self.k.device

        zero_point = torch.zeros(*size, dtype=dtype, device=device)
        zero_point[..., 0] = torch.sqrt(self.k)
        return geoopt.ManifoldTensor(zero_point, manifold=self)

    def mid_point(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the midpoint of points on the manifold.

        Parameters:
            x (torch.Tensor): Points on the manifold.
            w (torch.Tensor, optional): Weights for each point. Default is None.

        Returns:
            torch.Tensor: Midpoint.
        """
        if w is not None:
            ave = w.matmul(x)
        else:
            ave = x.mean(dim=-2)
        denom = (-self.inner(ave, ave, keepdim=True)).abs().clamp_min(1e-8).sqrt()
        return self.k.sqrt() * ave / denom

    def square_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the squared distance between two points on the manifold.

        Parameters:
            x (torch.Tensor): First point.
            y (torch.Tensor): Second point.

        Returns:
            torch.Tensor: Squared distance between x and y.
        """
        return -2 * self.k - 2 * self.inner(x, y, keepdim=True)
