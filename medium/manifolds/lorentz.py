import torch.nn
from typing import Tuple, Optional
import manifolds.lorentz_math as math
import geoopt
from geoopt import Manifold
from geoopt import Lorentz as LorentzOri
from geoopt.utils import size2shape


class Lorentz(LorentzOri):
    def __init__(self, k=1.0, learnable=False):
        """
        Initialize a Lorentz manifold with k, curvature is -1/k.

        Parameters:
            k (float): Curvature parameter of the manifold.
            learnable (bool): If True, k is learnable. Default is False.
        """
        super().__init__(k, learnable)

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

    def cdist(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        """
        Compute pairwise distances between points in the Lorentz model.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of points on the manifold.
        y : torch.Tensor
            Tensor of points on the manifold.
        dim : int
            Dimension along which the computation is performed.
        keepdim : bool
            Whether to keep the reduced dimension.

        Returns
        -------
        torch.Tensor
            Pairwise distances between points.
        """
        x = x.clone()
        x.narrow(dim, 0, 1).mul_(-1)
        return torch.sqrt(self.k) * math.acosh(-(x @ y.transpose(-1, -2)) / self.k)

    def lorentz_to_klein(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Convert points from Lorentz model to Klein model.

        Parameters
        ----------
        x : torch.Tensor
            Points in the Lorentz model.
        dim : int
            Dimension of the spatial coordinates.

        Returns
        -------
        torch.Tensor
            Points in the Klein model.
        """
        spatial_coords = x.narrow(dim, 1, x.size(dim) - 1)
        time_like = x.narrow(dim, 0, 1)
        klein_coords = spatial_coords / time_like
        return klein_coords

    def klein_to_lorentz(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Convert points from Klein model to Lorentz model.

        Parameters
        ----------
        x : torch.Tensor
            Points in the Klein model.
        dim : int
            Dimension of the spatial coordinates.

        Returns
        -------
        torch.Tensor
            Points in the Lorentz model.
        """
        norm = (x * x).sum(dim=dim, keepdim=True)
        time_like = torch.sqrt(self.k * (1 + norm))  # Ensure time-like dimension > 1
        spatial_coords = torch.sqrt(self.k) * x
        return torch.cat([time_like, spatial_coords], dim=dim)

    def lorentz_to_poincare(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Convert points from Lorentz model to Poincare model.

        Parameters
        ----------
        x : torch.Tensor
            Points in the Lorentz model.
        dim : int
            Dimension of the coordinates.

        Returns
        -------
        torch.Tensor
            Points in the Poincare model.
        """
        return math.lorentz_to_poincare(x, self.k, dim=dim)

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

    def cinner(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        """
        Compute the Lorentzian inner product.

        Parameters
        ----------
        x : torch.Tensor
            First tensor.
        y : torch.Tensor
            Second tensor.
        dim : int
            Dimension along which to compute.
        keepdim : bool
            Whether to keep the reduced dimension.

        Returns
        -------
        torch.Tensor
            Lorentzian inner product.
        """
        x = x.clone()
        x.narrow(dim, 0, 1).mul_(-1)
        return (x @ y.transpose(dim, -2)) / self.k

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

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        """
        Perform Möbius addition on the Lorentz manifold.

        Parameters
        ----------
        x : torch.Tensor
            First tensor.
        y : torch.Tensor
            Second tensor.
        dim : int
            Dimension of the coordinates.

        Returns
        -------
        torch.Tensor
            Result of Möbius addition.
        """
        v = self.logmap0(y, dim=dim)
        v = self.transp0(x, v, dim=dim)
        return self.expmap(x, v, dim=dim)

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

    def random_normal(
        self, *size, mean=0, std=1, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        r"""
        Create a point on the manifold, measure is induced by Normal distribution on the tangent space of zero.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        dtype: torch.dtype
            target dtype for sample, if not None, should match Manifold dtype
        device: torch.device
            target device for sample, if not None, should match Manifold device

        Returns
        -------
        ManifoldTensor
            random points on Hyperboloid

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        if device is not None and device != self.k.device:
            raise ValueError(
                "`device` does not match the projector `device`, set the `device` argument to None"
            )
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError(
                "`dtype` does not match the projector `dtype`, set the `dtype` arguement to None"
            )
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

    def mid_point(self, x: torch.Tensor, w: Optional[torch.Tensor] = None, *, dim=-1) -> torch.Tensor:
        """
        Compute the midpoint of points on the Lorentz manifold.

        Parameters
        ----------
        x : torch.Tensor
            Points on the manifold.
        w : Optional[torch.Tensor]
            Weights for the midpoint computation.
        dim : int
            Dimension of the coordinates.

        Returns
        -------
        torch.Tensor
            Midpoint on the manifold.
        """
        if w is not None:
            ave = w @ x
        else:
            ave = x.mean(dim=-2)
        denom = (-self.inner(ave, ave, dim=dim, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        return torch.sqrt(self.k) * ave / denom

    def square_dist(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, keepdim=True) -> torch.Tensor:
        """
        Compute squared distances between points on the Lorentz manifold.

        Parameters
        ----------
        x : torch.Tensor
            First tensor of points.
        y : torch.Tensor
            Second tensor of points.
        dim : int
            Dimension of the coordinates.
        keepdim : bool
            Whether to keep the reduced dimension.

        Returns
        -------
        torch.Tensor
            Squared distances between points.
        """
        return -2 * self.k - 2 * self.inner(x, y, dim=dim, keepdim=keepdim)