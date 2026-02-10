import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple, Self


class Rebar():
    """
        Base class describing a rebar axis in 3D space

        The rebar axis is estimated from a set of input points using PCA
    """

    def __init__(
            self,
            points: np.ndarray,
            volume_shape: Tuple[int, int, int],
            noise_msrt: int = 4,
            xlim: Optional[Tuple[int, int]] = None,
            ylim: Optional[Tuple[int, int]] = None,
            zlim: Optional[Tuple[int, int]] = None,
            mask: Optional[np.ndarray] = None,
            noise_mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Parameters
        ----------
        points : np.ndarray
            Array of 3D points lying on the rebar axis.
            Shape: (N, 3).

        volume_shape : tuple[int, int, int]
            Shape of the 3D volume (Y, X, Z).

        noise_msrt : float, optional
            Thickness of the noise measurement region around the rebar
            in voxels. Default is 4.

        xlim, ylim, zlim : tuple[int, int] or None, optional
            Spatial limits for the rebar mask along each axis.
            Defined as (min, max) voxel indices.

        mask : np.ndarray or None, optional
            Boolean mask of the rebar body.

        noise_mask : np.ndarray or None, optional
            Boolean mask of the noise measurement region.
        """
        self.points = points
        self.xlim = xlim if xlim else (0, volume_shape[0])
        self.ylim = ylim if ylim else (0, volume_shape[1])
        self.zlim = zlim if zlim else (0, volume_shape[2])
        self.volume_shape = volume_shape
        self.mask = mask
        self.noise_mask = noise_mask
        self.noise_msrt = noise_msrt
        self.direct()

    def direct(self) -> Self:
        """
        Estimate the rebar axis.

        Computes:
        - self.direction : unit direction vector of the rebar axis
        - self.point_on_line : a point lying on the axis
        """

        pca = PCA(n_components=1).fit(self.points)
        self.point_on_line = pca.mean_
        self.direction = pca.components_[0]
        return self

    def dir_validate(self) -> None:
        """
        Validate the estimated axis direction.

        Prints the RMSE of distances from the input points
        to the estimated rebar axis.
        """
        dists = np.array([self.distance_point_to_line(p) for p in self.points])
        print(f"Direction RMSE: {(np.sqrt((dists ** 2).mean())):.3f}")

    def distance_point_to_line(self, point) -> float:
        """
        Compute the perpendicular distance from a point to the rebar axis.

        Parameters
        ----------
        point : np.ndarray
            3D point coordinates (x, y, z).

        Returns
        -------
        float
            Distance from the point to the rebar axis.
        """
        v = point - self.point_on_line
        return np.linalg.norm(v - np.dot(v, self.direction) * self.direction)


class RoundRebar(Rebar):
    """
        Cylindrical (round) rebar representation.
    """

    def __init__(self, radius: float, **kwargs) -> None:
        """
            Parameters
            ----------
            radius : float
                Rebar radius in voxels.

            **kwargs :
                Arguments passed to the base Rebar class.
        """
        super().__init__(**kwargs)
        self.radius = radius
        if self.mask is None or self.noise_mask is None:
            self.create_mask()

    def apply_limits(self, arr: np.ndarray) -> Self:
        """
            Apply spatial limits to a boolean mask.

            Parameters
            ----------
            arr : np.ndarray
                Boolean mask array to be cropped.
        """
        arr[:self.ylim[0]] = False
        arr[self.ylim[1]:] = False
        arr[:, :self.xlim[0]] = False
        arr[:, self.xlim[1]:] = False
        arr[:, :, :self.zlim[0]] = False
        arr[:, :, self.zlim[1]:] = False
        return self

    def create_mask(self) -> Self:
        """
        Create volumetric masks for the rebar and noise region.

        - mask: voxels inside the rebar radius
        - noise_mask: voxels in the shell
          (radius, radius + noise_msrt]
        """
        Y, X, Z = np.meshgrid(
            np.arange(self.volume_shape[0]),  # y
            np.arange(self.volume_shape[1]),  # x
            np.arange(self.volume_shape[2]),  # z
            indexing='ij'
        )

        dx = X - self.point_on_line[0]
        dy = Y - self.point_on_line[1]
        dz = Z - self.point_on_line[2]

        cx = dy * self.direction[2] - dz * self.direction[1]
        cy = dz * self.direction[0] - dx * self.direction[2]
        cz = dx * self.direction[1] - dy * self.direction[0]

        distances = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)

        self.mask = np.zeros(self.volume_shape, dtype=bool)
        self.noise_mask = self.mask.copy()

        self.mask[distances <= self.radius] = True
        self.noise_mask[(distances > self.radius) & (distances <= self.radius + self.noise_msrt)] = True

        self.apply_limits(self.mask)
        self.apply_limits(self.noise_mask)

        return self