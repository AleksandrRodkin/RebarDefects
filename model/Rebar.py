import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple, Self


class Rebar():
    """
    Base class representing a rebar axis in 3D space.

    The rebar axis is estimated from a set of input points using PCA.
    Additionally, an orthonormal local coordinate system is constructed:
        - axis_direction: rebar axis direction
        - v, w: perpendicular unit vectors defining the cross-section plane
    """

    def __init__(
            self,
            points: np.ndarray,
            noise_area_thickness: int = 4,
            noise_area_indent: int = 1,
            profile_projection: Optional[np.ndarray] = None,
            noise_msrt_projection: Optional[np.ndarray] = None
    ) -> None:
        """
        Parameters
        ----------
        points : np.ndarray
            Array of 3D points lying on the rebar axis.
            Shape: (N, 3).

        noise_area_thickness : int, optional
            Thickness of the noise measurement region around the rebar
            in voxels. Default is 4.

        noise_area_indent : int, optional
            Gap between the rebar surface and noise region
            in voxels. Default is 1.

        profile_projection : np.ndarray, optional
            Precomputed local cross-section offsets for the rebar profile.

        noise_msrt_projection : np.ndarray, optional
            Precomputed local cross-section offsets for the noise measurement region.
        """
        self.points = points
        self.profile_projection = profile_projection
        self.noise_msrt_projection = noise_msrt_projection
        self.noise_area_thickness = noise_area_thickness
        self.noise_area_indent = noise_area_indent

        # Estimate axis and construct local basis
        self.direct()

    def orthonormal_basis(self, u):
        """
        Construct an orthonormal basis given a direction vector.

        Parameters
        ----------
        u : np.ndarray
            Direction vector.

        Returns
        -------
        u, v, w : np.ndarray
            Orthonormal basis vectors where v ⟂ u, w ⟂ u and v ⟂ w.
        """
        u = u / np.linalg.norm(u)
        # random vector not parallel to u
        a = np.array([1,0,0]) if abs(u[0]) < 0.9 else np.array([0,1,0])
        v = np.cross(u, a)
        v /= np.linalg.norm(v)
        w = np.cross(u, v)
        return u, v, w


    def direct(self) -> Self:
        """
        Estimate the rebar axis.

        Computes:
        - self.axis_direction : unit direction vector of the rebar axis
        - self.point_on_line : a point lying on the axis
        -self.v, self.w : perpendicular unit vectors defining local cross-section plane
        """

        pca = PCA(n_components=1).fit(self.points)
        self.point_on_line = pca.mean_
        self.axis_direction = pca.components_[0]
        self.axis_direction, self.v, self.w = self.orthonormal_basis(self.axis_direction)
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
            Euclidean distance from the point to the rebar axis.
        """
        v = point - self.point_on_line
        return np.linalg.norm(v - np.dot(v, self.axis_direction) * self.axis_direction)


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
        if self.profile_projection is None or self.noise_msrt_projection is None:
            self.create_projections()

    def precompute_circle(self, radius) -> np.ndarray:
        """
        Compute integer offsets inside a 2D disk.

        Parameters
        ----------
        radius : int
            Circle radius in pixels.

        Returns
        -------
        np.ndarray
            Array of (dx, dy) offsets within the disk.
        """
        offsets = []
        r2 = radius ** 2
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= r2:
                    offsets.append((dx, dy))
        return np.array(offsets, dtype=np.float32)

    def create_projections(self) -> Self:
        """
        Precompute local cross-section projections.

        - profile_projection: voxel offsets inside the rebar radius
        - noise_msrt_projection: voxel offsets in the noise measurement shell
          [radius + indent, radius + indent + thickness]
        """

        # rebar cross-section template
        circle = self.precompute_circle(self.radius)
        dv = circle[:, 0][:, None]
        dw = circle[:, 1][:, None]
        self.profile_projection  = dv * self.v + dw * self.w

        # noise measurement shell template
        ext_b = self.precompute_circle(self.radius + self.noise_area_indent + self.noise_area_thickness)
        int_b = self.precompute_circle(self.radius + self.noise_area_indent)

        noise_msrt = np.setdiff1d(
            ext_b.view(np.dtype((np.void, ext_b.dtype.itemsize * 2))),
            int_b.view(np.dtype((np.void, int_b.dtype.itemsize * 2)))
        ).view(np.float32).reshape(-1, 2)

        dv = noise_msrt[:, 0][:, None]
        dw = noise_msrt[:, 1][:, None]
        self.noise_msrt_projection = dv * self.v + dw * self.w

        return self