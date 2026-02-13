import numpy as np
import pandas as pd

from scipy.stats import t, shapiro, ttest_rel, wilcoxon

from typing import Optional, Tuple, Self

from model.Rebar import Rebar


class DefectFinder():
    """
    Detect discontinuities (gaps) along a rebar axis.

    The algorithm samples local cross-sectional profiles along the estimated
    rebar axis, computes signal-to-noise statistics, and estimates defect
    probabilities using robust statistical models (Student-t CDF or logistic model).

    Gaps are detected as contiguous regions where the rebar signal
    significantly deviates from the expected distribution.
    """

    def __init__(
            self,
            window : int = 5,
            stat: str = 'p95',
            mode: str = 'cdf',
            threshold: float = 0.05,
            min_def_size: int = 1,
            min_space_between_gaps: int = 2,
            steepness = 5,
            sigma_threshold = -3
    ) -> None:
        """
        Parameters
        ----------
        window : int
            Number of sampling steps aggregated along the rebar axis. #TODO

        stat : str
            Statistic used to characterize signal and noise:
            'mean', 'median', or 'p95'.

        mode : str
            Probability estimation method:
            - 'cdf'      : Student-t fitted core distribution tail probability
            - 'logistic' : robust logistic anomaly score in MAD units

        threshold : float
            Probability (or score) threshold for defect detection.

        min_def_size : int
            Minimum number of consecutive samples to form a defect segment.

        min_space_between_gaps : int
            Maximum allowed spacing between defect samples to merge them.

        steepness : float
            Logistic function slope (only for mode='logistic').

        sigma_threshold : float
            Logistic activation threshold in robust z-score units (only for mode='logistic').
        """

        self.volume = None
        self.rebar = None
        self.stat = stat
        self.mode = mode
        self.threshold = threshold
        self.min_def_size = min_def_size
        self.min_space_between_gaps = min_space_between_gaps
        self.df = None
        self.noise_checked = None
        self.window = window
        self.steepness = steepness
        self.sigma_threshold = sigma_threshold

    def fit(
            self,
            volume: np.ndarray,
            rebar: Rebar,
            xlim: Optional[Tuple[int, int]] = None,
            ylim: Optional[Tuple[int, int]] = None,
            zlim: Optional[Tuple[int, int]] = None,
    ) -> Self:
        """
        Compute statistics and prepare the model for defect detection.

        Parameters
        ----------
        volume : np.ndarray
            Original 3D volume data.

        rebar : Rebar
            Rebar object containing axis information and masks.

        xlim, ylim, zlim : tuple[int, int] or None, optional
            Spatial limits for the rebar along each axis.
            Defined as (min, max) voxel indices.

        Returns
        -------
        self
            Fitted DefectFinder instance.
        """
        volume_shape = volume.shape
        self.xlim = xlim if xlim else (0, volume_shape[0])
        self.ylim = ylim if ylim else (0, volume_shape[1])
        self.zlim = zlim if zlim else (0, volume_shape[2])
        self.volume = volume
        self.rebar = rebar
        self.axis_iteration()
        self.denoise()
        self.check_noise()
        if not self.noise_checked:
            return self
        self.estimate_confidences()
        return self

    def predict(self, grouped=True) -> Optional[pd.DataFrame]:
        """
        Detect and group defect regions along the rebar axis.

        Returns
        -------
        pd.DataFrame or None
            Each segment has its own segment_id with voxel coordinates.
        """
        if not self.noise_checked:
            print('Distributions are NOT separable - Rebar signal is indistinguishable from noise')
            return None

        if self.mode == 'cdf':
            df =  self.df[self.df.cdf <= self.threshold][['k', 'x', 'y', 'z', 'cdf']]
        elif self.mode == 'logistic':
            df =  self.df[self.df.logistic >= self.threshold][['k', 'x', 'y', 'z', 'logistic']]

        df = df.sort_values('k').reset_index(drop=True)

        df['segment_id'] = (df['k'].diff() >= self.min_space_between_gaps).cumsum()
        df = df[df.groupby('segment_id')['k'].transform('size') >= self.min_def_size]
        df['segment_id'] = df.groupby('segment_id').ngroup()

        if grouped:
            if self.mode == 'cdf':
                agg = df.groupby('segment_id', as_index=False).agg(
                    x=('x', 'mean'),
                    y=('y', 'mean'),
                    z=('z', 'mean'),
                    k=('k', 'mean'),
                    min_cdf=('cdf', 'min'),
                )
            elif self.mode == 'logistic':
                agg = df.groupby('segment_id', as_index=False).agg(
                    x=('x', 'mean'),
                    y=('y', 'mean'),
                    z=('z', 'mean'),
                    k=('k', 'mean'),
                    max_logistic=('logistic', 'max'),
                )
            agg[['x', 'y', 'z', 'k']] = agg[['x', 'y', 'z', 'k']].round().astype(int)

            return agg

        else:
            return df

    def estimate_confidences(self) -> None:
        """
        Estimate confidences.

        Two modes are supported:
        - 'cdf': Fit Student-t distribution to the core signal and compute tail probabilities.
        - 'logistic': Robust logistic anomaly score using MAD-normalized SNR.
        """
        if not self.noise_checked:
            print('Distributions are NOT separable - Rebar signal is indistinguishable from noise')
            return None

        med = self.df.snr.median()
        sigma = 1.4826 * np.abs((self.df.snr - med)).median()

        if self.mode == 'cdf':
            core = self.df[(self.df.snr > med - 3 * sigma) & (self.df.snr < med + 3 * sigma)].snr
            self.df['cdf'] = t.cdf(self.df.snr, *t.fit(core))
        elif self.mode == 'logistic':
            z_s = (self.df.snr - med) / sigma
            self.df['logistic'] = 1 / (1 + np.exp(self.steepness * (z_s - self.sigma_threshold)))
            self.df.loc[self.df[z_s > 0].index, 'logistic'] = 0
        else:
            raise AttributeError("mode must be in ('cdf', 'logistic')")


    def p5(self, x):
        """Return 5th percentile of the input array."""
        return np.percentile(x, 5)

    def p95(self, x):
        """Return 95th percentile of the input array."""
        return np.percentile(x, 95)

    def axis_iteration(self) -> None:
        """
        Sample signal and noise statistics along the rebar axis.

        For each step along the axis:
        - compute a local center point
        - project cross-section sampling kernels (rebar profile and noise shell)
        - extract voxel intensities
        - compute robust statistics over the sampling window

        The resulting DataFrame contains axis index k and voxel coordinates (x, y, z)
        together with signal and noise statistics.
        """

        if self.stat == 'mean':
            funcs = [np.mean, np.std]
        elif self.stat == 'median':
            funcs = [np.median, self.p5, self.p95]
        elif self.stat == 'p95':
            funcs = [self.p5, self.p95]
        else:
            raise AttributeError("stat must be in ('mean', 'median', 'p95')")


        nx, ny, nz = self.volume.shape
        if not self.xlim:
            self.xlim = (0, nx)
        if not self.ylim:
            self.ylim = (0, ny)
        if not self.zlim:
            self.zlim = (0, nz)

        max_len = int(np.linalg.norm(self.volume.shape))

        # normalized stepping direction along the axis
        step = self.rebar.axis_direction / np.max(np.abs(self.rebar.axis_direction))

        stats = []

        # iterate along the full possible axis extent
        for k in range(-max_len, max_len):

            # aggregate multiple centers for windowed smoothing
            centers = []
            values_in = np.array([])
            values_noise = np.array([])

            for s in range(-self.window//2 + 1, self.window//2 + 1):

                # center point
                p = np.round(self.rebar.point_on_line + (k + s) * step).astype(int)

                # skipping all points outside the volume
                if (
                        self.xlim[0] <= p[0] < self.xlim[1]
                        and self.ylim[0] <= p[1] < self.ylim[1]
                        and self.zlim[0] <= p[2] < self.zlim[1]
                ):
                    centers.append(p)
            if not centers:
                continue

            for p in centers:
                # coordinates of the points of a profile
                idx_in = np.round(p + self.rebar.profile_projection).astype(np.int32)  # shape (N,3)
                idx_noise = np.round(p + self.rebar.noise_msrt_projection).astype(np.int32)  # shape (N,3)

                # adjustment at the boundaries
                mask_in = (
                        (idx_in[:, 0] >= 0) & (idx_in[:, 0] < nx) &
                        (idx_in[:, 1] >= 0) & (idx_in[:, 1] < ny) &
                        (idx_in[:, 2] >= 0) & (idx_in[:, 2] < nz)
                )
                idx_in = idx_in[mask_in]
                if idx_in.shape[0] == 0:
                    continue

                mask_noise = (
                        (idx_noise[:, 0] >= 0) & (idx_noise[:, 0] < nx) &
                        (idx_noise[:, 1] >= 0) & (idx_noise[:, 1] < ny) &
                        (idx_noise[:, 2] >= 0) & (idx_noise[:, 2] < nz)
                )
                idx_noise = idx_noise[mask_noise]

                values_in = np.append(values_in, self.volume[idx_in[:, 0], idx_in[:, 1], idx_in[:, 2]])
                values_noise = np.append(values_noise, self.volume[idx_noise[:, 0], idx_noise[:, 1], idx_noise[:, 2]])

            stats.append((*p, *[f(values_in) for f in funcs], *[f(values_noise) for f in funcs]))

        self.df = pd.DataFrame(stats, columns=['x', 'y', 'z', *[f"in_{f.__name__}" for f in funcs],
                                            *[f"noise_{f.__name__}" for f in funcs]]).reset_index(names='k')


    def denoise(self) -> None:
        """
        Compute signal-to-noise ratio (SNR).

        SNR is defined differently depending on the chosen statistic:
        - mean   : normalized by noise standard deviation
        - median : normalized by inter-percentile range
        - p95    : contrast of high-percentile signal vs noise
        """

        if self.stat == 'mean':
            self.df['snr'] = (self.df['in_mean'] - self.df['noise_mean']) / (self.df['noise_std'] + 1e-16)
        elif self.stat == 'median':
            self.df['snr'] = (self.df['in_median'] - self.df['noise_median']) / (self.df['noise_p95'] - self.df['noise_p5'] + 1e-16)
        elif self.stat == 'p95':
            self.df['snr'] = (self.df['in_p95'] - self.df['noise_p95']) / (self.df['noise_p95'] - self.df['noise_p5'] + 1e-16)
        else:
            raise AttributeError("stat must be in ('mean', 'median', 'p95')")

    def check_noise(self) -> None:
        """
        Test whether rebar signal is statistically separable from noise.

        Shapiro-Wilk test is used to check normality.
        Depending on the result, either paired t-test or Wilcoxon signed-rank test
        is applied.

        Sets self.noise_checked flag.
        """

        in_stat = f'in_{self.stat}'
        noise_stat = f'noise_{self.stat}'

        if shapiro(self.df[in_stat]).pvalue > 0.05:
            stat, p = ttest_rel(
                self.df[in_stat],
                self.df[noise_stat],
                alternative='greater'
            )
            test = 't-test'
        else:
            stat, p = wilcoxon(
                self.df[in_stat],
                self.df[noise_stat],
                alternative='greater'
            )
            test = 'Wilcoxon'

        if p <= 0.05:
            print(
                f"{test} stat: {stat}, p-value: {p:.3f}. Distributions are separable - Rebar is clearly distinguishable from noise")
            self.noise_checked = True
        else:
            print(
                f"{test} stat: {stat}, p-value: {p:.3f}. Distributions are NOT separable - Rebar signal is indistinguishable from noise")
            self.noise_checked = False