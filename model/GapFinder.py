import numpy as np
import pandas as pd

from scipy.stats import norm, shapiro, ttest_rel, wilcoxon

from typing import Optional, List, Self

from model.Rebar import Rebar


class GapFinder():
    """
        Class for detecting discontinuities (gaps) in a rebar.

        The class analyzes signal-to-noise statistics along the rebar axis
        and identifies spatial regions that may correspond to defects.
    """

    def __init__(
            self,
            vol_mode: str = 'v5',
            stat: str = 'p95',
            defect_probs: str = 'mad_pval',
            threshold: float = 0.05,
            min_def_size: int = 1,
            max_gap: int = 2
    ) -> None:
        """
        Parameters
        ----------
        vol_mode : str, optional
            Window size along the x-axis used for statistics computation:
            'lin' (1 voxel), 'v3' (3 voxels), 'v5' (5 voxels).

        stat : str, optional
            Statistic type used for signal evaluation:
            'mean' or 'p95' (percentiles).

        defect_probs : str, optional
            Method for defect probability estimation:
            - 'empirical' : empirical p-value
            - 'pval' : p-value assuming normal distribution
            - 'mad_pval' : robust p-value using MAD estimator

        threshold : float, optional
            Probability threshold below which a location is considered
            a potential defect.

        min_def_size : int, optional
            Minimum defect length along the x-axis (in voxels).

        max_gap : int, optional
            Maximum allowed gap (in voxels) for merging neighboring defects.
        """
        self.volume = None
        self.rebar = None
        self.vol_mode = vol_mode
        self.stat = stat
        self.prob_stat = f'snr_{vol_mode}_{stat}'
        self.defect_probs = defect_probs
        self.threshold = threshold
        self.min_def_size = min_def_size
        self.max_gap = max_gap
        self.df = None
        self.noise_checked = None

    def fit(self, volume: np.ndarray, rebar: Rebar) -> Self:
        """
        Compute statistics and prepare the model for defect detection.

        Parameters
        ----------
        volume : np.ndarray
            Original 3D volume data.

        rebar : Rebar
            Rebar object containing axis information and masks.

        Returns
        -------
        self
            Fitted GapFinder instance.
        """
        self.volume = volume
        self.rebar = rebar
        self.calculate_cut_stats()
        self.denoise()
        self.check_noise()
        if not self.noise_checked:
            return self
        self.calculate_pval()
        return self

    def predict(self) -> Optional[List[pd.DataFrame]]:
        """
        Detect and group defect regions along the rebar axis.

        Returns
        -------
        list of pd.DataFrame or None
            List of defect segments. Each segment is a DataFrame with
            voxel coordinates. Returns None if signal is not separable
            from noise.
        """
        if not self.noise_checked:
            print('Distributions are NOT separable - Rebar signal is indistinguishable from noise')
            return None
        df = self.find_gap_coords().sort_values('x').reset_index(drop=True)
        segments = []
        current = [df.iloc[0]]

        for i in range(1, len(df)):
            if df.loc[i, 'x'] - df.loc[i - 1, 'x'] <= self.max_gap:
                current.append(df.iloc[i])
            else:
                if len(pd.DataFrame(current)) >= self.min_def_size:
                    segments.append(pd.DataFrame(current))
                current = [df.iloc[i]]
        if len(pd.DataFrame(current)) >= self.min_def_size:
            segments.append(pd.DataFrame(current))

        return segments

    def calculate_pval(self) -> None:
        """
        Calculate p-values for all supported statistics.

        Uses empirical, normal, and MAD-based probability estimators.
        """
        if not self.noise_checked:
            print('Distributions are NOT separable - Rebar signal is indistinguishable from noise')
            return None

        for stat in ['snr_v3_mean', 'snr_v5_mean', 'snr_v3_p95', 'snr_v5_p95']:
            self.df[f'p_empirical_{stat}'] = self.df[stat].apply(lambda x: (self.df[stat] <= x).mean())

            self.df[f'p_val_{stat}'] = norm.cdf(
                (self.df[stat] - self.df[stat].mean()) / self.df[stat].std()
            )

            self.df[f'p_val_mad_{stat}'] = norm.cdf(
                (self.df[stat] - self.df[stat].median()) / (
                            1.4826 * np.median(np.abs(self.df[stat] - self.df[stat].median())))
            )

    def find_gap_coords(self) -> pd.DataFrame:
        """
        Select voxel coordinates classified as defects.

        Returns
        -------
        pd.DataFrame
            DataFrame containing coordinates (x, y, z) and
            corresponding probability values.
        """

        if self.defect_probs == 'mad_pval':
            col = f'p_val_mad_snr_{self.vol_mode}_{self.stat}'
        elif self.defect_probs == 'empirical':
            print("""
                Empirical probability reflects an estimate of the likelihood of a defect occurring at a given point in space. 
                This value does not represent the actual probability of the defect's presence at a specific point.
            """)
            col = f'p_empirical_snr_{self.vol_mode}_{self.stat}'

        elif self.defect_probs == 'pval':
            shap = shapiro(self.df[self.prob_stat]).pvalue
            if shap < 0.05:
                raise AttributeError(
                    f"""Shapiro p-val: {shap:.5f} < 0.05, probability estimates will NOT be correct - 
                    try to use another 'stat', 'volume_mode', 'defect_probs'
                    """
                )
            else:
                print(f'Shapiro p-val: {shap:.5f} >= 0.05, probability estimates will be correct')
            col = f'p_val_snr_{self.vol_mode}_{self.stat}'
        else:
            raise AttributeError("defect_probs must be in ('mad_pval', 'pval', 'empirical')")

        return self.df[self.df[col] < self.threshold][['x', 'y', 'z', col]]

    def calculate_cut_stats(self) -> None:
        """
        Compute signal and noise statistics for cross-sections
        along the rebar axis.
        """
        cuts = []
        for x in range(self.rebar.xlim[0], self.rebar.xlim[1]):
            t = (x - self.rebar.point_on_line[0]) * self.rebar.direction[0]
            y = int(self.rebar.point_on_line[1] + t * self.rebar.direction[1])
            z = int(self.rebar.point_on_line[2] + t * self.rebar.direction[2])

            cut = {'x': x, 'y': y, 'z': z}

            for prefix, data in [
                ('lin_in', self.volume[:, x, :][self.rebar.mask[:, x, :]]),
                ('v3_in', self.volume[:, x - 1:x + 2, :][self.rebar.mask[:, x - 1:x + 2, :]]),
                ('v5_in', self.volume[:, x - 2:x + 3, :][self.rebar.mask[:, x - 2:x + 3, :]]),
                ('lin_noise', self.volume[:, x, :][self.rebar.noise_mask[:, x, :]]),
                ('v3_noise', self.volume[:, x - 1:x + 2, :][self.rebar.noise_mask[:, x - 1:x + 2, :]]),
                ('v5_noise', self.volume[:, x - 2:x + 3, :][self.rebar.noise_mask[:, x - 2:x + 3, :]]),
            ]:
                cut[f'{prefix}_mean'] = data.mean()
                cut[f'{prefix}_std'] = data.std(ddof=1)
                cut[f'{prefix}_min'] = data.min()
                cut[f'{prefix}_max'] = data.max()
                cut[f'{prefix}_p5'], cut[f'{prefix}_p95'] = np.percentile(data, (5, 95))

            cuts.append(cut)
        self.df = pd.DataFrame(cuts)

    def denoise(self) -> None:
        """
        Convert raw statistics into signal-to-noise ratio (SNR) metrics.
        """
        self.df['snr_lin_mean'] = (self.df['lin_in_mean'] - self.df['lin_noise_mean']) / (
                    self.df['lin_noise_std'] + 1e-16)
        self.df['snr_lin_p95'] = (self.df['lin_in_p95'] - self.df['lin_noise_p95']) / (
                self.df['lin_noise_p95'] - self.df['lin_noise_p5'] + 1e-16
        )

        self.df['snr_v3_mean'] = (self.df['v3_in_mean'] - self.df['v3_noise_mean']) / (self.df['lin_noise_std'] + 1e-16)
        self.df['snr_v3_p95'] = (self.df['v3_in_p95'] - self.df['v3_noise_p95']) / (
                self.df['v3_noise_p95'] - self.df['v3_noise_p5'] + 1e-16
        )

        self.df['snr_v5_mean'] = (self.df['v5_in_mean'] - self.df['v5_noise_mean']) / (self.df['lin_noise_std'] + 1e-16)
        self.df['snr_v5_p95'] = (self.df['v5_in_p95'] - self.df['v5_noise_p95']) / (
                self.df['v5_noise_p95'] - self.df['v5_noise_p5'] + 1e-16
        )

    def check_noise(self) -> None:
        """
        Test whether rebar signal is statistically separable from noise.
        """

        in_stat = f'{self.vol_mode}_in_{self.stat}'
        noise_stat = f'{self.vol_mode}_noise_{self.stat}'

        if shapiro(self.df[in_stat]).pvalue > 0.05:
            stat, p = ttest_rel(
                self.df[in_stat],
                self.df[noise_stat],
                alternative='greater'
            )
            test = 't-test'
        else:
            stat, p = wilcoxon(
                self.df.v5_in_mean,
                self.df.v5_noise_mean,
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