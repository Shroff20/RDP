import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_fake_data() -> pd.DataFrame:
    """_Make some fake data for testing purposes.

    Returns:
        df (pd.DataFrame): DataFrame with time as index
    """

    t = np.linspace(0, 3, 200) ** 2
    y1 = np.sin(t)
    y2 = np.cos(1.3 * t)
    y3 = np.heaviside(t - 1, 0) - 0.5
    y = np.vstack((y1, y2, y3)).T
    y[175:, :] = np.array([[-0.1, 0, 0.1]])
    df = pd.DataFrame(y, columns=["signal1", "signal2", "signal3"], index=t)
    df.index.name = "t"
    print(df)
    return df


class PointReducer:
    """Reduces the number of points in time series data using a linear interpolation error tolerance."""

    def __init__(self):
        pass

    def reduce_points(
        self,
        df,
        tolerance: float,
        plot: bool = True,
        normalize: bool = True,
        include_global_extrema: bool = True,
    ):
        """Reduces the number of points in the data using a linear interpolation error tolerance.

        Args:
            df (pd.DataFrame): DataFrame with time as index and signals as columns.
            tolerance (float, optional): The maximum allowable linear interpolation error. Defaults to 0.01 (1%).
            plot (bool, optional): Whether to create plots to visualize the original data, optimized data, and error. Defaults to True.
            normalize (bool, optional): Whether to normalize the data to range [0, 1] for each signal. Defaults to True.
            include_global_extrema (bool, optional): Whether to include global extrema as critical points. Defaults to True.

        Returns:
            df_optimized (pd.DataFrame): DataFrame with time as index and signals as columns containing the reduced points.
        """

        I_crit = PointReducer._get_crit_points(
            df, tolerance, normalize, include_global_extrema
        )
        df_optimized = df.loc[df.index[I_crit], :]
        df_error = PointReducer._calculate_linear_interp_error(df, I_crit)

        if plot:
            PointReducer._make_plots(df, df_optimized, df_error, tolerance)

        self.tolerance = tolerance
        self.df_orig = df
        self.normalize = normalize
        self.include_global_extrema = include_global_extrema
        self.I_crit = I_crit
        self.df_optimized = df_optimized
        self.df_error = df_error

        return df_optimized

    def _normalize_data(df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes the data to range [0, 1] for each signal.
        Args:
            df (pd.DataFrame): DataFrame with time as index and signals as columns.
        Returns:
            df (pd.DataFrame): Normalized DataFrame with time as index and signals as columns.
        """

        df = (df - df.min()) / (df.max() - df.min())
        return df

    def _get_distances(df) -> np.ndarray:
        """Calculates the linear interpolation error assuming fit based on first and last points.

        Args:
            df (pd.DataFrame): DataFrame with time as index and signals as columns.

        Returns:
            error (np.ndarray): The linear interpolation error for each intermediate point with shape (N_timepoints, N_signals).
        """

        t1 = df.index[0].reshape(-1, 1)
        t2 = df.index[-1].reshape(-1, 1)
        tn = df.index.values.reshape(-1, 1)
        values = df.values

        m = (values[-1, :] - values[0, :]) / (t2 - t1)
        y_fit = m * (tn - t1) + values[0, :]
        error = y_fit - values
        return error

    def _get_crit_points(
        df: pd.DataFrame,
        tolerance: float = 0.01,
        normalize: bool = True,
        include_global_extrema: bool = True,
    ) -> np.ndarray:
        """Finds the critical points in the data using a linear interpolation error tolerance.

        Args:
            df (pd.DataFrame): DataFrame with time as index and signals as columns.
            tolerance (float, optional): The maximum allowable linear interpolation error. Defaults to 0.01 (1%).
            normalize (bool, optional): Whether to normalize the data to range [0, 1] for each signal. Defaults to True.
            include_global_extrema (bool, optional): Whether to include global extrema as critical points. Defaults to True.

        Returns:
            I_crit (np.ndarray): Boolean array indicating critical points with shape (N_timepoints,).

        """

        if normalize == True:
            df = PointReducer._normalize_data(df)

        N_timepoints = len(df)
        N_signals = len(df.columns)
        I_crit = np.zeros(N_timepoints, dtype=bool)

        I_crit[0] = True
        I_crit[-1] = True

        if include_global_extrema:
            I_min = df.values.argmin(axis=0)
            I_max = df.values.argmax(axis=0)
            I_crit[I_min] = True
            I_crit[I_max] = True

        I_last_point = 0

        for i in range(1, N_timepoints):

            if I_crit[i - 1] == True:
                I_last_point = i - 1
            else:
                I_slice = df.index[I_last_point : i + 1]
                df_cur = df.loc[I_slice, :]
                error = PointReducer._get_distances(df_cur)

                if np.any(np.abs(error) > tolerance):
                    I_last_point = i - 1
                    I_crit[I_last_point] = True

        return I_crit

    def _calculate_linear_interp_error(
        df: pd.DataFrame, I_crit: np.ndarray
    ) -> pd.DataFrame:
        """Calculates the linear interpolation error assuming fit based on critical points.

        Args:
            df (pd.DataFrame): DataFrame with time as index and signals as columns.
            I_crit (np.ndarray): Boolean array indicating critical points with shape (N_timepoints,).

        Returns:
            df_error (pd.DataFrame): DataFrame with time as index and signals as columns containing the linear interpolation error.
        """

        df_reduced = df * np.nan
        df_reduced.loc[df.index[I_crit]] = df.loc[df.index[I_crit]]
        df_reduced = df_reduced.interpolate(method="index")
        df_error = df_reduced - df

        return df_error

    def _make_plots(
        df: pd.DataFrame,
        df_optimized: pd.DataFrame,
        df_error: pd.DataFrame,
        tolerance: float,
    ):
        """Creates plots to visualize the original data, optimized data, and error."""

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(df, marker="x", label=df.columns)
        ax[0].plot(df_optimized, marker="o", color="k")
        ax[0].set_title(
            f"reduced {len(df)} points to {len(df_optimized)} points using tolerance = {tolerance} ({tolerance*100:.1f}%)"
        )
        ax[0].set_xlabel("t")
        ax[0].set_ylabel("y")
        ax[1].plot(df_error, label=df.columns)
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("absolute error")
        ax[0].legend(loc="upper right")
        ax[1].legend(loc="upper right")
        fig.set_size_inches(10, 4)
        return fig
