class TimeScaler:

    def fit(self, t):
        """
        Fit the time scaler.

        Parameters:
            t (ndarray): Time indices for each series.

        Returns:
            TimeScaler: The fitted TimeScaler object.
        """
        
        if t.ndim == 1:
            t = t.reshape(1, -1)
        self.t_scale = (t[:, 1:] - t[:, :-1]).flatten().mean()
        self.t_min = t.min()
        return self

    def scale(self, t):
        """
        Transform the time indices.

        Parameters:
            t (ndarray): Time indices for each series.

        Returns:
            ndarray: Transformed time indices.
        """
        return (t - self.t_min) / self.t_scale

    def fit_scale(self, t):
        """
        Fit the time scaler and transform the time indices.

        Parameters:
            t (ndarray): Time indices for each series.

        Returns:
            ndarray: Transformed time indices.
        """
        self.fit(t)
        return self.scale(t)

