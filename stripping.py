import numpy as np

class YieldCurve:
    def __init__(self, dates, discount_factors, ref_date, n_months=360):
        self.ref_date = ref_date
        self.t_market = np.array([(d - ref_date).days / 360 for d in dates]) #in years
        self.df_market = np.asarray(discount_factors)
        self.dt = 1/12
        self.time = np.linspace(0, n_months/12, n_months + 1)
        log_df_interp = np.interp(self.time, self.t_market, np.log(self.df_market))
        self.discount_factors = np.exp(log_df_interp)
        self.forward_curve = -np.diff(log_df_interp) / self.dt
        self.forward_curve = np.append(self.forward_curve, self.forward_curve[-1])