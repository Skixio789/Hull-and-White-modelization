import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd

class HullWhite1F:
    def __init__(self, a: float, sigma: float, forward_curve: np.ndarray, time: list):
        self.a = a
        self.sigma = sigma
        self.forward_curve = np.asarray(forward_curve)
        self.time = time
        self._df_dt = np.empty(len(self.forward_curve))
        self._df_dt[1:-1] = (self.forward_curve[2:] - self.forward_curve[:-2])/ \
            (self.time[2:] - self.time[:-2])
        # left border (right derivative)
        self._df_dt[0]    = (self.forward_curve[1] - self.forward_curve[0]) / \
                            (self.time[1] - self.time[0])
        # right border (left derivative)
        self._df_dt[-1]   = (self.forward_curve[-1] - self.forward_curve[-2]) / \
                            (self.time[-1] - self.time[-2])
        self.dt = time[1]-time[0]
        self._log_P = np.concatenate([[0.0], -np.cumsum(self.forward_curve * self.dt)])[:-1]

    def _f(self,idx):
        return self.forward_curve[idx]
    
    def _df(self, idx):
        return self._df_dt[idx]
    
    def generate_path_HW(self,n_path):
        timestep=len(self.time)
        r=np.zeros((n_path,timestep))
        r[:,0]=self._f(0)
        alpha_t=self.forward_curve+self.sigma**2/(2*self.a**2)*(1-np.exp(-self.a*self.time))**2
        for i in range(timestep-1):
            mean=np.exp(-self.a*self.dt)*(r[:,i]-alpha_t[i])+alpha_t[i+1]
            var=self.sigma**2/(2*self.a)*(1-np.exp(-2*self.a*self.dt))
            r[:,i+1]=np.random.normal(mean,np.sqrt(var),n_path)
        return r

    def price_european_bond_option_HW(self,t_idx,T_idx,S_idx,strike,omega):
        #call: omega = 1, put: omega = -1
        B_T_S=(1-np.exp(-self.a*(self.time[S_idx]-self.time[T_idx])))/self.a
        sigma_p=self.sigma*np.sqrt((1-np.exp(-2*self.a*(self.time[T_idx]-self.time[t_idx])))/(2*self.a))*B_T_S
        P_t_S=self.discount_factor(t_idx,S_idx)
        P_t_T=self.discount_factor(t_idx,T_idx)

        if t_idx == T_idx:
            return np.maximum(omega*(P_t_S-strike),0)
        else:
            h=1/sigma_p*np.log(P_t_S/(P_t_T*strike))+sigma_p/2
            ZBO=omega*(P_t_S*norm.cdf(omega*h)-strike*P_t_T*norm.cdf(omega*(h-sigma_p)))
            return ZBO

    def discount_factor(self,t_idx, T_idx):
        return np.exp(self._log_P[T_idx] - self._log_P[t_idx])
    def discount_factor_HW(self, t_idx, T_idx, r=None):
        B_t_T = self.B_t_T_HW(t_idx, T_idx)  
        A_t_T = self.A_t_T_HW(t_idx, T_idx)

        # cas sans r : on utilise le forward rate
        if r is None:
            return A_t_T * np.exp(-B_t_T * self._f(0)) 

        r_t = r[:, t_idx] if np.ndim(r) == 2 else r[t_idx]  

        B_t_T = np.atleast_1d(B_t_T)  # (n,)
        A_t_T = np.atleast_1d(A_t_T)  # (n,)
        r_t   = np.atleast_1d(r_t)    # (n_paths,)

        return A_t_T[:, None] * np.exp(-B_t_T[:, None] * r_t[None, :])

    def black_formula(self,strike,F,sigma,omega):
        #sigma: implied vol * sqrt(time)
        #F: underlying asset
        #omega = 1 for a call, omega = -1 for a put
        d_1 = (np.log(F/strike)+sigma**2/2)/sigma
        d_2 = d_1-sigma
        return F*omega*norm.cdf(omega*d_1)-strike*omega*norm.cdf(omega*d_2)
    
    def swaption_black(self, strike, implied_vol, maturity, tenor, omega, N=1, period=3):
        # omega = 1 : payer swaption, omega = -1 : receiver swaption
        payment_dates = np.array(list(range(maturity + period, maturity + tenor + 1, period)))
        tau_i        = self.time[payment_dates[0]] - self.time[maturity]
        annuity      = tau_i * np.sum(self.discount_factor(0, payment_dates))
        S            = self.swap_rate(0,maturity, tenor, period)
        vol_sqrt_T   = implied_vol * np.sqrt(self.time[maturity])
        black_price  = self.black_formula(strike, S, vol_sqrt_T, omega)
        price        = N * annuity * black_price
        return price
    
    def B_t_T_HW(self,t_idx,T_idx):
        B_t_T=1/self.a*(1-np.exp(-self.a*(self.time[T_idx]-self.time[t_idx])))
        return B_t_T
    
    def A_t_T_HW(self, t_idx, T_idx):
        A_t_T = (self.discount_factor(0,T_idx) / self.discount_factor(0,t_idx)
                * np.exp(self.B_t_T_HW(t_idx, T_idx) * self._f(t_idx)
                        - self.sigma**2 / (4 * self.a) 
                        * (1 - np.exp(-2 * self.a * self.time[t_idx])) 
                        * self.B_t_T_HW(t_idx, T_idx)**2))
        return A_t_T
    
    def r_star_Jamshidian(self, maturity, tenor, strike, period=3):
        payment_dates = np.array(list(range(maturity + period, tenor + maturity + 1, period)))
        tau_i = self.time[payment_dates[0]] - self.time[maturity]
        c_i = np.ones(len(payment_dates)) * strike * tau_i
        c_i[-1] += 1
        A = self.A_t_T_HW(maturity, payment_dates)
        B = self.B_t_T_HW(maturity, payment_dates)
        def f(r_star):
            return sum(c_i * A * np.exp(-B * r_star)) - 1
        r_star = brentq(f, -0.5, 0.5)
        return r_star
    
    def swaption_Jamshidian(self,maturity, tenor, strike, omega, N=1, period=3):
        #omega = 1 for a payer swaption, omega = -1 for a receiver swaption
        payment_dates = np.array(list(range(maturity + period, tenor + maturity + 1, period)))
        tau_i = self.time[payment_dates[0]] - self.time[maturity]
        c_i = np.ones(len(payment_dates)) * strike * tau_i
        c_i[-1] += 1
        r_star = self.r_star_Jamshidian(maturity, tenor, strike, period)
        strike_i = self.A_t_T_HW(maturity, payment_dates)*np.exp(-self.B_t_T_HW(maturity, payment_dates)*r_star)
        price = np.sum(c_i*self.price_european_bond_option_HW(0,maturity,payment_dates,strike_i,-omega))#-omega because we price put for a payer swaption
        return N*price
    
    def swap(self,t,first_reset_day,tenor,omega,swap_rate,period=3):
        #omega = 1 for a payer swap, omega = -1 for a receiver swap
        payment_dates = np.array(list(range(first_reset_day + period, tenor+first_reset_day + 1, period)))
        tau_i = self.time[payment_dates[0]] - self.time[first_reset_day]
        price = self.discount_factor(t,first_reset_day)-self.discount_factor(t,payment_dates[-1])-swap_rate*np.sum(tau_i*self.discount_factor(t,payment_dates),0)
        return omega*price
    
    def swap_HW(self,t,first_reset_day,tenor,omega,swap_rate,r,period=3):
        #omega = 1 for a payer swap, omega = -1 for a receiver swap
        payment_dates = np.array(list(range(first_reset_day + period, tenor+first_reset_day + 1, period)))
        tau_i = self.time[payment_dates[0]] - self.time[first_reset_day]
        price = self.discount_factor_HW(t,first_reset_day,r)-self.discount_factor_HW(t,payment_dates[-1],r)-swap_rate*np.sum(tau_i*self.discount_factor_HW(t,payment_dates,r),0)
        return omega*price
    
    def swap_rate(self,t,first_reset_day, tenor, period=3):
        payment_dates = np.array(list(range(first_reset_day + period, tenor+first_reset_day + 1, period)))
        tau_i = self.time[payment_dates[0]] - self.time[first_reset_day]
        S_alpha_beta = (self.discount_factor(t,first_reset_day)-self.discount_factor(t,payment_dates[-1]))/np.sum(tau_i*self.discount_factor(t,payment_dates))
        return S_alpha_beta
    
    def calibration_report(model, CalibrationData):
        def implied_vol(model_price, maturity, tenor, strike):
            f = lambda v: model.swaption_black(strike, v, maturity, tenor, omega=1) - model_price
            return brentq(f, 1e-6, 2.0)
        
        data = {}
        for i in range(len(CalibrationData["maturity"])):
            maturity   = CalibrationData["maturity"][i]
            tenor      = CalibrationData["tenor"][i]
            market_vol = CalibrationData["volatility"][i]
            strike     = model.swap_rate(0,maturity, tenor)
            market_price = model.swaption_black(strike, market_vol, maturity, tenor, omega=1)
            model_price  = model.swaption_Jamshidian(maturity, tenor, strike, omega=1)

            data.setdefault("Model Price",  []).append(round(model_price,  6))
            data.setdefault("Market Price", []).append(round(market_price, 6))
            data.setdefault("Model Vol",    []).append(round(implied_vol(model_price, maturity, tenor, strike), 4))
            data.setdefault("Market Vol",   []).append(round(market_vol,   4))
            data.setdefault("Error Price",  []).append(round(model_price - market_price, 6))
            data.setdefault("Error Vol",    []).append(round(implied_vol(model_price, maturity, tenor, strike) - market_vol, 4))

        index = [f"{int(maturity/12)}Y x {int(tenor/12)}Y" 
                for maturity, tenor in zip(CalibrationData["maturity"], CalibrationData["tenor"])]
        df = pd.DataFrame(data, index=index)
        print(df.to_string())
        print(f"\nCumulated error price : {df['Error Price'].abs().sum():.6f}")
        print(f"Cumulated error vol   : {df['Error Vol'].abs().sum():.4f}")

    def constant_maturity_swap(self, cms_maturity, swap_tenor, swaption_surface, period=12):
        #We only price here the floating leg, as the fix leg is trivial
        payment_dates = np.array(list(range(period, cms_maturity + 1, period)))
        floating_leg = 0
        for payment_date in payment_dates:
            floating_leg+=self.constant_maturity_swap_coupon(payment_date,swap_tenor,swaption_surface,period)
        return floating_leg
    
    def constant_maturity_swap_coupon(self,payment_date,swap_tenor, swaption_surface,period = 12):
        #This function give the present value of one coupon payment from the CMS

        resetting_date = payment_date-period
        tau = self.time[payment_date]-self.time[resetting_date]
        discounting = self.discount_factor(0,payment_date)
        if resetting_date == 0:
            return tau*discounting*self.swap_rate(0,0,swap_tenor,period)
        else:
            swap_rate = self.swap_rate(0,resetting_date,swap_tenor,period)

            swaption_surface_row = int(resetting_date/3)-1
            swaption_surface_col = int(swap_tenor/3)-1
            swaption_vol = swaption_surface[swaption_surface_row,swaption_surface_col]/1E2

            epsilon = 1E-4 #1 bps
            def psi(y):
                return self.convexity_adjustment_psi(y,resetting_date,swap_tenor,period)
            first_derivative_psi = (psi(swap_rate+epsilon)-psi(swap_rate-epsilon))/(2*epsilon)
            second_derivative_psi = (psi(swap_rate + epsilon) - 2*psi(swap_rate) + psi(swap_rate - epsilon)) / (epsilon**2)
            
            cms_coupon = tau*discounting*(swap_rate-0.5*swap_rate**2*self.time[resetting_date]*swaption_vol**2*second_derivative_psi/first_derivative_psi)
            return cms_coupon
    
    def convexity_adjustment_psi(self,y,resetting_date,swap_tenor,period =12):
        #T_i : resetting date, T_{i+c} : swap tenor, 
        T_i = resetting_date
        T_i_c = swap_tenor
        swap_rate = self.swap_rate(0,T_i,T_i_c,period)
        payment_dates = np.array(list(range(T_i+period, T_i_c + T_i + 1, period)))
        tau_i = self.time[payment_dates[0]] - self.time[T_i]
        value = [tau_i * self.discount_factor(0, T_i) / (1+y)**(self.time[payment_dates[j]] - self.time[T_i]) 
                 for j in range(len(payment_dates))]
        return swap_rate*np.sum(value)
    
    def cap_Jamshidian(self,t,t_idx,T_maturity,K,N=1,period=3):
        if t>t_idx:
            raise Exception("Pricing date invalid")
        payment_dates = np.array(list(range(t_idx + period,t_idx + T_maturity + 1, period)))
        tau = self.time[payment_dates[0]]-self.time[t_idx]
        cap_value = 0
        for t_i in payment_dates:
            cap_value+=N*(1+tau*K)*self.price_european_bond_option_HW(t,t_i-period,t_i,1/(1+tau*K),-1)
        return cap_value   
     
    def floor_Jamshidian(self,t,t_idx,T_maturity,K,N=1,period=3):
        if t>t_idx:
            raise Exception("Pricing date invalid")
        payment_dates = np.array(list(range(t_idx + period, t_idx + T_maturity + 1, period)))
        tau = self.time[payment_dates[0]]-self.time[t_idx]
        floor_value = 0
        for t_i in payment_dates:
            floor_value+=N*(1+tau*K)*self.price_european_bond_option_HW(t,t_i-period,t_i,1/(1+tau*K),1)
        return floor_value  

    def caplet_black(self, first_reset_day, strike, implied_vol, period=3, N=1):
        payment_date = first_reset_day + period
        tau = self.time[payment_date] - self.time[first_reset_day]
        forward_rate = 1/tau * (self.discount_factor(0, first_reset_day) / self.discount_factor(0, payment_date) - 1)

        if first_reset_day == 0:
            return N * tau * self.discount_factor(0, payment_date) * max(forward_rate - strike, 0)
        else:
            annualized_vol = implied_vol * np.sqrt(self.time[first_reset_day])
            return N * self.discount_factor(0, payment_date) * tau * self.black_formula(strike, forward_rate, annualized_vol, 1)
        
    def floorlet_black(self, first_reset_day, strike, implied_vol_caplet, period=3, N=1):
        caplet = self.caplet_black(first_reset_day,strike,implied_vol_caplet,period,N)
        swap   = self.swap(0,first_reset_day,period,1,strike,period)
        return caplet-swap #caplet-floorlet parity: caplet-single period swap = floorlet
    
    def cap_black(self, first_reset_day,maturity,strike,vol_surface,period=3,N=1):
        payment_dates = np.array(list(range(first_reset_day + period,first_reset_day + maturity + 1, period)))
        cap_value = 0
        vol_caplet = self.cap_vol_surface(payment_dates,strike,vol_surface,period)
        for i,t_i in enumerate(payment_dates):
            cap_value += self.caplet_black(t_i-period,strike,vol_caplet[i],period,N)
        return cap_value
    
    def floor_black(self, first_reset_day, maturity, strike, vol_surface_cap, period=3, N=1):
        # vol_caplet is assumed to be an array
        floor_value = 0
        cap = self.cap_black(first_reset_day,maturity,strike,vol_surface_cap,period,N)
        swap = self.swap(0,first_reset_day,maturity,1,strike,period)
        return cap-swap #cap-floor parity: cap-swap = floor
    
    def cap_vol_surface(self,payment_dates,strike,vol_surface,period=3):
        vol_caplet = []
        strikes_surface = vol_surface.columns/100
        for t_i in payment_dates:
            vol_line = vol_surface.iloc[int(t_i//3-1)] #serie
            #Interpolation vol with respect to strike
            col = -1
            for i,k in enumerate(strikes_surface):
                if strike>k:
                    col = i
            if col==-1:
                vol_caplet.append(vol_line.iloc[0])
            elif col==len(strikes_surface)-1:
                vol_caplet.append(vol_line.iloc[-1])
            else:
                vol_1 = vol_line.iloc[col]
                vol_2 = vol_line.iloc[col+1]
                strike_1 = strikes_surface[col]
                strike_2 = strikes_surface[col+1]
                slope = (vol_2-vol_1)/(strike_2-strike_1)
                b = vol_1-slope*strike_1
                interpolated_vol = slope*strike+b
                vol_caplet.append(interpolated_vol)
        vol_caplet = np.array(vol_caplet)/1E2
        return vol_caplet
    
    def cap_black_flat(self,first_reset_day, maturity, strike, vol, period=3, N=1):
        """Cap price using a flat vol across all caplets"""
        payment_dates = np.array(list(range(first_reset_day + period, first_reset_day + maturity + 1, period)))
        cap_value = 0
        for t_i in payment_dates:
            cap_value += self.caplet_black(t_i - period, strike, vol, period, N)
        return cap_value
    
    def floor_black_flat(self, first_reset_day, maturity, strike, vol, period=3, N=1):
        #Cap - Floor parity
        cap  = self.cap_black_flat(first_reset_day, maturity, strike, vol, period, N)
        swap = self.swap(0, first_reset_day, maturity, 1, strike, period)
        return cap - swap