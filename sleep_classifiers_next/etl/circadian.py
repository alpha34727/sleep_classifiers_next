import numpy as np
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from sleep_classifiers_next.config import Settings

class CircadianProcessor:
    def __init__(self, settings: Settings):
        self.cfg = settings.features
        self.settings = settings

    def clean_steps_data(self, steps_data: np.ndarray) -> np.ndarray:
        # Remove data where timestamps == 0
        steps_data = steps_data[steps_data[:, 0] > 0]
        # Convert milliseconds to seconds if necessary
        if np.max(steps_data[:, 0]) > 1.5e10:
            steps_data[:, 0] = steps_data[:, 0] / 1000.0
        return steps_data

    def simulate_circadian_model(self, steps_data: np.ndarray, psg_timestamps: np.ndarray, is_mesa: bool = False) -> np.ndarray:
        """
        Simulates the Forger (1999) circadian clock model and returns normalized circadian phase (x)
        for each 30-second epoch of the PSG.
        """
        steps_data = self.clean_steps_data(steps_data)
        
        max_t = np.amax(steps_data[:, 0])
        min_t = np.amin(steps_data[:, 0])
        duration_days = (max_t - min_t) / self.cfg.seconds_per_day

        # Pad steps data if less than 5 days
        max_days_to_average = 5.0
        if duration_days < max_days_to_average:
            base_start = max_t - max_days_to_average * self.cfg.seconds_per_day
            padding_timestamps = []
            padding_values = []
            pad_t = base_start
            cumulative_sum = 0.0
            dt_bin = 600.0  # 10 minutes in seconds
            while pad_t < min_t:
                target_t = base_start + (cumulative_sum % self.cfg.seconds_per_day) + 4.0 * self.cfg.seconds_per_day
                val = np.interp(target_t, steps_data[:, 0], steps_data[:, 1])
                padding_timestamps.append(pad_t)
                padding_values.append(val)
                cumulative_sum += dt_bin
                pad_t = base_start + cumulative_sum
            
            pad_arr = np.column_stack((padding_timestamps, padding_values))
            steps_data = np.vstack((pad_arr, steps_data))
            steps_data = steps_data[np.argsort(steps_data[:, 0])]
            max_t = np.amax(steps_data[:, 0])
            min_t = np.amin(steps_data[:, 0])
            duration_days = (max_t - min_t) / self.cfg.seconds_per_day

        # Binning and averaging across days
        dt = 10.0 * self.cfg.seconds_per_minute  # 10 minutes (600s)
        num_bins_per_day = int(self.cfg.seconds_per_day / dt)  # 144
        
        cropped_start_point = max_t - self.cfg.seconds_per_day * np.floor(duration_days)
        end_point_light = max_t - dt / 2.0
        end_timestamp = max_t + 3.0 * self.cfg.seconds_per_day

        max_psg_t = np.max(psg_timestamps)
        min_psg_t = np.min(psg_timestamps)
        
        if max_psg_t < max_t:
            end_timestamp = max_psg_t + 3.0 * self.cfg.seconds_per_day
            end_point_light = max_psg_t - dt / 2.0
            cropped_start_point = max_psg_t - self.cfg.seconds_per_day * np.floor(duration_days)

        average_steps = np.zeros(num_bins_per_day)
        cumulative_bin_count = np.zeros(num_bins_per_day)

        # Loop to bin steps
        t_bins = np.arange(cropped_start_point + dt / 2.0, end_point_light + dt / 2.0, dt)
        for bin_idx, t in enumerate(t_bins):
            current_bin = bin_idx % num_bins_per_day
            # Find steps in [t - dt/2, t + dt/2)
            indices = np.where((steps_data[:, 0] >= t - dt / 2.0) & (steps_data[:, 0] < t + dt / 2.0))[0]
            average_steps[current_bin] += np.sum(steps_data[indices, 1])
            cumulative_bin_count[current_bin] += 1.0

        # Prevent division by zero
        cumulative_bin_count[cumulative_bin_count == 0] = 1.0
        average_steps = average_steps / cumulative_bin_count

        # Timezone offset calculations
        base_dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        dt_ob = (base_dt + timedelta(seconds=float(cropped_start_point))).astimezone(ZoneInfo("America/New_York"))
        tz_diff = dt_ob.utcoffset().total_seconds() / 3600.0
        start_time_for_avg_day = (dt_ob.hour + tz_diff + dt_ob.minute / 60.0) % 24.0

        # Steps to light mapping
        light_output = []
        threshold = 100 if is_mesa else 20
        for i in range(num_bins_per_day):
            val = 500.0
            hr_of_day = (start_time_for_avg_day + (i + 1) * dt / 3600.0) % 24.0
            if 10.0 < hr_of_day < 16.0:
                val = 1000.0
            elif hr_of_day < 7.0 or hr_of_day > 22.0:
                val = 50.0

            step_val = average_steps[i]
            if step_val < threshold:
                step_val = 0.0
            light_output.append(val * np.sign(step_val))

        light_output = np.array(light_output)

        # lux2alpha
        alpha_output = self.cfg.circadian_a0 * ((light_output / self.cfg.circadian_I0) ** self.cfg.circadian_p)

        # Repeat for 60 days (or numRepDays) to remove initial condition effects
        num_rep_days = 20 if is_mesa else 60
        light_for_sim = np.tile(alpha_output, num_rep_days)
        
        sim_dt = dt  # 600s
        time_for_sim = np.arange(0, len(light_for_sim) * sim_dt, sim_dt)
        shifted_times = time_for_sim - np.min(time_for_sim)
        sim_times_scaled_hours = shifted_times / self.cfg.seconds_per_hour
        
        # Integration parameters
        duration_hours = np.max(sim_times_scaled_hours)
        timestamps_for_sim = np.arange(end_timestamp - num_rep_days * self.cfg.seconds_per_day + sim_dt, end_timestamp + sim_dt, sim_dt)
        if len(timestamps_for_sim) > len(light_for_sim):
            timestamps_for_sim = timestamps_for_sim[:len(light_for_sim)]
        elif len(timestamps_for_sim) < len(light_for_sim):
            light_for_sim = light_for_sim[:len(timestamps_for_sim)]
            sim_times_scaled_hours = sim_times_scaled_hours[:len(timestamps_for_sim)]

        # Run RK4 fixed-step solver
        # Deterministic initial conditions (instead of random ics)
        ics = np.array([0.0, 0.0, 0.0])
        tc, Y = self._run_rk4(sim_times_scaled_hours, light_for_sim, duration_hours, ics)

        # Convert simulation times back to seconds
        timestamps_circadian = tc * self.cfg.seconds_per_hour + np.min(timestamps_for_sim)

        # Interpolate simulation outputs to the PSG epochs
        output_timestamps = np.arange(min_psg_t, min_psg_t + 9 * self.cfg.seconds_per_hour + 30, 30)
        # Ensure we don't go out of bounds of the actual PSG length
        output_timestamps = output_timestamps[output_timestamps <= max_psg_t]

        interp_x = np.interp(output_timestamps, timestamps_circadian, Y[:, 0])
        interp_xc = np.interp(output_timestamps, timestamps_circadian, Y[:, 1])

        # Normalize relative to the first epoch
        first_val = np.interp(psg_timestamps[0], timestamps_circadian, Y[:, 0])
        normalized_x = (interp_x - first_val) / np.amin(Y[:, 0] - first_val)
        normalized_x[normalized_x < self.cfg.circadian_lower_bound] = self.cfg.circadian_lower_bound

        # We need to return the normalized circadian feature aligned with the actual psg_timestamps
        final_feature = np.interp(psg_timestamps, output_timestamps, normalized_x)
        return np.expand_dims(final_feature, axis=1)

    def _run_rk4(self, u_time: np.ndarray, u_light: np.ndarray, dur: float, ics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dt = self.cfg.circadian_rk4_dt
        tspan = np.arange(0, dur + dt, dt)
        if tspan[-1] > dur:
            tspan[-1] = dur
            
        h = np.diff(tspan)
        N = len(tspan)
        Y = np.zeros((N, 3))
        Y[0] = ics

        for i in range(1, N):
            ti = tspan[i-1]
            hi = h[i-1]
            yi = Y[i-1]

            F1 = self._simple_ode(ti, yi, u_time, u_light)
            F2 = self._simple_ode(ti + 0.5 * hi, yi + 0.5 * hi * F1, u_time, u_light)
            F3 = self._simple_ode(ti + 0.5 * hi, yi + 0.5 * hi * F2, u_time, u_light)
            F4 = self._simple_ode(tspan[i], yi + hi * F3, u_time, u_light)

            Y[i] = yi + (hi / 6.0) * (F1 + 2.0 * F2 + 2.0 * F3 + F4)

        return tspan, Y

    def _simple_ode(self, t: float, y: np.ndarray, u_time: np.ndarray, u_light: np.ndarray) -> np.ndarray:
        x, xc, n = y[0], y[1], y[2]

        # Lookup light level
        idx = np.searchsorted(u_time, t, side='left') - 1
        alph = u_light[idx] if idx >= 0 else 0.0

        tx = self.cfg.circadian_tau
        G = self.cfg.circadian_G
        k = self.cfg.circadian_k
        mu = self.cfg.circadian_mu
        b = self.cfg.circadian_b

        Bh = G * (1.0 - n) * alph
        B = Bh * (1.0 - 0.4 * x) * (1.0 - 0.4 * xc)

        dxdt = np.pi / 12.0 * (xc + B)
        dxcdt = np.pi / 12.0 * (mu * (xc - 4.0 * (xc**3) / 3.0) - x * ((24.0 / (0.99669 * tx))**2 + k * B))
        dndt = 60.0 * (alph * (1.0 - n) - b * n)

        return np.array([dxdt, dxcdt, dndt])
