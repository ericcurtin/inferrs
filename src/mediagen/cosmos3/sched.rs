//! diffusers' `UniPCMultistepScheduler` as Cosmos3 configures it: Karras sigmas
//! mapped to flow sigmas, `predict_x0`, order 2, `bh2`, corrector after step 0.

/// `sigmas` (n+1, last 0) and the integer `timesteps` (n) the transformer is conditioned on.
pub fn schedule(n_steps: usize) -> (Vec<f32>, Vec<i64>) {
    let n = n_steps.max(1);
    let (smin, smax, rho) = (0.147f64, 200.0f64, 7.0f64);
    let (a, b) = (smax.powf(1.0 / rho), smin.powf(1.0 / rho));
    let mut sigmas = Vec::with_capacity(n + 1);
    let mut timesteps = Vec::with_capacity(n);
    for i in 0..n {
        let ramp = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.0
        };
        let k = (a + ramp * (b - a)).powf(rho);
        let s = k / (k + 1.0);
        sigmas.push(s as f32);
        timesteps.push((s * 1000.0).trunc() as i64);
    }
    sigmas.push(0.0);
    (sigmas, timesteps)
}

/// The `use_karras_sigmas=False` schedule the action examples run with:
/// flow sigmas shifted by `shift` (`s' = shift*s / (1 + (shift-1)*s)`), from the
/// pipeline's own linspace when `native` (`Cosmos3-Edge`) or the scheduler's.
pub fn schedule_flow(n_steps: usize, shift: f32, native: bool) -> (Vec<f32>, Vec<i64>) {
    let n = n_steps.max(1);
    let shift = shift as f64;
    let mut sigmas = Vec::with_capacity(n + 1);
    let mut timesteps = Vec::with_capacity(n);
    for i in 0..n {
        // native: linspace(1 - 1/1000, 0, n+1)[:-1]; scheduler: linspace(1, 1/1000, n+1)[:-1]
        let s = if native {
            (1.0 - 1.0 / 1000.0) * (1.0 - i as f64 / n as f64)
        } else {
            1.0 - (1.0 - 1.0 / 1000.0) * i as f64 / n as f64
        };
        let mut s = shift * s / (1.0 + (shift - 1.0) * s);
        if i == 0 && (s - 1.0).abs() < 1e-6 {
            s -= 1e-6;
        }
        sigmas.push(s as f32);
        timesteps.push((s * 1000.0).trunc() as i64);
    }
    sigmas.push(0.0);
    (sigmas, timesteps)
}

/// Per-modality solver state; one per latent stream (vision, sound, action).
pub struct UniPC {
    sigmas: Vec<f32>,
    step: usize,
    /// x0 predictions of the last two steps: `[older, newest]`.
    outputs: [Option<Vec<f32>>; 2],
    last_sample: Option<Vec<f32>>,
    lower_order_nums: usize,
    this_order: usize,
}

fn lam(sigma: f32) -> f32 {
    (1.0 - sigma).ln() - sigma.ln()
}

impl UniPC {
    pub fn new(sigmas: Vec<f32>) -> Self {
        UniPC {
            sigmas,
            step: 0,
            outputs: [None, None],
            last_sample: None,
            lower_order_nums: 0,
            this_order: 1,
        }
    }

    pub fn n_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    pub fn sigma(&self, i: usize) -> f32 {
        self.sigmas[i]
    }

    /// One `scheduler.step(velocity, t, sample)`: returns the next sample.
    pub fn step(&mut self, velocity: &[f32], sample: &[f32]) -> Vec<f32> {
        let i = self.step;
        let s_i = self.sigmas[i];
        // convert_model_output: flow prediction -> x0
        let x0: Vec<f32> = sample
            .iter()
            .zip(velocity)
            .map(|(x, v)| x - s_i * v)
            .collect();

        let mut sample = sample.to_vec();
        if i > 0 && self.last_sample.is_some() {
            sample = self.corrector(&x0, &sample);
        }

        self.outputs[0] = self.outputs[1].take();
        self.outputs[1] = Some(x0);

        let n = self.n_steps();
        let mut order = 2.min(n - i);
        order = order.min(self.lower_order_nums + 1);
        self.this_order = order.max(1);

        self.last_sample = Some(sample.clone());
        let prev = self.predictor(&sample, self.this_order);
        if self.lower_order_nums < 2 {
            self.lower_order_nums += 1;
        }
        self.step += 1;
        prev
    }

    /// multistep_uni_p_bh_update
    fn predictor(&self, x: &[f32], order: usize) -> Vec<f32> {
        let i = self.step;
        let m0 = self.outputs[1].as_ref().expect("model output");
        let sigma_t = self.sigmas[i + 1];
        let sigma_s0 = self.sigmas[i];
        if sigma_t == 0.0 {
            // lambda_t = +inf: the predictor returns the x0 estimate exactly
            return m0.clone();
        }
        let alpha_t = 1.0 - sigma_t;
        let h = lam(sigma_t) - lam(sigma_s0);
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = h_phi_1; // bh2
        let ratio = sigma_t / sigma_s0;
        let mut out: Vec<f32> = x
            .iter()
            .zip(m0)
            .map(|(x, m)| ratio * x - alpha_t * h_phi_1 * m)
            .collect();
        if order == 2 {
            let m1 = self.outputs[0].as_ref().expect("previous model output");
            let rk = (lam(self.sigmas[i - 1]) - lam(sigma_s0)) / h;
            // D1 = (m1 - m0) / rk; pred_res = 0.5 * D1
            for ((o, a), b) in out.iter_mut().zip(m1).zip(m0) {
                let d1 = (a - b) / rk;
                *o -= alpha_t * b_h * 0.5 * d1;
            }
        }
        out
    }

    /// multistep_uni_c_bh_update with `this_model_output = x0` of the current step
    fn corrector(&self, model_t: &[f32], this_sample: &[f32]) -> Vec<f32> {
        let _ = this_sample;
        let i = self.step;
        let order = self.this_order;
        let x = self.last_sample.as_ref().expect("last sample");
        let m0 = self.outputs[1].as_ref().expect("model output");
        let sigma_t = self.sigmas[i];
        let sigma_s0 = self.sigmas[i - 1];
        let alpha_t = 1.0 - sigma_t;
        let h = lam(sigma_t) - lam(sigma_s0);
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = h_phi_1;
        let ratio = sigma_t / sigma_s0;
        let x_t_: Vec<f32> = x
            .iter()
            .zip(m0)
            .map(|(x, m)| ratio * x - alpha_t * h_phi_1 * m)
            .collect();
        if order == 1 || i < 2 || self.outputs[0].is_none() {
            // rhos_c = [0.5]
            return x_t_
                .iter()
                .zip(model_t)
                .zip(m0)
                .map(|((xt, mt), m)| xt - alpha_t * b_h * 0.5 * (mt - m))
                .collect();
        }
        let m1 = self.outputs[0].as_ref().expect("previous model output");
        let r0 = (lam(self.sigmas[i - 2]) - lam(sigma_s0)) / h;
        let b0 = (h_phi_1 / hh - 1.0) / b_h;
        let b1 = 2.0 * ((h_phi_1 / hh - 1.0) / hh - 0.5) / b_h;
        let rho0 = (b1 - b0) / (r0 - 1.0);
        let rho1 = b0 - rho0;
        x_t_.iter()
            .zip(model_t)
            .zip(m0)
            .zip(m1)
            .map(|(((xt, mt), m), m1)| {
                let d1 = (m1 - m) / r0;
                let d1_t = mt - m;
                xt - alpha_t * b_h * (rho0 * d1 + rho1 * d1_t)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_diffusers() {
        let (s, t) = schedule(4);
        let want = [0.99502486f32, 0.9736331, 0.7985969, 0.12816042, 0.0];
        for (a, b) in s.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        assert_eq!(t, vec![995, 973, 798, 128]);
        let (s8, t8) = schedule(8);
        assert_eq!(s8.len(), 9);
        assert_eq!(t8, vec![995, 990, 979, 954, 890, 729, 422, 128]);
    }

    #[test]
    fn flow_schedule_shift() {
        let (s, t) = schedule_flow(4, 10.0, false);
        // sigma_0 = 1 - eps; sigma_1 = 10*0.75025/(1+9*0.75025)
        assert!((s[0] - (1.0 - 1e-6)).abs() < 1e-6);
        let s1 = 10.0 * 0.75025 / (1.0 + 9.0 * 0.75025);
        assert!((s[1] as f64 - s1).abs() < 1e-6, "{}", s[1]);
        assert_eq!(s[4], 0.0);
        assert_eq!(t[0], 999);
        let (sn, _) = schedule_flow(4, 10.0, true);
        assert!((sn[0] as f64 - 10.0 * 0.999 / (1.0 + 9.0 * 0.999)).abs() < 1e-6);
    }

    #[test]
    fn last_step_returns_x0() {
        let (s, _) = schedule(1);
        let mut u = UniPC::new(s);
        let x = vec![1.0f32, -2.0];
        let v = vec![0.5f32, 0.25];
        let out = u.step(&v, &x);
        let s0 = 0.99502486f32;
        assert!((out[0] - (1.0 - s0 * 0.5)).abs() < 1e-6);
        assert!((out[1] - (-2.0 - s0 * 0.25)).abs() < 1e-6);
    }

    #[test]
    fn multistep_runs_without_nan() {
        let (s, _) = schedule(6);
        let mut u = UniPC::new(s);
        let mut x = vec![0.3f32, -1.1, 2.0];
        for i in 0..6 {
            let v: Vec<f32> = x.iter().map(|a| a * 0.1 + i as f32 * 0.01).collect();
            x = u.step(&v, &x);
            assert!(x.iter().all(|a| a.is_finite()), "step {i}: {x:?}");
        }
    }
}
