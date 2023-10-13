pub enum Activation {
    Sigmoid,
    ReLU,
    Tanh,
    LeakyReLU(f64),
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::ReLU => x.max(0.0),
            Self::Tanh => x.tanh(),
            Self::LeakyReLU(alpha) => {
                if x > 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => {
                let sx = self.apply(x);
                sx * (1.0 - sx)
            }
            Self::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Tanh => 1.0 - x.tanh().powi(2),
            Self::LeakyReLU(alpha) => {
                if x > 0.0 {
                    1.0
                } else {
                    *alpha
                }
            }
        }
    }
}
