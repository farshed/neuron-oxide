use crate::{activation::Activation, matrix::Matrix};

pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation,
    learning_rate: f64,
}

impl Network {
    pub fn new(layers: Vec<usize>, activation: Activation, learning_rate: f64) -> Self {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Self {
            layers,
            weights,
            biases,
            data: vec![],
            activation,
            learning_rate,
        }
    }

    pub fn feed_forward(&mut self, inputs: Matrix) -> Matrix {
        assert!(
            self.layers[0] == inputs.data.len(),
            "Invalid Number of Inputs"
        );

        let mut current = inputs;

        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .dot_product(&current)
                .add(&self.biases[i])
                .map(|x| self.activation.apply(x));

            self.data.push(current.clone());
        }

        current
    }

    pub fn back_propogate(&mut self, inputs: Matrix, targets: Matrix) {
        let mut errors = targets.subtract(&inputs);

        let mut gradients = inputs.clone().map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.hadamard_product(&errors).map(|x| x * 0.5); // learning rate

            self.weights[i] =
                self.weights[i].add(&gradients.dot_product(&self.data[i].transpose()));

            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].transpose().dot_product(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
        for i in 1..=epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
                println!("Epoch {} of {}", i, epochs);
            }
            for j in 0..inputs.len() {
                let outputs = self.feed_forward(Matrix::from(inputs[j].clone()));
                self.back_propogate(outputs, Matrix::from(targets[j].clone()));
            }
        }
    }
}
