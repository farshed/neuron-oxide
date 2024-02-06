use rand::Rng;

#[derive(Debug)]
struct NeuralNetwork {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

impl NeuralNetwork {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            weights: vec![rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)],
            bias: rng.gen_range(0.0..1.0),
            learning_rate: 0.1,
        }
    }

    fn train(&mut self, inputs: Vec<Vec<f64>>, outputs: Vec<f64>, epochs: usize) {
        for _ in 0..epochs {
            for (i, input) in inputs.iter().enumerate() {
                let output = self.predict(input.clone());

                let error = outputs[i] - output;
                let delta = derivative(output);

                for j in 0..self.weights.len() {
                    self.weights[j] += self.learning_rate * error * input[j] * delta;
                }
                self.bias += self.learning_rate * error * delta;
            }
        }
    }

    fn predict(&self, input: Vec<f64>) -> f64 {
        let mut sum = self.bias;
        for (i, weight) in self.weights.iter().enumerate() {
            sum += input[i] * weight;
        }

        sigmoid(sum)
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

fn main() {
    let inputs = vec![
        vec![0.5, 0.5],
        vec![0.2, 0.8],
        vec![0.1, 0.9],
        vec![0.9, 0.0],
        vec![0.2, 0.3],
        vec![0.4, 0.4],
        vec![0.6, 0.3],
        vec![0.3, 0.7],
        vec![0.5, 0.2],
        vec![0.3, 0.4],
        vec![0.2, 0.3],
    ];
    let outputs = vec![1.0, 1.0, 1.0, 0.9, 0.5, 0.8, 0.9, 1.0, 0.7, 0.7, 0.5];

    let test_inputs = vec![
        vec![0.1, 0.1],
        vec![0.0, 0.8],
        vec![0.1, 0.6],
        vec![0.5, 0.4],
        vec![0.0, 0.6],
        vec![0.4, 0.4],
        vec![0.7, 0.1],
        vec![0.2, 0.1],
        vec![0.3, 0.4],
        vec![0.2, 0.2],
        vec![0.7, 0.3],
    ];

    let mut neural_net = NeuralNetwork::new();
    neural_net.train(inputs, outputs, 10000);

    for input in test_inputs.iter() {
        let prediction = neural_net.predict(input.clone());
        println!("Input: {:?}, Prediction: {:.1}", input, prediction);
    }
}
