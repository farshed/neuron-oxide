const data = require('./data.json');

class NeuralNetwork {
	constructor() {
		this.weights = [Math.random(), Math.random()];
		this.bias = Math.random();
		this.learningRate = 0.1;
	}

	predict(input) {
		const sum = this.weights.reduce((acc, weight, i) => {
			acc += input[i] * weight;
			return acc;
		}, this.bias);

		return sigmoid(sum);
	}

	train(inputs, outputs, epochs) {
		for (let i = 0; i < epochs; i++) {
			inputs.forEach((input, j) => {
				const output = this.predict(input);

				const error = outputs[j] - output;
				const gradient = derivative(output);

				this.weights.forEach((_, k) => {
					this.weights[k] += this.learningRate * error * input[k] * gradient;
				});

				this.bias += this.learningRate * error * gradient;
			});
		}
	}
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const derivative = (x) => x * (1 - x);

// Train and test

const {
	training_inputs: trainingInputs,
	training_outputs: trainingOutputs,
	test_inputs: testInputs
} = data;

const neuralNet = new NeuralNetwork();
neuralNet.train(trainingInputs, trainingOutputs, 1e5);

for (const input of testInputs) {
	const prediction = neuralNet.predict(input);
	console.log(`Input: [${input}], Prediction: ${prediction.toFixed(1)}`);
}
