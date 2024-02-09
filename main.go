package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
)

type NeuralNetwork struct {
	weights      []float64
	bias         float64
	learningRate float64
}

func NewNeuralNetwork() *NeuralNetwork {
	return &NeuralNetwork{
		weights:      []float64{rand.Float64(), rand.Float64()},
		bias:         rand.Float64(),
		learningRate: 0.1,
	}
}

func (nn *NeuralNetwork) predict(input []float64) float64 {
	sum := nn.bias

	for i, weight := range nn.weights {
		sum += input[i] * weight
	}
	return sigmoid(sum)
}

func (nn *NeuralNetwork) Train(inputs [][]float64, outputs []float64, epochs int) {
	for range epochs {
		for j, input := range inputs {
			output := nn.predict(input)

			err := outputs[j] - output
			gradient := derivative(output)

			for k := range nn.weights {
				nn.weights[k] += nn.learningRate * err * input[k] * gradient
			}
			nn.bias += nn.learningRate * err * gradient
		}
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func derivative(x float64) float64 {
	return x * (1 - x)
}

type Data struct {
	TrainingInputs  [][]float64 `json:"training_inputs"`
	TrainingOutputs []float64   `json:"training_outputs"`
	TestInputs      [][]float64 `json:"test_inputs"`
}

func main() {
	content, _ := os.ReadFile("data.json")
	data := Data{}
	json.Unmarshal(content, &data)

	nn := NewNeuralNetwork()
	nn.Train(data.TrainingInputs, data.TrainingOutputs, 10000)

	for _, input := range data.TestInputs {
		prediction := fmt.Sprintf("%.1f", nn.predict(input))
		fmt.Println("Input:", input, "Predicted Output:", prediction)
	}
}
