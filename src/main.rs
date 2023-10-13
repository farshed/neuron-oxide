extern crate neuron_oxide;

use neuron_oxide::mat;
use neuron_oxide::matrix::Matrix;

fn main() {
    let mat = Matrix::zero(3, 3);
    let mat2 = mat.map(|&x| x + 3.0);
    println!("{}", mat2);

    let m = mat![
        1.0, 2.0, 3.0;
        4.0, 5.0, 6.0;
        7.0, 8.0, 9.0
    ];

    println!("{}", (1.0f64).exp());
}
