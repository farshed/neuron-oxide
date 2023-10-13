use rand::{self, Rng};
use std::fmt;
use std::ops::{Add, Mul};

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    #[inline]
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        Self { rows, cols, data }
    }

    #[inline]
    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Creates and returns a matrix of size rows * cols with random entries
    #[inline]
    pub fn random(rows: usize, cols: usize) -> Self {
        let data = vec![0.0; rows * cols]
            .iter()
            .map(|_| rand::thread_rng().gen_range(0.0..1.0))
            .collect();

        Self { rows, cols, data }
    }

    /// Returns an element-wise product with another matrix of the same dimensions.
    /// https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
    #[inline]
    pub fn hadamard_product(&self, other: &Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Cannot multiply matrices of different dimensions");
        }

        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(i, &x)| x * other.data[i])
            .collect();

        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Cannot add matrices of different dimensions");
        }

        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(i, &x)| x + other.data[i])
            .collect();

        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    #[inline]
    pub fn subtract(&self, other: &Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Cannot add matrices of different dimensions");
        }

        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(i, &x)| x - other.data[i])
            .collect();

        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    #[inline]
    pub fn dot_product(&self, other: &Self) -> Self {
        if self.cols != other.rows {
            panic!("Cannot create a dot product of matrices with incorrect dimensions");
        }

        let mut data = vec![0.0; self.rows * other.cols];

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                data[i * other.cols + j] = sum;
            }
        }

        Self {
            rows: self.rows,
            cols: other.cols,
            data,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut data = vec![0.0; self.data.len()];

        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }

        Self {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    pub fn map(&self, func: fn(&f64) -> f64) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(func).collect(),
        }
    }
}

impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        self.dot_product(rhs)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                if j == 0 {
                    write!(formatter, "{}", "⎥")?;
                }
                write!(formatter, "{}", self.data[i * self.cols + j])?;
                if j < self.cols - 1 {
                    write!(formatter, "  ")?;
                } else {
                    write!(formatter, "{}", "⎥")?;
                }
            }
            writeln!(formatter)?;
        }
        Ok(())
    }
}
