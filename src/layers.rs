use candle_core::{quantized::QMatMul, DType, Module, Result, Tensor};
use candle_nn::Linear;

#[derive(Clone, Debug)]
pub struct QuantLinear {
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl QuantLinear {
    pub fn new(weight: QMatMul, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
}

#[derive(Clone, Debug)]
pub enum LinearLayer {
    Float(Linear),
    Quantized(QuantLinear),
}

impl LinearLayer {
    pub fn from_linear(linear: Linear) -> Self {
        Self::Float(linear)
    }

    pub fn from_quantized(weight: QMatMul, bias: Option<Tensor>) -> Self {
        Self::Quantized(QuantLinear::new(weight, bias))
    }
}

#[derive(Clone, Debug)]
pub enum OutputProjection {
    Tensor(Tensor),
    Quantized(QMatMul),
}

impl OutputProjection {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = match self {
            Self::Tensor(weight) => {
                let xs = to_dtype_if_needed(xs, weight.dtype())?;
                xs.matmul(&weight.t()?)?
            }
            Self::Quantized(weight) => forward_qmatmul(weight, xs)?,
        };
        if ys.dtype() == DType::F32 {
            Ok(ys)
        } else {
            ys.to_dtype(DType::F32)
        }
    }
}

impl Module for QuantLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = forward_qmatmul(&self.weight, xs)?;
        let ys = match &self.bias {
            Some(bias) => {
                let ys = to_dtype_if_needed(&ys, bias.dtype())?;
                ys.broadcast_add(bias)?
            }
            None => ys,
        };
        if ys.dtype() == DType::F32 {
            Ok(ys)
        } else {
            ys.to_dtype(DType::F32)
        }
    }
}

impl Module for LinearLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Float(linear) => linear.forward(xs),
            Self::Quantized(linear) => linear.forward(xs),
        }
    }
}

fn forward_qmatmul(weight: &QMatMul, xs: &Tensor) -> Result<Tensor> {
    match weight {
        QMatMul::QTensor(_) => {
            let xs = to_dtype_if_needed(xs, DType::F32)?;
            weight.forward(&xs)
        }
        QMatMul::Tensor(tensor) => {
            let xs = to_dtype_if_needed(xs, tensor.dtype())?;
            weight.forward(&xs)
        }
        QMatMul::TensorF16(_) => {
            let xs = to_dtype_if_needed(xs, DType::F16)?;
            weight.forward(&xs)
        }
    }
}

fn to_dtype_if_needed(xs: &Tensor, dtype: DType) -> Result<Tensor> {
    if xs.dtype() == dtype {
        Ok(xs.clone())
    } else {
        xs.to_dtype(dtype)
    }
}
