use crate::Tensor;

/// Max pool 2D. Returns (output, argmax_indices for backward).
/// input: [batch, C, H, W], kernel_size: square pool window.
pub fn max_pool2d(input: &Tensor, kernel_size: usize) -> (Tensor, Vec<usize>) {
    let batch = input.shape[0];
    let c = input.shape[1];
    let h_in = input.shape[2];
    let w_in = input.shape[3];
    let h_out = h_in / kernel_size;
    let w_out = w_in / kernel_size;

    let out_size = batch * c * h_out * w_out;
    let mut output = vec![f32::NEG_INFINITY; out_size];
    let mut indices = vec![0usize; out_size];

    for b in 0..batch {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let out_idx = b * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                    for kh in 0..kernel_size {
                        for kw in 0..kernel_size {
                            let ih = oh * kernel_size + kh;
                            let iw = ow * kernel_size + kw;
                            let in_idx = b * c * h_in * w_in + ch * h_in * w_in + ih * w_in + iw;
                            if input.data[in_idx] > output[out_idx] {
                                output[out_idx] = input.data[in_idx];
                                indices[out_idx] = in_idx;
                            }
                        }
                    }
                }
            }
        }
    }

    (
        Tensor::from_data(output, &[batch, c, h_out, w_out]),
        indices,
    )
}

/// Max pool 2D backward.
pub fn max_pool2d_backward(d_output: &Tensor, indices: &[usize], input_shape: &[usize]) -> Tensor {
    let mut d_input = vec![0.0f32; input_shape.iter().product()];
    for (i, &idx) in indices.iter().enumerate() {
        d_input[idx] += d_output.data[i];
    }
    Tensor::from_data(d_input, input_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool2d() {
        let input = Tensor::from_data(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[1, 1, 4, 4],
        );
        let (out, _) = max_pool2d(&input, 2);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        assert_eq!(out.data, vec![6.0, 8.0, 14.0, 16.0]);
    }
}
