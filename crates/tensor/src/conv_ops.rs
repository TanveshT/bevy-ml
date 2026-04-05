use crate::Tensor;

/// Direct 2D convolution (forward). No im2col — zero temp memory.
/// input: [batch, C_in, H, W]
/// weight: [C_out, C_in, kH, kW]
/// bias: [C_out]
/// padding: same on all sides
/// Returns: [batch, C_out, H_out, W_out]
pub fn direct_conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    padding: usize,
    stride: usize,
) -> Tensor {
    let batch = input.shape[0];
    let c_in = input.shape[1];
    let h_in = input.shape[2];
    let w_in = input.shape[3];
    let c_out = weight.shape[0];
    let kh = weight.shape[2];
    let kw = weight.shape[3];
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;

    let mut output = vec![0.0f32; batch * c_out * h_out * w_out];

    for b in 0..batch {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = bias.data[oc];
                    for ic in 0..c_in {
                        for fh in 0..kh {
                            for fw in 0..kw {
                                let ih = (oh * stride + fh) as isize - padding as isize;
                                let iw = (ow * stride + fw) as isize - padding as isize;
                                if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                    let in_idx = b * c_in * h_in * w_in
                                        + ic * h_in * w_in
                                        + ih as usize * w_in
                                        + iw as usize;
                                    let w_idx = oc * c_in * kh * kw + ic * kh * kw + fh * kw + fw;
                                    acc += input.data[in_idx] * weight.data[w_idx];
                                }
                            }
                        }
                    }
                    output[b * c_out * h_out * w_out + oc * h_out * w_out + oh * w_out + ow] = acc;
                }
            }
        }
    }

    Tensor::from_data(output, &[batch, c_out, h_out, w_out])
}

/// Backward for direct conv2d. Returns (d_input, d_weight, d_bias).
pub fn direct_conv2d_backward(
    d_output: &Tensor, // [batch, C_out, H_out, W_out]
    input: &Tensor,    // [batch, C_in, H_in, W_in]
    weight: &Tensor,   // [C_out, C_in, kH, kW]
    padding: usize,
    stride: usize,
) -> (Tensor, Tensor, Tensor) {
    let batch = input.shape[0];
    let c_in = input.shape[1];
    let h_in = input.shape[2];
    let w_in = input.shape[3];
    let c_out = weight.shape[0];
    let kh = weight.shape[2];
    let kw = weight.shape[3];
    let h_out = d_output.shape[2];
    let w_out = d_output.shape[3];

    let mut d_input = vec![0.0f32; input.data.len()];
    let mut d_weight = vec![0.0f32; weight.data.len()];
    let mut d_bias = vec![0.0f32; c_out];

    for b in 0..batch {
        for (oc, d_bias_oc) in d_bias.iter_mut().enumerate() {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let d_val = d_output.data
                        [b * c_out * h_out * w_out + oc * h_out * w_out + oh * w_out + ow];
                    *d_bias_oc += d_val;

                    for ic in 0..c_in {
                        for fh in 0..kh {
                            for fw in 0..kw {
                                let ih = (oh * stride + fh) as isize - padding as isize;
                                let iw = (ow * stride + fw) as isize - padding as isize;
                                if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                    let in_idx = b * c_in * h_in * w_in
                                        + ic * h_in * w_in
                                        + ih as usize * w_in
                                        + iw as usize;
                                    let w_idx = oc * c_in * kh * kw + ic * kh * kw + fh * kw + fw;
                                    d_input[in_idx] += d_val * weight.data[w_idx];
                                    d_weight[w_idx] += d_val * input.data[in_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (
        Tensor::from_data(d_input, &input.shape),
        Tensor::from_data(d_weight, &weight.shape),
        Tensor::from_data(d_bias, &[c_out]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_identity_kernel() {
        // 1x1 conv with identity kernel should copy input channels
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let weight = Tensor::from_data(vec![1.0], &[1, 1, 1, 1]);
        let bias = Tensor::from_data(vec![0.0], &[1]);
        let out = direct_conv2d(&input, &weight, &bias, 0, 1);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_conv2d_with_padding() {
        let input = Tensor::from_data(vec![1.0; 9], &[1, 1, 3, 3]);
        let weight = Tensor::from_data(vec![1.0; 9], &[1, 1, 3, 3]);
        let bias = Tensor::from_data(vec![0.0], &[1]);
        let out = direct_conv2d(&input, &weight, &bias, 1, 1);
        assert_eq!(out.shape, vec![1, 1, 3, 3]);
        // Center pixel sees all 9 ones → 9.0
        assert_eq!(out.data[4], 9.0);
    }
}
