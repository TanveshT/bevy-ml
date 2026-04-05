use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::dataset::Dataset;

pub struct MnistDataset {
    pub images: Vec<f32>, // N × 784, normalized [0,1]
    pub labels: Vec<u8>,
    pub count: usize,
}

impl MnistDataset {
    pub fn load_train(dir: &Path) -> Self {
        let images = read_idx_images(&dir.join("train-images.idx3-ubyte"));
        let labels = read_idx_labels(&dir.join("train-labels.idx1-ubyte"));
        let count = labels.len();
        Self {
            images,
            labels,
            count,
        }
    }

    pub fn load_test(dir: &Path) -> Self {
        let images = read_idx_images(&dir.join("t10k-images.idx3-ubyte"));
        let labels = read_idx_labels(&dir.join("t10k-labels.idx1-ubyte"));
        let count = labels.len();
        Self {
            images,
            labels,
            count,
        }
    }
}

impl Dataset for MnistDataset {
    fn len(&self) -> usize {
        self.count
    }

    fn get_features(&self, idx: usize) -> &[f32] {
        &self.images[idx * 784..(idx + 1) * 784]
    }

    fn get_label(&self, idx: usize) -> usize {
        self.labels[idx] as usize
    }

    fn feature_dim(&self) -> usize {
        784
    }
    fn num_classes(&self) -> usize {
        10
    }
}

fn read_idx_images(path: &Path) -> Vec<f32> {
    let mut file =
        File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {e}", path.display()));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let magic = u32::from_be_bytes(buf[0..4].try_into().unwrap());
    assert_eq!(magic, 2051, "Invalid image file magic: {magic}");
    let _count = u32::from_be_bytes(buf[4..8].try_into().unwrap()) as usize;
    assert_eq!(u32::from_be_bytes(buf[8..12].try_into().unwrap()), 28);
    assert_eq!(u32::from_be_bytes(buf[12..16].try_into().unwrap()), 28);

    buf[16..].iter().map(|&b| b as f32 / 255.0).collect()
}

fn read_idx_labels(path: &Path) -> Vec<u8> {
    let mut file =
        File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {e}", path.display()));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let magic = u32::from_be_bytes(buf[0..4].try_into().unwrap());
    assert_eq!(magic, 2049, "Invalid label file magic: {magic}");

    buf[8..].to_vec()
}
