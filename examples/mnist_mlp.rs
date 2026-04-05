use std::path::Path;
use std::time::Instant;

use bevy_app::App;
use ecs_ml_core::plugin::CorePlugin;
use ecs_ml_core::resources::{InputBatch, LossOutput, MetricsHistory, TrainingState};
use ecs_ml_data::dataset::Dataset;
use ecs_ml_data::mnist::MnistDataset;
use ecs_ml_nn::network::NetworkBuilder;
use ecs_ml_optim::plugin::{OptimPlugin, OptimizerChoice};
use ecs_ml_tensor::Tensor;
use ecs_ml_viz::plugin::TerminalHandle;

fn main() {
    let json_mode = std::env::args().any(|a| a == "--json");

    let train = MnistDataset::load_train(Path::new("data"));
    let test = MnistDataset::load_test(Path::new("data"));

    let batch_size = 32;
    let num_epochs = 10;
    let lr = 0.01;

    let mut app = App::new();
    app.add_plugins(CorePlugin);
    app.add_plugins(OptimPlugin {
        choice: OptimizerChoice::Sgd { momentum: 0.0 },
    });

    NetworkBuilder::new()
        .linear(784, 256)
        .relu()
        .linear(256, 128)
        .relu()
        .linear(128, 10)
        .log_softmax()
        .build(&mut app, batch_size);

    app.world_mut().insert_resource(TrainingState {
        epoch: 0,
        batch_idx: 0,
        learning_rate: lr,
        eval_mode: false,
        total_batches: train.len() / batch_size,
        total_epochs: num_epochs,
    });

    app.update();

    let mut tui = if json_mode {
        TerminalHandle::Headless
    } else {
        TerminalHandle::init()
    };

    let batches_per_epoch = train.len() / batch_size;
    let render_every = 50;

    // For JSON output
    let mut epoch_results: Vec<String> = Vec::new();
    let total_start = Instant::now();
    let mem_before = get_rss_mb();

    for epoch in 0..num_epochs {
        let epoch_start = Instant::now();

        {
            let mut state = app.world_mut().resource_mut::<TrainingState>();
            state.epoch = epoch;
            state.batch_idx = 0;
            state.eval_mode = false;
        }

        let mut epoch_loss = 0.0f32;
        let mut epoch_correct = 0usize;

        for batch in 0..batches_per_epoch {
            {
                let mut state = app.world_mut().resource_mut::<TrainingState>();
                state.batch_idx = batch;
            }

            load_batch(&train, &mut app, batch * batch_size, batch_size);
            app.update();

            let loss = app.world().resource::<LossOutput>();
            epoch_loss += loss.batch_loss;
            epoch_correct += loss.correct;

            if batch % render_every == 0 {
                let metrics = app.world().resource::<MetricsHistory>().clone();
                if tui.render_tick(&metrics) {
                    drop(tui);
                    eprintln!("Training interrupted at epoch {epoch}, batch {batch}");
                    return;
                }
            }
        }

        let avg_loss = epoch_loss / batches_per_epoch as f32;
        let train_acc = epoch_correct as f32 / (batches_per_epoch * batch_size) as f32 * 100.0;
        let test_acc = evaluate(&test, &mut app, batch_size);
        let elapsed = epoch_start.elapsed().as_millis() as u64;

        {
            let mut metrics = app.world_mut().resource_mut::<MetricsHistory>();
            metrics.train_losses.push(avg_loss);
            metrics.train_accuracies.push(train_acc);
            metrics.test_accuracies.push(test_acc);
            metrics.epoch_times_ms.push(elapsed);
        }

        let metrics = app.world().resource::<MetricsHistory>().clone();
        tui.render_tick(&metrics);

        let mem_now = get_rss_mb();

        if json_mode {
            epoch_results.push(format!(
                r#"    {{"epoch": {epoch}, "loss": {avg_loss:.4}, "train_acc": {train_acc:.1}, "test_acc": {test_acc:.1}, "time_ms": {elapsed}, "memory_mb": {mem_now:.1}}}"#
            ));
            eprintln!(
                "Epoch {epoch}: loss={avg_loss:.4} train_acc={train_acc:.1}% test_acc={test_acc:.1}%"
            );
        } else if !tui.is_tui() {
            println!(
                "Epoch {epoch}: loss={avg_loss:.4} train_acc={train_acc:.1}% test_acc={test_acc:.1}%"
            );
        }
    }

    if json_mode {
        let total_ms = total_start.elapsed().as_millis();
        let peak_mem = get_rss_mb();
        println!("{{");
        println!(r#"  "framework": "ecs-ml","#);
        println!(r#"  "architecture": "784->256->ReLU->128->ReLU->10->LogSoftmax","#);
        println!(r#"  "optimizer": "SGD(lr=0.01)","#);
        println!(r#"  "batch_size": 32,"#);
        println!(
            r#"  "total_params": {},"#,
            784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
        );
        println!(r#"  "total_time_ms": {total_ms},"#);
        println!(r#"  "peak_memory_mb": {peak_mem:.1},"#);
        println!(r#"  "memory_before_mb": {mem_before:.1},"#);
        println!(r#"  "epochs": ["#);
        println!("{}", epoch_results.join(",\n"));
        println!("  ]");
        println!("}}");
    } else if tui.is_tui() {
        loop {
            let metrics = app.world().resource::<MetricsHistory>().clone();
            if tui.render_tick(&metrics) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

fn get_rss_mb() -> f64 {
    // Read RSS from /proc/self/status on Linux, or use sysctl on macOS
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let pid = std::process::id();
        if let Ok(output) = Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
            && let Ok(s) = String::from_utf8(output.stdout)
            && let Ok(kb) = s.trim().parse::<f64>()
        {
            return kb / 1024.0;
        }
        0.0
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            return kb / 1024.0;
                        }
                    }
                }
            }
        }
        0.0
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0.0
    }
}

fn load_batch(dataset: &MnistDataset, app: &mut App, offset: usize, batch_size: usize) {
    let mut input = app.world_mut().resource_mut::<InputBatch>();
    input.batch_size = batch_size;

    let dim = dataset.feature_dim();
    let total = dataset.len();
    let mut data = vec![0.0f32; batch_size * dim];
    let mut targets = vec![0usize; batch_size];

    for s in 0..batch_size {
        let idx = (offset + s) % total;
        data[s * dim..(s + 1) * dim].copy_from_slice(dataset.get_features(idx));
        targets[s] = dataset.get_label(idx);
    }
    input.data = Tensor::from_data(data, &[batch_size, dim]);
    input.targets = targets;
}

fn evaluate(dataset: &MnistDataset, app: &mut App, batch_size: usize) -> f32 {
    {
        let mut state = app.world_mut().resource_mut::<TrainingState>();
        state.eval_mode = true;
    }

    let total = dataset.len();
    let mut correct = 0usize;
    let num_batches = total.div_ceil(batch_size);

    for b in 0..num_batches {
        let start = b * batch_size;
        let this_bs = batch_size.min(total - start);
        load_batch(dataset, app, start, this_bs);
        app.update();

        let loss = app.world().resource::<LossOutput>();
        correct += loss.correct;
    }

    {
        let mut state = app.world_mut().resource_mut::<TrainingState>();
        state.eval_mode = false;
    }

    correct as f32 / total as f32 * 100.0
}
