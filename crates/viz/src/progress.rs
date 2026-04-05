use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph};

use ecs_ml_core::resources::MetricsHistory;

pub fn render_progress(f: &mut Frame, area: Rect, metrics: &MetricsHistory) {
    let block = Block::default().title(" Progress ").borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(2),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(inner);

    // Epoch progress
    let epoch_ratio = if metrics.total_epochs > 0 {
        (metrics.current_epoch as f64 + 1.0) / metrics.total_epochs as f64
    } else {
        0.0
    };
    let epoch_gauge = Gauge::default()
        .label(format!(
            "Epoch: {}/{}",
            metrics.current_epoch + 1,
            metrics.total_epochs
        ))
        .ratio(epoch_ratio.min(1.0))
        .gauge_style(Style::default().fg(Color::Cyan));
    f.render_widget(epoch_gauge, chunks[0]);

    // Batch progress
    let batch_ratio = if metrics.total_batches > 0 {
        (metrics.current_batch as f64 + 1.0) / metrics.total_batches as f64
    } else {
        0.0
    };
    let batch_gauge = Gauge::default()
        .label(format!(
            "Batch: {}/{}",
            metrics.current_batch + 1,
            metrics.total_batches
        ))
        .ratio(batch_ratio.min(1.0))
        .gauge_style(Style::default().fg(Color::Green));
    f.render_widget(batch_gauge, chunks[1]);

    // LR
    let lr_text = Paragraph::new(format!("LR: {:.6}", metrics.learning_rate));
    f.render_widget(lr_text, chunks[2]);

    // Current loss
    let loss_text = Paragraph::new(format!("Loss: {:.4}", metrics.current_loss));
    f.render_widget(loss_text, chunks[3]);

    // Latest accuracies
    let train_acc = metrics.train_accuracies.last().copied().unwrap_or(0.0);
    let test_acc = metrics.test_accuracies.last().copied().unwrap_or(0.0);
    let acc_text = Paragraph::new(format!("Train: {train_acc:.1}%  Test: {test_acc:.1}%"));
    f.render_widget(acc_text, chunks[4]);
}
