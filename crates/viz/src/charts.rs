use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::symbols::Marker;
use ratatui::text::Line;
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset as RDataset, GraphType};

use ecs_ml_core::resources::MetricsHistory;

pub fn render_loss_chart(f: &mut Frame, area: Rect, metrics: &MetricsHistory) {
    let points: Vec<(f64, f64)> = metrics
        .batch_losses
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v as f64))
        .collect();

    if points.is_empty() {
        let block = Block::default()
            .title(" Training Loss ")
            .borders(Borders::ALL);
        f.render_widget(block, area);
        return;
    }

    let max_loss = points
        .iter()
        .map(|(_, y)| *y)
        .fold(0.0f64, f64::max)
        .max(0.1);
    let max_x = points.len() as f64;

    let datasets = vec![
        RDataset::default()
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&points),
    ];

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(" Training Loss ")
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("batch")
                .bounds([0.0, max_x])
                .labels::<Vec<Line>>(vec![
                    Line::from("0"),
                    Line::from(format!("{}", max_x as usize)),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("loss")
                .bounds([0.0, max_loss * 1.1])
                .labels::<Vec<Line>>(vec![
                    Line::from("0"),
                    Line::from(format!("{:.2}", max_loss / 2.0)),
                    Line::from(format!("{:.2}", max_loss)),
                ]),
        );

    f.render_widget(chart, area);
}

pub fn render_accuracy_chart(f: &mut Frame, area: Rect, metrics: &MetricsHistory) {
    let train_pts: Vec<(f64, f64)> = metrics
        .train_accuracies
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v as f64))
        .collect();
    let test_pts: Vec<(f64, f64)> = metrics
        .test_accuracies
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v as f64))
        .collect();

    if train_pts.is_empty() && test_pts.is_empty() {
        let block = Block::default()
            .title(" Accuracy (%) — waiting for epoch 1 ")
            .borders(Borders::ALL);
        f.render_widget(block, area);
        return;
    }

    let max_x = train_pts.len().max(test_pts.len()) as f64;

    let mut datasets = Vec::new();
    if !train_pts.is_empty() {
        datasets.push(
            RDataset::default()
                .name("train")
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Green))
                .data(&train_pts),
        );
    }
    if !test_pts.is_empty() {
        datasets.push(
            RDataset::default()
                .name("test")
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Yellow))
                .data(&test_pts),
        );
    }

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(" Accuracy (%) ")
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("epoch")
                .bounds([0.0, max_x.max(1.0)])
                .labels::<Vec<Line>>(vec![
                    Line::from("0"),
                    Line::from(format!("{}", max_x as usize)),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("%")
                .bounds([0.0, 100.0])
                .labels::<Vec<Line>>(vec![Line::from("0"), Line::from("50"), Line::from("100")]),
        );

    f.render_widget(chart, area);
}
