use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, Cell, Row, Table};

use ecs_ml_core::resources::MetricsHistory;

pub fn render_layer_stats(f: &mut Frame, area: Rect, metrics: &MetricsHistory) {
    let header = Row::new(vec![
        Cell::from("Layer"),
        Cell::from("Mean"),
        Cell::from("Std"),
        Cell::from("|grad|"),
    ])
    .style(
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
    );

    let rows: Vec<Row> = metrics
        .layer_stats
        .iter()
        .map(|(name, mean, std, grad_norm)| {
            Row::new(vec![
                Cell::from(name.clone()),
                Cell::from(format!("{mean:>8.5}")),
                Cell::from(format!("{std:>8.5}")),
                Cell::from(format!("{grad_norm:>8.4}")),
            ])
        })
        .collect();

    let widths = [
        ratatui::layout::Constraint::Min(20),
        ratatui::layout::Constraint::Length(10),
        ratatui::layout::Constraint::Length(10),
        ratatui::layout::Constraint::Length(10),
    ];

    let table = Table::new(rows, widths).header(header).block(
        Block::default()
            .title(" Layer Stats ")
            .borders(Borders::ALL),
    );

    f.render_widget(table, area);
}
