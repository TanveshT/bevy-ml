use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};

use ecs_ml_core::resources::MetricsHistory;

use crate::charts::{render_accuracy_chart, render_loss_chart};
use crate::progress::render_progress;
use crate::stats::render_layer_stats;

/// Render the full dashboard into the given frame.
///
/// Layout:
/// ```text
/// +------------ Loss -----------+------- Accuracy --------+
/// |                             |                          |
/// +--------- Layer Stats -------+------- Progress --------+
/// |                             |                          |
/// +-----------------------------+--------------------------+
/// ```
pub fn render_dashboard(f: &mut Frame, metrics: &MetricsHistory) {
    let rows =
        Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).split(f.area());

    let top =
        Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).split(rows[0]);
    let bottom =
        Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).split(rows[1]);

    render_loss_chart(f, top[0], metrics);
    render_accuracy_chart(f, top[1], metrics);
    render_layer_stats(f, bottom[0], metrics);
    render_progress(f, bottom[1], metrics);
}
