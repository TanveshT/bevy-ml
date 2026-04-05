use bevy_app::{App, Plugin, Update};
use bevy_ecs::schedule::IntoScheduleConfigs;

use crate::resources::MetricsHistory;
use crate::training::{backward_pass, compute_loss, forward_pass, record_metrics};

pub struct CorePlugin;

impl Plugin for CorePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MetricsHistory>().add_systems(
            Update,
            (forward_pass, compute_loss, backward_pass, record_metrics).chain(),
        );
    }
}
