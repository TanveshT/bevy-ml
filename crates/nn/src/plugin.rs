use bevy_app::{App, Plugin};

/// NnPlugin — currently a marker. Network building is done via NetworkBuilder.
pub struct NnPlugin;

impl Plugin for NnPlugin {
    fn build(&self, _app: &mut App) {
        // NetworkBuilder handles entity spawning and resource allocation.
        // This plugin exists for future nn-specific systems (e.g., batch norm running stats).
    }
}
