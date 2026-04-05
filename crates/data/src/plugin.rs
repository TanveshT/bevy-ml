use bevy_app::{App, Plugin};

/// DataPlugin — marker for data loading systems.
/// Currently, data loading is handled by examples directly.
pub struct DataPlugin;

impl Plugin for DataPlugin {
    fn build(&self, _app: &mut App) {
        // Future: add batch loading system, shuffling, etc.
    }
}
