use bevy_app::{App, Plugin, Startup, Update};

use crate::adam::{AdamConfig, adam_step, init_adam_state};
use crate::sgd::{SgdConfig, init_momentum, sgd_step};

pub enum OptimizerChoice {
    Sgd { momentum: f32 },
    Adam { beta1: f32, beta2: f32, eps: f32 },
}

pub struct OptimPlugin {
    pub choice: OptimizerChoice,
}

impl Plugin for OptimPlugin {
    fn build(&self, app: &mut App) {
        match &self.choice {
            OptimizerChoice::Sgd { momentum } => {
                app.insert_resource(SgdConfig {
                    momentum: *momentum,
                });
                if *momentum > 0.0 {
                    app.add_systems(Startup, init_momentum);
                }
                app.add_systems(Update, sgd_step);
            }
            OptimizerChoice::Adam { beta1, beta2, eps } => {
                app.insert_resource(AdamConfig {
                    beta1: *beta1,
                    beta2: *beta2,
                    eps: *eps,
                    step: 0,
                });
                app.add_systems(Startup, init_adam_state);
                app.add_systems(Update, adam_step);
            }
        }
    }
}
