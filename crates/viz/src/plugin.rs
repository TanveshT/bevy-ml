use std::io::{self, Stdout};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use ecs_ml_core::resources::MetricsHistory;

use crate::dashboard::render_dashboard;

/// TUI handle — either a real terminal or a headless fallback for non-TTY contexts.
pub enum TerminalHandle {
    Tui(Terminal<CrosstermBackend<Stdout>>),
    Headless,
}

impl TerminalHandle {
    pub fn init() -> Self {
        // Check if stdout is a TTY
        if !std::io::IsTerminal::is_terminal(&io::stdout()) {
            return Self::Headless;
        }
        match Self::init_tui() {
            Ok(t) => Self::Tui(t),
            Err(_) => Self::Headless,
        }
    }

    fn init_tui() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
        ctrlc_restore_terminal();
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        Terminal::new(backend)
    }

    pub fn restore(&mut self) {
        if let Self::Tui(terminal) = self {
            let _ = disable_raw_mode();
            let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
            let _ = terminal.show_cursor();
        }
    }

    /// Render one frame. Returns true if user wants to quit (q / Ctrl-C).
    pub fn render_tick(&mut self, metrics: &MetricsHistory) -> bool {
        match self {
            Self::Tui(terminal) => {
                let _ = terminal.draw(|f| render_dashboard(f, metrics));

                if let Ok(true) = event::poll(std::time::Duration::from_millis(0))
                    && let Ok(Event::Key(key)) = event::read()
                {
                    match key {
                        KeyEvent {
                            code: KeyCode::Char('q'),
                            ..
                        } => return true,
                        KeyEvent {
                            code: KeyCode::Char('c'),
                            modifiers,
                            ..
                        } if modifiers.contains(KeyModifiers::CONTROL) => {
                            return true;
                        }
                        _ => {}
                    }
                }
                false
            }
            Self::Headless => false,
        }
    }

    pub fn is_tui(&self) -> bool {
        matches!(self, Self::Tui(_))
    }
}

impl Drop for TerminalHandle {
    fn drop(&mut self) {
        self.restore();
    }
}

/// Install a signal handler so Ctrl-C restores the terminal before exit.
fn ctrlc_restore_terminal() {
    let _ = std::thread::spawn(|| {
        // Listen for SIGINT via a simple flag
        let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let r = running.clone();
        if signal_hook::flag::register(signal_hook::consts::SIGINT, r).is_err() {
            return; // signal_hook not available, fall through
        }
        while running.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        // SIGINT received — restore terminal and exit
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        std::process::exit(130);
    });
}
