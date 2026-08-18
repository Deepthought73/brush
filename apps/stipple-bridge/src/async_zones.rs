//! Tracy zones that survive `.await`.
//!
//! `tracing-tracy` opens its zone in `Layer::on_enter` and closes it in
//! `on_exit`. For an `.instrument()`ed future those fire once per `poll`
//! (`Instrumented::poll` does `let _enter = span.enter();` around the inner
//! poll), so one call becomes N short zones: the Tracy *count* is the poll
//! count, and the durations only cover the synchronous work between awaits.
//! Anything dominated by awaiting a GPU readback looks nearly free.
//!
//! This layer instead opens a zone when the span is *created* and closes it
//! when the span is *closed*. For `fut.instrument(span).await` that is exactly
//! "immediately before the call" and "once it has completed", so the count
//! equals the number of calls and the duration is wall time.
//!
//! `tracy_client::Span` is `!Send` and zones must nest LIFO on the thread that
//! opened them, so a zone cannot be opened on one tokio worker and closed on
//! another after the task migrates. Every zone is therefore owned by one
//! dedicated thread whose only job is to hold them open; they show up under
//! its own track in Tracy.

use std::sync::OnceLock;
use std::sync::mpsc::{Sender, channel};

use tracing::Subscriber;
use tracing::span::{Attributes, Id};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

enum Cmd {
    Open {
        id: u64,
        name: &'static str,
        file: &'static str,
        line: u32,
    },
    Close {
        id: u64,
    },
}

/// Channel to the zone-owning thread, started on first use.
fn sender() -> &'static Sender<Cmd> {
    static SENDER: OnceLock<Sender<Cmd>> = OnceLock::new();
    SENDER.get_or_init(|| {
        let (tx, rx) = channel::<Cmd>();
        std::thread::Builder::new()
            .name("tracy-async-zones".to_owned())
            .spawn(move || {
                if let Some(client) = tracing_tracy::client::Client::running() {
                    client.set_thread_name("async spans");
                }
                // Open zones, innermost last. Dropping a `Span` closes it.
                let mut open: Vec<(u64, tracing_tracy::client::Span)> = Vec::new();
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        Cmd::Open {
                            id,
                            name,
                            file,
                            line,
                        } => {
                            let Some(client) = tracing_tracy::client::Client::running() else {
                                continue;
                            };
                            open.push((id, client.span_alloc(Some(name), name, file, line, 0)));
                        }
                        Cmd::Close { id } => {
                            // Normally the top of the stack. If it isn't, a
                            // child span outlived its parent; close everything
                            // above it too rather than leaking those zones and
                            // desynchronising the stack for good.
                            if let Some(pos) = open.iter().rposition(|(open_id, _)| *open_id == id)
                            {
                                open.truncate(pos);
                            }
                        }
                    }
                }
            })
            .expect("failed to spawn tracy-async-zones thread");
        tx
    })
}

/// See the module docs. Attach with a narrow per-layer filter — every span it
/// sees costs a channel send.
pub struct AsyncZoneLayer;

impl<S> Layer<S> for AsyncZoneLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let meta = span.metadata();
        let _ = sender().send(Cmd::Open {
            id: id.into_u64(),
            name: meta.name(),
            file: meta.file().unwrap_or("<unknown>"),
            line: meta.line().unwrap_or(0),
        });
    }

    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        let _ = sender().send(Cmd::Close { id: id.into_u64() });
    }
}
