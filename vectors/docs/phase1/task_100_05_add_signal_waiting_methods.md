# Task 100_05: Add Signal Waiting Methods

## Prerequisites Check
- [ ] Task 100_04 completed: Cleanup registration method added
- [ ] Run: `cargo check` (should pass)

## Context
Add cross-platform signal handling for graceful shutdown (SIGTERM/Ctrl+C).

## Task Objective
Implement wait_for_shutdown_signal and platform-specific signal handlers.

## Steps
1. Add signal handling methods to ShutdownHandler:
   ```rust
   impl ShutdownHandler {
       pub async fn wait_for_shutdown_signal(&self) {
           let shutdown_requested = self.shutdown_requested.clone();
           
           tokio::select! {
               _ = signal::ctrl_c() => {
                   info!("Received Ctrl+C signal");
               }
               _ = Self::wait_for_sigterm() => {
                   info!("Received SIGTERM signal");
               }
           }
           
           shutdown_requested.store(true, Ordering::Relaxed);
           info!("Shutdown requested, starting cleanup...");
           
           self.execute_cleanup().await;
       }
       
       #[cfg(unix)]
       async fn wait_for_sigterm() {
           signal::unix::signal(signal::unix::SignalKind::terminate())
               .expect("Failed to register SIGTERM handler")
               .recv()
               .await;
       }
       
       #[cfg(windows)]
       async fn wait_for_sigterm() {
           // Windows doesn't have SIGTERM, wait indefinitely
           std::future::pending::<()>().await;
       }
   }
   ```

## Success Criteria
- [ ] Cross-platform signal handling implemented
- [ ] Shutdown state properly set
- [ ] Cleanup execution triggered
- [ ] Compiles without errors

## Time: 5 minutes