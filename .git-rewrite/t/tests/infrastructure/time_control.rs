//! Time Control System
//! 
//! Provides deterministic time control for reproducible testing of time-dependent operations.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::collections::BTreeMap;

/// Controlled time environment for deterministic testing
#[derive(Debug)]
pub struct ControlledTime {
    /// Shared time state
    state: Arc<RwLock<TimeState>>,
    /// Time event scheduler
    scheduler: Arc<Mutex<TimeScheduler>>,
}

/// Internal time state
#[derive(Debug, Clone)]
struct TimeState {
    /// Base reference time
    base_time: SystemTime,
    /// Virtual time offset from base
    virtual_offset: Duration,
    /// Time scaling factor (1.0 = real time, 0.0 = frozen)
    time_scale: f64,
    /// Whether time is frozen
    frozen: bool,
    /// Last real time when state was updated
    last_real_update: SystemTime,
    /// Accumulated virtual time when frozen
    frozen_virtual_time: Duration,
}

/// Scheduled time events
#[derive(Debug)]
struct TimeScheduler {
    /// Scheduled events by virtual time
    events: BTreeMap<Duration, Vec<TimeEvent>>,
    /// Next event ID
    next_event_id: u64,
}

/// A scheduled time event
#[derive(Debug, Clone)]
pub struct TimeEvent {
    /// Unique event ID
    pub id: u64,
    /// Event name/description
    pub name: String,
    /// Scheduled virtual time
    pub scheduled_time: Duration,
    /// Event callback type
    pub event_type: TimeEventType,
}

/// Types of time events
#[derive(Debug, Clone)]
pub enum TimeEventType {
    /// One-time event
    OneTime,
    /// Recurring event with interval
    Recurring(Duration),
    /// Conditional event (execute when condition is met)
    Conditional(String),
}

/// Time measurement utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeMeasurement {
    /// Start time (virtual)
    pub start_time: Duration,
    /// End time (virtual)
    pub end_time: Duration,
    /// Duration
    pub duration: Duration,
    /// Measurement label
    pub label: String,
}

/// Time statistics for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeStatistics {
    /// Total virtual time elapsed
    pub total_virtual_time: Duration,
    /// Total real time elapsed
    pub total_real_time: Duration,
    /// Time scaling factor used
    pub average_time_scale: f64,
    /// Number of time jumps/freezes
    pub time_manipulations: u32,
    /// Time measurements taken
    pub measurements: Vec<TimeMeasurement>,
}

impl ControlledTime {
    /// Create a new controlled time environment
    pub fn new() -> Self {
        let now = SystemTime::now();
        Self {
            state: Arc::new(RwLock::new(TimeState {
                base_time: now,
                virtual_offset: Duration::from_secs(0),
                time_scale: 1.0,
                frozen: false,
                last_real_update: now,
                frozen_virtual_time: Duration::from_secs(0),
            })),
            scheduler: Arc::new(Mutex::new(TimeScheduler {
                events: BTreeMap::new(),
                next_event_id: 1,
            })),
        }
    }

    /// Create a controlled time environment starting at a specific time
    pub fn new_at_time(base_time: SystemTime) -> Self {
        Self {
            state: Arc::new(RwLock::new(TimeState {
                base_time,
                virtual_offset: Duration::from_secs(0),
                time_scale: 1.0,
                frozen: false,
                last_real_update: SystemTime::now(),
                frozen_virtual_time: Duration::from_secs(0),
            })),
            scheduler: Arc::new(Mutex::new(TimeScheduler {
                events: BTreeMap::new(),
                next_event_id: 1,
            })),
        }
    }

    /// Get the current virtual time
    pub fn now(&self) -> SystemTime {
        let state = self.state.read().unwrap();
        self.compute_virtual_time(&state)
    }

    /// Get the current virtual time as duration since UNIX epoch
    pub fn now_duration(&self) -> Duration {
        self.now().duration_since(UNIX_EPOCH).unwrap_or_default()
    }

    /// Freeze time at the current virtual time
    pub fn freeze(&self) -> Result<()> {
        let mut state = self.state.write()
            .map_err(|_| anyhow!("Failed to acquire time state lock"))?;
        
        if !state.frozen {
            // Update virtual time before freezing
            let current_virtual = self.compute_virtual_time(&state);
            state.frozen_virtual_time = current_virtual.duration_since(state.base_time).unwrap_or_default();
            state.frozen = true;
            state.last_real_update = SystemTime::now();
        }
        
        Ok(())
    }

    /// Freeze time at a specific virtual time
    pub fn freeze_at(&self, timestamp: SystemTime) -> Result<()> {
        let mut state = self.state.write()
            .map_err(|_| anyhow!("Failed to acquire time state lock"))?;
        
        state.base_time = timestamp;
        state.virtual_offset = Duration::from_secs(0);
        state.frozen = true;
        state.frozen_virtual_time = Duration::from_secs(0);
        state.last_real_update = SystemTime::now();
        
        Ok(())
    }

    /// Unfreeze time and resume normal progression
    pub fn unfreeze(&self) -> Result<()> {
        let mut state = self.state.write()
            .map_err(|_| anyhow!("Failed to acquire time state lock"))?;
        
        if state.frozen {
            // Set new base time to maintain continuity
            state.base_time = state.base_time + state.frozen_virtual_time;
            state.virtual_offset = Duration::from_secs(0);
            state.frozen = false;
            state.last_real_update = SystemTime::now();
        }
        
        Ok(())
    }

    /// Advance virtual time by a specific duration
    pub fn advance_by(&self, duration: Duration) -> Result<()> {
        let mut state = self.state.write()
            .map_err(|_| anyhow!("Failed to acquire time state lock"))?;
        
        if state.frozen {
            state.frozen_virtual_time += duration;
        } else {
            state.virtual_offset += duration;
        }
        
        Ok(())
    }

    /// Jump to a specific virtual time
    pub fn jump_to(&self, target_time: SystemTime) -> Result<()> {
        let mut state = self.state.write()
            .map_err(|_| anyhow!("Failed to acquire time state lock"))?;
        
        let current_virtual = self.compute_virtual_time(&state);
        let jump_duration = target_time.duration_since(current_virtual)
            .unwrap_or_else(|_| Duration::from_secs(0));
        
        if state.frozen {
            state.frozen_virtual_time += jump_duration;
        } else {
            state.virtual_offset += jump_duration;
        }
        
        Ok(())
    }

    /// Set time scaling factor (1.0 = normal, 0.0 = frozen, 2.0 = 2x speed)
    pub fn set_scale(&self, scale: f64) -> Result<()> {
        if scale < 0.0 {
            return Err(anyhow!("Time scale must be non-negative"));
        }
        
        let mut state = self.state.write()
            .map_err(|_| anyhow!("Failed to acquire time state lock"))?;
        
        // Update virtual time with old scale before changing
        let current_virtual = self.compute_virtual_time(&state);
        state.base_time = current_virtual;
        state.virtual_offset = Duration::from_secs(0);
        state.time_scale = scale;
        state.last_real_update = SystemTime::now();
        
        // Handle freezing/unfreezing
        if scale == 0.0 && !state.frozen {
            state.frozen = true;
            state.frozen_virtual_time = Duration::from_secs(0);
        } else if scale > 0.0 && state.frozen {
            state.frozen = false;
        }
        
        Ok(())
    }

    /// Get the current time scale
    pub fn get_scale(&self) -> f64 {
        let state = self.state.read().unwrap();
        state.time_scale
    }

    /// Check if time is currently frozen
    pub fn is_frozen(&self) -> bool {
        let state = self.state.read().unwrap();
        state.frozen
    }

    /// Reset to normal time progression
    pub fn reset(&self) -> Result<()> {
        let mut state = self.state.write()
            .map_err(|_| anyhow!("Failed to acquire time state lock"))?;
        
        let now = SystemTime::now();
        state.base_time = now;
        state.virtual_offset = Duration::from_secs(0);
        state.time_scale = 1.0;
        state.frozen = false;
        state.last_real_update = now;
        state.frozen_virtual_time = Duration::from_secs(0);
        
        Ok(())
    }

    /// Schedule a time event
    pub fn schedule_event(&self, name: String, delay: Duration, event_type: TimeEventType) -> Result<u64> {
        let current_time = self.now_duration();
        let scheduled_time = current_time + delay;
        
        let mut scheduler = self.scheduler.lock()
            .map_err(|_| anyhow!("Failed to acquire scheduler lock"))?;
        
        let event_id = scheduler.next_event_id;
        scheduler.next_event_id += 1;
        
        let event = TimeEvent {
            id: event_id,
            name,
            scheduled_time,
            event_type,
        };
        
        scheduler.events.entry(scheduled_time).or_insert_with(Vec::new).push(event);
        
        Ok(event_id)
    }

    /// Get events that should trigger at current time
    pub fn get_triggered_events(&self) -> Result<Vec<TimeEvent>> {
        let current_time = self.now_duration();
        let mut scheduler = self.scheduler.lock()
            .map_err(|_| anyhow!("Failed to acquire scheduler lock"))?;
        
        let mut triggered = Vec::new();
        let mut keys_to_remove = Vec::new();
        
        for (&event_time, events) in scheduler.events.iter() {
            if event_time <= current_time {
                triggered.extend(events.clone());
                keys_to_remove.push(event_time);
            } else {
                break; // BTreeMap is ordered, so we can stop here
            }
        }
        
        // Remove triggered events and reschedule recurring ones
        for key in keys_to_remove {
            if let Some(events) = scheduler.events.remove(&key) {
                for event in events {
                    if let TimeEventType::Recurring(interval) = event.event_type {
                        let next_time = event.scheduled_time + interval;
                        let new_event = TimeEvent {
                            id: event.id,
                            name: event.name,
                            scheduled_time: next_time,
                            event_type: event.event_type,
                        };
                        scheduler.events.entry(next_time).or_insert_with(Vec::new).push(new_event);
                    }
                }
            }
        }
        
        Ok(triggered)
    }

    /// Cancel a scheduled event
    pub fn cancel_event(&self, event_id: u64) -> Result<bool> {
        let mut scheduler = self.scheduler.lock()
            .map_err(|_| anyhow!("Failed to acquire scheduler lock"))?;
        
        for events in scheduler.events.values_mut() {
            events.retain(|event| event.id != event_id);
        }
        
        // Remove empty entries
        scheduler.events.retain(|_, events| !events.is_empty());
        
        Ok(true)
    }

    /// Get all scheduled events
    pub fn get_scheduled_events(&self) -> Result<Vec<TimeEvent>> {
        let scheduler = self.scheduler.lock()
            .map_err(|_| anyhow!("Failed to acquire scheduler lock"))?;
        
        let mut all_events = Vec::new();
        for events in scheduler.events.values() {
            all_events.extend(events.clone());
        }
        
        Ok(all_events)
    }

    /// Start measuring time for a labeled operation
    pub fn start_measurement(&self, label: String) -> TimeMeasurement {
        TimeMeasurement {
            start_time: self.now_duration(),
            end_time: Duration::from_secs(0),
            duration: Duration::from_secs(0),
            label,
        }
    }

    /// Finish a time measurement
    pub fn finish_measurement(&self, mut measurement: TimeMeasurement) -> TimeMeasurement {
        measurement.end_time = self.now_duration();
        measurement.duration = measurement.end_time - measurement.start_time;
        measurement
    }

    /// Measure the duration of a closure execution
    pub fn measure<T, F>(&self, label: String, f: F) -> (T, TimeMeasurement)
    where
        F: FnOnce() -> T,
    {
        let measurement = self.start_measurement(label);
        let result = f();
        let measurement = self.finish_measurement(measurement);
        (result, measurement)
    }

    /// Get time statistics
    pub fn get_statistics(&self) -> TimeStatistics {
        let state = self.state.read().unwrap();
        let current_virtual = self.compute_virtual_time(&state);
        let current_real = SystemTime::now();
        
        TimeStatistics {
            total_virtual_time: current_virtual.duration_since(state.base_time).unwrap_or_default(),
            total_real_time: current_real.duration_since(state.last_real_update).unwrap_or_default(),
            average_time_scale: state.time_scale,
            time_manipulations: 0, // Would need to track this
            measurements: Vec::new(), // Would need to store measurements
        }
    }

    /// Compute the current virtual time based on state
    fn compute_virtual_time(&self, state: &TimeState) -> SystemTime {
        if state.frozen {
            state.base_time + state.frozen_virtual_time
        } else {
            let real_elapsed = SystemTime::now().duration_since(state.last_real_update).unwrap_or_default();
            let virtual_elapsed = Duration::from_secs_f64(real_elapsed.as_secs_f64() * state.time_scale);
            state.base_time + state.virtual_offset + virtual_elapsed
        }
    }

    /// Create a snapshot of the current time state
    pub fn snapshot(&self) -> TimeSnapshot {
        let state = self.state.read().unwrap();
        TimeSnapshot {
            base_time: state.base_time,
            virtual_offset: state.virtual_offset,
            time_scale: state.time_scale,
            frozen: state.frozen,
            frozen_virtual_time: state.frozen_virtual_time,
        }
    }

    /// Restore time state from a snapshot
    pub fn restore(&self, snapshot: &TimeSnapshot) -> Result<()> {
        let mut state = self.state.write()
            .map_err(|_| anyhow!("Failed to acquire time state lock"))?;
        
        state.base_time = snapshot.base_time;
        state.virtual_offset = snapshot.virtual_offset;
        state.time_scale = snapshot.time_scale;
        state.frozen = snapshot.frozen;
        state.frozen_virtual_time = snapshot.frozen_virtual_time;
        state.last_real_update = SystemTime::now();
        
        Ok(())
    }
}

impl Default for ControlledTime {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ControlledTime {
    fn clone(&self) -> Self {
        let state = self.state.read().unwrap().clone();
        let now = SystemTime::now();
        
        Self {
            state: Arc::new(RwLock::new(TimeState {
                base_time: state.base_time,
                virtual_offset: state.virtual_offset,
                time_scale: state.time_scale,
                frozen: state.frozen,
                last_real_update: now,
                frozen_virtual_time: state.frozen_virtual_time,
            })),
            scheduler: Arc::new(Mutex::new(TimeScheduler {
                events: BTreeMap::new(),
                next_event_id: 1,
            })),
        }
    }
}

/// Snapshot of time state for restoration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSnapshot {
    base_time: SystemTime,
    virtual_offset: Duration,
    time_scale: f64,
    frozen: bool,
    frozen_virtual_time: Duration,
}

/// Utility functions for time testing
pub mod utils {
    use super::*;

    /// Create a controlled time environment for a specific test date
    pub fn time_for_test(year: i32, month: u32, day: u32) -> ControlledTime {
        // Simple date calculation (not accounting for leap years, etc.)
        let days_since_epoch = (year - 1970) * 365 + (month - 1) * 30 + day;
        let seconds_since_epoch = days_since_epoch as u64 * 24 * 60 * 60;
        let test_time = UNIX_EPOCH + Duration::from_secs(seconds_since_epoch);
        ControlledTime::new_at_time(test_time)
    }

    /// Simulate time passing at high speed
    pub fn simulate_time_passage(time_control: &ControlledTime, duration: Duration, steps: u32) -> Result<()> {
        let step_duration = duration / steps;
        
        for _ in 0..steps {
            time_control.advance_by(step_duration)?;
            // Allow for event processing
            let _ = time_control.get_triggered_events()?;
        }
        
        Ok(())
    }

    /// Run a closure with time frozen
    pub fn with_frozen_time<T, F>(time_control: &ControlledTime, f: F) -> Result<T>
    where
        F: FnOnce() -> T,
    {
        time_control.freeze()?;
        let result = f();
        time_control.unfreeze()?;
        Ok(result)
    }

    /// Run a closure with accelerated time
    pub fn with_accelerated_time<T, F>(time_control: &ControlledTime, scale: f64, f: F) -> Result<T>
    where
        F: FnOnce() -> T,
    {
        let original_scale = time_control.get_scale();
        time_control.set_scale(scale)?;
        let result = f();
        time_control.set_scale(original_scale)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_time_creation() {
        let time_control = ControlledTime::new();
        let now = time_control.now();
        
        // Should be close to real time
        let real_now = SystemTime::now();
        let diff = real_now.duration_since(now).unwrap_or_else(|_| now.duration_since(real_now).unwrap());
        assert!(diff < Duration::from_millis(100));
    }

    #[test]
    fn test_time_freeze() {
        let time_control = ControlledTime::new();
        let frozen_time = time_control.now();
        
        time_control.freeze().unwrap();
        thread::sleep(Duration::from_millis(10));
        
        let still_frozen = time_control.now();
        assert_eq!(frozen_time.duration_since(UNIX_EPOCH).unwrap().as_secs(),
                  still_frozen.duration_since(UNIX_EPOCH).unwrap().as_secs());
    }

    #[test]
    fn test_time_advance() {
        let time_control = ControlledTime::new();
        time_control.freeze().unwrap();
        
        let start_time = time_control.now();
        time_control.advance_by(Duration::from_secs(3600)).unwrap(); // 1 hour
        let end_time = time_control.now();
        
        let elapsed = end_time.duration_since(start_time).unwrap();
        assert_eq!(elapsed, Duration::from_secs(3600));
    }

    #[test]
    fn test_time_scale() {
        let time_control = ControlledTime::new();
        time_control.set_scale(2.0).unwrap(); // 2x speed
        
        let start_time = time_control.now();
        thread::sleep(Duration::from_millis(100));
        let end_time = time_control.now();
        
        let virtual_elapsed = end_time.duration_since(start_time).unwrap();
        // Should be approximately 2x real time (allowing for timing variations)
        assert!(virtual_elapsed >= Duration::from_millis(150));
        assert!(virtual_elapsed <= Duration::from_millis(250));
    }

    #[test]
    fn test_event_scheduling() {
        let time_control = ControlledTime::new();
        time_control.freeze().unwrap();
        
        let event_id = time_control.schedule_event(
            "test_event".to_string(),
            Duration::from_secs(60),
            TimeEventType::OneTime,
        ).unwrap();
        
        // Event shouldn't trigger yet
        let events = time_control.get_triggered_events().unwrap();
        assert!(events.is_empty());
        
        // Advance time past the event
        time_control.advance_by(Duration::from_secs(120)).unwrap();
        let events = time_control.get_triggered_events().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, event_id);
        assert_eq!(events[0].name, "test_event");
    }

    #[test]
    fn test_recurring_events() {
        let time_control = ControlledTime::new();
        time_control.freeze().unwrap();
        
        time_control.schedule_event(
            "recurring_event".to_string(),
            Duration::from_secs(10),
            TimeEventType::Recurring(Duration::from_secs(10)),
        ).unwrap();
        
        // Advance time to trigger multiple occurrences
        time_control.advance_by(Duration::from_secs(35)).unwrap();
        let events = time_control.get_triggered_events().unwrap();
        
        // Should have triggered 3 times (at 10s, 20s, 30s)
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_measurement() {
        let time_control = ControlledTime::new();
        time_control.freeze().unwrap();
        
        let measurement = time_control.start_measurement("test_op".to_string());
        time_control.advance_by(Duration::from_millis(500)).unwrap();
        let measurement = time_control.finish_measurement(measurement);
        
        assert_eq!(measurement.duration, Duration::from_millis(500));
        assert_eq!(measurement.label, "test_op");
    }

    #[test]
    fn test_measure_closure() {
        let time_control = ControlledTime::new();
        time_control.freeze().unwrap();
        
        let (result, measurement) = time_control.measure("closure_test".to_string(), || {
            time_control.advance_by(Duration::from_millis(100)).unwrap();
            42
        });
        
        assert_eq!(result, 42);
        assert_eq!(measurement.duration, Duration::from_millis(100));
    }

    #[test]
    fn test_snapshot_restore() {
        let time_control = ControlledTime::new();
        time_control.freeze().unwrap();
        
        let snapshot = time_control.snapshot();
        
        time_control.advance_by(Duration::from_secs(100)).unwrap();
        time_control.set_scale(5.0).unwrap();
        
        time_control.restore(&snapshot).unwrap();
        
        assert_eq!(time_control.get_scale(), 1.0);
        assert!(time_control.is_frozen());
    }

    #[test]
    fn test_utils_frozen_time() {
        let time_control = ControlledTime::new();
        
        let result = utils::with_frozen_time(&time_control, || {
            assert!(time_control.is_frozen());
            42
        }).unwrap();
        
        assert_eq!(result, 42);
        assert!(!time_control.is_frozen());
    }

    #[test]
    fn test_utils_accelerated_time() {
        let time_control = ControlledTime::new();
        
        let result = utils::with_accelerated_time(&time_control, 10.0, || {
            assert_eq!(time_control.get_scale(), 10.0);
            42
        }).unwrap();
        
        assert_eq!(result, 42);
        assert_eq!(time_control.get_scale(), 1.0);
    }

    #[test]
    fn test_time_for_test() {
        let time_control = utils::time_for_test(2023, 6, 15);
        let test_time = time_control.now();
        
        // Should be a specific date
        let duration_since_epoch = test_time.duration_since(UNIX_EPOCH).unwrap();
        assert!(duration_since_epoch.as_secs() > 0);
    }
}