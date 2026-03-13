use anyhow::{anyhow, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Copy)]
pub enum ExecutionPath {
    ScheduledForGpu,
    ExecutedOnGpu,
    ScheduledForCpu,
    ExecutedOnCpu,
    GpuToCpuFallback { reason: FallbackReason },
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum FallbackReason {
    GpuUnavailable,
    GpuExecutionFailed,
    GpuTooSlow,
    HardwareIncompatibility,
    DriverIssues,
    MonomorphizationFailed,
    OptimizationFailed,
    CodeGenerationFailed,
}

impl std::fmt::Display for FallbackReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            FallbackReason::GpuUnavailable => "GPU unavailable",
            FallbackReason::GpuExecutionFailed => "GPU execution failed",
            FallbackReason::GpuTooSlow => "GPU too slow",
            FallbackReason::HardwareIncompatibility => "Hardware incompatibility",
            FallbackReason::DriverIssues => "Driver issues",
            FallbackReason::MonomorphizationFailed => "Monomorphization failed",
            FallbackReason::OptimizationFailed => "Optimization failed",
            FallbackReason::CodeGenerationFailed => "Code generation failed",
        };
        write!(f, "{}", text)
    }
}

pub struct PerformanceMonitor {
    metrics: Arc<Mutex<PerformanceMetrics>>,
    history: Arc<Mutex<VecDeque<TaskMetrics>>>,
    scheduler: Arc<Mutex<AdaptiveScheduler>>,
    config: MonitoringConfig,
}

#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    // Original metrics
    pub total_tasks: u64,
    pub gpu_tasks: u64,
    pub cpu_tasks: u64,
    pub gpu_success_rate: f64,
    pub cpu_success_rate: f64,
    pub avg_gpu_time_ms: f64,
    pub avg_cpu_time_ms: f64,
    pub gpu_utilization: f64,
    pub memory_usage_mb: f64,
    pub cache_hit_rate: f64,

    // New tracking metrics
    pub gpu_attempts: u64,
    pub gpu_executed: u64,
    pub gpu_fallbacks: u64,
    pub fallback_reasons: HashMap<FallbackReason, u64>,
    pub fallback_rate: f64,

    // Success tracking
    pub gpu_successes: u64,
    pub cpu_successes: u64,
}

#[derive(Debug, Clone)]
pub struct TaskMetrics {
    pub task_id: u64,
    pub task_type: String,
    pub execution_path: Option<ExecutionPath>,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub success: bool,
    pub data_size_bytes: usize,
    pub gpu_utilization_during: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum ExecutionTarget {
    GPU,
    CPU,
}

#[derive(Debug)]
pub struct AdaptiveScheduler {
    pub gpu_success_rate: f64,
    pub cpu_success_rate: f64,
    pub avg_gpu_time: Duration,
    pub avg_cpu_time: Duration,
    pub gpu_load_factor: f64,
    pub cpu_load_factor: f64,
    pub task_success_history: VecDeque<bool>,
    pub decision_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub history_size: usize,
    pub update_interval_ms: u64,
    pub adaptive_threshold: f64,
    pub performance_window_size: usize,
    pub enable_auto_tuning: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            history_size: 1000,
            update_interval_ms: 100,
            adaptive_threshold: 0.8,
            performance_window_size: 100,
            enable_auto_tuning: true,
        }
    }
}

impl PerformanceMonitor {
    pub fn new(config: MonitoringConfig) -> Self {
        info!("Initializing PerformanceMonitor with adaptive scheduling");

        Self {
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            history: Arc::new(Mutex::new(VecDeque::with_capacity(config.history_size))),
            scheduler: Arc::new(Mutex::new(AdaptiveScheduler::new(
                config.adaptive_threshold,
            ))),
            config,
        }
    }

    pub fn start_task_monitoring(
        &self,
        task_id: u64,
        task_type: &str,
        data_size: usize,
    ) -> TaskHandle {
        let metrics = TaskMetrics {
            task_id,
            task_type: task_type.to_string(),
            execution_path: Some(ExecutionPath::ScheduledForGpu), // Default, will be updated
            start_time: Instant::now(),
            end_time: None,
            success: false,
            data_size_bytes: data_size,
            gpu_utilization_during: 0.0,
        };

        debug!("Started monitoring task {} ({})", task_id, task_type);

        TaskHandle {
            task_id,
            metrics: Arc::clone(&self.metrics),
            history: Arc::clone(&self.history),
            scheduler: Arc::clone(&self.scheduler),
            task_metrics: metrics,
            start_time: Instant::now(),
        }
    }

    pub fn should_use_gpu(&self, task_type: &str, data_size: usize) -> bool {
        let scheduler = self.scheduler.lock().unwrap();
        scheduler.should_schedule_to_gpu(task_type, data_size)
    }

    pub fn get_performance_summary(&self) -> PerformanceSummary {
        let metrics = self.metrics.lock().unwrap();
        let scheduler = self.scheduler.lock().unwrap();
        let history = self.history.lock().unwrap();

        PerformanceSummary {
            // Original fields for backward compatibility
            total_tasks: metrics.total_tasks,
            gpu_tasks: metrics.gpu_tasks,
            cpu_tasks: metrics.cpu_tasks,
            gpu_success_rate: metrics.gpu_success_rate,
            cpu_success_rate: metrics.cpu_success_rate,
            avg_gpu_time_ms: metrics.avg_gpu_time_ms,
            avg_cpu_time_ms: metrics.avg_cpu_time_ms,
            gpu_utilization: metrics.gpu_utilization,
            memory_usage_mb: metrics.memory_usage_mb,
            cache_hit_rate: metrics.cache_hit_rate,
            scheduler_efficiency: scheduler.get_efficiency_score(),
            adaptive_decisions: scheduler.task_success_history.len() as u64,

            // New detailed tracking fields
            gpu_attempts: metrics.gpu_attempts,
            gpu_executed: metrics.gpu_executed,
            gpu_fallbacks: metrics.gpu_fallbacks,
            fallback_rate: metrics.fallback_rate,
            gpu_successes: metrics.gpu_successes,
            cpu_successes: metrics.cpu_successes,
            fallback_reasons: metrics.fallback_reasons.clone(),
        }
    }

    pub fn update_system_metrics(
        &self,
        gpu_utilization: f64,
        memory_usage_mb: f64,
        cache_hit_rate: f64,
    ) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.gpu_utilization = gpu_utilization;
        metrics.memory_usage_mb = memory_usage_mb;
        metrics.cache_hit_rate = cache_hit_rate;

        // Trigger adaptive tuning if enabled
        if self.config.enable_auto_tuning {
            drop(metrics); // Release lock before calling adaptive tuning
            self.adaptive_tuning();
        }
    }

    fn adaptive_tuning(&self) {
        debug!("Performing adaptive tuning based on current metrics");

        let metrics = self.metrics.lock().unwrap();
        let mut scheduler = self.scheduler.lock().unwrap();

        // Update scheduler with current metrics
        scheduler.update_performance_metrics(
            metrics.gpu_success_rate,
            Duration::from_millis(metrics.avg_gpu_time_ms as u64),
            Duration::from_millis(metrics.avg_cpu_time_ms as u64),
            metrics.gpu_utilization,
        );

        // Adjust decision threshold based on performance
        if metrics.gpu_success_rate < 0.7 && metrics.avg_gpu_time_ms > metrics.avg_cpu_time_ms {
            // GPU performing poorly, favor CPU
            scheduler.decision_threshold = (scheduler.decision_threshold * 1.1).min(0.95);
            warn!(
                "Adjusting scheduler threshold to favor CPU: {:.3}",
                scheduler.decision_threshold
            );
        } else if metrics.gpu_success_rate > 0.9
            && metrics.avg_gpu_time_ms < metrics.avg_cpu_time_ms * 0.8
        {
            // GPU performing well, favor GPU
            scheduler.decision_threshold = (scheduler.decision_threshold * 0.9).max(0.5);
            info!(
                "Adjusting scheduler threshold to favor GPU: {:.3}",
                scheduler.decision_threshold
            );
        }
    }

    pub fn get_task_metrics(&self, task_id: u64) -> Option<TaskMetrics> {
        let history = self.history.lock().unwrap();
        history.iter().find(|m| m.task_id == task_id).cloned()
    }

    pub fn get_recent_performance(&self, window_size: Option<usize>) -> PerformanceWindow {
        let history = self.history.lock().unwrap();
        let window = window_size.unwrap_or(self.config.performance_window_size);

        let recent_tasks: Vec<_> = history.iter().rev().take(window).collect();

        let gpu_tasks: Vec<_> = recent_tasks
            .iter()
            .filter(|m| matches!(m.execution_path, Some(ExecutionPath::ExecutedOnGpu)))
            .collect();

        let cpu_tasks: Vec<_> = recent_tasks
            .iter()
            .filter(|m| {
                matches!(
                    m.execution_path,
                    Some(ExecutionPath::ExecutedOnCpu)
                        | Some(ExecutionPath::GpuToCpuFallback { .. })
                )
            })
            .collect();

        PerformanceWindow {
            total_tasks: recent_tasks.len(),
            gpu_tasks: gpu_tasks.len(),
            cpu_tasks: cpu_tasks.len(),
            gpu_success_rate: if gpu_tasks.is_empty() {
                0.0
            } else {
                gpu_tasks
                    .iter()
                    .map(|m| if m.success { 1.0 } else { 0.0 })
                    .sum::<f64>()
                    / gpu_tasks.len() as f64
            },
            cpu_success_rate: if cpu_tasks.is_empty() {
                0.0
            } else {
                cpu_tasks
                    .iter()
                    .map(|m| if m.success { 1.0 } else { 0.0 })
                    .sum::<f64>()
                    / cpu_tasks.len() as f64
            },
            avg_gpu_time_ms: if gpu_tasks.is_empty() {
                0.0
            } else {
                gpu_tasks
                    .iter()
                    .filter_map(|m| {
                        m.end_time
                            .map(|end| end.duration_since(m.start_time).as_millis() as f64)
                    })
                    .sum::<f64>()
                    / gpu_tasks.len() as f64
            },
            avg_cpu_time_ms: if cpu_tasks.is_empty() {
                0.0
            } else {
                cpu_tasks
                    .iter()
                    .filter_map(|m| {
                        m.end_time
                            .map(|end| end.duration_since(m.start_time).as_millis() as f64)
                    })
                    .sum::<f64>()
                    / cpu_tasks.len() as f64
            },
        }
    }
}

pub struct TaskHandle {
    task_id: u64,
    metrics: Arc<Mutex<PerformanceMetrics>>,
    history: Arc<Mutex<VecDeque<TaskMetrics>>>,
    scheduler: Arc<Mutex<AdaptiveScheduler>>,
    task_metrics: TaskMetrics,
    start_time: Instant,
}

impl TaskHandle {
    // New execution path tracking methods
    pub fn record_gpu_attempt(&mut self) {
        self.task_metrics.execution_path = Some(ExecutionPath::ScheduledForGpu);
    }

    pub fn record_gpu_execution(&mut self) {
        self.task_metrics.execution_path = Some(ExecutionPath::ExecutedOnGpu);
    }

    pub fn record_gpu_fallback(&mut self, reason: FallbackReason) {
        self.task_metrics.execution_path = Some(ExecutionPath::GpuToCpuFallback { reason });
    }

    pub fn record_cpu_scheduling(&mut self) {
        self.task_metrics.execution_path = Some(ExecutionPath::ScheduledForCpu);
    }

    pub fn record_cpu_execution(&mut self) {
        self.task_metrics.execution_path = Some(ExecutionPath::ExecutedOnCpu);
    }

    // Legacy method for backward compatibility
    pub fn set_execution_target(&mut self, target: ExecutionTarget) {
        match target {
            ExecutionTarget::GPU => self.record_gpu_attempt(),
            ExecutionTarget::CPU => self.record_cpu_scheduling(),
        }
    }

    // Helper method to get execution path with proper access
    pub fn get_execution_path(&self) -> Option<ExecutionPath> {
        self.task_metrics.execution_path
    }

    pub fn complete_task(
        mut self,
        execution_path: ExecutionPath,
        success: bool,
        gpu_utilization: f64,
    ) {
        let end_time = Instant::now();
        self.task_metrics.end_time = Some(end_time);
        self.task_metrics.success = success;
        self.task_metrics.gpu_utilization_during = gpu_utilization;

        // Update global metrics based on actual execution path
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.total_tasks += 1;

            match execution_path {
                ExecutionPath::ExecutedOnGpu => {
                    metrics.gpu_tasks += 1;
                    metrics.gpu_executed += 1;
                    metrics.gpu_successes += if success { 1 } else { 0 };

                    let duration = end_time.duration_since(self.start_time).as_millis() as f64;
                    metrics.avg_gpu_time_ms =
                        (metrics.avg_gpu_time_ms * (metrics.gpu_tasks - 1) as f64 + duration)
                            / metrics.gpu_tasks as f64;
                }
                ExecutionPath::ExecutedOnCpu => {
                    metrics.cpu_tasks += 1;
                    metrics.cpu_successes += if success { 1 } else { 0 };

                    let duration = end_time.duration_since(self.start_time).as_millis() as f64;
                    metrics.avg_cpu_time_ms =
                        (metrics.avg_cpu_time_ms * (metrics.cpu_tasks - 1) as f64 + duration)
                            / metrics.cpu_tasks as f64;
                }
                ExecutionPath::GpuToCpuFallback { ref reason } => {
                    metrics.cpu_tasks += 1;
                    metrics.gpu_attempts += 1;
                    metrics.gpu_fallbacks += 1;
                    metrics.cpu_successes += if success { 1 } else { 0 };

                    // Track fallback reasons
                    *metrics.fallback_reasons.entry(*reason).or_insert(0) += 1;

                    // Update fallback rate
                    metrics.fallback_rate =
                        metrics.gpu_fallbacks as f64 / metrics.gpu_attempts as f64;

                    // Update CPU timing for fallback execution
                    let duration = end_time.duration_since(self.start_time).as_millis() as f64;
                    metrics.avg_cpu_time_ms =
                        (metrics.avg_cpu_time_ms * (metrics.cpu_tasks - 1) as f64 + duration)
                            / metrics.cpu_tasks as f64;
                }
                _ => {
                    // Should not reach here - indicates incomplete tracking
                    warn!(
                        "Task {} completed without proper execution path tracking",
                        self.task_id
                    );
                }
            }
        }

        // Update scheduler based on execution
        {
            let mut scheduler = self.scheduler.lock().unwrap();
            match execution_path {
                ExecutionPath::ExecutedOnGpu => {
                    scheduler.record_gpu_execution(
                        success,
                        end_time - self.start_time,
                        gpu_utilization,
                    );
                }
                ExecutionPath::ExecutedOnCpu | ExecutionPath::GpuToCpuFallback { .. } => {
                    scheduler.record_cpu_execution(success, end_time - self.start_time);
                }
                _ => {}
            }
        }

        // Add to history
        {
            let mut history = self.history.lock().unwrap();
            if history.len() >= 1000 {
                history.pop_front();
            }
            history.push_back(self.task_metrics.clone());
        }

        debug!(
            "Completed task {} in {:?} (path: {:?}, success: {})",
            self.task_id,
            end_time - self.start_time,
            execution_path,
            success
        );
    }

    // Legacy methods for backward compatibility
    pub fn complete_gpu(mut self, success: bool, gpu_utilization: f64) {
        self.complete_task(ExecutionPath::ExecutedOnGpu, success, gpu_utilization);
    }

    pub fn complete_cpu(mut self, success: bool) {
        self.complete_task(ExecutionPath::ExecutedOnCpu, success, 0.0);
    }
}

impl AdaptiveScheduler {
    pub fn new(initial_threshold: f64) -> Self {
        Self {
            gpu_success_rate: 1.0,
            cpu_success_rate: 1.0,
            avg_gpu_time: Duration::from_millis(50),
            avg_cpu_time: Duration::from_millis(100),
            gpu_load_factor: 0.5,
            cpu_load_factor: 0.5,
            task_success_history: VecDeque::with_capacity(100),
            decision_threshold: initial_threshold,
        }
    }

    pub fn should_schedule_to_gpu(&self, task_type: &str, data_size: usize) -> bool {
        // Calculate task-specific factors
        let task_suitability = self.calculate_task_suitability(task_type, data_size);
        let load_factor = self.calculate_load_factor();
        let performance_factor = self.calculate_performance_factor();

        // Final decision
        let decision_score = task_suitability * performance_factor * load_factor;

        decision_score >= self.decision_threshold
    }

    fn calculate_task_suitability(&self, task_type: &str, data_size: usize) -> f64 {
        // Different tasks have different GPU suitability
        let base_suitability = match task_type {
            "monomorphization" => 0.9,
            "optimization" => 0.85,
            "code_generation" => 0.7,
            "register_allocation" => 0.6,
            "link_preparation" => 0.4,
            "parsing" => 0.3,
            _ => 0.5,
        };

        // Larger data sizes benefit more from GPU parallelism
        let size_factor = if data_size > 1024 * 1024 {
            1.2
        }
        // >1MB
        else if data_size > 1024 * 100 {
            1.1
        }
        // >100KB
        else if data_size > 1024 * 10 {
            1.0
        }
        // >10KB
        else {
            0.8
        }; // Small tasks

        base_suitability * size_factor
    }

    fn calculate_load_factor(&self) -> f64 {
        // Balance between GPU and CPU load
        1.0 - (self.gpu_load_factor * 0.3 + self.cpu_load_factor * 0.1)
    }

    fn calculate_performance_factor(&self) -> f64 {
        // Factor in recent performance
        let success_factor = (self.gpu_success_rate + self.cpu_success_rate) / 2.0;
        let speed_factor = if self.avg_gpu_time.as_millis() > 0 && self.avg_cpu_time.as_millis() > 0
        {
            self.avg_cpu_time.as_millis() as f64 / self.avg_gpu_time.as_millis() as f64
        } else {
            1.0
        };

        (success_factor + speed_factor) / 2.0
    }

    pub fn record_gpu_execution(&mut self, success: bool, duration: Duration, utilization: f64) {
        self.task_success_history.push_back(success);
        if self.task_success_history.len() > 100 {
            self.task_success_history.pop_front();
        }

        // Update rolling averages
        if success {
            self.gpu_success_rate = self.gpu_success_rate * 0.9 + 1.0 * 0.1;
        } else {
            self.gpu_success_rate = self.gpu_success_rate * 0.9 + 0.0 * 0.1;
        }

        self.avg_gpu_time = Duration::from_millis(
            (self.avg_gpu_time.as_millis() as f64 * 0.9 + duration.as_millis() as f64 * 0.1) as u64,
        );

        self.gpu_load_factor = self.gpu_load_factor * 0.8 + utilization * 0.2;
    }

    pub fn record_cpu_execution(&mut self, success: bool, duration: Duration) {
        if success {
            self.cpu_success_rate = self.cpu_success_rate * 0.9 + 1.0 * 0.1;
        } else {
            self.cpu_success_rate = self.cpu_success_rate * 0.9 + 0.0 * 0.1;
        }

        self.avg_cpu_time = Duration::from_millis(
            (self.avg_cpu_time.as_millis() as f64 * 0.9 + duration.as_millis() as f64 * 0.1) as u64,
        );

        // Simulate CPU load reduction
        self.cpu_load_factor = self.cpu_load_factor * 0.9;
    }

    pub fn update_performance_metrics(
        &mut self,
        gpu_success: f64,
        avg_gpu: Duration,
        avg_cpu: Duration,
        gpu_util: f64,
    ) {
        self.gpu_success_rate = gpu_success;
        self.avg_gpu_time = avg_gpu;
        self.avg_cpu_time = avg_cpu;
        self.gpu_load_factor = gpu_util;
    }

    pub fn get_efficiency_score(&self) -> f64 {
        let success_balance = (self.gpu_success_rate + self.cpu_success_rate) / 2.0;
        let speed_balance =
            if self.avg_gpu_time.as_millis() > 0 && self.avg_cpu_time.as_millis() > 0 {
                self.avg_gpu_time.as_millis() as f64 / self.avg_cpu_time.as_millis() as f64
            } else {
                1.0
            };
        let load_balance = 1.0 - (self.gpu_load_factor - self.cpu_load_factor).abs();

        (success_balance + speed_balance.min(2.0) / 2.0 + load_balance) / 3.0
    }
}

#[derive(Debug)]
pub struct PerformanceSummary {
    // Original metrics for backward compatibility
    pub total_tasks: u64,
    pub gpu_tasks: u64,
    pub cpu_tasks: u64,
    pub gpu_success_rate: f64,
    pub cpu_success_rate: f64,
    pub avg_gpu_time_ms: f64,
    pub avg_cpu_time_ms: f64,
    pub gpu_utilization: f64,
    pub memory_usage_mb: f64,
    pub cache_hit_rate: f64,
    pub scheduler_efficiency: f64,
    pub adaptive_decisions: u64,

    // New detailed tracking metrics
    pub gpu_attempts: u64,
    pub gpu_executed: u64,
    pub gpu_fallbacks: u64,
    pub fallback_rate: f64,
    pub gpu_successes: u64,
    pub cpu_successes: u64,
    pub fallback_reasons: std::collections::HashMap<FallbackReason, u64>,
}

#[derive(Debug)]
pub struct PerformanceWindow {
    pub total_tasks: usize,
    pub gpu_tasks: usize,
    pub cpu_tasks: usize,
    pub gpu_success_rate: f64,
    pub cpu_success_rate: f64,
    pub avg_gpu_time_ms: f64,
    pub avg_cpu_time_ms: f64,
}
