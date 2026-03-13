use anyhow::{anyhow, Result};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CarGPUError {
    #[error("GPU initialization failed: {0}")]
    GpuInitializationFailed(String),
    
    #[error("Compilation task failed: {0}")]
    CompilationFailed(String),
    
    #[error("GPU task execution failed: {0}")]
    GpuTaskExecutionFailed(String),
    
    #[error("Fallback to CPU failed: {0}")]
    CpuFallbackFailed(String),
    
    #[error("Work-stealing buffer error: {0}")]
    WorkStealingBufferError(String),
    
    #[error("Cranelift translation failed: {0}")]
    CraneliftTranslationFailed(String),
    
    #[error("Cargo compatibility error: {0}")]
    CargoCompatibilityError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

pub struct FallbackHandler {
    enable_gpu: bool,
    auto_fallback: bool,
    max_gpu_failures: u32,
    gpu_failure_count: u32,
}

impl FallbackHandler {
    pub fn new(enable_gpu: bool, auto_fallback: bool) -> Self {
        Self {
            enable_gpu,
            auto_fallback,
            max_gpu_failures: 3,
            gpu_failure_count: 0,
        }
    }
    
    pub fn should_use_gpu(&self) -> bool {
        self.enable_gpu && self.gpu_failure_count < self.max_gpu_failures
    }
    
    pub fn record_gpu_success(&mut self) {
        // Reset failure count on success
        self.gpu_failure_count = 0;
    }
    
    pub fn record_gpu_failure(&mut self) -> bool {
        self.gpu_failure_count += 1;
        tracing::warn!(
            "GPU failure recorded ({}/{}), auto_fallback: {}",
            self.gpu_failure_count,
            self.max_gpu_failures,
            self.auto_fallback
        );
        
        self.gpu_failure_count >= self.max_gpu_failures
    }
    
    pub async fn handle_gpu_failure<T>(
        &mut self,
        error: CarGPUError,
        cpu_fallback: impl std::future::Future<Output = Result<T>>,
    ) -> Result<T> {
        tracing::error!("GPU operation failed: {}", error);
        
        if self.auto_fallback {
            tracing::info!("Automatically falling back to CPU");
            match cpu_fallback.await {
                Ok(result) => {
                    tracing::info!("CPU fallback succeeded");
                    Ok(result)
                }
                Err(fallback_error) => {
                    tracing::error!("CPU fallback also failed: {}", fallback_error);
                    Err(anyhow!("Both GPU and CPU fallback failed: GPU error: {}, CPU error: {}", error, fallback_error))
                }
            }
        } else {
            Err(anyhow!("GPU operation failed and auto-fallback is disabled: {}", error))
        }
    }
    
    pub fn get_failure_stats(&self) -> FailureStats {
        FailureStats {
            gpu_enabled: self.enable_gpu,
            current_failure_count: self.gpu_failure_count,
            max_failures: self.max_gpu_failures,
            gpu_available: self.should_use_gpu(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FailureStats {
    pub gpu_enabled: bool,
    pub current_failure_count: u32,
    pub max_failures: u32,
    pub gpu_available: bool,
}

pub trait ErrorRecovery {
    async fn retry_with_fallback<T, F, Fut>(
        &mut self,
        primary: F,
        fallback: impl Fn() -> Fut,
        max_retries: u32,
    ) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>> + Copy,
        Fut: std::future::Future<Output = Result<T>>;
}

impl ErrorRecovery for FallbackHandler {
    async fn retry_with_fallback<T, F, Fut>(
        &mut self,
        primary: F,
        fallback: impl Fn() -> Fut,
        max_retries: u32,
    ) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>> + Copy,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut last_error = None;
        
        for attempt in 1..=max_retries {
            match primary.await {
                Ok(result) => {
                    self.record_gpu_success();
                    tracing::debug!("Operation succeeded on attempt {}", attempt);
                    return Ok(result);
                }
                Err(e) => {
                    last_error = Some(e);
                    tracing::warn!("Operation failed on attempt {}: {}", attempt, last_error.as_ref().unwrap());
                    
                    if attempt == max_retries {
                        break;
                    }
                    
                    // Exponential backoff
                    let delay = std::time::Duration::from_millis(100 * (2_u64.pow(attempt - 1)));
                    tokio::time::sleep(delay).await;
                }
            }
        }
        
        // All retries failed, try fallback
        tracing::info!("All retries exhausted, trying fallback");
        match fallback().await {
            Ok(result) => Ok(result),
            Err(fallback_error) => {
                Err(anyhow!(
                    "All retries and fallback failed. Last primary error: {}, Fallback error: {}",
                    last_error.unwrap(),
                    fallback_error
                ))
            }
        }
    }
}

pub mod prelude {
    pub use super::{CarGPUError, ErrorRecovery, FallbackHandler, FailureStats};
}