pub mod compiler;

#[cfg(feature = "rustc-internals")]
pub mod mir_extractor;

pub mod codegen_backend;
pub mod incremental;

pub use compiler::{CarGPCompiler, BuildConfig, CheckConfig};
pub use codegen_backend::{RealCodegenBackend, CodegenConfig, OptLevel, GeneratedCode};
pub use incremental::{IncrementalCompiler, CacheConfig, IncrementalResult, cache_maintenance_task};