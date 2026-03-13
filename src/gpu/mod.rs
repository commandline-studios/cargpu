// pub mod translator; // Temporarily disabled due to API compatibility issues
pub mod dispatcher;
pub mod buffer;
pub mod monomorphizer;
pub mod codegen_units;
pub mod lowering;
pub mod optimizations;
pub mod monitoring;
pub mod mir_processor;
pub mod compute_shaders;
pub mod memory_manager;

// pub use translator::CraneliftTranslator; // Temporarily disabled
pub use dispatcher::GpuDispatcher;
pub use buffer::WorkStealingBuffer;
pub use monomorphizer::{Monomorphizer, MonomorphizedInstance, GenericFunction, TypeInfo};
pub use codegen_units::{CodegenUnitManager, CodegenUnit, CGUFunction, CompilationWave};
pub use lowering::{FunctionLowerer, LoweredFunction, LoweringConfig};
pub use optimizations::{PeepholeOptimizer, OptimizationConfig};
pub use monitoring::{PerformanceMonitor, TaskHandle};
pub use mir_processor::{MirProcessor, ProcessedCrate, ProcessedFunction, MirFunction};
pub use compute_shaders::{GpuComputeShader, CompilationShader, ShaderType};
pub use memory_manager::{MemoryManager, MemoryLayout, BufferHandle, FieldType};