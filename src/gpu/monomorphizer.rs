use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::gpu::buffer::CompilationTask;

const COMPILATION_SHADER: &str = include_str!("compilation.wgsl");

pub struct Monomorphizer {
    instantiations: HashMap<String, MonomorphizedInstance>,
    generic_functions: HashMap<String, GenericFunction>,
    config: MonomorphizerConfig,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<Arc<wgpu::Queue>>,
}

#[derive(Debug, Clone)]
pub struct MonomorphizerConfig {
    pub max_instantiations: usize,
    pub enable_gpu_optimization: bool,
    pub cache_instantiations: bool,
}

impl Default for MonomorphizerConfig {
    fn default() -> Self {
        Self {
            max_instantiations: 10000,
            enable_gpu_optimization: true,
            cache_instantiations: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MonomorphizedInstance {
    pub function_name: String,
    pub concrete_types: Vec<TypeInfo>,
    pub optimized_code: Vec<u8>,
    pub dependency_graph: Vec<String>,
    pub size_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct GenericFunction {
    pub name: String,
    pub type_params: Vec<String>,
    pub param_types: Vec<TypeInfo>,
    pub return_type: TypeInfo,
    pub body_ir: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum TypeInfo {
    I32,
    I64,
    F32,
    F64,
    Bool,
    String,
    Generic(String),
    Struct { name: String, fields: Vec<(String, TypeInfo)> },
    Array { element: Box<TypeInfo>, size: usize },
    Slice(Box<TypeInfo>),
    Reference(Box<TypeInfo>),
    MutReference(Box<TypeInfo>),
}

impl Monomorphizer {
    pub fn new(config: MonomorphizerConfig, device: Option<Arc<wgpu::Device>>, queue: Option<Arc<wgpu::Queue>>) -> Self {
        info!("Initializing GPU Monomorphizer with config: {:?}", config);
        
        Self {
            instantiations: HashMap::new(),
            generic_functions: HashMap::new(),
            config,
            device,
            queue,
        }
    }

    pub fn register_generic_function(&mut self, func: GenericFunction) -> Result<()> {
        debug!("Registering generic function: {}", func.name);
        
        if self.generic_functions.contains_key(&func.name) {
            return Err(anyhow!("Generic function {} already registered", func.name));
        }
        
        self.generic_functions.insert(func.name.clone(), func);
        Ok(())
    }

    pub async fn monomorphize_function(
        &mut self,
        generic_name: &str,
        concrete_types: Vec<TypeInfo>,
    ) -> Result<String> {
        debug!(
            "Monomorphizing {} with types: {:?}",
            generic_name, concrete_types
        );

        let generic_func = self.generic_functions
            .get(generic_name)
            .ok_or_else(|| anyhow!("Generic function {} not found", generic_name))?;

        if concrete_types.len() != generic_func.type_params.len() {
            return Err(anyhow!(
                "Type parameter count mismatch for {}: expected {}, got {}",
                generic_name,
                generic_func.type_params.len(),
                concrete_types.len()
            ));
        }

        let instance_key = self.generate_instance_key(generic_name, &concrete_types);
        
        if let Some(instance) = self.instantiations.get(&instance_key) {
            debug!("Reusing existing monomorphization: {}", instance.function_name);
            return Ok(instance.function_name.clone());
        }

        if self.instantiations.len() >= self.config.max_instantiations {
            warn!("Reached maximum instantiations limit, consider increasing cache size");
            return Err(anyhow!("Maximum monomorphization limit reached"));
        }

        let instance = self.create_monomorphized_instance(generic_func, &concrete_types).await?;
        let instance_name = instance.function_name.clone();
        
        self.instantiations.insert(instance_key, instance);
        
        info!("Created new monomorphization: {}", instance_name);
        Ok(instance_name)
    }

    async fn create_monomorphized_instance(
        &self,
        generic_func: &GenericFunction,
        concrete_types: &[TypeInfo],
    ) -> Result<MonomorphizedInstance> {
        let instance_name = self.generate_instance_name(&generic_func.name, concrete_types);
        
        debug!("Creating instance: {}", instance_name);

        let type_substitutions: HashMap<String, TypeInfo> = generic_func
            .type_params
            .iter()
            .zip(concrete_types.iter())
            .map(|(param, concrete)| (param.clone(), concrete.clone()))
            .collect();

        let mut optimized_code = generic_func.body_ir.clone();
        
        if self.config.enable_gpu_optimization {
            optimized_code = self.optimize_for_gpu(&optimized_code, &type_substitutions).await?;
        }

        let dependency_graph = self.analyze_dependencies(&optimized_code, &type_substitutions)?;

        let size_bytes = optimized_code.len();

        Ok(MonomorphizedInstance {
            function_name: instance_name,
            concrete_types: concrete_types.to_vec(),
            optimized_code,
            dependency_graph,
            size_bytes,
        })
    }

    fn generate_instance_key(&self, generic_name: &str, concrete_types: &[TypeInfo]) -> String {
        let type_strings: Vec<String> = concrete_types
            .iter()
            .map(|t| self.type_to_string(t))
            .collect();
        format!("{}__{}", generic_name, type_strings.join("_"))
    }

    fn generate_instance_name(&self, generic_name: &str, concrete_types: &[TypeInfo]) -> String {
        let type_strings: Vec<String> = concrete_types
            .iter()
            .map(|t| self.type_to_string(t))
            .collect();
        format!("{}_monomorphized_{}", generic_name, type_strings.join("_"))
    }

    fn type_to_string(&self, type_info: &TypeInfo) -> String {
        match type_info {
            TypeInfo::I32 => "i32".to_string(),
            TypeInfo::I64 => "i64".to_string(),
            TypeInfo::F32 => "f32".to_string(),
            TypeInfo::F64 => "f64".to_string(),
            TypeInfo::Bool => "bool".to_string(),
            TypeInfo::String => "str".to_string(),
            TypeInfo::Generic(name) => name.clone(),
            TypeInfo::Struct { name, fields } => {
                let field_strings: Vec<String> = fields
                    .iter()
                    .map(|(name, ty)| format!("{}:{}", name, self.type_to_string(ty)))
                    .collect();
                format!("{}{{{}}}", name, field_strings.join(","))
            }
            TypeInfo::Array { element, size } => format!("[{};{}]", self.type_to_string(element), size),
            TypeInfo::Slice(element) => format!("[{}]", self.type_to_string(element)),
            TypeInfo::Reference(inner) => format!("&{}", self.type_to_string(inner)),
            TypeInfo::MutReference(inner) => format!("&mut {}", self.type_to_string(inner)),
        }
    }

    async fn optimize_for_gpu(
        &self,
        code: &[u8],
        _type_substitutions: &HashMap<String, TypeInfo>,
    ) -> Result<Vec<u8>> {
        debug!("Optimizing {} bytes of IR for GPU", code.len());

        let mut optimized = Vec::with_capacity(code.len());
        
        for chunk in code.chunks(64) {
            let chunk_result = self.process_code_chunk_gpu(chunk).await?;
            optimized.extend_from_slice(&chunk_result);
        }

        info!("GPU optimization completed: {} -> {} bytes", code.len(), optimized.len());
        Ok(optimized)
    }

    async fn process_code_chunk_gpu(&self, chunk: &[u8]) -> Result<Vec<u8>> {
        let device = self.device.as_ref().ok_or_else(|| anyhow!("GPU device not available for monomorphization"))?;
        let queue = self.queue.as_ref().ok_or_else(|| anyhow!("GPU queue not available for monomorphization"))?;

        debug!("Processing {} bytes of code on GPU", chunk.len());

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Monomorphization Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPILATION_SHADER.into()),
        });

        let aligned_size = ((chunk.len() + 3) / 4) * 4;
        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Monomorphization Input"),
            size: aligned_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_size = std::cmp::max(chunk.len() * 2, 2048);
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Monomorphization Output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut aligned_data = vec![0u8; aligned_size];
        aligned_data[..chunk.len()].copy_from_slice(chunk);
        queue.write_buffer(&input_buffer, 0, &aligned_data);

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Monomorphization Pipeline"),
            layout: None,
            module: &shader_module,
            entry_point: "main",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Monomorphization Bind Group"),
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Monomorphization Encoder"),
        });

        let workgroup_count = std::cmp::max(1, (chunk.len() + 255) / 256);
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Monomorphization Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Monomorphization Readback"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size as u64);

        let command_buffer = encoder.finish();
        queue.submit(Some(command_buffer));

        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);

        receiver.await??;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<u8> = data.to_vec();

        debug!("GPU monomorphization completed: {} -> {} bytes", chunk.len(), result.len());
        Ok(result)
    }

    fn analyze_dependencies(
        &self,
        code: &[u8],
        _type_substitutions: &HashMap<String, TypeInfo>,
    ) -> Result<Vec<String>> {
        debug!("Analyzing dependencies in {} bytes of code", code.len());

        let mut dependencies = Vec::new();
        
        if code.len() > 1000 {
            dependencies.push("std::mem".to_string());
        }
        
        if code.iter().any(|&b| b == 0xFF) {
            dependencies.push("core::intrinsics".to_string());
        }

        Ok(dependencies)
    }

    pub fn get_instantiation_count(&self) -> usize {
        self.instantiations.len()
    }

    pub fn get_cached_instance(&self, instance_key: &str) -> Option<&MonomorphizedInstance> {
        self.instantiations.get(instance_key)
    }

    pub fn get_all_instances(&self) -> Vec<&MonomorphizedInstance> {
        self.instantiations.values().collect()
    }

    pub fn clear_cache(&mut self) {
        info!("Clearing monomorphization cache");
        self.instantiations.clear();
    }

    pub async fn batch_monomorphize(
        &mut self,
        functions: Vec<(String, Vec<TypeInfo>)>,
    ) -> Result<Vec<String>> {
        debug!("Batch monomorphizing {} functions", functions.len());
        
        let mut results = Vec::new();
        
        for (generic_name, types) in functions {
            let instance_name = self.monomorphize_function(&generic_name, types).await?;
            results.push(instance_name);
        }

        info!("Batch monomorphization completed: {} instances", results.len());
        Ok(results)
    }

    pub fn get_memory_usage(&self) -> MemoryUsage {
        let total_bytes: usize = self.instantiations
            .values()
            .map(|inst| inst.size_bytes)
            .sum();

        let generic_ir_bytes: usize = self.generic_functions
            .values()
            .map(|func| func.body_ir.len())
            .sum();

        MemoryUsage {
            instantiation_cache_bytes: total_bytes,
            generic_function_bytes: generic_ir_bytes,
            total_instances: self.instantiations.len(),
            total_generic_functions: self.generic_functions.len(),
        }
    }
}

#[derive(Debug)]
pub struct MemoryUsage {
    pub instantiation_cache_bytes: usize,
    pub generic_function_bytes: usize,
    pub total_instances: usize,
    pub total_generic_functions: usize,
}