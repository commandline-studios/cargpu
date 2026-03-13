use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use wgpu::{Buffer, Device, Queue, ShaderModule, ComputePipeline, BindGroup};

pub struct GpuComputeShader {
    device: Arc<Device>,
    queue: Arc<Queue>,
    shaders: HashMap<String, Arc<ShaderModule>>,
    pipelines: HashMap<String, Arc<ComputePipeline>>,
    config: GpuComputeConfig,
}

#[derive(Debug, Clone)]
pub struct GpuComputeConfig {
    pub max_workgroup_size: u32,
    pub enable_simd_optimization: bool,
    pub shader_cache_size: usize,
    pub workgroup_multiplier: u32,
}

impl Default for GpuComputeConfig {
    fn default() -> Self {
        Self {
            max_workgroup_size: 256,
            enable_simd_optimization: true,
            shader_cache_size: 1000,
            workgroup_multiplier: 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompilationShader {
    pub name: String,
    pub shader_type: ShaderType,
    pub workgroup_size: (u32, u32, u32),
    pub bindings: Vec<ShaderBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShaderType {
    CodeGeneration,
    Optimization,
    Monomorphization,
    LinkPreparation,
    RegisterAllocation,
    TypeChecking,
    BorrowChecking,
}

#[derive(Debug, Clone)]
pub struct ShaderBinding {
    pub name: String,
    pub binding_type: BindingType,
    pub size: Option<usize>,
    pub visibility: wgpu::ShaderStages,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BindingType {
    ReadOnly,
    WriteOnly,
    ReadWrite,
    Uniform,
    Storage,
}

impl GpuComputeShader {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, config: GpuComputeConfig) -> Result<Self> {
        info!("Initializing GpuComputeShader with config: {:?}", config);

        let mut instance = Self {
            device,
            queue,
            shaders: HashMap::new(),
            pipelines: HashMap::new(),
            config,
        };

        // Initialize built-in compilation shaders
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(instance.initialize_built_in_shaders())
        })?;

        Ok(instance)
    }

    async fn initialize_built_in_shaders(&mut self) -> Result<()> {
        info!("Initializing built-in compilation shaders");

        // Code generation shader
        self.create_code_generation_shader().await?;
        
        // Optimization shader  
        self.create_optimization_shader().await?;
        
        // Monomorphization shader
        self.create_monomorphization_shader().await?;
        
        // Link preparation shader
        self.create_link_preparation_shader().await?;
        
        // Register allocation shader
        self.create_register_allocation_shader().await?;
        
        // Type checking shader
        self.create_type_checking_shader().await?;
        
        // Borrow checking shader
        self.create_borrow_checking_shader().await?;

        info!("Built-in shaders initialized successfully");
        Ok(())
    }

    async fn create_code_generation_shader(&mut self) -> Result<()> {
        debug!("Creating code generation shader");

        let shader_source = r#"
@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(2) var<uniform> config: CodeGenConfig;

struct CodeGenConfig {
    input_size: u32,
    optimization_level: u32,
    target_arch: u32,
};

@compute @workgroup_size(64, 1, 1)
fn code_generation_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= config.input_size {
        return;
    }
    
    let input_val = input_data[idx];
    
    // Code generation optimizations
    var optimized_val = input_val;
    
    // Dead code elimination simulation
    if (input_val & 0x80000000u) == 0u {
        optimized_val = input_val & 0x7FFFFFFFu;
    }
    
    // Constant folding simulation
    if (input_val & 0x40000000u) != 0u {
        optimized_val = optimized_val ^ 0x12345678u;
    }
    
    // Instruction selection simulation
    switch (config.optimization_level) {
        case 0u: { break; }
        case 1u: { optimized_val = optimized_val | 0x00000001u; }
        case 2u: { optimized_val = optimized_val | 0x00000003u; }
        case 3u: { optimized_val = optimized_val | 0x00000007u; }
        default: { break; }
    }
    
    output_data[idx] = optimized_val;
}
"#;

        let shader = self.compile_shader("code_generation", shader_source).await?;
        self.shaders.insert("code_generation".to_string(), shader.clone());
        
        let pipeline = self.create_compute_pipeline("code_generation", &shader).await?;
        self.pipelines.insert("code_generation".to_string(), pipeline);

        Ok(())
    }

async fn create_optimization_shader(&mut self) -> Result<()> {
        debug!("Creating optimization shader");

        let shader_source = r#"
@group(0) @binding(0) var<storage, read> input_ir: array<IRInstruction>;
@group(0) @binding(1) var<storage, read> dominator_tree: array<DominatorInfo>;
@group(0) @binding(2) var<storage, read_write> output_ir: array<IRInstruction>;
@group(0) @binding(3) var<atomic> optimization_stats: array<u32, 16>; // stats counters
@group(0) @binding(4) var<uniform> config: OptimizationConfig;

struct IRInstruction {
    opcode: u32,
    operands: array<u32, 4>,
    result_type: u32,
    flags: u32,
    basic_block: u32,
    instruction_id: u32,
};

struct DominatorInfo {
    block_id: u32,
    dominator: u32,
    depth: u32,
    is_loop_header: u32,
};

struct OptimizationConfig {
    input_size: u32,
    pass_type: u32, // 0=DCE, 1=GVN, 2=StrengthReduction, 3=CF, 4=LVN
    threshold: u32,
    enable_value_numbering: u32,
    enable_loop_optimizations: u32,
};

// Value numbering table for global value numbering
struct ValueNumberEntry {
    hash: u32,
    instruction_id: u32,
    is_valid: u32,
};

var<private> value_table: array<ValueNumberEntry, 1024>;

// Hash function for instructions
fn hash_instruction(inst: IRInstruction) -> u32 {
    var hash = inst.opcode * 2654435761u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        hash = hash ^ (inst.operands[i] * 2654435761u);
        hash = (hash << 3) | (hash >> 29);
    }
    return hash;
}

// Check if instruction is dead (no side effects and unused)
fn is_dead_code(inst: IRInstruction) -> u32 {
    // Instructions with side effects
    if (inst.opcode == 1u || inst.opcode == 2u || inst.opcode == 3u) { // Store, Call, Atomic
        return 0u;
    }
    
    // Instructions marked as used
    if ((inst.flags & 0x00000001u) != 0u) {
        return 0u;
    }
    
    return 1u;
}

// Dead Code Elimination
fn eliminate_dead_code(inst: IRInstruction, output_idx: u32) -> IRInstruction {
    var optimized_inst = inst;
    
    if (is_dead_code(inst) == 1u) {
        optimized_inst.opcode = 255u; // NOP opcode
        atomicAdd(&optimization_stats[0], 1u); // DCE counter
    }
    
    return optimized_inst;
}

// Global Value Numbering
fn apply_gvn(inst: IRInstruction, output_idx: u32) -> IRInstruction {
    let hash = hash_instruction(inst);
    let table_idx = hash % 1024u;
    
    // Check if we already have an equivalent instruction
    let entry = value_table[table_idx];
    if (entry.is_valid != 0u && entry.hash == hash) {
        // Replace with existing instruction
        var optimized_inst = inst;
        optimized_inst.opcode = 254u; // COPY opcode  
        optimized_inst.operands[0] = entry.instruction_id;
        atomicAdd(&optimization_stats[1], 1u); // GVN counter
        return optimized_inst;
    }
    
    // Add to value table
    value_table[table_idx] = ValueNumberEntry(hash, inst.instruction_id, 1u);
    return inst;
}

// Strength Reduction
fn apply_strength_reduction(inst: IRInstruction, output_idx: u32) -> IRInstruction {
    var optimized_inst = inst;
    
    // Replace multiplication by power of 2 with shift
    if (inst.opcode == 10u) { // MUL
        let operand = inst.operands[1];
        // Check if operand is a power of 2
        if ((operand & (operand - 1u)) == 0u && operand != 0u) {
            // Count trailing zeros to get shift amount
            var shift = 0u;
            var temp = operand;
            while (temp > 1u) {
                temp = temp >> 1u;
                shift = shift + 1u;
            }
            optimized_inst.opcode = 11u; // SHL
            optimized_inst.operands[1] = shift;
            atomicAdd(&optimization_stats[2], 1u); // SR counter
        }
    }
    
    // Replace division by power of 2 with shift
    if (inst.opcode == 12u) { // DIV
        let operand = inst.operands[1];
        if ((operand & (operand - 1u)) == 0u && operand != 0u) {
            var shift = 0u;
            var temp = operand;
            while (temp > 1u) {
                temp = temp >> 1u;
                shift = shift + 1u;
            }
            optimized_inst.opcode = 13u; // SHR
            optimized_inst.operands[1] = shift;
            atomicAdd(&optimization_stats[2], 1u); // SR counter
        }
    }
    
    return optimized_inst;
}

// Local Value Numbering (simplified)
fn apply_lvn(inst: IRInstruction, output_idx: u32) -> IRInstruction {
    // Similar to GVN but limited to current basic block
    return apply_gvn(inst, output_idx);
}

@compute @workgroup_size(256, 1, 1)
fn optimization_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= config.input_size {
        return;
    }
    
    let input_inst = input_ir[idx];
    var output_inst = input_inst;
    
    // Apply selected optimization pass
    switch (config.pass_type) {
        case 0u: { // Dead Code Elimination
            output_inst = eliminate_dead_code(input_inst, idx);
        }
        case 1u: { // Global Value Numbering
            output_inst = apply_gvn(input_inst, idx);
        }
        case 2u: { // Strength Reduction
            output_inst = apply_strength_reduction(input_inst, idx);
        }
        case 3u: { // Constant Folding (simplified)
            if (input_inst.opcode == 10u && input_inst.operands[1] < 65536u) { // Small constant mul
                output_inst.opcode = 254u; // COPY
                output_inst.operands[0] = input_inst.operands[0] * input_inst.operands[1];
                atomicAdd(&optimization_stats[3], 1u);
            }
        }
        case 4u: { // Local Value Numbering
            output_inst = apply_lvn(input_inst, idx);
        }
        default: {
            // Pass through
        }
    }
    
    output_ir[idx] = output_inst;
}
"#;

        let shader = self.compile_shader("optimization", shader_source).await?;
        self.shaders.insert("optimization".to_string(), shader.clone());
        
        let pipeline = self.create_compute_pipeline("optimization", &shader).await?;
        self.pipelines.insert("optimization".to_string(), pipeline);

        Ok(())
    }

async fn create_monomorphization_shader(&mut self) -> Result<()> {
        debug!("Creating monomorphization shader");

        let shader_source = r#"
@group(0) @binding(0) var<storage, read> input_functions: array<GenericFunction>;
@group(0) @binding(1) var<storage, read> type_constraints: array<TypeConstraint>;
@group(0) @binding(2) var<storage, read_write> monomorphized_instances: array<MonoInstance>;
@group(0) @binding(3) var<atomic> instance_counter: atomic<u32>;
@group(0) @binding(4) var<uniform> config: MonomorphizationConfig;

struct GenericFunction {
    function_id: u32,
    num_params: u32,
    param_types: array<u32, 8>,
    return_type: u32,
    function_hash: u32,
    complexity_score: u32,
};

struct TypeConstraint {
    generic_param: u32,
    concrete_type: u32,
    trait_bound: u32,
    variance: u32, // 0=covariant, 1=contravariant, 2=invariant
};

struct MonoInstance {
    function_id: u32,
    instance_id: u32,
    concrete_params: array<u32, 8>,
    num_concrete_params: u32,
    is_valid: u32,
    specialization_depth: u32,
};

struct MonomorphizationConfig {
    max_instances: u32,
    target_types: array<u32, 32>,
    max_specialization_depth: u32,
    enable_trait_specialization: u32,
};

// Hash function for deduplication
fn hash_instance(func_id: u32, params: array<u32, 8>, num_params: u32) -> u32 {
    var hash = func_id * 2654435761u;
    for (var i = 0u; i < num_params; i = i + 1u) {
        hash = hash ^ (params[i] * 2654435761u);
        hash = (hash << 5) | (hash >> 27); // rotate left 5
    }
    return hash;
}

// Check if type constraints are satisfied
fn check_constraints(params: array<u32, 8>, num_params: u32) -> u32 {
    for (var i = 0u; i < arrayLength(&type_constraints); i = i + 1u) {
        let constraint = type_constraints[i];
        if (constraint.generic_param < num_params) {
            let param_type = params[constraint.generic_param];
            // Simple trait bound checking (simplified)
            if (constraint.trait_bound != 0u) {
                // Check if param_type implements required trait
                if (param_type == 1u && (constraint.trait_bound & 0x01u) == 0u) {
                    return 0u; // Integer doesn't implement float trait
                }
                if (param_type == 2u && (constraint.trait_bound & 0x02u) == 0u) {
                    return 0u; // Float doesn't implement integer trait
                }
            }
        }
    }
    return 1u;
}

@compute @workgroup_size(256, 1, 1)
fn monomorphization_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let func_idx = global_id.x;
    if func_idx >= arrayLength(&input_functions) {
        return;
    }
    
    let function = input_functions[func_idx];
    
    // Skip non-generic functions or those too complex
    if (function.num_params == 0u || function.complexity_score > 1000u) {
        return;
    }
    
    // Generate combinations of concrete types
    let num_combinations = u32(pow(f32(arrayLength(&config.target_types)), f32(function.num_params)));
    let max_combos = min(num_combinations, 8u); // Limit combinations per function
    
    for (var combo = 0u; combo < max_combos; combo = combo + 1u) {
        var instance = MonoInstance(
            function.function_id,
            0u, // Will be set atomically
            array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 0),
            function.num_params,
            1u,
            0u
        );
        
        // Generate concrete type combination
        var remaining = combo;
        for (var param = 0u; param < function.num_params; param = param + 1u) {
            let type_idx = remaining % arrayLength(&config.target_types);
            instance.concrete_params[param] = config.target_types[type_idx];
            remaining = remaining / arrayLength(&config.target_types);
        }
        
        // Check if this instance satisfies type constraints
        if (check_constraints(instance.concrete_params, instance.num_concrete_params) == 0u) {
            continue;
        }
        
        // Check for duplicates using hash
        let instance_hash = hash_instance(function.function_id, instance.concrete_params, instance.num_concrete_params);
        
        // Atomically get a unique instance ID
        let instance_id = atomicAdd(&instance_counter, 1u);
        if (instance_id >= config.max_instances) {
            return;
        }
        
        instance.instance_id = instance_id;
        monomorphized_instances[instance_id] = instance;
    }
}
"#;

        let shader = self.compile_shader("monomorphization", shader_source).await?;
        self.shaders.insert("monomorphization".to_string(), shader.clone());
        
        let pipeline = self.create_compute_pipeline("monomorphization", &shader).await?;
        self.pipelines.insert("monomorphization".to_string(), pipeline);

        Ok(())
    }

    async fn create_link_preparation_shader(&mut self) -> Result<()> {
        debug!("Creating link preparation shader");

        let shader_source = r#"
@group(0) @binding(0) var<storage, read> object_files: array<u32>;
@group(0) @binding(1) var<storage, read_write> symbol_table: array<u32>;
@group(0) @binding(2) var<uniform> link_config: LinkConfig;

struct LinkConfig {
    object_count: u32,
    symbol_count: u32,
    relocation_entries: u32,
};

fn hash_symbol(symbol_data: u32) -> u32 {
    var hash = 2166136261u;
    hash = hash ^ (symbol_data & 0xFFu);
    hash = hash * 16777619u;
    hash = hash ^ ((symbol_data >> 8u) & 0xFFu);
    hash = hash * 16777619u;
    return hash;
}

@compute @workgroup_size(512, 1, 1)
fn link_preparation_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= link_config.object_count {
        return;
    }
    
    let object_data = object_files[idx];
    var symbol_entry = object_data;
    
    // Extract symbols from object file
    let symbol_hash = hash_symbol(object_data);
    
    // Check for duplicate symbols
    if (symbol_hash & 0x80000000u) == 0u {
        symbol_entry = symbol_entry | 0x80000000u; // Mark as resolved
    }
    
    // Prepare relocations
    if (object_data & 0x00FF0000u) != 0u {
        symbol_entry = symbol_entry | 0x00800000u; // Has relocations
    }
    
    // Update symbol table
    if idx < link_config.symbol_count {
        symbol_table[idx] = symbol_entry;
    }
}
"#;

        let shader = self.compile_shader("link_preparation", shader_source).await?;
        self.shaders.insert("link_preparation".to_string(), shader.clone());
        
        let pipeline = self.create_compute_pipeline("link_preparation", &shader).await?;
        self.pipelines.insert("link_preparation".to_string(), pipeline);

        Ok(())
    }

    async fn create_register_allocation_shader(&mut self) -> Result<()> {
        debug!("Creating register allocation shader");

        let shader_source = r#"
@group(0) @binding(0) var<storage, read> live_ranges: array<u32>;
@group(0) @binding(1) var<storage, read_write> register_map: array<u32>;
@group(0) @binding(2) var<uniform> reg_config: RegConfig;

struct RegConfig {
    variable_count: u32,
    register_count: u32,
    allocation_strategy: u32,
};

@compute @workgroup_size(128, 1, 1)
fn register_allocation_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= reg_config.variable_count {
        return;
    }
    
    let live_range = live_ranges[idx];
    var allocated_register = 0u;
    
    // Linear scan allocation
    if reg_config.allocation_strategy == 0u {
        allocated_register = idx % reg_config.register_count;
    }
    // Graph coloring allocation simulation
    else if reg_config.allocation_strategy == 1u {
        let conflicts = (live_range >> 16u) & 0xFFu;
        allocated_register = (idx + conflicts) % reg_config.register_count;
    }
    // Greedy allocation with interference
    else if reg_config.allocation_strategy == 2u {
        let interference = live_range & 0x0000FFFFu;
        allocated_register = (idx * 7u + interference) % reg_config.register_count;
    }
    
    // Mark register as allocated
    let allocation_entry = (allocated_register << 24u) | (idx & 0x00FFFFFFu);
    register_map[idx] = allocation_entry;
}
"#;

        let shader = self.compile_shader("register_allocation", shader_source).await?;
        self.shaders.insert("register_allocation".to_string(), shader.clone());
        
        let pipeline = self.create_compute_pipeline("register_allocation", &shader).await?;
        self.pipelines.insert("register_allocation".to_string(), pipeline);

        Ok(())
    }

    async fn create_type_checking_shader(&mut self) -> Result<()> {
        debug!("Creating type checking shader");

        let shader_source = r#"
@group(0) @binding(0) var<storage, read> type_constraints: array<u32>;
@group(0) @binding(1) var<storage, read_write> type_errors: array<u32>;
@group(0) @binding(2) var<uniform> type_config: TypeConfig;

struct TypeConfig {
    constraint_count: u32,
    max_type_depth: u32,
    checking_mode: u32,
};

fn type_compatibility(expected: u32, actual: u32) -> bool {
    // Simple type compatibility check
    if expected == actual {
        return true;
    }
    
    // Subtype checking simulation
    let expected_base = expected & 0xF0u;
    let actual_base = actual & 0xF0u;
    
    return expected_base == actual_base;
}

@compute @workgroup_size(256, 1, 1)
fn type_checking_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= type_config.constraint_count {
        return;
    }
    
    let constraint = type_constraints[idx];
    var error_flag = 0u;
    
    let expected_type = (constraint >> 16u) & 0xFFFFu;
    let actual_type = constraint & 0xFFFFu;
    
    // Check type compatibility
    if !type_compatibility(expected_type, actual_type) {
        error_flag = 1u;
    }
    
    // Check generic constraints
    if (expected_type & 0x8000u) != 0u && (actual_type & 0x8000u) == 0u {
        error_flag = 2u; // Generic constraint violation
    }
    
    // Check lifetime constraints
    if (constraint & 0x40000000u) != 0u {
        let lifetime_mismatch = ((expected_type ^ actual_type) & 0x0F00u) != 0u;
        if lifetime_mismatch {
            error_flag = 3u; // Lifetime error
        }
    }
    
    // Record error
    let error_entry = (error_flag << 24u) | idx;
    type_errors[idx] = error_entry;
}
"#;

        let shader = self.compile_shader("type_checking", shader_source).await?;
        self.shaders.insert("type_checking".to_string(), shader.clone());
        
        let pipeline = self.create_compute_pipeline("type_checking", &shader).await?;
        self.pipelines.insert("type_checking".to_string(), pipeline);

        Ok(())
    }

    async fn create_borrow_checking_shader(&mut self) -> Result<()> {
        debug!("Creating borrow checking shader");

        let shader_source = r#"
@group(0) @binding(0) var<storage, read> borrow_operations: array<u32>;
@group(0) @binding(1) var<storage, read_write> borrow_errors: array<u32>;
@group(0) @binding(2) var<uniform> borrow_config: BorrowConfig;

struct BorrowConfig {
    operation_count: u32,
    region_count: u32,
    checking_mode: u32,
};

fn check_borrow_conflict(op1: u32, op2: u32) -> bool {
    let op1_kind = (op1 >> 28u) & 0xFu;
    let op2_kind = (op2 >> 28u) & 0xFu;
    let op1_region = (op1 >> 16u) & 0xFFFu;
    let op2_region = (op2 >> 16u) & 0xFFFu;
    
    // Different regions don't conflict
    if op1_region != op2_region {
        return false;
    }
    
    // Immutable borrows can coexist
    if op1_kind == 1u && op2_kind == 1u {
        return false;
    }
    
    // Mutable borrows conflict with any other borrow
    if op1_kind == 2u || op2_kind == 2u {
        return true;
    }
    
    return false;
}

@compute @workgroup_size(256, 1, 1)
fn borrow_checking_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= borrow_config.operation_count {
        return;
    }
    
    let current_op = borrow_operations[idx];
    var error_flag = 0u;
    
    // Check against all other operations
    for (var i = 0u; i < borrow_config.operation_count; i = i + 1u) {
        if i == idx {
            continue;
        }
        
        let other_op = borrow_operations[i];
        
        if check_borrow_conflict(current_op, other_op) {
            error_flag = 1u;
            break;
        }
    }
    
    // Check for use-after-move
    let op_kind = (current_op >> 28u) & 0xFu;
    if op_kind == 4u { // Move operation
        // Check if there are any later uses
        for (var i = idx + 1u; i < borrow_config.operation_count; i = i + 1u) {
            let later_op = borrow_operations[i];
            let later_region = (later_op >> 16u) & 0xFFFu;
            let current_region = (current_op >> 16u) & 0xFFFu;
            
            if later_region == current_region {
                error_flag = 2u; // Use after move
                break;
            }
        }
    }
    
    // Record error
    let error_entry = (error_flag << 24u) | idx;
    borrow_errors[idx] = error_entry;
}
"#;

        let shader = self.compile_shader("borrow_checking", shader_source).await?;
        self.shaders.insert("borrow_checking".to_string(), shader.clone());
        
        let pipeline = self.create_compute_pipeline("borrow_checking", &shader).await?;
        self.pipelines.insert("borrow_checking".to_string(), pipeline);

        Ok(())
    }

    async fn compile_shader(&self, name: &str, source: &str) -> Result<Arc<ShaderModule>> {
        debug!("Compiling shader: {}", name);

        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        Ok(Arc::new(shader_module))
    }

    async fn create_compute_pipeline(
        &self,
        name: &str,
        shader: &ShaderModule,
    ) -> Result<Arc<ComputePipeline>> {
        debug!("Creating compute pipeline: {}", name);

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_layout", name)),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "main",
        });

        Ok(Arc::new(pipeline))
    }

    pub async fn dispatch_compilation_task(
        &self,
        shader_name: &str,
        input_data: &[u8],
        config: &CompilationTaskConfig,
    ) -> Result<Vec<u8>> {
        debug!("Dispatching compilation task with shader: {}", shader_name);

        let shader = self.shaders
            .get(shader_name)
            .ok_or_else(|| anyhow!("Shader '{}' not found", shader_name))?;
        
        let pipeline = self.pipelines
            .get(shader_name)
            .ok_or_else(|| anyhow!("Pipeline '{}' not found", shader_name))?;

        // Create input buffer
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}_input", shader_name)),
            size: input_data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output buffer
        let output_buffer_size = (input_data.len() * config.output_size_multiplier) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}_output", shader_name)),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Copy input data to GPU
        self.queue.write_buffer(&input_buffer, 0, input_data);

        // Compute workgroup count
        let workgroup_count = ((input_data.len() as u32 + self.config.max_workgroup_size - 1) 
            / self.config.max_workgroup_size) * self.config.workgroup_multiplier;

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{}_encoder", shader_name)),
        });

        {
        let bind_group = self.create_bind_group(shader_name, &input_buffer, &output_buffer, config)?;
        
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{}_pass", shader_name)),
                timestamp_writes: None,
            });
            
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        }

        // Copy output back to CPU
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &output_buffer, 0, output_buffer_size);

        // Submit commands
        let command_buffer = encoder.finish();
        self.queue.submit(Some(command_buffer));

        // Read back results
        let result_data = self.read_buffer_data(&output_buffer).await?;
        
        Ok(result_data)
    }

    fn create_bind_group(
        &self,
        _shader_name: &str,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        config: &CompilationTaskConfig,
    ) -> Result<BindGroup> {
        // Create uniform buffer for config
        let config_data = config.to_bytes();
        let config_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config_buffer"),
            size: config_data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(bind_group)
    }

    async fn read_buffer_data(&self, buffer: &Buffer) -> Result<Vec<u8>> {
        let buffer_slice = buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        
        self.device.poll(wgpu::Maintain::Wait);
        
        let _ = rx.await??;
        let data = buffer_slice.get_mapped_range().to_vec();
        
        Ok(data)
    }

    pub fn get_available_shaders(&self) -> Vec<&str> {
        self.shaders.keys().map(|s| s.as_str()).collect()
    }

    pub fn get_shader_info(&self, shader_name: &str) -> Option<CompilationShader> {
        // Return shader metadata based on the shader name
        match shader_name {
            "code_generation" => Some(CompilationShader {
                name: shader_name.to_string(),
                shader_type: ShaderType::CodeGeneration,
                workgroup_size: (64, 1, 1),
                bindings: vec![
                    ShaderBinding {
                        name: "input_data".to_string(),
                        binding_type: BindingType::ReadOnly,
                        size: None,
                        visibility: wgpu::ShaderStages::COMPUTE,
                    },
                    ShaderBinding {
                        name: "output_data".to_string(),
                        binding_type: BindingType::ReadWrite,
                        size: None,
                        visibility: wgpu::ShaderStages::COMPUTE,
                    },
                ],
            }),
            "optimization" => Some(CompilationShader {
                name: shader_name.to_string(),
                shader_type: ShaderType::Optimization,
                workgroup_size: (128, 1, 1),
                bindings: vec![
                    ShaderBinding {
                        name: "ir_instructions".to_string(),
                        binding_type: BindingType::ReadOnly,
                        size: None,
                        visibility: wgpu::ShaderStages::COMPUTE,
                    },
                    ShaderBinding {
                        name: "optimized_ir".to_string(),
                        binding_type: BindingType::ReadWrite,
                        size: None,
                        visibility: wgpu::ShaderStages::COMPUTE,
                    },
                ],
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompilationTaskConfig {
    pub input_size: u32,
    pub output_size_multiplier: usize,
    pub optimization_level: u32,
    pub flags: u32,
}

impl CompilationTaskConfig {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.input_size.to_le_bytes());
        bytes.extend_from_slice(&(self.output_size_multiplier as u32).to_le_bytes());
        bytes.extend_from_slice(&self.optimization_level.to_le_bytes());
        bytes.extend_from_slice(&self.flags.to_le_bytes());
        bytes
    }
}

// Helper trait for buffer initialization
trait BufferInitExt {
    fn create_buffer_init(&self, descriptor: &wgpu::util::BufferInitDescriptor) -> Buffer;
}

impl BufferInitExt for Device {
    fn create_buffer_init(&self, descriptor: &wgpu::util::BufferInitDescriptor) -> Buffer {
        self.create_buffer(&wgpu::BufferDescriptor {
            label: descriptor.label,
            size: descriptor.contents.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        })
    }
}