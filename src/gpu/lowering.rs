use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::gpu::monomorphizer::{MonomorphizedInstance, TypeInfo};

pub struct FunctionLowerer {
    config: LoweringConfig,
    type_converter: TypeConverter,
    instruction_selector: InstructionSelector,
}

#[derive(Debug, Clone)]
pub struct LoweringConfig {
    pub enable_vectorization: bool,
    pub enable_loop_unrolling: bool,
    pub enable_inlining: bool,
    pub max_inline_size: usize,
    pub optimization_level: OptimizationLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizationLevel {
    O0,
    O1,
    O2,
    O3,
}

impl Default for LoweringConfig {
    fn default() -> Self {
        Self {
            enable_vectorization: true,
            enable_loop_unrolling: true,
            enable_inlining: true,
            max_inline_size: 50,
            optimization_level: OptimizationLevel::O2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoweredFunction {
    pub name: String,
    pub original_name: String,
    pub parameters: Vec<LoweredParameter>,
    pub return_type: LoweredType,
    pub basic_blocks: Vec<BasicBlock>,
    pub register_count: usize,
    pub stack_size: usize,
    pub calls_other_functions: bool,
    pub is_gpu_kernel: bool,
}

#[derive(Debug, Clone)]
pub struct LoweredParameter {
    pub name: String,
    pub type_info: LoweredType,
    pub is_by_reference: bool,
    pub register_location: Option<RegisterLocation>,
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: usize,
    pub label: String,
    pub instructions: Vec<LoweredInstruction>,
    pub predecessors: Vec<usize>,
    pub successors: Vec<usize>,
    pub phi_nodes: Vec<PhiNode>,
}

#[derive(Debug, Clone)]
pub struct LoweredInstruction {
    pub opcode: Opcode,
    pub operands: Vec<Operand>,
    pub result: Option<RegisterLocation>,
    pub metadata: InstructionMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegisterLocation {
    pub register: u32,
    pub virtual_register: bool,
}

#[derive(Debug, Clone)]
pub struct PhiNode {
    pub result: RegisterLocation,
    pub incoming: Vec<(BasicBlockId, Operand)>,
}

#[derive(Debug, Clone)]
pub struct InstructionMetadata {
    pub source_line: Option<u32>,
    pub optimization_hints: Vec<String>,
    pub performance_critical: bool,
}

#[derive(Debug, Clone)]
pub enum LoweredType {
    I8,
    I32,
    I64,
    F32,
    F64,
    Bool,
    Vector { element: Box<LoweredType>, size: usize },
    Struct { name: String, fields: Vec<(String, LoweredType)> },
    Array { element: Box<LoweredType>, size: usize },
    Pointer(Box<LoweredType>),
    Void,
}



#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Register(RegisterLocation),
    Immediate(Value),
    Memory(MemoryLocation),
    Label(BasicBlockId),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryLocation {
    pub base: Option<RegisterLocation>,
    pub offset: i64,
    pub size: usize,
}



#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasicBlockId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Opcode {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    
    // Bitwise
    And,
    Or,
    Xor,
    Shl,
    Shr,
    
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    
    // Control flow
    Br,
    CondBr,
    Ret,
    Call,
    IndirectCall,
    
    // Memory
    Load,
    Store,
    Alloca,
    GetElementPtr,
    
    // Vector operations
    VectorAdd,
    VectorSub,
    VectorMul,
    VectorDiv,
    
    // GPU-specific
    ThreadId,
    BlockId,
    GridId,
    SyncThreads,
    
    // Special
    Phi,
    Cast,
}

impl FunctionLowerer {
    pub fn new(config: LoweringConfig) -> Self {
        info!("Initializing FunctionLowerer with config: {:?}", config);
        
        Self {
            type_converter: TypeConverter::new(),
            instruction_selector: InstructionSelector::new(config.clone()),
            config,
        }
    }

    pub async fn lower_function(
        &mut self,
        instance: &MonomorphizedInstance,
    ) -> Result<LoweredFunction> {
        debug!("Lowering function: {}", instance.function_name);
        
        let mut lowered = LoweredFunction {
            name: instance.function_name.clone(),
            original_name: instance.function_name.clone(),
            parameters: Vec::new(),
            return_type: LoweredType::Void,
            basic_blocks: Vec::new(),
            register_count: 0,
            stack_size: 0,
            calls_other_functions: false,
            is_gpu_kernel: self.is_gpu_kernel_candidate(instance),
        };

        self.analyze_function_signature(instance, &mut lowered)?;
        self.lower_function_body(instance, &mut lowered).await?;
        self.optimize_lowered_function(&mut lowered).await?;
        
        info!("Lowered function {} to {} basic blocks, {} registers",
              lowered.name, lowered.basic_blocks.len(), lowered.register_count);
        
        Ok(lowered)
    }

    fn analyze_function_signature(
        &mut self,
        instance: &MonomorphizedInstance,
        lowered: &mut LoweredFunction,
    ) -> Result<()> {
        debug!("Analyzing function signature for {}", instance.function_name);

        let param_count = (instance.optimized_code.len() / 8).min(10);
        
        for i in 0..param_count {
            let param_type = if i % 3 == 0 { 
                LoweredType::I32 
            } else if i % 3 == 1 { 
                LoweredType::F32 
            } else { 
                LoweredType::I64 
            };

            lowered.parameters.push(LoweredParameter {
                name: format!("param_{}", i),
                type_info: param_type.clone(),
                is_by_reference: false,
                register_location: Some(RegisterLocation {
                    register: i as u32,
                    virtual_register: true,
                }),
            });
        }

        lowered.return_type = if instance.concrete_types.len() > 0 {
            self.type_converter.convert_type(&instance.concrete_types[0])?
        } else {
            LoweredType::Void
        };

        Ok(())
    }

    async fn lower_function_body(
        &mut self,
        instance: &MonomorphizedInstance,
        lowered: &mut LoweredFunction,
    ) -> Result<()> {
        debug!("Lowering function body for {}", instance.function_name);

        let mut entry_block = BasicBlock {
            id: 0,
            label: "entry".to_string(),
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: vec![1],
            phi_nodes: Vec::new(),
        };

        let body_blocks = self.create_basic_blocks_from_ir(&instance.optimized_code).await?;
        
        let mut next_block_id = 1;
        let chunks: Vec<&[u8]> = instance.optimized_code.chunks(64).collect();
        for (chunk_idx, code_chunk) in chunks.iter().enumerate() {
            let mut block = BasicBlock {
                id: next_block_id,
                label: format!("block_{}", next_block_id),
                instructions: Vec::new(),
                predecessors: vec![next_block_id - 1],
                successors: if chunk_idx < chunks.len() - 1 {
                    vec![next_block_id + 1]
                } else {
                    Vec::new()
                },
                phi_nodes: Vec::new(),
            };

            let instructions = self.lower_code_chunk(code_chunk, &mut lowered.register_count).await?;
            block.instructions.extend(instructions);

            lowered.basic_blocks.push(block);
            next_block_id += 1;
        }

        if let Some(last_block) = lowered.basic_blocks.last_mut() {
            last_block.instructions.push(LoweredInstruction {
                opcode: Opcode::Ret,
                operands: Vec::new(),
                result: None,
                metadata: InstructionMetadata {
                    source_line: None,
                    optimization_hints: Vec::new(),
                    performance_critical: false,
                },
            });
        }

        lowered.basic_blocks.insert(0, entry_block);
        
        Ok(())
    }

    async fn create_basic_blocks_from_ir(
        &self,
        ir_code: &[u8],
    ) -> Result<Vec<Vec<u8>>> {
        let mut blocks = Vec::new();
        
        let block_size = 64;
        for chunk in ir_code.chunks(block_size) {
            blocks.push(chunk.to_vec());
        }

        Ok(blocks)
    }

    async fn lower_code_chunk(
        &mut self,
        chunk: &[u8],
        register_count: &mut usize,
    ) -> Result<Vec<LoweredInstruction>> {
        let mut instructions = Vec::new();

        for (i, &byte) in chunk.iter().enumerate() {
            let opcode = match byte % 10 {
                0 => Opcode::Add,
                1 => Opcode::Sub,
                2 => Opcode::Mul,
                3 => Opcode::Load,
                4 => Opcode::Store,
                5 => Opcode::Br,
                6 => Opcode::CondBr,
                7 => Opcode::Call,
                8 => Opcode::Cast,
                _ => Opcode::Add,
            };

            let instruction = LoweredInstruction {
                opcode,
                operands: vec![
                    Operand::Register(RegisterLocation {
                        register: (i * 2) as u32,
                        virtual_register: true,
                    }),
                    Operand::Immediate(Value::I32(byte as i32)),
                ],
                result: Some(RegisterLocation {
                    register: (*register_count) as u32,
                    virtual_register: true,
                }),
                metadata: InstructionMetadata {
                    source_line: Some(i as u32),
                    optimization_hints: if byte % 7 == 0 {
                        vec!["vectorizable".to_string()]
                    } else {
                        Vec::new()
                    },
                    performance_critical: byte % 13 == 0,
                },
            };

            instructions.push(instruction);
            *register_count += 1;
        }

        Ok(instructions)
    }

    async fn optimize_lowered_function(&mut self, lowered: &mut LoweredFunction) -> Result<()> {
        debug!("Optimizing lowered function {}", lowered.name);

        if matches!(self.config.optimization_level, OptimizationLevel::O2 | OptimizationLevel::O3) {
            self.apply_local_optimizations(lowered).await?;
        }

        if self.config.enable_vectorization {
            self.apply_vectorization(lowered).await?;
        }

        if self.config.enable_inlining {
            self.mark_inlinable_functions(lowered);
        }

        self.calculate_stack_usage(lowered)?;

        Ok(())
    }

    async fn apply_local_optimizations(&self, lowered: &mut LoweredFunction) -> Result<()> {
        debug!("Applying local optimizations to {}", lowered.name);

        for block in &mut lowered.basic_blocks {
            let mut i = 0;
            while i < block.instructions.len() {
                let instruction = &block.instructions[i];
                
                if self.is_redundant_instruction(instruction, &block.instructions, i) {
                    block.instructions.remove(i);
                    continue;
                }

                if let Some(optimized) = self.try_constant_folding(instruction, &block.instructions, i) {
                    block.instructions[i] = optimized;
                }

                i += 1;
            }
        }

        Ok(())
    }

    async fn apply_vectorization(&self, lowered: &mut LoweredFunction) -> Result<()> {
        debug!("Applying vectorization to {}", lowered.name);

        for block in &mut lowered.basic_blocks {
            let mut vectorizable_ops = Vec::new();

            for (i, instruction) in block.instructions.iter().enumerate() {
                if instruction.metadata.optimization_hints.contains(&"vectorizable".to_string()) {
                    vectorizable_ops.push(i);
                }
            }

            for chunk in vectorizable_ops.chunks(4) {
                if chunk.len() >= 2 {
                    let first_idx = chunk[0];
                    if let Some(first_instr) = block.instructions.get(first_idx) {
                        let vectorized = LoweredInstruction {
                            opcode: match first_instr.opcode {
                                Opcode::Add => Opcode::VectorAdd,
                                Opcode::Sub => Opcode::VectorSub,
                                Opcode::Mul => Opcode::VectorMul,
                                Opcode::Div => Opcode::VectorDiv,
                                _ => first_instr.opcode,
                            },
                            operands: first_instr.operands.clone(),
                            result: first_instr.result,
                            metadata: InstructionMetadata {
                                source_line: first_instr.metadata.source_line,
                                optimization_hints: vec!["vectorized".to_string()],
                                performance_critical: true,
                            },
                        };

                        for &idx in chunk.iter().skip(1).rev() {
                            if idx < block.instructions.len() {
                                block.instructions.remove(idx);
                            }
                        }

                        block.instructions[first_idx] = vectorized;
                    }
                }
            }
        }

        Ok(())
    }

    fn mark_inlinable_functions(&self, lowered: &mut LoweredFunction) {
        debug!("Marking inlinable functions for {}", lowered.name);

        let instruction_count = lowered.basic_blocks
            .iter()
            .map(|b| b.instructions.len())
            .sum::<usize>();

        lowered.calls_other_functions = instruction_count < self.config.max_inline_size;
    }

    fn calculate_stack_usage(&self, lowered: &mut LoweredFunction) -> Result<()> {
        let mut stack_size = 0;

        for block in &lowered.basic_blocks {
            for instruction in &block.instructions {
                if instruction.opcode == Opcode::Alloca {
                    stack_size += 8;
                }
            }
        }

        lowered.stack_size = stack_size;
        Ok(())
    }

    fn is_gpu_kernel_candidate(&self, instance: &MonomorphizedInstance) -> bool {
        instance.dependency_graph.iter().any(|dep| 
            dep.contains("cuda") || dep.contains("gpu") || dep.contains("kernel")
        ) || instance.function_name.contains("kernel")
    }

    fn is_redundant_instruction(
        &self,
        instruction: &LoweredInstruction,
        block_instructions: &[LoweredInstruction],
        instruction_index: usize,
    ) -> bool {
        if instruction_index == 0 {
            return false;
        }

        if let Some(prev_instr) = block_instructions.get(instruction_index - 1) {
            if prev_instr.opcode == instruction.opcode 
                && prev_instr.operands == instruction.operands
                && instruction.opcode != Opcode::Store
                && instruction.opcode != Opcode::Call {
                return true;
            }
        }

        false
    }

    fn try_constant_folding(
        &self,
        instruction: &LoweredInstruction,
        block_instructions: &[LoweredInstruction],
        instruction_index: usize,
    ) -> Option<LoweredInstruction> {
        if instruction.operands.len() < 2 {
            return None;
        }

        match (&instruction.operands[0], &instruction.operands[1]) {
            (Operand::Immediate(val1), Operand::Immediate(val2)) => {
                let result = match (instruction.opcode, val1, val2) {
                    (Opcode::Add, Value::I32(a), Value::I32(b)) => {
                        Some(Value::I32(a.wrapping_add(*b)))
                    }
                    (Opcode::Mul, Value::I32(a), Value::I32(b)) => {
                        Some(Value::I32(a.wrapping_mul(*b)))
                    }
                    _ => None,
                };

                result.map(|constant_value| LoweredInstruction {
                    opcode: Opcode::Add,
                    operands: vec![Operand::Immediate(constant_value)],
                    result: instruction.result,
                    metadata: InstructionMetadata {
                        source_line: instruction.metadata.source_line,
                        optimization_hints: vec!["constant_folded".to_string()],
                        performance_critical: false,
                    },
                })
            }
            _ => None,
        }
    }

    pub fn get_lowering_statistics(&self) -> LoweringStats {
        LoweringStats {
            functions_lowered: 0,
            total_instructions_generated: 0,
            total_registers_used: 0,
            vectorized_operations: 0,
            constant_folded_operations: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoweringStats {
    pub functions_lowered: usize,
    pub total_instructions_generated: usize,
    pub total_registers_used: usize,
    pub vectorized_operations: usize,
    pub constant_folded_operations: usize,
}

struct TypeConverter {
    type_cache: HashMap<String, LoweredType>,
}

impl TypeConverter {
    fn new() -> Self {
        Self {
            type_cache: HashMap::new(),
        }
    }

    fn convert_type(&mut self, type_info: &TypeInfo) -> Result<LoweredType> {
        let type_key = format!("{:?}", type_info);
        
        if let Some(cached) = self.type_cache.get(&type_key) {
            return Ok(cached.clone());
        }

        let lowered = match type_info {
            TypeInfo::I32 => LoweredType::I32,
            TypeInfo::I64 => LoweredType::I64,
            TypeInfo::F32 => LoweredType::F32,
            TypeInfo::F64 => LoweredType::F64,
            TypeInfo::Bool => LoweredType::Bool,
            TypeInfo::String => LoweredType::Pointer(Box::new(LoweredType::I8)),
            TypeInfo::Reference(inner) => {
                let inner_lowered = self.convert_type(inner)?;
                LoweredType::Pointer(Box::new(inner_lowered))
            }
            TypeInfo::MutReference(inner) => {
                let inner_lowered = self.convert_type(inner)?;
                LoweredType::Pointer(Box::new(inner_lowered))
            }
            TypeInfo::Array { element, size } => {
                let element_lowered = self.convert_type(element)?;
                LoweredType::Array {
                    element: Box::new(element_lowered),
                    size: *size,
                }
            }
            TypeInfo::Struct { name, fields } => {
                let lowered_fields: Result<Vec<_>> = fields
                    .iter()
                    .map(|(fname, ftype)| {
                        let lowered_ftype = self.convert_type(ftype)?;
                        Ok((fname.clone(), lowered_ftype))
                    })
                    .collect();
                LoweredType::Struct {
                    name: name.clone(),
                    fields: lowered_fields?,
                }
            }
            _ => LoweredType::Void,
        };

        self.type_cache.insert(type_key, lowered.clone());
        Ok(lowered)
    }
}

struct InstructionSelector {
    config: LoweringConfig,
}

impl InstructionSelector {
    fn new(config: LoweringConfig) -> Self {
        Self { config }
    }

    fn select_instruction(
        &self,
        operation: &str,
        operands: &[Operand],
        target_type: &LoweredType,
    ) -> Opcode {
        match operation {
            "add" => match target_type {
                LoweredType::Vector { .. } => Opcode::VectorAdd,
                _ => Opcode::Add,
            },
            "sub" => match target_type {
                LoweredType::Vector { .. } => Opcode::VectorSub,
                _ => Opcode::Sub,
            },
            "mul" => match target_type {
                LoweredType::Vector { .. } => Opcode::VectorMul,
                _ => Opcode::Mul,
            },
            "div" => match target_type {
                LoweredType::Vector { .. } => Opcode::VectorDiv,
                _ => Opcode::Div,
            },
            "load" => Opcode::Load,
            "store" => Opcode::Store,
            "call" => Opcode::Call,
            "branch" => Opcode::Br,
            "cond_branch" => Opcode::CondBr,
            "return" => Opcode::Ret,
            _ => Opcode::Add,
        }
    }
}