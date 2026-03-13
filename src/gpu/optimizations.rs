use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::gpu::lowering::{
    LoweredFunction, LoweredInstruction, BasicBlock, Opcode, Operand, 
    RegisterLocation, Value, MemoryLocation, BasicBlockId
};

pub struct PeepholeOptimizer {
    config: OptimizationConfig,
    pattern_matcher: PatternMatcher,
    value_numberer: ValueNumberer,
}

#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub enable_constant_folding: bool,
    pub enable_dead_code_elimination: bool,
    pub enable_strength_reduction: bool,
    pub enable_algebraic_simplification: bool,
    pub enable_copy_propagation: bool,
    pub max_passes: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_constant_folding: true,
            enable_dead_code_elimination: true,
            enable_strength_reduction: true,
            enable_algebraic_simplification: true,
            enable_copy_propagation: true,
            max_passes: 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub optimized_function: LoweredFunction,
    pub passes_completed: usize,
    pub instructions_removed: usize,
    pub instructions_modified: usize,
    pub constants_folded: usize,
    pub dead_code_eliminated: usize,
}

impl PeepholeOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        info!("Initializing PeepholeOptimizer with config: {:?}", config);
        
        Self {
            pattern_matcher: PatternMatcher::new(),
            value_numberer: ValueNumberer::new(),
            config,
        }
    }

    pub async fn optimize_function(&mut self, function: LoweredFunction) -> Result<OptimizationResult> {
        debug!("Starting peephole optimization for function: {}", function.name);
        
        let mut optimized_function = function.clone();
        let mut optimization_stats = OptimizationStats::new();
        
        for pass in 0..self.config.max_passes {
            debug!("Starting optimization pass {}", pass + 1);
            
            let pass_stats = self.run_optimization_pass(&mut optimized_function).await?;
            
            if !pass_stats.made_changes() {
                debug!("No changes in pass {}, stopping optimization", pass + 1);
                break;
            }
            
            optimization_stats.combine(pass_stats);
            
            if pass >= 2 {
                debug!("Completed {} passes with diminishing returns", pass + 1);
                break;
            }
        }

        let result = OptimizationResult {
            optimized_function,
            passes_completed: optimization_stats.passes_completed,
            instructions_removed: optimization_stats.instructions_removed,
            instructions_modified: optimization_stats.instructions_modified,
            constants_folded: optimization_stats.constants_folded,
            dead_code_eliminated: optimization_stats.dead_code_eliminated,
        };

        info!("Peephole optimization completed for {}: {} passes, {} instructions removed",
              function.name, result.passes_completed, result.instructions_removed);
        
        Ok(result)
    }

    async fn run_optimization_pass(&mut self, function: &mut LoweredFunction) -> Result<OptimizationStats> {
        let mut stats = OptimizationStats::new();
        stats.passes_completed = 1;

        if self.config.enable_dead_code_elimination {
            stats.dead_code_eliminated += self.eliminate_dead_code(function).await?;
        }

        if self.config.enable_constant_folding {
            stats.constants_folded += self.fold_constants(function).await?;
        }

        if self.config.enable_strength_reduction {
            stats.instructions_modified += self.apply_strength_reduction(function).await?;
        }

        if self.config.enable_algebraic_simplification {
            stats.instructions_modified += self.simplify_algebraic_expressions(function).await?;
        }

        if self.config.enable_copy_propagation {
            stats.instructions_modified += self.propagate_copies(function).await?;
        }

        stats.instructions_removed = function.basic_blocks
            .iter()
            .map(|b| b.instructions.len())
            .sum::<usize>();

        Ok(stats)
    }

    async fn eliminate_dead_code(&self, function: &mut LoweredFunction) -> Result<usize> {
        debug!("Eliminating dead code in {}", function.name);
        
        let mut live_values = HashSet::new();
        let mut instructions_removed = 0;

        for block in &function.basic_blocks {
            for instruction in &block.instructions {
                self.collect_live_values(instruction, &mut live_values);
            }
        }

        for block in &mut function.basic_blocks {
            let mut i = 0;
            while i < block.instructions.len() {
                let instruction = &block.instructions[i];
                
                if let Some(result) = &instruction.result {
                    let value_id = self.get_value_id(result);
                    if !live_values.contains(&value_id) && self.is_instruction_safe_to_remove(instruction) {
                        block.instructions.remove(i);
                        instructions_removed += 1;
                        continue;
                    }
                }
                
                i += 1;
            }
        }

        debug!("Removed {} dead instructions", instructions_removed);
        Ok(instructions_removed)
    }

    async fn fold_constants(&self, function: &mut LoweredFunction) -> Result<usize> {
        debug!("Folding constants in {}", function.name);
        
        let mut constants_folded = 0;
        let mut constant_values: HashMap<u32, Value> = HashMap::new();

        for block in &mut function.basic_blocks {
            for instruction in &mut block.instructions {
                if let Some(folded_value) = self.try_fold_constant(instruction, &constant_values) {
                    if let Some(result) = &instruction.result {
                        constant_values.insert(result.register, folded_value.clone());
                    }
                    
                    *instruction = LoweredInstruction {
                        opcode: Opcode::Add,
                        operands: vec![Operand::Immediate(folded_value)],
                        result: instruction.result.clone(),
                        metadata: instruction.metadata.clone(),
                    };
                    
                    constants_folded += 1;
                }
            }
        }

        debug!("Folded {} constant expressions", constants_folded);
        Ok(constants_folded)
    }

    async fn apply_strength_reduction(&self, function: &mut LoweredFunction) -> Result<usize> {
        debug!("Applying strength reduction in {}", function.name);
        
        let mut instructions_modified = 0;

        for block in &mut function.basic_blocks {
            for instruction in &mut block.instructions {
                if self.can_apply_strength_reduction(instruction) {
                    self.apply_strength_reduction_to_instruction(instruction);
                    instructions_modified += 1;
                }
            }
        }

        debug!("Applied strength reduction to {} instructions", instructions_modified);
        Ok(instructions_modified)
    }

    async fn simplify_algebraic_expressions(&self, function: &mut LoweredFunction) -> Result<usize> {
        debug!("Simplifying algebraic expressions in {}", function.name);
        
        let mut instructions_modified = 0;

        for block in &mut function.basic_blocks {
            for instruction in &mut block.instructions {
                if let Some(simplified) = self.try_algebraic_simplification(instruction) {
                    *instruction = simplified;
                    instructions_modified += 1;
                }
            }
        }

        debug!("Simplified {} algebraic expressions", instructions_modified);
        Ok(instructions_modified)
    }

    async fn propagate_copies(&self, function: &mut LoweredFunction) -> Result<usize> {
        debug!("Propagating copies in {}", function.name);
        
        let mut instructions_modified = 0;
        let mut copy_map: HashMap<u32, u32> = HashMap::new();

        for block in &mut function.basic_blocks {
            for instruction in &mut block.instructions {
                self.replace_copies_with_originals(instruction, &copy_map);
                
                if self.is_copy_instruction(instruction) {
                    self.update_copy_map(instruction, &mut copy_map);
                } else {
                    self.invalidate_invalid_copies(instruction, &mut copy_map);
                }
            }
        }

        instructions_modified = copy_map.len();
        debug!("Propagated {} copies", instructions_modified);
        Ok(instructions_modified)
    }

    fn collect_live_values(&self, instruction: &LoweredInstruction, live_values: &mut HashSet<u32>) {
        for operand in &instruction.operands {
            if let Operand::Register(reg) = operand {
                live_values.insert(reg.register);
            }
        }

        match instruction.opcode {
            Opcode::CondBr | Opcode::Br => {
                if let Some(Operand::Label(BasicBlockId(block_id))) = instruction.operands.get(1) {
                }
            }
            _ => {}
        }
    }

    fn get_value_id(&self, register: &RegisterLocation) -> u32 {
        register.register
    }

    fn is_instruction_safe_to_remove(&self, instruction: &LoweredInstruction) -> bool {
        match instruction.opcode {
            Opcode::Call | Opcode::Store | Opcode::Ret | Opcode::CondBr | Opcode::Br => false,
            Opcode::Load => false,
            _ => true,
        }
    }

    fn try_fold_constant(
        &self,
        instruction: &LoweredInstruction,
        constant_values: &HashMap<u32, Value>,
    ) -> Option<Value> {
        if instruction.operands.len() < 2 {
            return None;
        }

        let eval_operands: Vec<Option<Value>> = instruction.operands
            .iter()
            .map(|op| {
                match op {
                    Operand::Immediate(val) => Some(val.clone()),
                    Operand::Register(reg) => constant_values.get(&reg.register).cloned(),
                    _ => None,
                }
            })
            .collect();

        if eval_operands.iter().any(|opt| opt.is_none()) {
            return None;
        }

        let values: Vec<Value> = eval_operands.into_iter().flatten().collect();

        match (instruction.opcode, &values[0], &values[1]) {
            (Opcode::Add, Value::I32(a), Value::I32(b)) => Some(Value::I32(a.wrapping_add(*b))),
            (Opcode::Sub, Value::I32(a), Value::I32(b)) => Some(Value::I32(a.wrapping_sub(*b))),
            (Opcode::Mul, Value::I32(a), Value::I32(b)) => Some(Value::I32(a.wrapping_mul(*b))),
            (Opcode::And, Value::I32(a), Value::I32(b)) => Some(Value::I32(a & b)),
            (Opcode::Or, Value::I32(a), Value::I32(b)) => Some(Value::I32(a | b)),
            (Opcode::Xor, Value::I32(a), Value::I32(b)) => Some(Value::I32(a ^ b)),
            
            (Opcode::Eq, Value::I32(a), Value::I32(b)) => Some(Value::Bool(a == b)),
            (Opcode::Ne, Value::I32(a), Value::I32(b)) => Some(Value::Bool(a != b)),
            (Opcode::Lt, Value::I32(a), Value::I32(b)) => Some(Value::Bool(a < b)),
            (Opcode::Le, Value::I32(a), Value::I32(b)) => Some(Value::Bool(a <= b)),
            (Opcode::Gt, Value::I32(a), Value::I32(b)) => Some(Value::Bool(a > b)),
            (Opcode::Ge, Value::I32(a), Value::I32(b)) => Some(Value::Bool(a >= b)),
            
            _ => None,
        }
    }

    fn can_apply_strength_reduction(&self, instruction: &LoweredInstruction) -> bool {
        match instruction.opcode {
            Opcode::Mul | Opcode::Div | Opcode::Rem => {
                instruction.operands.len() == 2 && 
                matches!(&instruction.operands[1], Operand::Immediate(Value::I32(_)))
            }
            _ => false,
        }
    }

    fn apply_strength_reduction_to_instruction(&self, instruction: &mut LoweredInstruction) {
        if let Some(Operand::Immediate(Value::I32(constant))) = instruction.operands.get(1) {
            match (instruction.opcode, *constant) {
                (Opcode::Mul, 2) => {
                    instruction.opcode = Opcode::Shl;
                    instruction.operands[1] = Operand::Immediate(Value::I32(1));
                }
                (Opcode::Mul, 4) => {
                    instruction.opcode = Opcode::Shl;
                    instruction.operands[1] = Operand::Immediate(Value::I32(2));
                }
                (Opcode::Mul, 8) => {
                    instruction.opcode = Opcode::Shl;
                    instruction.operands[1] = Operand::Immediate(Value::I32(3));
                }
                (Opcode::Div, 2) => {
                    instruction.opcode = Opcode::Shr;
                    instruction.operands[1] = Operand::Immediate(Value::I32(1));
                }
                (Opcode::Div, 4) => {
                    instruction.opcode = Opcode::Shr;
                    instruction.operands[1] = Operand::Immediate(Value::I32(2));
                }
                (Opcode::Div, 8) => {
                    instruction.opcode = Opcode::Shr;
                    instruction.operands[1] = Operand::Immediate(Value::I32(3));
                }
                _ => {}
            }
        }
    }

    fn try_algebraic_simplification(&self, instruction: &LoweredInstruction) -> Option<LoweredInstruction> {
        if instruction.operands.len() < 2 {
            return None;
        }

        let left = &instruction.operands[0];
        let right = &instruction.operands[1];

        match (instruction.opcode, left, right) {
            (Opcode::Add, Operand::Immediate(Value::I32(0)), _) => {
                Some(LoweredInstruction {
                    opcode: Opcode::Add,
                    operands: vec![right.clone()],
                    result: instruction.result.clone(),
                    metadata: instruction.metadata.clone(),
                })
            }
            (Opcode::Add, _, Operand::Immediate(Value::I32(0))) => {
                Some(LoweredInstruction {
                    opcode: Opcode::Add,
                    operands: vec![left.clone()],
                    result: instruction.result.clone(),
                    metadata: instruction.metadata.clone(),
                })
            }
            (Opcode::Mul, Operand::Immediate(Value::I32(1)), _) => {
                Some(LoweredInstruction {
                    opcode: Opcode::Add,
                    operands: vec![right.clone()],
                    result: instruction.result.clone(),
                    metadata: instruction.metadata.clone(),
                })
            }
            (Opcode::Mul, _, Operand::Immediate(Value::I32(1))) => {
                Some(LoweredInstruction {
                    opcode: Opcode::Add,
                    operands: vec![left.clone()],
                    result: instruction.result.clone(),
                    metadata: instruction.metadata.clone(),
                })
            }
            (Opcode::Mul, Operand::Immediate(Value::I32(0)), _) => {
                Some(LoweredInstruction {
                    opcode: Opcode::Add,
                    operands: vec![Operand::Immediate(Value::I32(0))],
                    result: instruction.result.clone(),
                    metadata: instruction.metadata.clone(),
                })
            }
            (Opcode::Mul, _, Operand::Immediate(Value::I32(0))) => {
                Some(LoweredInstruction {
                    opcode: Opcode::Add,
                    operands: vec![Operand::Immediate(Value::I32(0))],
                    result: instruction.result.clone(),
                    metadata: instruction.metadata.clone(),
                })
            }
            _ => None,
        }
    }

    fn replace_copies_with_originals(
        &self,
        instruction: &mut LoweredInstruction,
        copy_map: &HashMap<u32, u32>,
    ) {
        for operand in &mut instruction.operands {
            if let Operand::Register(reg) = operand {
                if let Some(&original) = copy_map.get(&reg.register) {
                    reg.register = original;
                }
            }
        }
    }

    fn is_copy_instruction(&self, instruction: &LoweredInstruction) -> bool {
        instruction.opcode == Opcode::Add && 
        instruction.operands.len() == 2 &&
        matches!(&instruction.operands[0], Operand::Register(_)) &&
        matches!(&instruction.operands[1], Operand::Immediate(Value::I32(0)))
    }

    fn update_copy_map(&self, instruction: &LoweredInstruction, copy_map: &mut HashMap<u32, u32>) {
        if let (Some(result), Operand::Register(source)) = (&instruction.result, &instruction.operands[0]) {
            copy_map.insert(result.register, source.register);
        }
    }

    fn invalidate_invalid_copies(&self, instruction: &LoweredInstruction, copy_map: &mut HashMap<u32, u32>) {
        if let Some(result) = &instruction.result {
            copy_map.remove(&result.register);
        }

        for operand in &instruction.operands {
            if let Operand::Register(reg) = operand {
                copy_map.retain(|_, &mut original| original != reg.register);
            }
        }
    }

    pub fn get_optimization_statistics(&self) -> OptimizationStats {
        OptimizationStats::new()
    }
}

#[derive(Debug, Clone)]
struct OptimizationStats {
    pub passes_completed: usize,
    pub instructions_removed: usize,
    pub instructions_modified: usize,
    pub constants_folded: usize,
    pub dead_code_eliminated: usize,
}

impl OptimizationStats {
    fn new() -> Self {
        Self {
            passes_completed: 0,
            instructions_removed: 0,
            instructions_modified: 0,
            constants_folded: 0,
            dead_code_eliminated: 0,
        }
    }

    fn combine(&mut self, other: OptimizationStats) {
        self.instructions_removed += other.instructions_removed;
        self.instructions_modified += other.instructions_modified;
        self.constants_folded += other.constants_folded;
        self.dead_code_eliminated += other.dead_code_eliminated;
    }

    fn made_changes(&self) -> bool {
        self.instructions_removed > 0 || 
        self.instructions_modified > 0 || 
        self.constants_folded > 0 || 
        self.dead_code_eliminated > 0
    }
}

struct PatternMatcher {
    patterns: Vec<OptimizationPattern>,
}

impl PatternMatcher {
    fn new() -> Self {
        Self {
            patterns: vec![
                OptimizationPattern::ConstantFolding,
                OptimizationPattern::StrengthReduction,
                OptimizationPattern::AlgebraicSimplification,
            ],
        }
    }

    fn find_matching_pattern(&self, instruction: &LoweredInstruction) -> Option<&OptimizationPattern> {
        self.patterns.iter().find(|pattern| pattern.matches(instruction))
    }
}

#[derive(Debug, Clone)]
enum OptimizationPattern {
    ConstantFolding,
    StrengthReduction,
    AlgebraicSimplification,
}

impl OptimizationPattern {
    fn matches(&self, instruction: &LoweredInstruction) -> bool {
        match self {
            OptimizationPattern::ConstantFolding => {
                instruction.operands.len() >= 2 &&
                instruction.operands.iter().any(|op| matches!(op, Operand::Immediate(_)))
            }
            OptimizationPattern::StrengthReduction => {
                matches!(instruction.opcode, Opcode::Mul | Opcode::Div) &&
                instruction.operands.len() == 2 &&
                matches!(&instruction.operands[1], Operand::Immediate(Value::I32(n)) if *n > 0 && *n <= 8)
            }
            OptimizationPattern::AlgebraicSimplification => {
                matches!(instruction.opcode, Opcode::Add | Opcode::Mul) &&
                instruction.operands.len() == 2 &&
                (matches!(&instruction.operands[0], Operand::Immediate(Value::I32(0) | Value::I32(1))) ||
                 matches!(&instruction.operands[1], Operand::Immediate(Value::I32(0) | Value::I32(1))))
            }
        }
    }
}

struct ValueNumberer {
    next_value_number: u32,
}

impl ValueNumberer {
    fn new() -> Self {
        Self {
            next_value_number: 0,
        }
    }

    fn assign_value_number(&mut self) -> u32 {
        let number = self.next_value_number;
        self.next_value_number += 1;
        number
    }

    fn reset(&mut self) {
        self.next_value_number = 0;
    }
}