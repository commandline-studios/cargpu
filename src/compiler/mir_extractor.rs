//! Real MIR extraction using rustc internals
//!
//! This module provides actual MIR extraction from rustc instead of text-based parsing.

use anyhow::{anyhow, Result};
use rustc_data_structures::fx::FxHashMap;
use rustc_interface::interface;
use rustc_middle::{
    mir::{
        self, BasicBlock, Body, Location, Operand, Place, ProjectionElem, Rvalue, Statement,
        StatementKind, Terminator, TerminatorKind,
    },
    thir,
    ty::{EarlyBinder, ParamEnv, Ty, TyCtxt},
};
use rustc_session::config::{self, Input, OutputType};
use rustc_span::{
    def_id::{DefId, LocalDefId},
    symbol::Symbol,
    Span,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use crate::gpu::mir_processor::{MirFunction, ProcessedCrate, ProcessedFunction};

pub struct RealMirExtractor {
    tcx: Option<TyCtxt<'_>>,
    functions: Vec<ProcessedFunction>,
    mir_data: FxHashMap<DefId, mir::Body<'_>>,
}

impl RealMirExtractor {
    pub fn new() -> Self {
        Self {
            tcx: None,
            functions: Vec::new(),
            mir_data: FxHashMap::default(),
        }
    }

    /// Extract MIR from a Rust crate using real rustc APIs
    pub fn extract_crate_mir(&mut self, crate_root: &Path) -> Result<ProcessedCrate> {
        info!("Extracting real MIR from crate at {:?}", crate_root);

        // Configure rustc to compile the crate and give us access to its MIR
        let config = self.create_rustc_config(crate_root)?;

        // Run rustc with our custom callback
        interface::run_compiler(config, &mut self)?;

        // Process the extracted MIR data
        let processed_crate = self.process_extracted_mir()?;

        Ok(processed_crate)
    }

    /// Create rustc configuration for MIR extraction
    fn create_rustc_config(&self, crate_root: &Path) -> interface::Config {
        let args = vec![
            "rustc".to_string(),
            crate_root.to_string_lossy().to_string(),
            "--emit=mir".to_string(),
            "--crate-type=lib".to_string(),
        ];

        interface::Config {
            args: args,
            // Use default file loader and input
            input: Input::File(crate_root.to_path_buf()),
            input_path: Some(crate_root.to_path_buf()),
            output_dir: None,
            output_file: None,
            file_loader: None,
            locale_resources: rustc_driver::DEFAULT_LOCALE_RESOURCES.to_vec(),
            lint_caps: rustc_session::lint::LintCaps::default(),
            // We want to parse but not generate code
            parse_only: false,
            no_input: false,
            // Don't actually generate code
            dont_output_codegen_items: true,
            // Capture our callbacks for MIR extraction
            make_crate_callback: Box::new(|_| Ok(())),
            make_codegen_backend: None,
            override_queries: Some(Arc::new(move |_, _, _| {})),
            // Use default registry
            registry: rustc_driver::cli::registry(),
        }
    }

    /// Process the extracted MIR data into our internal format
    fn process_extracted_mir(&mut self) -> Result<ProcessedCrate> {
        info!("Processing {} MIR bodies", self.mir_data.len());

        let mut functions = Vec::new();
        let mut dependencies = Vec::new();

        for (&def_id, mir_body) in &self.mir_data {
            let processed_fn = self.process_mir_function(def_id, mir_body)?;
            functions.push(processed_fn);

            // Extract function dependencies
            let fn_deps = self.extract_function_dependencies(mir_body);
            dependencies.extend(fn_deps);
        }

        // Create generic functions for monomorphization
        let generic_functions = self.create_generic_functions(&functions)?;

        Ok(ProcessedCrate {
            functions,
            generic_functions,
            monomorphized_instances: Vec::new(), // Will be filled later
            dependencies,
        })
    }

    /// Process a single MIR function into our internal format
    fn process_mir_function(
        &self,
        def_id: DefId,
        mir_body: &mir::Body<'_>,
    ) -> Result<ProcessedFunction> {
        let name = self.get_function_name(def_id)?;

        // Analyze basic blocks
        let basic_blocks = mir_body.basic_blocks.len();
        let statements = mir_body
            .basic_blocks
            .iter()
            .map(|bb| bb.statements.len())
            .sum();

        // Determine if this function is suitable for GPU acceleration
        let is_gpu_suitable = self.analyze_gpu_suitability(mir_body);

        // Extract function complexity metrics
        let complexity = self.analyze_function_complexity(mir_body);

        Ok(ProcessedFunction {
            name,
            def_id: def_id.index.as_u32(),
            is_gpu_suitable,
            basic_blocks: basic_blocks as u32,
            statements: statements as u32,
            complexity,
            // Simplified MIR data for GPU processing
            mir_data: self.serialize_mir_for_gpu(mir_body)?,
        })
    }

    /// Analyze if a function is suitable for GPU execution
    fn analyze_gpu_suitability(&self, mir_body: &mir::Body<'_>) -> bool {
        let mut arithmetic_ops = 0;
        let mut memory_ops = 0;
        let mut control_flow = 0;

        for bb in &mir_body.basic_blocks {
            for stmt in &bb.statements {
                match &stmt.kind {
                    StatementKind::Assign(box (_, Rvalue::BinaryOp(..))) => arithmetic_ops += 1,
                    StatementKind::Assign(box (_, Rvalue::UnaryOp(..))) => arithmetic_ops += 1,
                    StatementKind::Assign(box (_, Rvalue::Len(..)))
                    | StatementKind::Assign(box (_, Rvalue::Ref(..)))
                    | StatementKind::Assign(box (_, Rvalue::AddressOf(..))) => memory_ops += 1,
                    StatementKind::Nop => {}
                    _ => {}
                }
            }

            if let Some(term) = &bb.terminator {
                match &term.kind {
                    TerminatorKind::SwitchInt { .. } => control_flow += 1,
                    TerminatorKind::Call { .. } => control_flow += 1,
                    TerminatorKind::Goto { .. }
                    | TerminatorKind::Return
                    | TerminatorKind::Unreachable => {}
                    _ => control_flow += 1,
                }
            }
        }

        // GPU-suitable if it has arithmetic and memory operations but not too complex control flow
        (arithmetic_ops > 0 || memory_ops > 0) && control_flow < 10
    }

    /// Analyze function complexity for scheduling
    fn analyze_function_complexity(&self, mir_body: &mir::Body<'_>) -> u32 {
        let mut complexity = 0u32;

        for bb in &mir_body.basic_blocks {
            complexity += bb.statements.len() as u32;
            if bb.terminator.is_some() {
                complexity += 1;
            }
        }

        // Add complexity based on variable count
        complexity += mir_body.local_decls.len() as u32;

        complexity
    }

    /// Extract function call dependencies
    fn extract_function_dependencies(&self, mir_body: &mir::Body<'_>) -> Vec<(u32, u32)> {
        let mut deps = Vec::new();

        for bb in &mir_body.basic_blocks {
            for stmt in &bb.statements {
                if let StatementKind::Assign(box (_, Rvalue::Call(func, _))) = &stmt.kind {
                    if let Some(callee_def_id) = self.extract_callee_def_id(func) {
                        deps.push((
                            mir_body.source.def_id().index.as_u32(),
                            callee_def_id.as_u32(),
                        ));
                    }
                }
            }

            if let Some(term) = &bb.terminator {
                if let TerminatorKind::Call { func, .. } = &term.kind {
                    if let Some(callee_def_id) = self.extract_callee_def_id(func) {
                        deps.push((
                            mir_body.source.def_id().index.as_u32(),
                            callee_def_id.as_u32(),
                        ));
                    }
                }
            }
        }

        deps
    }

    /// Extract callee DefId from function operand
    fn extract_callee_def_id(&self, func: &Operand<'_>) -> Option<u32> {
        match func {
            Operand::Constant(box constant) => {
                // Try to extract function from constant
                if let mir::ConstantKind::Val(_, ty) = constant.literal {
                    if let ty::FnDef(def_id, _) = ty.kind() {
                        return Some(def_id.index.as_u32());
                    }
                }
            }
            Operand::Copy(_) | Operand::Move(_) => {
                // Function through place - would need more complex analysis
            }
        }
        None
    }

    /// Get function name from DefId
    fn get_function_name(&self, def_id: DefId) -> Result<String> {
        let tcx = self.tcx.ok_or_else(|| anyhow!("TyCtxt not initialized"))?;

        let name = tcx.item_name(def_id).to_string();
        let parent = tcx.parent_module_from_def_id(def_id);

        Ok(format!("{}::{}", tcx.item_name(parent.to_def_id()), name))
    }

    /// Serialize MIR for GPU processing
    fn serialize_mir_for_gpu(&self, mir_body: &mir::Body<'_>) -> Result<Vec<u8>> {
        let mut serialized = Vec::new();

        // Basic block count
        serialized.extend_from_slice(&(mir_body.basic_blocks.len() as u32).to_le_bytes());

        // Local declarations count
        serialized.extend_from_slice(&(mir_body.local_decls.len() as u32).to_le_bytes());

        // Serialize each basic block
        for (bb_idx, bb) in mir_body.basic_blocks.iter().enumerate() {
            // Statement count for this block
            serialized.extend_from_slice(&(bb.statements.len() as u32).to_le_bytes());

            // Serialize statements
            for stmt in &bb.statements {
                let stmt_code = self.encode_statement(stmt);
                serialized.extend_from_slice(&stmt_code.to_le_bytes());
            }

            // Serialize terminator
            if let Some(term) = &bb.terminator {
                let term_code = self.encode_terminator(term);
                serialized.extend_from_slice(&term_code.to_le_bytes());
            } else {
                serialized.extend_from_slice(&0u32.to_le_bytes());
            }
        }

        Ok(serialized)
    }

    /// Encode statement as compact code for GPU
    fn encode_statement(&self, stmt: &Statement<'_>) -> u32 {
        match &stmt.kind {
            StatementKind::Assign(box (_, Rvalue::BinaryOp(op, _))) => 1 + (op.to_u32() << 8),
            StatementKind::Assign(box (_, Rvalue::UnaryOp(op, _))) => 2 + (op.to_u32() << 8),
            StatementKind::Assign(box (_, Rvalue::Use(_))) => 3,
            StatementKind::Assign(box (_, Rvalue::Ref(_, _, _))) => 4,
            StatementKind::Assign(box (_, Rvalue::Len(_))) => 5,
            StatementKind::StorageLive(_) => 6,
            StatementKind::StorageDead(_) => 7,
            StatementKind::Nop => 8,
            _ => 99, // Unknown/other
        }
    }

    /// Encode terminator as compact code for GPU
    fn encode_terminator(&self, term: &Terminator<'_>) -> u32 {
        match &term.kind {
            TerminatorKind::Return => 1,
            TerminatorKind::Unreachable => 2,
            TerminatorKind::Goto { .. } => 3,
            TerminatorKind::SwitchInt { .. } => 4,
            TerminatorKind::Call { .. } => 5,
            TerminatorKind::Assert { .. } => 6,
            TerminatorKind::Yield { .. } => 7,
            TerminatorKind::Drop { .. } => 8,
            TerminatorKind::DropAndReplace { .. } => 9,
            TerminatorKind::FalseEdge { .. } => 10,
            TerminatorKind::FalseUnwind { .. } => 11,
            TerminatorKind::GeneratorDrop => 12,
            TerminatorKind::InlineAsm { .. } => 13,
        }
    }

    /// Create generic function representations for monomorphization
    fn create_generic_functions(
        &self,
        functions: &[ProcessedFunction],
    ) -> Result<Vec<MirFunction>> {
        let mut generic_functions = Vec::new();

        // Identify generic functions based on name patterns
        for func in functions {
            if func.name.contains('<') || func.name.contains("::") {
                // This looks like a generic function
                generic_functions.push(MirFunction {
                    name: func.name.clone(),
                    is_generic: true,
                    type_params: self.extract_type_parameters(&func.name)?,
                    instance_count: 0, // Will be calculated during monomorphization
                });
            }
        }

        Ok(generic_functions)
    }

    /// Extract type parameters from function name (simplified)
    fn extract_type_parameters(&self, name: &str) -> Result<Vec<String>> {
        let mut params = Vec::new();

        if let Some(start) = name.find('<') {
            if let Some(end) = name.rfind('>') {
                let param_str = &name[start + 1..end];
                for param in param_str.split(',') {
                    params.push(param.trim().to_string());
                }
            }
        }

        Ok(params)
    }
}

/// Helper trait for binary operations
trait BinOpExt {
    fn to_u32(&self) -> u32;
}

impl BinOpExt for mir::BinOp {
    fn to_u32(&self) -> u32 {
        match self {
            mir::BinOp::Add => 1,
            mir::BinOp::Sub => 2,
            mir::BinOp::Mul => 3,
            mir::BinOp::Div => 4,
            mir::BinOp::Rem => 5,
            mir::BinOp::BitXor => 6,
            mir::BinOp::BitAnd => 7,
            mir::BinOp::BitOr => 8,
            mir::BinOp::Shl => 9,
            mir::BinOp::Shr => 10,
            mir::BinOp::Eq => 11,
            mir::BinOp::Lt => 12,
            mir::BinOp::Le => 13,
            mir::BinOp::Ne => 14,
            mir::BinOp::Ge => 15,
            mir::BinOp::Gt => 16,
            mir::BinOp::Offset => 17,
            mir::BinOp::AddUnchecked => 18,
            mir::BinOp::SubUnchecked => 19,
            mir::BinOp::MulUnchecked => 20,
            mir::BinOp::ShlUnchecked => 21,
            mir::BinOp::ShrUnchecked => 22,
        }
    }
}

/// Helper trait for unary operations
trait UnOpExt {
    fn to_u32(&self) -> u32;
}

impl UnOpExt for mir::UnOp {
    fn to_u32(&self) -> u32 {
        match self {
            mir::UnOp::Not => 1,
            mir::UnOp::Neg => 2,
        }
    }
}

impl rustc_driver::Callbacks for RealMirExtractor {
    fn config(&mut self, config: &mut interface::Config) {
        // Configure for MIR extraction
        config.opts.output_types = vec![OutputType::Mir];
        config.opts.debug_assertions = true;
        config.opts.trim_diagnostic_paths = false;
    }

    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &interface::Compiler,
        queries: &'tcx rustc_interface::Queries<'tcx>,
    ) -> interface::Compilation {
        // Extract the TyCtxt
        queries.global_ctxt().unwrap().enter(|tcx| {
            self.tcx = Some(tcx);

            // Collect all MIR bodies
            let mut mir_bodies = FxHashMap::default();

            // Iterate over all items in the crate
            for def_id in tcx.hir().body_owners() {
                let def_id = def_id.to_def_id();

                // Get the optimized MIR for this function
                if let Some(mir_body) = tcx.optimized_mir(def_id) {
                    mir_bodies.insert(def_id, mir_body.clone());
                }
            }

            self.mir_data = mir_bodies;
        });

        interface::Compilation::Stop
    }
}
