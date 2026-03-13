use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::gpu::monomorphizer::{MonomorphizedInstance, TypeInfo};

pub struct CodegenUnitManager {
    units: Vec<CodegenUnit>,
    config: CGUConfig,
    next_unit_id: usize,
}

#[derive(Debug, Clone)]
pub struct CGUConfig {
    pub max_functions_per_unit: usize,
    pub max_unit_size_bytes: usize,
    pub enable_parallel_compilation: bool,
    pub prioritize_hot_functions: bool,
}

impl Default for CGUConfig {
    fn default() -> Self {
        Self {
            max_functions_per_unit: 1000,
            max_unit_size_bytes: 10 * 1024 * 1024, // 10MB
            enable_parallel_compilation: true,
            prioritize_hot_functions: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CodegenUnit {
    pub id: usize,
    pub name: String,
    pub functions: Vec<CGUFunction>,
    pub dependencies: HashSet<usize>,
    pub size_bytes: usize,
    pub priority: CGUPriority,
    pub compilation_stage: CompilationStage,
}

#[derive(Debug, Clone)]
pub struct CGUFunction {
    pub name: String,
    pub instance: Option<MonomorphizedInstance>,
    pub type_info: TypeInfo,
    pub call_frequency: f64,
    pub size_bytes: usize,
    pub optimization_level: OptimizationLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CGUPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompilationStage {
    Pending,
    Lowering,
    Optimization,
    CodeGeneration,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizationLevel {
    O0,
    O1,
    O2,
    O3,
}

impl CodegenUnitManager {
    pub fn new(config: CGUConfig) -> Self {
        info!("Initializing CodegenUnitManager with config: {:?}", config);

        Self {
            units: Vec::new(),
            config,
            next_unit_id: 0,
        }
    }

    pub fn create_codegen_unit(&mut self, name: String) -> Result<usize> {
        debug!("Creating new CGU: {}", name);

        let unit_id = self.next_unit_id;
        self.next_unit_id += 1;

        let unit = CodegenUnit {
            id: unit_id,
            name,
            functions: Vec::new(),
            dependencies: HashSet::new(),
            size_bytes: 0,
            priority: CGUPriority::Medium,
            compilation_stage: CompilationStage::Pending,
        };

        self.units.push(unit);
        Ok(unit_id)
    }

    pub fn add_function_to_unit(&mut self, unit_id: usize, function: CGUFunction) -> Result<()> {
        debug!("Adding function {} to CGU {}", function.name, unit_id);

        let unit = self
            .units
            .get_mut(unit_id)
            .ok_or_else(|| anyhow!("CGU {} not found", unit_id))?;

        if unit.functions.len() >= self.config.max_functions_per_unit {
            return Err(anyhow!("CGU {} is at capacity", unit_id));
        }

        let new_size = unit.size_bytes + function.size_bytes;
        if new_size > self.config.max_unit_size_bytes {
            return Err(anyhow!("Adding function would exceed CGU size limit"));
        }

        unit.size_bytes = new_size;
        unit.functions.push(function);

        Ok(())
    }

    pub fn add_dependency(&mut self, from_unit: usize, to_unit: usize) -> Result<()> {
        debug!("Adding dependency: {} -> {}", from_unit, to_unit);

        let from = self
            .units
            .get_mut(from_unit)
            .ok_or_else(|| anyhow!("Source CGU {} not found", from_unit))?;

        from.dependencies.insert(to_unit);
        Ok(())
    }

    pub fn optimize_cgu_layout(&mut self) -> Result<()> {
        info!("Optimizing CGU layout for parallel compilation");

        if !self.config.enable_parallel_compilation {
            debug!("Parallel compilation disabled, skipping layout optimization");
            return Ok(());
        }

        self.sort_units_by_priority();
        self.balance_unit_sizes();
        self.minimize_dependencies();

        info!("CGU layout optimization completed");
        Ok(())
    }

    fn sort_units_by_priority(&mut self) {
        debug!("Sorting CGUs by priority and size");

        self.units.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| b.size_bytes.cmp(&a.size_bytes))
        });
    }

    fn balance_unit_sizes(&mut self) {
        debug!("Balancing CGU sizes for optimal parallelism");

        if self.units.len() < 2 {
            return;
        }

        let total_size: usize = self.units.iter().map(|u| u.size_bytes).sum();
        let target_size = total_size / self.units.len();

        for i in 0..self.units.len().saturating_sub(1) {
            let (current_size, current_func_count) = {
                let current = &self.units[i];
                (current.size_bytes, current.functions.len())
            };

            if current_size > target_size * 2 && current_func_count > 10 {
                if let Some((func_idx, func)) = self.units[i]
                    .functions
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| f.size_bytes < target_size / 4)
                    .min_by_key(|(_, f)| f.size_bytes)
                {
                    let func_size = func.size_bytes;
                    let removed_func = self.units[i].functions.remove(func_idx);
                    self.units[i + 1].functions.push(removed_func);

                    self.units[i].size_bytes -= func_size;
                    self.units[i + 1].size_bytes += func_size;
                }
            }
        }
    }

    fn minimize_dependencies(&mut self) {
        debug!("Minimizing cross-CGU dependencies");

        let mut dependency_chains: HashMap<usize, Vec<usize>> = HashMap::new();

        for unit in &self.units {
            dependency_chains.insert(unit.id, Vec::new());
        }

        for (unit_idx, unit) in self.units.iter().enumerate() {
            for &dep in &unit.dependencies {
                dependency_chains.entry(unit_idx).or_default().push(dep);
            }
        }

        for unit in &mut self.units {
            let mut removable_deps = Vec::new();

            for &dep in &unit.dependencies {
                if !unit.functions.iter().any(|f| {
                    if let Some(instance) = &f.instance {
                        instance
                            .dependency_graph
                            .iter()
                            .any(|d| d.contains(&format!("cgu_{}", dep)))
                    } else {
                        false
                    }
                }) {
                    removable_deps.push(dep);
                }
            }

            for dep in removable_deps {
                unit.dependencies.remove(&dep);
                debug!("Removed unnecessary dependency {} -> {}", unit.id, dep);
            }
        }
    }

    pub fn get_ready_units(&self) -> Vec<&CodegenUnit> {
        debug!("Finding ready CGUs for compilation");

        let mut ready_units = Vec::new();

        for unit in &self.units {
            if unit.compilation_stage == CompilationStage::Pending {
                let deps_completed = unit.dependencies.iter().all(|&dep_id| {
                    self.units
                        .get(dep_id)
                        .map(|dep_unit| dep_unit.compilation_stage == CompilationStage::Completed)
                        .unwrap_or(false)
                });

                if deps_completed {
                    ready_units.push(unit);
                }
            }
        }

        ready_units.sort_by(|a, b| b.priority.cmp(&a.priority));
        ready_units
    }

    pub fn update_unit_stage(&mut self, unit_id: usize, stage: CompilationStage) -> Result<()> {
        debug!("Updating CGU {} to stage: {:?}", unit_id, stage);

        let unit = self
            .units
            .get_mut(unit_id)
            .ok_or_else(|| anyhow!("CGU {} not found", unit_id))?;

        unit.compilation_stage = stage;
        Ok(())
    }

    pub fn get_compilation_statistics(&self) -> CompilationStats {
        let total_units = self.units.len();
        let completed_units = self
            .units
            .iter()
            .filter(|u| u.compilation_stage == CompilationStage::Completed)
            .count();

        let total_functions: usize = self.units.iter().map(|u| u.functions.len()).sum();

        let total_bytes: usize = self.units.iter().map(|u| u.size_bytes).sum();

        let avg_functions_per_unit = if total_units > 0 {
            total_functions as f64 / total_units as f64
        } else {
            0.0
        };

        CompilationStats {
            total_units,
            completed_units,
            total_functions,
            total_bytes,
            avg_functions_per_unit,
        }
    }

    pub fn get_parallel_compilation_plan(&self) -> Vec<CompilationWave> {
        debug!("Creating parallel compilation plan");

        let mut waves = Vec::new();
        let mut completed_units = HashSet::new();

        while completed_units.len() < self.units.len() {
            let mut current_wave = Vec::new();

            for unit in &self.units {
                if !completed_units.contains(&unit.id)
                    && unit.compilation_stage == CompilationStage::Pending
                {
                    let deps_completed = unit
                        .dependencies
                        .iter()
                        .all(|&dep| completed_units.contains(&dep));

                    if deps_completed {
                        current_wave.push(unit.id);
                    }
                }
            }

            if current_wave.is_empty() {
                warn!("No more units can be compiled, but compilation is not complete");
                break;
            }

            waves.push(CompilationWave {
                wave_number: waves.len(),
                unit_ids: current_wave.clone(),
                can_run_in_parallel: true,
            });

            for unit_id in &current_wave {
                completed_units.insert(*unit_id);
            }
        }

        info!("Created compilation plan with {} waves", waves.len());
        waves
    }

    pub fn merge_small_units(&mut self, min_functions: usize) -> Result<()> {
        debug!("Merging small CGUs (< {} functions)", min_functions);

        let mut small_unit_indices: Vec<usize> = self
            .units
            .iter()
            .enumerate()
            .filter(|(_, u)| u.functions.len() < min_functions)
            .map(|(i, _)| i)
            .collect();

        small_unit_indices.sort_by_key(|&i| self.units[i].functions.len());
        small_unit_indices.reverse();

        while let Some(small_idx) = small_unit_indices.pop() {
            if small_idx >= self.units.len() {
                continue;
            }

            let small_unit_info = {
                let unit = &self.units[small_idx];
                (
                    unit.functions.len(),
                    unit.size_bytes,
                    unit.dependencies.clone(),
                )
            };

            if small_unit_info.0 >= min_functions {
                continue;
            }

            let mut best_merge_candidate = None;
            let mut best_score = -1.0;

            for (other_idx, other_unit) in self.units.iter().enumerate() {
                if other_idx == small_idx {
                    continue;
                }

                if other_unit.functions.len() + small_unit_info.0
                    > self.config.max_functions_per_unit
                {
                    continue;
                }

                if other_unit.size_bytes + small_unit_info.1 > self.config.max_unit_size_bytes {
                    continue;
                }

                let shared_deps = small_unit_info
                    .2
                    .intersection(&other_unit.dependencies)
                    .count();
                let score = shared_deps as f64 - (other_unit.functions.len() as f64 * 0.1);

                if score > best_score {
                    best_score = score;
                    best_merge_candidate = Some(other_idx);
                }
            }

            if let Some(merge_idx) = best_merge_candidate {
                debug!("Merging CGU {} into {}", small_idx, merge_idx);

                let small_size = self.units[small_idx].size_bytes;
                let small_functions = self.units[small_idx]
                    .functions
                    .drain(..)
                    .collect::<Vec<_>>();
                let small_deps: HashSet<_> = self.units[small_idx].dependencies.drain().collect();

                let target_unit = &mut self.units[merge_idx];
                target_unit.functions.extend(small_functions);
                target_unit.dependencies.extend(small_deps);
                target_unit.size_bytes += small_size;

                self.units.swap_remove(small_idx);

                for idx in &mut small_unit_indices {
                    if *idx > small_idx {
                        *idx -= 1;
                    }
                    if *idx == merge_idx {
                        *idx -= 1;
                    }
                }
            }
        }

        info!(
            "CGU merging completed, {} units remaining",
            self.units.len()
        );
        Ok(())
    }

    pub fn get_units(&self) -> &[CodegenUnit] {
        &self.units
    }

    pub fn get_unit_mut(&mut self, unit_id: usize) -> Option<&mut CodegenUnit> {
        self.units.get_mut(unit_id)
    }
}

#[derive(Debug, Clone)]
pub struct CompilationWave {
    pub wave_number: usize,
    pub unit_ids: Vec<usize>,
    pub can_run_in_parallel: bool,
}

#[derive(Debug)]
pub struct CompilationStats {
    pub total_units: usize,
    pub completed_units: usize,
    pub total_functions: usize,
    pub total_bytes: usize,
    pub avg_functions_per_unit: f64,
}
