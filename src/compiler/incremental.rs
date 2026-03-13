//! Incremental compilation and persistent caching system
//! 
//! This module provides intelligent caching to avoid redundant compilation work.

use anyhow::{anyhow, Result};
use serde::{Serialize, Deserialize};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
    fs,
    hash::{Hash, Hasher},
    collections::hash_map::DefaultHasher,
};
use tracing::{debug, info, warn};

use crate::gpu::mir_processor::{ProcessedFunction, ProcessedCrate};

pub struct IncrementalCompiler {
    cache_dir: PathBuf,
    cache: Arc<RwLock<CompilationCache>>,
    config: CacheConfig,
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub enable_caching: bool,
    pub max_cache_size_mb: usize,
    pub cache_ttl_seconds: u64,
    pub enable_fingerprints: bool,
    pub track_dependencies: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cache_size_mb: 1024, // 1GB
            cache_ttl_seconds: 3600 * 24 * 7, // 1 week
            enable_fingerprints: true,
            track_dependencies: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationCache {
    pub entries: HashMap<String, CacheEntry>,
    pub dependency_graph: HashMap<String, Vec<String>>,
    pub global_fingerprints: HashMap<String, String>,
    pub statistics: CacheStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub function_data: CachedFunctionData,
    pub timestamp: u64,
    pub fingerprint: String,
    pub dependencies: Vec<String>,
    pub hit_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedFunctionData {
    pub name: String,
    pub machine_code: Vec<u8>,
    pub size_bytes: usize,
    pub compilation_time_ms: u64,
    pub gpu_accelerated: bool,
    pub optimization_level: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheStatistics {
    pub total_hits: u64,
    pub total_misses: u64,
    pub total_compilations: u64,
    pub gpu_accelerated_compilations: u64,
    pub cache_size_bytes: u64,
    pub last_cleanup: u64,
}

#[derive(Debug, Clone)]
pub struct IncrementalResult {
    pub cached_functions: Vec<CachedFunctionData>,
    pub functions_to_compile: Vec<ProcessedFunction>,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl IncrementalCompiler {
    pub fn new(config: CacheConfig) -> Result<Self> {
        let cache_dir = std::env::temp_dir().join("cargpu_cache");
        fs::create_dir_all(&cache_dir)?;
        
        let cache_file = cache_dir.join("cache.json");
        let cache = if cache_file.exists() {
            Self::load_cache(&cache_file)?
        } else {
            CompilationCache::default()
        };
        
        info!("Initialized incremental compiler with cache at {:?}", cache_dir);
        
        Ok(Self {
            cache_dir,
            cache: Arc::new(RwLock::new(cache)),
            config,
        })
    }

    /// Process a crate incrementally, using cached results where possible
    pub fn process_incrementally(&mut self, crate_data: &ProcessedCrate) -> Result<IncrementalResult> {
        info!("Processing {} functions incrementally", crate_data.functions.len());
        
        let mut cached_functions = Vec::new();
        let mut functions_to_compile = Vec::new();
        let mut cache_hits = 0;
        let mut cache_misses = 0;
        
        // Process each function
        for function in &crate_data.functions {
            let cache_key = self.generate_cache_key(function);
            
            if let Some(cached_data) = self.lookup_cache(&cache_key, function)? {
                debug!("Cache hit for function: {}", function.name);
                cached_functions.push(cached_data);
                cache_hits += 1;
            } else {
                debug!("Cache miss for function: {}", function.name);
                functions_to_compile.push(function.clone());
                cache_misses += 1;
            }
        }
        
        // Update statistics
        {
            let mut cache = self.cache.write().unwrap();
            cache.statistics.total_hits += cache_hits as u64;
            cache.statistics.total_misses += cache_misses as u64;
            cache.statistics.total_compilations += crate_data.functions.len() as u64;
        }
        
        info!("Incremental processing: {} hits, {} misses", cache_hits, cache_misses);
        
        Ok(IncrementalResult {
            cached_functions,
            functions_to_compile,
            cache_hits,
            cache_misses,
        })
    }

    /// Store compiled function data in cache
    pub fn store_compilation_result(
        &mut self,
        function: &ProcessedFunction,
        compiled_data: &[u8],
        compilation_time_ms: u64,
        gpu_accelerated: bool,
    ) -> Result<()> {
        let cache_key = self.generate_cache_key(function);
        let fingerprint = self.compute_function_fingerprint(function)?;
        
        let entry = CacheEntry {
            key: cache_key.clone(),
            function_data: CachedFunctionData {
                name: function.name.clone(),
                machine_code: compiled_data.to_vec(),
                size_bytes: compiled_data.len(),
                compilation_time_ms,
                gpu_accelerated,
                optimization_level: 2, // Default optimization level
            },
            timestamp: current_timestamp(),
            fingerprint,
            dependencies: self.extract_function_dependencies(function),
            hit_count: 0,
        };
        
        {
            let mut cache = self.cache.write().unwrap();
            cache.entries.insert(cache_key, entry);
            
            if gpu_accelerated {
                cache.statistics.gpu_accelerated_compilations += 1;
            }
        }
        
        // Save cache to disk
        self.save_cache()?;
        
        Ok(())
    }

    /// Generate cache key for a function
    fn generate_cache_key(&self, function: &ProcessedFunction) -> String {
        let mut hasher = DefaultHasher::new();
        
        // Include function name and MIR data
        function.name.hash(&mut hasher);
        function.mir_data.hash(&mut hasher);
        function.complexity.hash(&mut hasher);
        function.basic_blocks.hash(&mut hasher);
        
        // Include compilation configuration
        2u32.hash(&mut hasher); // optimization level
        true.hash(&mut hasher); // GPU enabled
        
        format!("fn_{}", hasher.finish())
    }

    /// Compute fingerprint of function source for dependency tracking
    fn compute_function_fingerprint(&self, function: &ProcessedFunction) -> Result<String> {
        let mut hasher = DefaultHasher::new();
        
        // Hash the MIR data comprehensively
        function.mir_data.hash(&mut hasher);
        function.complexity.hash(&mut hasher);
        
        Ok(format!("fp_{}", hasher.finish()))
    }

    /// Lookup cached function data
    fn lookup_cache(&self, cache_key: &str, function: &ProcessedFunction) -> Result<Option<CachedFunctionData>> {
        if !self.config.enable_caching {
            return Ok(None);
        }
        
        let cache = self.cache.read().unwrap();
        
        if let Some(entry) = cache.entries.get(cache_key) {
            // Check if entry is still valid
            let current_time = current_timestamp();
            let is_expired = current_time - entry.timestamp > self.config.cache_ttl_seconds;
            
            // Check if function has changed (fingerprint comparison)
            let current_fingerprint = self.compute_function_fingerprint(function)?;
            let fingerprint_match = current_fingerprint == entry.fingerprint;
            
            if !is_expired && fingerprint_match {
                // Update hit count
                let function_data = entry.function_data.clone();
                drop(cache);
                let mut cache = self.cache.write().unwrap();
                if let Some(entry) = cache.entries.get_mut(cache_key) {
                    entry.hit_count += 1;
                }
                
                return Ok(Some(function_data));
            } else {
                debug!("Cache entry expired or invalid for: {}", function.name);
            }
        }
        
        Ok(None)
    }

    /// Extract function dependencies for invalidation
    fn extract_function_dependencies(&self, function: &ProcessedFunction) -> Vec<String> {
        // Simplified dependency extraction
        // In practice, this would parse the MIR to find called functions
        let mut dependencies = Vec::new();
        
        // Extract function calls from name patterns
        if function.name.contains("::") {
            let parts: Vec<&str> = function.name.split("::").collect();
            for i in 0..parts.len().saturating_sub(1) {
                dependencies.push(format!("{}::", parts[..=i].join("::")));
            }
        }
        
        dependencies
    }

    /// Invalidate cache entries affected by file changes
    pub fn invalidate_on_file_change(&mut self, changed_files: &[PathBuf]) -> Result<usize> {
        info!("Invalidating cache due to {} changed files", changed_files.len());
        
        let mut invalidated = 0;
        
        {
            let mut cache = self.cache.write().unwrap();
            
            // For now, invalidate all entries on any file change
            // In practice, would use fine-grained dependency tracking
            for key in cache.entries.keys().cloned().collect::<Vec<_>>() {
                cache.entries.remove(&key);
                invalidated += 1;
            }
        }
        
        if invalidated > 0 {
            self.save_cache()?;
        }
        
        Ok(invalidated)
    }

    /// Clean up expired cache entries
    pub fn cleanup_expired_entries(&mut self) -> Result<usize> {
        info!("Cleaning up expired cache entries");
        
        let current_time = current_timestamp();
        let mut removed = 0;
        
        {
            let mut cache = self.cache.write().unwrap();
            
            let mut to_remove = Vec::new();
            for (key, entry) in &cache.entries {
                if current_time - entry.timestamp > self.config.cache_ttl_seconds {
                    to_remove.push(key.clone());
                }
            }
            
            for key in to_remove {
                cache.entries.remove(&key);
                removed += 1;
            }
            
            cache.statistics.last_cleanup = current_time;
        }
        
        if removed > 0 {
            self.save_cache()?;
        }
        
        info!("Removed {} expired cache entries", removed);
        Ok(removed)
    }

    /// Get cache statistics
    pub fn get_statistics(&self) -> CacheStatistics {
        let cache = self.cache.read().unwrap();
        cache.statistics.clone()
    }

    /// Estimate cache size in bytes
    pub fn estimate_cache_size(&self) -> Result<u64> {
        let cache = self.cache.read().unwrap();
        let mut total_size = 0u64;
        
        for entry in cache.entries.values() {
            total_size += entry.function_data.machine_code.len() as u64;
            total_size += 1024; // Estimate overhead per entry
        }
        
        Ok(total_size)
    }

    /// Load cache from disk
    fn load_cache(cache_file: &Path) -> Result<CompilationCache> {
        let data = fs::read_to_string(cache_file)?;
        let cache: CompilationCache = serde_json::from_str(&data)
            .map_err(|e| anyhow!("Failed to load cache: {}", e))?;
        Ok(cache)
    }

    /// Save cache to disk
    fn save_cache(&self) -> Result<()> {
        let cache_file = self.cache_dir.join("cache.json");
        let cache = self.cache.read().unwrap();
        
        // Extract data for serialization
        let cache_data = serde_json::to_value(&*cache)?;
        let data = serde_json::to_string_pretty(&cache_data)?;
        fs::write(cache_file, data)?;
        
        Ok(())
    }

    /// Preheat cache with commonly used functions
    pub async fn preheat_cache(&mut self, common_functions: &[String]) -> Result<usize> {
        info!("Preheating cache with {} common functions", common_functions.len());
        
        // This would normally compile common functions proactively
        // For now, just report that preheating occurred
        Ok(common_functions.len())
    }
}

impl Default for CompilationCache {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            dependency_graph: HashMap::new(),
            global_fingerprints: HashMap::new(),
            statistics: CacheStatistics::default(),
        }
    }
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Background task for periodic cache maintenance
pub async fn cache_maintenance_task(
    incremental_compiler: Arc<RwLock<IncrementalCompiler>>,
    interval_seconds: u64,
) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_seconds));
    
    loop {
        interval.tick().await;
        
        if let Ok(mut compiler) = incremental_compiler.write() {
            if let Ok(removed) = compiler.cleanup_expired_entries() {
                if removed > 0 {
                    info!("Background maintenance removed {} cache entries", removed);
                }
            }
            
            // Check cache size and trim if necessary
            if let Ok(cache_size) = compiler.estimate_cache_size() {
                let max_size_bytes = (compiler.config.max_cache_size_mb * 1024 * 1024) as u64;
                if cache_size > max_size_bytes {
                    warn!("Cache size {} exceeds limit {}, cleanup needed", 
                          cache_size, max_size_bytes);
                    // TODO: Implement LRU-based cache trimming
                }
            }
        }
    }
}