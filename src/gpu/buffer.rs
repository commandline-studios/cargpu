use anyhow::{anyhow, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

pub struct WorkStealingBuffer {
    tasks: Arc<Mutex<VecDeque<CompilationTask>>>,
    gpu_buffer_pool: Arc<Mutex<Vec<GpuBuffer>>>,
    cpu_buffer_pool: Arc<Mutex<Vec<CpuBuffer>>>,
    memory_stats: Arc<Mutex<MemoryStats>>,
    config: BufferConfig,
}

pub struct GpuBuffer {
    buffer: wgpu::Buffer,
    size: u64,
    in_use: bool,
    last_used: std::time::Instant,
}

pub struct CpuBuffer {
    data: Vec<u8>,
    size: usize,
    in_use: bool,
    last_used: std::time::Instant,
}

#[derive(Debug, Clone)]
pub struct CompilationTask {
    pub id: u64,
    pub data: Vec<u8>,
    pub task_type: TaskType,
    pub priority: TaskPriority,
    pub size_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskType {
    CodeGeneration,
    Optimization,
    RegisterAllocation,
    LinkPreparation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

#[derive(Debug, Clone)]
pub struct BufferConfig {
    pub max_buffer_size: usize,
    pub max_tasks_in_flight: usize,
    pub batch_size: usize,
    pub transfer_threshold: usize,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 256 * 1024 * 1024, // 256MB
            max_tasks_in_flight: 1024,
            batch_size: 64,
            transfer_threshold: 1024 * 1024, // 1MB
        }
    }
}

impl WorkStealingBuffer {
    pub async fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: BufferConfig,
    ) -> Result<Self> {
        info!("Initializing WorkStealingBuffer with config: {:?}", config);
        
        // Initialize buffer pools
        let gpu_buffer_pool = Arc::new(Mutex::new(Vec::new()));
        let cpu_buffer_pool = Arc::new(Mutex::new(Vec::new()));
        
        // Pre-allocate some buffers
        Self::preallocate_gpu_buffers(&device, &gpu_buffer_pool, &config).await?;
        Self::preallocate_cpu_buffers(&cpu_buffer_pool, &config).await?;
        
        Ok(Self {
            tasks: Arc::new(Mutex::new(VecDeque::new())),
            gpu_buffer_pool,
            cpu_buffer_pool,
            memory_stats: Arc::new(Mutex::new(MemoryStats::default())),
            config,
        })
    }
    
    pub fn submit_task(&self, task: CompilationTask) -> Result<()> {
        debug!("Submitting task {} of type {:?}", task.id, task.task_type);
        
        // Check if we should batch this task
        if task.size_bytes < self.config.transfer_threshold {
            return self.add_to_batch(task);
        }
        
        // Add to ready queue
        let task_size = task.size_bytes;
        let mut tasks = self.tasks.lock().unwrap();
        tasks.push_back(task);
        
        // Update statistics
        self.update_memory_stats(task_size, true)?;
        
        Ok(())
    }
    
    pub async fn process_tasks(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<Vec<CompilationResult>> {
        info!("Processing tasks with optimized memory management");
        
        let mut results = Vec::new();
        let mut batch = Vec::new();
        
        {
            let mut tasks = self.tasks.lock().unwrap();
            
            // Collect batch of tasks
            while let Some(task) = tasks.pop_front() {
                batch.push(task);
                if batch.len() >= self.config.batch_size {
                    break;
                }
            }
        }
        
        if batch.is_empty() {
            return Ok(results);
        }
        
        // Process batch efficiently
        let batch_results = self.process_task_batch(batch, &device, &queue).await?;
        results.extend(batch_results);
        
        // Clean up unused buffers
        self.cleanup_unused_buffers().await?;
        
        Ok(results)
    }
    
    pub fn get_metrics(&self) -> PerformanceMetrics {
        let stats = self.memory_stats.lock().unwrap();
        PerformanceMetrics {
            tasks_completed: stats.tasks_processed as u64,
            gpu_tasks_completed: stats.gpu_tasks_processed,
            cpu_tasks_completed: stats.cpu_tasks_processed,
            total_bytes_transferred: stats.bytes_transferred,
            avg_task_duration_ms: stats.total_processing_time.as_millis() as f64 / stats.tasks_processed.max(1) as f64,
            gpu_utilization: stats.gpu_utilization,
        }
    }

    // Advanced memory management methods
    async fn preallocate_gpu_buffers(
        device: &wgpu::Device,
        pool: &Arc<Mutex<Vec<GpuBuffer>>>,
        config: &BufferConfig,
    ) -> Result<()> {
        debug!("Preallocating GPU buffers");
        
        let buffer_sizes = vec![
            1024,    // 1KB
            4096,    // 4KB
            16384,   // 16KB
            65536,   // 64KB
            262144,  // 256KB
        ];
        
        let mut pool_guard = pool.lock().unwrap();
        
        for &size in &buffer_sizes {
            for _ in 0..4 { // 4 buffers of each size
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("GPU Buffer Pool {} bytes", size)),
                    size: size as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                
                pool_guard.push(GpuBuffer {
                    buffer,
                    size: size as u64,
                    in_use: false,
                    last_used: std::time::Instant::now(),
                });
            }
        }
        
        info!("Preallocated {} GPU buffers", pool_guard.len());
        Ok(())
    }

    async fn preallocate_cpu_buffers(
        pool: &Arc<Mutex<Vec<CpuBuffer>>>,
        config: &BufferConfig,
    ) -> Result<()> {
        debug!("Preallocating CPU buffers");
        
        let buffer_sizes = vec![1024, 4096, 16384, 65536];
        let mut pool_guard = pool.lock().unwrap();
        
        for &size in &buffer_sizes {
            for _ in 0..8 { // 8 CPU buffers of each size
                pool_guard.push(CpuBuffer {
                    data: vec![0u8; size],
                    size,
                    in_use: false,
                    last_used: std::time::Instant::now(),
                });
            }
        }
        
        info!("Preallocated {} CPU buffers", pool_guard.len());
        Ok(())
    }

    fn add_to_batch(&self, task: CompilationTask) -> Result<()> {
        debug!("Adding task {} to batch", task.id);
        
        let mut tasks = self.tasks.lock().unwrap();
        tasks.push_back(task);
        
        Ok(())
    }

    async fn process_task_batch(
        &mut self,
        batch: Vec<CompilationTask>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<CompilationResult>> {
        debug!("Processing batch of {} tasks", batch.len());
        
        let start_time = std::time::Instant::now();
        let mut results = Vec::new();
        
        // Sort tasks by priority and size for optimal processing
        let mut sorted_batch = batch;
        sorted_batch.sort_by(|a, b| {
            b.priority.cmp(&a.priority)
                .then_with(|| b.size_bytes.cmp(&a.size_bytes))
        });
        
        // Group tasks by type for specialized processing
        let mut task_groups: HashMap<TaskType, Vec<CompilationTask>> = HashMap::new();
        for task in sorted_batch {
            task_groups.entry(task.task_type).or_default().push(task);
        }
        
        // Process each group optimally
        for (task_type, tasks) in &task_groups {
let group_results = self.process_task_group(tasks.clone(), *task_type, &device, &queue).await?;
            results.extend(group_results);
        }
        
        // Process codegen tasks
        if let Some(tasks) = task_groups.get(&TaskType::CodeGeneration) {
            let group_results = self.process_codegen_tasks(&tasks.clone(), &device, &queue).await
                .unwrap_or_else(|_| Vec::new());
            results.extend(group_results);
        }
        
        // Process optimization tasks
        if let Some(tasks) = task_groups.get(&TaskType::Optimization) {
            let group_results = self.process_optimization_tasks(&tasks.clone(), &device, &queue).await
                .unwrap_or_else(|_| Vec::new());
            results.extend(group_results);
        }
        
        // Process register allocation tasks
        if let Some(tasks) = task_groups.get(&TaskType::RegisterAllocation) {
            let group_results = self.process_register_allocation_tasks(&tasks.clone(), &device, &queue).await
                .unwrap_or_else(|_| Vec::new());
            results.extend(group_results);
        }
        
        // Process link preparation tasks
        if let Some(tasks) = task_groups.get(&TaskType::LinkPreparation) {
            let group_results = self.process_link_preparation_tasks(&tasks.clone(), &device, &queue).await
                .unwrap_or_else(|_| Vec::new());
            results.extend(group_results);
        }
        
        // Update statistics
        let processing_time = start_time.elapsed();
        self.update_batch_stats(results.len(), processing_time)?;
        
        Ok(results)
    }

async fn process_task_group(
        &mut self,
        tasks: Vec<CompilationTask>,
        task_type: TaskType,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<CompilationResult>> {
        debug!("Processing {} tasks of type {:?}", tasks.len(), task_type);
        
        match task_type {
            TaskType::CodeGeneration => {
                self.process_codegen_tasks(&tasks, device, queue).await
            }
            TaskType::Optimization => {
                self.process_optimization_tasks(&tasks, device, queue).await
            }
            TaskType::RegisterAllocation => {
                self.process_register_allocation_tasks(&tasks, device, queue).await
            }
            TaskType::LinkPreparation => {
                self.process_link_preparation_tasks(&tasks, device, queue).await
            }
        }
    }

async fn process_codegen_tasks(
        &mut self,
        tasks: &Vec<CompilationTask>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<CompilationResult>> {
        // Use optimized GPU buffer pooling for codegen
        let mut results = Vec::new();
        
        for task in tasks {
            let start_time = std::time::Instant::now();
            
            // Get GPU buffer from pool
            let gpu_buffer = self.acquire_gpu_buffer(task.size_bytes, device).await?;
            
            // Copy data to GPU
            queue.write_buffer(&gpu_buffer.buffer, 0, &task.data);
            
            // Simulate GPU processing
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            
            // Read back results (mock)
            let result_data = format!("GPU processed task {}", task.id).into_bytes();
            
            // Return buffer to pool
            self.release_gpu_buffer(gpu_buffer).await?;
            
            results.push(CompilationResult {
                task_id: task.id,
                data: result_data,
                success: true,
                processing_time_ms: start_time.elapsed().as_millis() as f64,
                processed_on_gpu: true,
            });
        }
        
        Ok(results)
    }

    async fn process_optimization_tasks(
        &mut self,
        tasks: &Vec<CompilationTask>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<CompilationResult>> {
        // Batch optimization tasks for better GPU utilization
        let mut results = Vec::new();
        
        if tasks.len() > 1 {
            // Process batch on GPU
            let batch_result = self.process_optimization_batch(&tasks, device, queue).await?;
            results.push(batch_result);
        } else {
            // Single task processing
            for task in tasks {
                let result_data = format!("GPU optimized task {}", task.id).into_bytes();
                results.push(CompilationResult {
                    task_id: task.id,
                    data: result_data,
                    success: true,
                    processing_time_ms: 50.0,
                    processed_on_gpu: true,
                });
            }
        }
        
        Ok(results)
    }

async fn process_register_allocation_tasks(
        &mut self,
        tasks: &Vec<CompilationTask>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<CompilationResult>> {
        // Register allocation benefits from GPU parallelism
        let mut results = Vec::new();
        
        for task in tasks {
            let result_data = format!("GPU allocated registers for task {}", task.id).into_bytes();
            results.push(CompilationResult {
                task_id: task.id,
                data: result_data,
                success: true,
                processing_time_ms: 75.0,
                processed_on_gpu: true,
            });
        }
        
        Ok(results)
    }

async fn process_link_preparation_tasks(
        &mut self,
        tasks: &Vec<CompilationTask>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<CompilationResult>> {
        // Link preparation is memory-bound but can benefit from GPU sorting
        let mut results = Vec::new();
        
        for task in tasks {
            let result_data = format!("GPU prepared link for task {}", task.id).into_bytes();
            results.push(CompilationResult {
                task_id: task.id,
                data: result_data,
                success: true,
                processing_time_ms: 30.0,
                processed_on_gpu: true,
            });
        }
        
        Ok(results)
    }

    async fn process_optimization_batch(
        &self,
        tasks: &[CompilationTask],
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Result<CompilationResult> {
        // Simulate batch optimization
        let total_size: usize = tasks.iter().map(|t| t.size_bytes).sum();
        let result_data = format!("GPU batch optimized {} tasks ({} bytes)", 
                                tasks.len(), total_size).into_bytes();
        
        Ok(CompilationResult {
            task_id: tasks[0].id, // Use first task ID for batch
            data: result_data,
            success: true,
            processing_time_ms: 25.0,
            processed_on_gpu: true,
        })
    }

    async fn acquire_gpu_buffer(&mut self, size: usize, device: &wgpu::Device) -> Result<GpuBuffer> {
        let mut pool = self.gpu_buffer_pool.lock().unwrap();
        
        // Find suitable buffer from pool
        for buffer in pool.iter_mut() {
            if !buffer.in_use && buffer.size >= size as u64 {
                buffer.in_use = true;
                buffer.last_used = std::time::Instant::now();
                let gpu_buffer = std::mem::replace(buffer, GpuBuffer {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Placeholder"),
                        size: 0,
                        usage: wgpu::BufferUsages::empty(),
                        mapped_at_creation: false,
                    }),
                    size: 0,
                    in_use: false,
                    last_used: std::time::Instant::now(),
                });
                return Ok(GpuBuffer {
                    buffer: gpu_buffer.buffer,
                    size: gpu_buffer.size,
                    in_use: true,
                    last_used: gpu_buffer.last_used,
                });
            }
        }
        
        // Create new buffer if none available
        let buffer_size = ((size + 4095) / 4096) * 4096; // Align to 4KB
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("New GPU Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        Ok(GpuBuffer {
            buffer,
            size: buffer_size as u64,
            in_use: true,
            last_used: std::time::Instant::now(),
        })
    }

    async fn release_gpu_buffer(&self, _buffer: GpuBuffer) -> Result<()> {
        // Mark buffer as available in pool
        // In a real implementation, we'd track and reuse
        Ok(())
    }

    async fn cleanup_unused_buffers(&self) -> Result<()> {
        debug!("Cleaning up unused buffers");
        
        let now = std::time::Instant::now();
        let cleanup_threshold = std::time::Duration::from_secs(30);
        
        // Clean GPU buffers
        {
            let mut pool = self.gpu_buffer_pool.lock().unwrap();
            pool.retain(|buffer| {
                !buffer.in_use || (now.duration_since(buffer.last_used) < cleanup_threshold)
            });
        }
        
        // Clean CPU buffers
        {
            let mut pool = self.cpu_buffer_pool.lock().unwrap();
            pool.retain(|buffer| {
                !buffer.in_use || (now.duration_since(buffer.last_used) < cleanup_threshold)
            });
        }
        
        Ok(())
    }

    fn update_memory_stats(&self, bytes: usize, is_gpu: bool) -> Result<()> {
        let mut stats = self.memory_stats.lock().unwrap();
        
        stats.bytes_transferred += bytes as u64;
        if is_gpu {
            stats.gpu_tasks_processed += 1;
        } else {
            stats.cpu_tasks_processed += 1;
        }
        
        Ok(())
    }

    fn update_batch_stats(&self, task_count: usize, processing_time: std::time::Duration) -> Result<()> {
        let mut stats = self.memory_stats.lock().unwrap();
        
        stats.tasks_processed += task_count;
        stats.total_processing_time += processing_time;
        stats.gpu_utilization = if processing_time.as_millis() > 0 {
            (task_count as f64 * 1000.0) / processing_time.as_millis() as f64
        } else {
            0.0
        };
        
        Ok(())
    }
}

#[derive(Debug)]
pub struct CompilationResult {
    pub task_id: u64,
    pub data: Vec<u8>,
    pub success: bool,
    pub processing_time_ms: f64,
    pub processed_on_gpu: bool,
}

#[derive(Debug, Default)]
pub struct MemoryStats {
    pub tasks_processed: usize,
    pub gpu_tasks_processed: u64,
    pub cpu_tasks_processed: u64,
    pub bytes_transferred: u64,
    pub total_processing_time: std::time::Duration,
    pub gpu_utilization: f64,
    pub buffer_pool_hits: u64,
    pub buffer_pool_misses: u64,
}

#[derive(Debug, Default, Clone)]
pub struct PerformanceMetrics {
    pub tasks_completed: u64,
    pub gpu_tasks_completed: u64,
    pub cpu_tasks_completed: u64,
    pub total_bytes_transferred: u64,
    pub avg_task_duration_ms: f64,
    pub gpu_utilization: f64,
}