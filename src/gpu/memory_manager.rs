use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use wgpu::{Buffer, BufferAddress, Device, Queue};

pub struct MemoryManager {
    device: Arc<Device>,
    queue: Arc<Queue>,
    buffers: HashMap<String, Arc<Buffer>>,
    memory_layouts: HashMap<String, MemoryLayout>,
    config: MemoryManagerConfig,
    total_allocated: usize,
    transfer_queue: Vec<TransferOperation>,
}

#[derive(Debug, Clone)]
pub struct MemoryManagerConfig {
    pub max_memory_mb: usize,
    pub enable_persistent_mappings: bool,
    pub transfer_chunk_size: usize,
    pub enable_async_transfers: bool,
    pub alignment_bytes: usize,
}

impl Default for MemoryManagerConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024, // 1GB default
            enable_persistent_mappings: true,
            transfer_chunk_size: 64 * 1024, // 64KB chunks
            enable_async_transfers: true,
            alignment_bytes: 256, // GPU cache line alignment
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryLayout {
    pub name: String,
    pub size_bytes: usize,
    pub alignment: usize,
    pub layout_type: LayoutType,
    pub fields: Vec<LayoutField>,
    pub padding_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayoutType {
    Struct,
    Array,
    Buffer,
    Uniform,
    Storage,
    PushConstant,
}

#[derive(Debug, Clone)]
pub struct LayoutField {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub field_type: FieldType,
    pub alignment: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
    Vec2,
    Vec3,
    Vec4,
    Mat3,
    Mat4,
    Custom { size: usize, alignment: usize },
}

impl FieldType {
    pub fn size(&self) -> usize {
        match self {
            FieldType::I8 | FieldType::U8 | FieldType::Bool => 1,
            FieldType::I16 | FieldType::U16 => 2,
            FieldType::I32 | FieldType::U32 | FieldType::F32 => 4,
            FieldType::I64 | FieldType::U64 | FieldType::F64 => 8,
            FieldType::Vec2 => 8,
            FieldType::Vec3 => 12,
            FieldType::Vec4 => 16,
            FieldType::Mat3 => 36,
            FieldType::Mat4 => 64,
            FieldType::Custom { size, .. } => *size,
        }
    }

    pub fn alignment(&self) -> usize {
        match self {
            FieldType::I8 | FieldType::U8 | FieldType::Bool => 1,
            FieldType::I16 | FieldType::U16 => 2,
            FieldType::I32 | FieldType::U32 | FieldType::F32 | FieldType::Vec2 => 4,
            FieldType::I64
            | FieldType::U64
            | FieldType::F64
            | FieldType::Vec3
            | FieldType::Vec4 => 8,
            FieldType::Mat3 | FieldType::Mat4 => 16,
            FieldType::Custom { alignment, .. } => *alignment,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransferOperation {
    pub id: String,
    pub operation_type: TransferType,
    pub source: TransferSource,
    pub destination: TransferDestination,
    pub size_bytes: usize,
    pub priority: TransferPriority,
    pub timestamp: std::time::Instant,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransferType {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

#[derive(Debug, Clone)]
pub enum TransferSource {
    CpuData(Vec<u8>),
    GpuBuffer(String),
}

#[derive(Debug, Clone)]
pub enum TransferDestination {
    CpuBuffer,
    GpuBuffer(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransferPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

#[derive(Debug, Clone)]
pub struct BufferHandle {
    pub name: String,
    pub buffer: Arc<Buffer>,
    pub size_bytes: usize,
    pub usage: wgpu::BufferUsages,
    pub memory_type: MemoryType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryType {
    DeviceLocal,
    HostVisible,
    HostCoherent,
    Cached,
}

impl MemoryManager {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, config: MemoryManagerConfig) -> Self {
        info!("Initializing MemoryManager with config: {:?}", config);

        Self {
            device,
            queue,
            buffers: HashMap::new(),
            memory_layouts: HashMap::new(),
            config,
            total_allocated: 0,
            transfer_queue: Vec::new(),
        }
    }

    pub fn create_memory_layout(&mut self, layout: MemoryLayout) -> Result<()> {
        debug!(
            "Creating memory layout: {} ({} bytes)",
            layout.name, layout.size_bytes
        );

        // Validate layout
        self.validate_layout(&layout)?;

        // Calculate total size including padding
        let total_size = self.calculate_layout_size(&layout);

        let mut optimized_layout = layout.clone();
        optimized_layout.size_bytes = total_size;
        optimized_layout.padding_bytes = total_size - layout.size_bytes;

        self.memory_layouts
            .insert(layout.name.clone(), optimized_layout);
        self.total_allocated += total_size;

        info!(
            "Created memory layout: {} ({} bytes total)",
            layout.name, total_size
        );
        Ok(())
    }

    fn validate_layout(&self, layout: &MemoryLayout) -> Result<()> {
        if layout.name.is_empty() {
            return Err(anyhow!("Layout name cannot be empty"));
        }

        if layout.size_bytes == 0 {
            return Err(anyhow!("Layout size must be greater than 0"));
        }

        // Validate field alignment and offsets
        for field in &layout.fields {
            if field.offset % field.alignment != 0 {
                return Err(anyhow!(
                    "Field '{}' offset {} is not aligned to {} bytes",
                    field.name,
                    field.offset,
                    field.alignment
                ));
            }

            if field.offset + field.size > layout.size_bytes {
                return Err(anyhow!("Field '{}' exceeds layout bounds", field.name));
            }
        }

        Ok(())
    }

    fn calculate_layout_size(&self, layout: &MemoryLayout) -> usize {
        let mut current_offset = 0;
        let mut max_alignment = 1;

        for field in &layout.fields {
            // Align current offset to field alignment
            current_offset = (current_offset + field.alignment - 1) & !(field.alignment - 1);
            current_offset += field.size;
            max_alignment = max_alignment.max(field.alignment);
        }

        // Align total size to maximum alignment
        current_offset = (current_offset + max_alignment - 1) & !(max_alignment - 1);

        // Also align to global configuration
        (current_offset + self.config.alignment_bytes - 1) & !(self.config.alignment_bytes - 1)
    }

    pub fn create_buffer(&mut self, handle: BufferHandle) -> Result<()> {
        debug!(
            "Creating buffer: {} ({} bytes)",
            handle.name, handle.size_bytes
        );

        // Check if buffer already exists
        if self.buffers.contains_key(&handle.name) {
            return Err(anyhow!("Buffer '{}' already exists", handle.name));
        }

        // Create GPU buffer
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&handle.name),
            size: handle.size_bytes as BufferAddress,
            usage: handle.usage,
            mapped_at_creation: false,
        });

        self.buffers.insert(handle.name.clone(), handle.buffer);

        info!(
            "Created buffer: {} ({} bytes)",
            handle.name, handle.size_bytes
        );
        Ok(())
    }

    pub fn create_struct_buffer<T: bytemuck::Pod>(
        &mut self,
        name: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> Result<BufferHandle> {
        debug!(
            "Creating struct buffer: {} with {} elements",
            name,
            data.len()
        );

        let size_bytes = std::mem::size_of_val(data);
        let aligned_size =
            (size_bytes + self.config.alignment_bytes - 1) & !(self.config.alignment_bytes - 1);

        // Create memory layout for the struct
        let layout = MemoryLayout {
            name: format!("{}_layout", name),
            size_bytes,
            alignment: std::mem::align_of::<T>(),
            layout_type: LayoutType::Struct,
            fields: vec![], // Could be populated with field info if needed
            padding_bytes: aligned_size - size_bytes,
        };

        self.create_memory_layout(layout)?;

        // Create buffer
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: aligned_size as BufferAddress,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let handle = BufferHandle {
            name: name.to_string(),
            buffer: buffer.into(),
            size_bytes: aligned_size,
            usage,
            memory_type: MemoryType::DeviceLocal,
        };

        // Transfer data if provided
        if !data.is_empty() {
            self.transfer_host_to_device(name, bytemuck::cast_slice(data))?;
        }

        Ok(handle)
    }

    pub fn transfer_host_to_device(&mut self, buffer_name: &str, data: &[u8]) -> Result<()> {
        debug!(
            "Transferring {} bytes from host to device buffer '{}'",
            data.len(),
            buffer_name
        );

        let buffer = self
            .buffers
            .get(buffer_name)
            .ok_or_else(|| anyhow!("Buffer '{}' not found", buffer_name))?;

        if data.len() > buffer.size() as usize {
            return Err(anyhow!("Data size exceeds buffer size"));
        }

        if self.config.enable_async_transfers {
            let transfer = TransferOperation {
                id: format!("h2d_{}_{}", buffer_name, uuid::Uuid::new_v4()),
                operation_type: TransferType::HostToDevice,
                source: TransferSource::CpuData(data.to_vec()),
                destination: TransferDestination::GpuBuffer(buffer_name.to_string()),
                size_bytes: data.len(),
                priority: TransferPriority::Medium,
                timestamp: std::time::Instant::now(),
            };

            self.transfer_queue.push(transfer);
            self.process_transfer_queue_async()?;
        } else {
            // Synchronous transfer
            self.queue.write_buffer(buffer, 0, data);
        }

        Ok(())
    }

    pub fn transfer_device_to_host(
        &mut self,
        buffer_name: &str,
        size: Option<usize>,
    ) -> Result<Vec<u8>> {
        debug!(
            "Transferring data from device buffer '{}' to host",
            buffer_name
        );

        let buffer = self
            .buffers
            .get(buffer_name)
            .ok_or_else(|| anyhow!("Buffer '{}' not found", buffer_name))?;

        let transfer_size = size.unwrap_or(buffer.size() as usize);
        if transfer_size > buffer.size() as usize {
            return Err(anyhow!("Requested size exceeds buffer size"));
        }

        // Create staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}_staging", buffer_name)),
            size: transfer_size as BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder for copy operation
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{}_copy_encoder", buffer_name)),
            });

        encoder.copy_buffer_to_buffer(
            buffer,
            0,
            &staging_buffer,
            0,
            transfer_size as BufferAddress,
        );

        let command_buffer = encoder.finish();
        self.queue.submit(Some(command_buffer));

        // Map and read the staging buffer
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        let result =
            tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(rx))??;
        let data = buffer_slice.get_mapped_range().to_vec();

        Ok(data)
    }

    fn process_transfer_queue_async(&mut self) -> Result<()> {
        if self.transfer_queue.is_empty() {
            return Ok(());
        }

        debug!(
            "Processing async transfer queue with {} operations",
            self.transfer_queue.len()
        );

        // Sort transfers by priority
        self.transfer_queue
            .sort_by(|a, b| b.priority.cmp(&a.priority));

        // Process transfers in chunks
        let chunk_size = self.config.transfer_chunk_size;
        let mut processed_transfers = Vec::new();

        for transfer in self.transfer_queue.drain(..) {
            match transfer.operation_type {
                TransferType::HostToDevice => {
                    if let TransferSource::CpuData(data) = transfer.source {
                        if let TransferDestination::GpuBuffer(buffer_name) = transfer.destination {
                            if let Some(buffer) = self.buffers.get(&buffer_name) {
                                // Process in chunks
                                for (chunk_idx, chunk) in data.chunks(chunk_size).enumerate() {
                                    let offset = chunk_idx * chunk_size;
                                    self.queue
                                        .write_buffer(buffer, offset as BufferAddress, chunk);
                                }
                                processed_transfers.push(transfer.id);
                            }
                        }
                    }
                }
                TransferType::DeviceToHost | TransferType::DeviceToDevice => {
                    // TODO: Implement device-to-host and device-to-device transfers
                    warn!(
                        "Transfer type {:?} not yet implemented",
                        transfer.operation_type
                    );
                }
            }
        }

        debug!(
            "Processed {} transfer operations",
            processed_transfers.len()
        );
        Ok(())
    }

    pub fn get_buffer(&self, name: &str) -> Option<&Arc<Buffer>> {
        self.buffers.get(name)
    }

    pub fn get_memory_layout(&self, name: &str) -> Option<&MemoryLayout> {
        self.memory_layouts.get(name)
    }

    pub fn get_memory_usage(&self) -> MemoryUsage {
        let total_buffer_size: usize = self.buffers.values().map(|b| b.size() as usize).sum();

        let total_layout_size: usize = self.memory_layouts.values().map(|l| l.size_bytes).sum();

        MemoryUsage {
            allocated_bytes: total_buffer_size,
            layout_bytes: total_layout_size,
            buffer_count: self.buffers.len(),
            layout_count: self.memory_layouts.len(),
            pending_transfers: self.transfer_queue.len(),
            utilization_percent: if self.config.max_memory_mb > 0 {
                (total_buffer_size as f64 / (self.config.max_memory_mb * 1024 * 1024) as f64)
                    * 100.0
            } else {
                0.0
            },
        }
    }

    pub fn optimize_memory_layouts(&mut self) -> Result<()> {
        info!("Optimizing memory layouts");

        // Find fragmented layouts that can be merged
        let mut layouts_to_merge: Vec<(String, String)> = Vec::new();

        let layout_names: Vec<&String> = self.memory_layouts.keys().collect();

        for (i, name1) in layout_names.iter().enumerate() {
            for name2 in layout_names.iter().skip(i + 1) {
                let layout1 = &self.memory_layouts[*name1];
                let layout2 = &self.memory_layouts[*name2];

                // Check if layouts can be merged (same type, compatible alignment)
                if layout1.layout_type == layout2.layout_type {
                    let combined_size = layout1.size_bytes + layout2.size_bytes;
                    let combined_alignment = layout1.alignment.max(layout2.alignment);

                    // Merge if combined size is significantly less than separate allocations
                    if combined_size <= (layout1.size_bytes + layout2.size_bytes) * 110 / 100 {
                        layouts_to_merge.push(((*name1).clone(), (*name2).clone()));
                    }
                }
            }
        }

        // Perform merges
        for (name1, name2) in layouts_to_merge {
            self.merge_layouts(&name1, &name2)?;
        }

        info!("Memory layout optimization completed");
        Ok(())
    }

    fn merge_layouts(&mut self, name1: &str, name2: &str) -> Result<()> {
        debug!("Merging layouts '{}' and '{}'", name1, name2);

        let layout1 = self
            .memory_layouts
            .remove(name1)
            .ok_or_else(|| anyhow!("Layout '{}' not found", name1))?;
        let layout2 = self
            .memory_layouts
            .remove(name2)
            .ok_or_else(|| anyhow!("Layout '{}' not found", name2))?;

        let merged_layout = MemoryLayout {
            name: format!("{}_{}", name1, name2),
            size_bytes: layout1.size_bytes + layout2.size_bytes,
            alignment: layout1.alignment.max(layout2.alignment),
            layout_type: layout1.layout_type,
            fields: [layout1.fields, layout2.fields].concat(),
            padding_bytes: layout1.padding_bytes + layout2.padding_bytes,
        };

        self.memory_layouts
            .insert(merged_layout.name.clone(), merged_layout);

        debug!("Successfully merged layouts");
        Ok(())
    }

    pub fn cleanup_unused_resources(&mut self) -> Result<()> {
        info!("Cleaning up unused resources");

        let buffers_to_remove: Vec<String> = self
            .buffers
            .keys()
            .filter(|name| {
                // Simple heuristic: buffers with "temp" in name are candidates for cleanup
                name.contains("temp") || name.contains("staging")
            })
            .cloned()
            .collect();

        for buffer_name in buffers_to_remove {
            self.buffers.remove(&buffer_name);
            debug!("Removed buffer: {}", buffer_name);
        }

        let layouts_to_remove: Vec<String> = self
            .memory_layouts
            .keys()
            .filter(|name| name.contains("temp") || name.contains("staging"))
            .cloned()
            .collect();

        for layout_name in layouts_to_remove {
            self.memory_layouts.remove(&layout_name);
            debug!("Removed layout: {}", layout_name);
        }

        info!("Cleanup completed");
        Ok(())
    }

    pub fn create_compilation_data_layout(
        &mut self,
        function_name: &str,
        ir_size: usize,
        optimization_level: u32,
    ) -> Result<String> {
        debug!(
            "Creating compilation data layout for function: {}",
            function_name
        );

        let layout_name = format!("{}_compilation_data", function_name);

        let fields = vec![
            LayoutField {
                name: "ir_data".to_string(),
                offset: 0,
                size: ir_size,
                field_type: FieldType::Custom {
                    size: ir_size,
                    alignment: 8,
                },
                alignment: 8,
            },
            LayoutField {
                name: "metadata".to_string(),
                offset: ir_size,
                size: 32,
                field_type: FieldType::Custom {
                    size: 32,
                    alignment: 8,
                },
                alignment: 8,
            },
            LayoutField {
                name: "optimization_flags".to_string(),
                offset: ir_size + 32,
                size: 4,
                field_type: FieldType::U32,
                alignment: 4,
            },
            LayoutField {
                name: "target_info".to_string(),
                offset: ir_size + 36,
                size: 16,
                field_type: FieldType::Custom {
                    size: 16,
                    alignment: 8,
                },
                alignment: 8,
            },
        ];

        let total_size = ir_size + 52; // Include padding

        let layout = MemoryLayout {
            name: layout_name.clone(),
            size_bytes: total_size,
            alignment: 8,
            layout_type: LayoutType::Storage,
            fields,
            padding_bytes: 0,
        };

        self.create_memory_layout(layout)?;

        Ok(layout_name)
    }

    pub fn align_size(&self, size: usize) -> usize {
        (size + self.config.alignment_bytes - 1) & !(self.config.alignment_bytes - 1)
    }
}

#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub allocated_bytes: usize,
    pub layout_bytes: usize,
    pub buffer_count: usize,
    pub layout_count: usize,
    pub pending_transfers: usize,
    pub utilization_percent: f64,
}

impl std::fmt::Display for MemoryUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Memory Usage: {:.2} MB allocated ({} buffers, {} layouts), {:.1}% utilization, {} pending transfers",
            self.allocated_bytes as f64 / (1024.0 * 1024.0),
            self.buffer_count,
            self.layout_count,
            self.utilization_percent,
            self.pending_transfers
        )
    }
}
