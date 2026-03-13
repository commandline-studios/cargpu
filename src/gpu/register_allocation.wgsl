// Register allocation shader for GPU-accelerated compilation
@group(0) @binding(0)
var<storage, read> register_graph: array<u32>;

@group(0) @binding(1)
var<storage, read> function_blocks: array<u32>;

@group(0) @binding(2)
var<storage, read_write> allocation_result: array<u32>;

@group(0) @binding(3)
var<storage, read_write> spill_info: array<u32>;

// Register allocation constants
const MAX_REGISTERS: u32 = 32u;
const INTERFERENCE_MASK_SIZE: u32 = 16u;
const SPILL_COST_THRESHOLD: u32 = 1000u;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block_id = global_id.x;
    let num_blocks = arrayLength(&function_blocks);
    
    if (block_id >= num_blocks) {
        return;
    }
    
    let block_info = function_blocks[block_id];
    let virtual_reg_count = (block_info & 0xFFFFu);
    let block_size = ((block_info >> 16u) & 0xFFFFu);
    
    // Initialize allocation result
    allocation_result[block_id] = 0u;
    spill_info[block_id] = 0u;
    
    if (virtual_reg_count == 0u || block_size == 0u) {
        return;
    }
    
    // Perform register allocation for this block
    let allocation = allocate_registers_for_block(block_id, virtual_reg_count);
    let spills = calculate_spill_cost(block_id, virtual_reg_count, allocation);
    
    allocation_result[block_id] = allocation;
    spill_info[block_id] = spills;
}

fn allocate_registers_for_block(block_id: u32, reg_count: u32) -> u32 {
    // Greedy graph coloring register allocation
    var allocation: u32 = 0u;
    var used_registers: u32 = 0u;
    
    // Process virtual registers in order of priority
    for (var vreg = 0u; vreg < reg_count; vreg = vreg + 1u) {
        let interference = get_interference_mask(block_id, vreg);
        let available = find_available_register(interference, used_registers);
        
        if (available < MAX_REGISTERS) {
            // Allocate virtual register to physical register
            allocation = allocation | (available << (vreg * 3u)); // 3 bits per register
            used_registers = used_registers | (1u << available);
        } else {
            // Need to spill
            allocation = allocation | (0x7u << (vreg * 3u)); // Mark as spilled
        }
    }
    
    return allocation;
}

fn get_interference_mask(block_id: u32, vreg: u32) -> u32 {
    // Get interference mask for virtual register
    let graph_offset = block_id * INTERFERENCE_MASK_SIZE + vreg / 32u;
    if (graph_offset >= arrayLength(&register_graph)) {
        return 0xFFFFFFFFu;
    }
    
    let mask_data = register_graph[graph_offset];
    let bit_offset = vreg % 32u;
    return (mask_data >> bit_offset) & 0xFFFFFFFFu;
}

fn find_available_register(interference: u32, used: u32) -> u32 {
    // Find first available physical register
    let conflict_mask = interference | used;
    
    for (var reg = 0u; reg < MAX_REGISTERS; reg = reg + 1u) {
        if ((conflict_mask & (1u << reg)) == 0u) {
            return reg;
        }
    }
    
    return MAX_REGISTERS; // No register available
}

fn calculate_spill_cost(block_id: u32, reg_count: u32, allocation: u32) -> u32 {
    // Calculate spill cost for the block
    var spill_count: u32 = 0u;
    var total_cost: u32 = 0u;
    
    for (var vreg = 0u; vreg < reg_count; vreg = vreg + 1u) {
        let phys_reg = (allocation >> (vreg * 3u)) & 0x7u;
        
        if (phys_reg == 0x7u) { // Spilled register
            spill_count = spill_count + 1u;
            // Estimate spill cost based on register usage frequency
            let freq = get_register_frequency(block_id, vreg);
            total_cost = total_cost + freq;
        }
    }
    
    return (spill_count << 16u) | (total_cost & 0xFFFFu);
}

fn get_register_frequency(block_id: u32, vreg: u32) -> u32 {
    // Estimate register usage frequency
    // Higher frequency = higher spill cost
    let base_freq = (vreg + 1u) * 10u;
    let block_factor = (block_id + 1u) * 2u;
    
    return (base_freq + block_factor) % 1000u + 1u;
}