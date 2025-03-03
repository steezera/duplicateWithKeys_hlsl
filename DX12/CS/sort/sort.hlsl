#define WORKGROUP_SIZE 256 // assert WORKGROUP_SIZE >= RADIX_SORT_BINS
#define RADIX_SORT_BINS 256u
#define SUBGROUP_SIZE 32 // 32 NVIDIA; Default Wave Size in HLSL is 32

#define BITS 64 // sorting uint64_t

typedef uint64_t key_t;

struct PushConstants {
    uint g_num_elements;
    uint g_shift;
    uint g_num_workgroups;
    uint g_num_blocks_per_workgroup;
};

ConstantBuffer<PushConstants> pushConstants : register(b0);

RWStructuredBuffer<key_t> g_elements_in : register(u0);
RWStructuredBuffer<key_t> g_elements_out : register(u1);
RWStructuredBuffer<uint> g_payload_in : register(u2);
RWStructuredBuffer<uint> g_payload_out : register(u3);
RWStructuredBuffer<uint> g_histograms : register(u4); // [histogram_of_workgroup_0 | histogram_of_workgroup_1 | ... ]

groupshared uint sums[RADIX_SORT_BINS / SUBGROUP_SIZE]; // subgroup reductions
groupshared uint global_offsets[RADIX_SORT_BINS]; // global exclusive scan (prefix sum)

struct BinFlags {
    key_t flags[WORKGROUP_SIZE / BITS];
};
groupshared BinFlags bin_flags[RADIX_SORT_BINS];

// Helper functions to implement subgroup operations in HLSL
uint WavePrefixSum(uint value) {
    uint sum = 0;
    uint mask = WaveReadLaneFirst(WaveActiveBallot(true));
    for (uint i = 0; i < WaveGetLaneCount(); i++) {
        uint laneValue = WaveReadLaneAt(value, i);
        if (WaveGetLaneIndex() > i) {
            sum += laneValue;
        }
    }
    return sum;
}

uint WaveSum(uint value) {
    return WaveActiveSum(value);
}

uint countBits(uint value) {
    return countbits(value);
}

uint countBits(uint64_t value) {
    return countbits((uint)value) + countbits((uint)(value >> 32));
}

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint gID = DTid.x;
    uint lID = GTid.x;
    uint wID = Gid.x;
    uint sID = lID / SUBGROUP_SIZE; // WaveGetLaneIndex()
    uint lsID = lID % SUBGROUP_SIZE; // WaveGetLaneIndex()

    uint local_histogram = 0;
    uint prefix_sum = 0;
    uint histogram_count = 0;

    if (lID < RADIX_SORT_BINS) {
        uint count = 0;
        for (uint j = 0; j < pushConstants.g_num_workgroups; j++) {
            const uint t = g_histograms[RADIX_SORT_BINS * j + lID];
            local_histogram = (j == wID) ? count : local_histogram;
            count += t;
        }
        histogram_count = count;
        
        // Using Wave functions for subgroup operations
        const uint sum = WaveSum(histogram_count);
        prefix_sum = WavePrefixSum(histogram_count);
        
        // One thread inside the wave enters this section
        if (WaveIsFirstLane()) {
            sums[sID] = sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (lID < RADIX_SORT_BINS) {
        // Using wave functions for prefix sum
        const uint wave_prefix_sum = WavePrefixSum(sums[lsID]);
        const uint sums_prefix_sum = WaveReadLaneAt(wave_prefix_sum, sID);
        const uint global_histogram = sums_prefix_sum + prefix_sum;
        global_offsets[lID] = global_histogram + local_histogram;
    }

    // ==== scatter keys according to global offsets =====
    const uint flags_bin = lID / BITS;
    const uint64_t flags_bit = 1ULL << (lID % BITS);

    for (uint index = 0; index < pushConstants.g_num_blocks_per_workgroup; index++) {
        uint elementId = wID * pushConstants.g_num_blocks_per_workgroup * WORKGROUP_SIZE + index * WORKGROUP_SIZE + lID;

        // initialize bin flags
        if (lID < RADIX_SORT_BINS) {
            for (int i = 0; i < WORKGROUP_SIZE / BITS; i++) {
                bin_flags[lID].flags[i] = 0u; // init all bin flags to 0
            }
        }
        GroupMemoryBarrierWithGroupSync();

        key_t element_in = 0;
        uint payload_in = 0;
        uint binID = 0;
        uint binOffset = 0;
        if (elementId < pushConstants.g_num_elements) {
            element_in = g_elements_in[elementId];
            payload_in = g_payload_in[elementId];
            binID = uint(element_in >> pushConstants.g_shift) & uint(RADIX_SORT_BINS - 1);
            // offset for group
            binOffset = global_offsets[binID];
            // add bit to flag using atomic operations
            InterlockedOr(bin_flags[binID].flags[flags_bin], flags_bit);
        }
        GroupMemoryBarrierWithGroupSync();

        if (elementId < pushConstants.g_num_elements) {
            // calculate output index of element
            uint prefix = 0;
            uint count = 0;
            for (uint i = 0; i < WORKGROUP_SIZE / BITS; i++) {
                const key_t bits = bin_flags[binID].flags[i];
                const uint full_count = countBits(bits);
                const key_t partial_bits = bits & (flags_bit - 1);
                const uint partial_count = countBits(partial_bits);
                
                prefix += (i < flags_bin) ? full_count : 0u;
                prefix += (i == flags_bin) ? partial_count : 0u;
                count += full_count;
            }
            g_elements_out[binOffset + prefix] = element_in;
            g_payload_out[binOffset + prefix] = payload_in;
            if (prefix == count - 1) {
                InterlockedAdd(global_offsets[binID], count);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}