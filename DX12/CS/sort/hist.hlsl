#define WORKGROUP_SIZE 256 // assert WORKGROUP_SIZE >= RADIX_SORT_BINS
#define RADIX_SORT_BINS 256u

#define BITS 64

typedef uint64_t key_t;

struct PushConstants {
    uint g_num_elements;
    uint g_shift;
    uint g_num_workgroups;
    uint g_num_blocks_per_workgroup;
};

ConstantBuffer<PushConstants> pushConstants : register(b0);

RWStructuredBuffer<key_t> g_elements_in : register(u0);
RWStructuredBuffer<uint> g_histograms : register(u1); // [histogram_of_workgroup_0 | histogram_of_workgroup_1 | ... ]

groupshared uint histogram[RADIX_SORT_BINS];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint gID = DTid.x;
    uint lID = GTid.x;
    uint wID = Gid.x;

    // initialize histogram
    if (lID < RADIX_SORT_BINS) {
        histogram[lID] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint index = 0; index < pushConstants.g_num_blocks_per_workgroup; index++) {
        uint elementId = wID * pushConstants.g_num_blocks_per_workgroup * WORKGROUP_SIZE + index * WORKGROUP_SIZE + lID;
        if (elementId < pushConstants.g_num_elements) {
            // determine the bin
            const uint bin = uint(g_elements_in[elementId] >> pushConstants.g_shift) & (RADIX_SORT_BINS - 1);
            // increment the histogram
            InterlockedAdd(histogram[bin], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (lID < RADIX_SORT_BINS) {
        g_histograms[RADIX_SORT_BINS * wID + lID] = histogram[lID];
    }
}