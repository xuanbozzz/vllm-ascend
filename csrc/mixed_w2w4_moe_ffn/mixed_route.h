#ifndef MIXED_MOE_ROUTE_H
#define MIXED_MOE_ROUTE_H

#include <cstdint>

// uint32_t bit masks make each route flag a real one-bit GM value.  Expert i
// is stored in bit (i & 31) of word (i >> 5).
__aicore__ inline bool MixedRouteBit(__gm__ uint32_t *mask, uint64_t expert_id)
{
    const uint32_t word = mask[expert_id >> 5];
    return ((word >> (expert_id & 31)) & 1U) != 0;
}

#endif
