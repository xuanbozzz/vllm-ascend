#ifndef MIXED_MOE_TILING_BUILDERS_H
#define MIXED_MOE_TILING_BUILDERS_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

// Each original Host tiling implementation is compiled in this single Host
// translation unit.  The namespaces preserve the four native strategies and
// avoid the former four helper object files.
namespace GateW2TilingBuilder {
#undef MIXED_MOE_GATE_W2_TILING_H
#undef MIXED_MOE_GATE_W2_POST_TILING_H
#include "components/gate_w2/gate_w2_tiling_builder.h"
} // namespace GateW2TilingBuilder

namespace GateW4TilingBuilder {
#undef MIXED_MOE_GATE_W4_TILING_H
#undef MIXED_MOE_GATE_W4_POST_TILING_H
#include "components/gate_w4/gate_w4_tiling_builder.h"
} // namespace GateW4TilingBuilder

namespace DownW2TilingBuilder {
#undef MIXED_MOE_DOWN_W2_TILING_H
#include "components/down_w2/down_w2_tiling_builder.h"
} // namespace DownW2TilingBuilder

namespace DownW4TilingBuilder {
#undef MIXED_MOE_DOWN_W4_TILING_H
#include "components/down_w4/down_w4_tiling_builder.h"
} // namespace DownW4TilingBuilder

inline std::vector<uint8_t> BuildGateW2TilingBlob(uint64_t k, uint64_t n)
{
    return GateW2TilingBuilder::BuildGateW2TilingBlob(k, n);
}

inline std::vector<uint8_t> BuildGateW4TilingBlob(uint64_t k, uint64_t n)
{
    return GateW4TilingBuilder::BuildGateW4TilingBlob(k, n);
}

inline std::vector<uint8_t> BuildDownW2TilingBlob(uint64_t k, uint64_t n)
{
    return DownW2TilingBuilder::BuildDownW2TilingBlob(k, n);
}

inline std::vector<uint8_t> BuildDownW4TilingBlob(uint64_t k, uint64_t n)
{
    return DownW4TilingBuilder::BuildDownW4TilingBlob(k, n);
}

#endif
