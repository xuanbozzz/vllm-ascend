#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "data_utils.h"
#include "hacl_rt.h"
#include "mixed_tiling.h"
#include "tiling_builders.h"

namespace {

constexpr uint64_t kMaxGroupCount = 256;
constexpr uint64_t kKAlignment = 64;
constexpr uint64_t kZnN0 = 32;
constexpr uint64_t kMxScaleGroupSize = 32;
constexpr uint32_t kKernelCoreCount = 32;
using OutputStorage = uint16_t;
static_assert(sizeof(OutputStorage) == 2, "BF16 output storage must occupy 2 bytes");

constexpr const char *kKernelName = "mixed_w2w4_moe_ffn";
constexpr const char *kKernelBinary = "mixed_w2w4_moe_ffn.o";

struct MatFP4Args {
    uint64_t group_count;
    uint64_t k;
    uint64_t n;
    uint64_t d;
    void *group_list;
    void *a;
    void *a_scale;
    void *gateup_w;
    void *gateup_table;
    void *down_w;
    void *down_table;
    void *scale_upgate;
    void *scale_down;
    void *is_expert_w4;
    void *is_down_w4;
    void *h;
    void *h_scale;
    void *c;
    void *tiling;
};

static_assert(sizeof(MatFP4Args) == 19 * sizeof(uint64_t), "unexpected kernel argument layout");

[[noreturn]] void ThrowError(const std::string &action, int64_t code)
{
    throw std::runtime_error(action + " failed, error code=" + std::to_string(code));
}

void CheckRt(rtError_t status, const std::string &action)
{
    if (status != RT_ERROR_NONE) ThrowError(action, status);
}

void CheckAcl(aclError status, const std::string &action)
{
    if (status != ACL_SUCCESS) ThrowError(action, status);
}

uint64_t CheckedProduct(std::initializer_list<uint64_t> values)
{
    uint64_t result = 1;
    for (uint64_t value : values) {
        if (value != 0 && result > std::numeric_limits<uint64_t>::max() / value) {
            throw std::overflow_error("tensor byte size overflow");
        }
        result *= value;
    }
    return result;
}

uint64_t CheckedSum(std::initializer_list<uint64_t> values)
{
    uint64_t result = 0;
    for (uint64_t value : values) {
        if (result > std::numeric_limits<uint64_t>::max() - value) {
            throw std::overflow_error("tensor byte size overflow");
        }
        result += value;
    }
    return result;
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    if (divisor == 0) {
        throw std::logic_error("division by zero while building tiling");
    }
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

size_t ToHostSize(uint64_t size)
{
    if (size > std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("tensor is too large for host address space");
    }
    return static_cast<size_t>(size);
}

void ReadExactFile(const std::string &path, void *destination, uint64_t expected_size)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open input file: " + path);
    }

    const std::streamoff actual_size = file.tellg();
    if (actual_size < 0 || static_cast<uint64_t>(actual_size) != expected_size) {
        throw std::runtime_error(path + " has " + std::to_string(actual_size) + " bytes; expected " +
                                 std::to_string(expected_size));
    }

    file.seekg(0, std::ios::beg);
    file.read(static_cast<char *>(destination), static_cast<std::streamsize>(expected_size));
    if (!file) {
        throw std::runtime_error("failed to read complete input file: " + path);
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(uint64_t bytes) : bytes_(bytes)
    {
        CheckRt(rtMalloc(&pointer_, bytes_, RT_MEMORY_HBM), "rtMalloc");
    }

    ~DeviceBuffer()
    {
        if (pointer_ != nullptr) {
            (void)rtFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    void *data() const { return pointer_; }
    uint64_t size() const { return bytes_; }

private:
    void *pointer_ = nullptr;
    uint64_t bytes_ = 0;
};

class RuntimeStream {
public:
    RuntimeStream()
    {
        CheckRt(rtStreamCreate(&stream_, 0), "rtStreamCreate");
    }

    ~RuntimeStream()
    {
        if (stream_ != nullptr) {
            (void)rtStreamDestroy(stream_);
        }
    }

    RuntimeStream(const RuntimeStream &) = delete;
    RuntimeStream &operator=(const RuntimeStream &) = delete;

    rtStream_t get() const { return stream_; }

    aclrtStream acl_get() const { return reinterpret_cast<aclrtStream>(stream_); }

    void Synchronize(const char *phase) const
    {
        CheckRt(rtStreamSynchronize(stream_), std::string("rtStreamSynchronize(") + phase + ")");
    }

private:
    rtStream_t stream_ = nullptr;
};

class RuntimeEvent {
public:
    RuntimeEvent()
    {
        CheckAcl(aclrtCreateEvent(&event_), "aclrtCreateEvent");
    }

    ~RuntimeEvent()
    {
        if (event_ != nullptr) {
            (void)aclrtDestroyEvent(event_);
        }
    }

    RuntimeEvent(const RuntimeEvent &) = delete;
    RuntimeEvent &operator=(const RuntimeEvent &) = delete;

    void Record(const RuntimeStream &stream) const
    {
        CheckAcl(aclrtRecordEvent(event_, stream.acl_get()), "aclrtRecordEvent");
    }

    float ElapsedMillisecondsTo(const RuntimeEvent &end) const
    {
        float elapsed_ms = 0.0F;
        CheckAcl(aclrtEventElapsedTime(&elapsed_ms, event_, end.event_), "aclrtEventElapsedTime");
        return elapsed_ms;
    }

private:
    aclrtEvent event_ = nullptr;
};

class RegisteredKernel {
public:
    RegisteredKernel(const char *kernel_name, const char *binary_path)
    {
        std::ifstream file(binary_path, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error(std::string("cannot open kernel binary: ") + binary_path);
        }
        const std::streamoff binary_size = file.tellg();
        if (binary_size <= 0) {
            throw std::runtime_error(std::string("empty kernel binary: ") + binary_path);
        }
        binary_.resize(static_cast<size_t>(binary_size));
        file.seekg(0, std::ios::beg);
        file.read(binary_.data(), binary_size);
        if (!file) {
            throw std::runtime_error(std::string("failed to read kernel binary: ") + binary_path);
        }

        rtDevBinary_t binary_desc {};
        binary_desc.magic = RT_DEV_BINARY_MAGIC_ELF;
        binary_desc.version = 0;
        binary_desc.data = binary_.data();
        binary_desc.length = binary_.size();
        CheckRt(rtDevBinaryRegister(&binary_desc, &binary_handle_), "rtDevBinaryRegister");

        const void *stub = reinterpret_cast<const void *>(kernel_name);
        CheckRt(rtFunctionRegister(binary_handle_, stub, kernel_name, const_cast<void *>(stub), 0), "rtFunctionRegister");
    }

    RegisteredKernel(const RegisteredKernel &) = delete;
    RegisteredKernel &operator=(const RegisteredKernel &) = delete;

private:
    std::vector<char> binary_;
    void *binary_handle_ = nullptr;
};

void ValidateShape(uint64_t group_count, uint64_t k, uint64_t n, uint64_t d)
{
    if (group_count == 0 || group_count > kMaxGroupCount) {
        throw std::invalid_argument("group count G must be in [1, 256]");
    }
    if (k == 0 || k % kKAlignment != 0) {
        throw std::invalid_argument("GateUp K must be a positive multiple of 64");
    }
    // Down K is N/2, so N must be a multiple of 128 rather than merely 64.
    if (n == 0 || n % 128 != 0) {
        throw std::invalid_argument("total Gate+Up width N must be a positive multiple of 128");
    }
    if (d == 0 || d % kZnN0 != 0) {
        throw std::invalid_argument("Down output width D must be a positive multiple of 32");
    }
}

bool HostRouteBit(const std::vector<uint32_t> &mask, uint64_t expert_id)
{
    return ((mask[ToHostSize(expert_id >> 5)] >> (expert_id & 31)) & 1U) != 0;
}

void CopyTilingBlob(std::vector<uint8_t> &destination, uint64_t offset,
                    uint64_t expected_bytes, const std::vector<uint8_t> &source,
                    const char *name)
{
    if (source.size() != ToHostSize(expected_bytes)) {
        throw std::logic_error(std::string(name) + " tiling has " +
                               std::to_string(source.size()) + " bytes; expected " +
                               std::to_string(expected_bytes));
    }
    std::memcpy(destination.data() + ToHostSize(offset), source.data(), source.size());
}

std::vector<uint8_t> BuildMixedTiling(uint64_t k, uint64_t n, uint64_t d,
                                      uint64_t stage_mask)
{
    std::vector<uint8_t> result(ToHostSize(kMixedTilingBytes), 0);
    const MixedTilingHeader header {
        kMixedTilingMagic,
        kMixedTilingVersion,
        kMixedGateW2TilingOffset,
        kMixedGateW4TilingOffset,
        kMixedDownW2TilingOffset,
        kMixedDownW4TilingOffset,
        kMixedTilingBytes,
        stage_mask,
    };
    std::memcpy(result.data(), &header, sizeof(header));
    CopyTilingBlob(result, kMixedGateW2TilingOffset, kMixedGateW2TilingBytes,
                   BuildGateW2TilingBlob(k, n), "Gate W2");
    CopyTilingBlob(result, kMixedGateW4TilingOffset, kMixedGateW4TilingBytes,
                   BuildGateW4TilingBlob(k, n), "Gate W4");
    CopyTilingBlob(result, kMixedDownW2TilingOffset, kMixedDownW2TilingBytes,
                   BuildDownW2TilingBlob(n / 2, d), "Down W2");
    CopyTilingBlob(result, kMixedDownW4TilingOffset, kMixedDownW4TilingBytes,
                   BuildDownW4TilingBlob(n / 2, d), "Down W4");
    return result;
}

void CopyHostToDevice(DeviceBuffer &destination, const void *source)
{
    CheckRt(rtMemcpy(destination.data(), destination.size(), source, destination.size(), RT_MEMCPY_HOST_TO_DEVICE),
            "rtMemcpy(H2D)");
}

void LaunchFusedPass(const RuntimeStream &stream, uint64_t group_count,
                     uint64_t k, uint64_t n, uint64_t d,
                     const DeviceBuffer &group_list, const DeviceBuffer &a,
                     const DeviceBuffer &a_scale, const DeviceBuffer &gateup_w,
                     const DeviceBuffer &gateup_table, const DeviceBuffer &down_w,
                     const DeviceBuffer &down_table, const DeviceBuffer &scale_upgate,
                     const DeviceBuffer &scale_down, const DeviceBuffer &is_expert_w4,
                     const DeviceBuffer &is_down_w4, DeviceBuffer &h,
                     DeviceBuffer &h_scale, DeviceBuffer &c,
                     const DeviceBuffer &tiling)
{
    MatFP4Args args {
        group_count, k, n, d, group_list.data(), a.data(), a_scale.data(),
        gateup_w.data(), gateup_table.data(), down_w.data(), down_table.data(),
        scale_upgate.data(), scale_down.data(),
        is_expert_w4.data(), is_down_w4.data(), h.data(), h_scale.data(),
        c.data(), tiling.data(),
    };
    // This is intentionally the only kernel launch in the complete GateUp +
    // SwiGLU/quant + Down operator path.
    CheckRt(rtKernelLaunch(reinterpret_cast<const void *>(kKernelName), kKernelCoreCount,
                           &args, sizeof(args), nullptr, stream.get()),
            "rtKernelLaunch fused mixed-W2/W4 MoE pass");
}

void RunTest(uint64_t group_count, uint64_t k, uint64_t n, uint64_t d,
             uint64_t warmup_cycles, uint64_t test_cycles)
{
    ValidateShape(group_count, k, n, d);
    if (test_cycles == 0) {
        throw std::invalid_argument("test_cycles must be at least 1");
    }

    const uint64_t group_list_bytes = CheckedProduct({group_count, sizeof(int64_t)});
    const uint64_t mask_words = CeilDiv(group_count, uint64_t {32});
    const uint64_t mask_bytes = CheckedProduct({mask_words, sizeof(uint32_t)});
    std::vector<int64_t> host_group_list(ToHostSize(group_count));
    std::vector<uint32_t> host_is_expert_w4(ToHostSize(mask_words), 0);
    std::vector<uint32_t> host_is_down_w4(ToHostSize(mask_words), 0);
    ReadExactFile("input/input_group_list.bin", host_group_list.data(), group_list_bytes);
    ReadExactFile("input/input_is_expert_w4.bin", host_is_expert_w4.data(), mask_bytes);
    ReadExactFile("input/input_is_down_w4.bin", host_is_down_w4.data(), mask_bytes);

    if ((group_count & 31) != 0) {
        const uint32_t valid = (uint32_t {1} << (group_count & 31)) - 1U;
        if ((host_is_expert_w4.back() & ~valid) != 0 ||
            (host_is_down_w4.back() & ~valid) != 0) {
            throw std::invalid_argument("unused high bits in route masks must be zero");
        }
    }

    uint64_t total_m = 0;
    uint64_t active_group_count = 0;
    uint64_t mode_22 = 0;
    uint64_t mode_24 = 0;
    uint64_t mode_44 = 0;
    uint64_t gateup_w_bytes = 0;
    uint64_t gateup_table_bytes = 0;
    uint64_t down_w_bytes = 0;
    uint64_t down_table_bytes = 0;
    uint64_t stage_mask = 0;
    const uint64_t hidden = n / 2;
    for (uint64_t group_id = 0; group_id < group_count; ++group_id) {
        const int64_t signed_group_m = host_group_list[ToHostSize(group_id)];
        if (signed_group_m < 0) {
            throw std::invalid_argument("every group_list[i] must be non-negative; group_list[" +
                                        std::to_string(group_id) + "]=" + std::to_string(signed_group_m));
        }
        const uint64_t group_m = static_cast<uint64_t>(signed_group_m);
        total_m = CheckedSum({total_m, group_m});
        active_group_count += group_m != 0;

        const bool expert_w4 = HostRouteBit(host_is_expert_w4, group_id);
        const bool down_w4 = HostRouteBit(host_is_down_w4, group_id);
        if (expert_w4 && !down_w4) {
            throw std::invalid_argument("invalid route 10 at expert " + std::to_string(group_id) +
                                        ": W4 GateUp requires W4 Down");
        }
        mode_22 += !expert_w4 && !down_w4;
        mode_24 += !expert_w4 && down_w4;
        mode_44 += expert_w4 && down_w4;
        if (group_m != 0) {
            stage_mask |= expert_w4 ? kMixedStageGateW4 : kMixedStageGateW2;
            stage_mask |= down_w4 ? kMixedStageDownW4 : kMixedStageDownW2;
        }

        gateup_w_bytes = CheckedSum({gateup_w_bytes,
            CheckedProduct({k, n}) / (expert_w4 ? 2 : 4)});
        gateup_table_bytes = CheckedSum({gateup_table_bytes,
            CheckedProduct({k, n}) / 32});
        down_w_bytes = CheckedSum({down_w_bytes,
            CheckedProduct({hidden, d}) / (down_w4 ? 2 : 4)});
        down_table_bytes = CheckedSum({down_table_bytes,
            CheckedProduct({hidden, d}) / 32});
    }
    if (total_m == 0) {
        throw std::invalid_argument("group_list must contain at least one token row");
    }

    const uint64_t a_bytes = CheckedProduct({total_m, k});
    const uint64_t a_scale_bytes = CheckedProduct({total_m, k}) / kMxScaleGroupSize;
    const uint64_t h_bytes = CheckedProduct({total_m, hidden});
    const uint64_t h_scale_bytes = h_bytes / kMxScaleGroupSize;
    const uint64_t c_bytes = CheckedProduct({total_m, d, sizeof(OutputStorage)});
    const uint64_t scale_upgate_bytes = CheckedProduct({group_count, n, sizeof(uint64_t)});
    const uint64_t scale_down_bytes = CheckedProduct({group_count, d, sizeof(uint64_t)});
    const std::vector<uint8_t> host_tiling = BuildMixedTiling(k, n, d, stage_mask);

    std::vector<uint8_t> host_a(ToHostSize(a_bytes));
    std::vector<uint8_t> host_a_scale(ToHostSize(a_scale_bytes));
    std::vector<uint8_t> host_gateup_w(ToHostSize(gateup_w_bytes));
    std::vector<uint8_t> host_gateup_table(ToHostSize(gateup_table_bytes));
    std::vector<uint8_t> host_down_w(ToHostSize(down_w_bytes));
    std::vector<uint8_t> host_down_table(ToHostSize(down_table_bytes));
    std::vector<uint64_t> host_scale_upgate(ToHostSize(group_count * n));
    std::vector<uint64_t> host_scale_down(ToHostSize(group_count * d));
    std::vector<uint8_t> host_h(ToHostSize(h_bytes), 0);
    std::vector<uint8_t> host_h_scale(ToHostSize(h_scale_bytes), 0);
    std::vector<OutputStorage> host_c(ToHostSize(CheckedProduct({total_m, d})), 0);

    ReadExactFile("input/input_a.bin", host_a.data(), a_bytes);
    ReadExactFile("input/input_a_scale.bin", host_a_scale.data(), a_scale_bytes);
    ReadExactFile("input/input_gateup_w.bin", host_gateup_w.data(), gateup_w_bytes);
    ReadExactFile("input/input_gateup_table.bin", host_gateup_table.data(), gateup_table_bytes);
    ReadExactFile("input/input_down_w.bin", host_down_w.data(), down_w_bytes);
    ReadExactFile("input/input_down_table.bin", host_down_table.data(), down_table_bytes);
    ReadExactFile("input/input_scale_upgate.bin", host_scale_upgate.data(), scale_upgate_bytes);
    ReadExactFile("input/input_scale_down.bin", host_scale_down.data(), scale_down_bytes);

    RegisteredKernel registered_kernel(kKernelName, kKernelBinary);
    DeviceBuffer device_group_list(group_list_bytes);
    DeviceBuffer device_a(a_bytes);
    DeviceBuffer device_gateup_w(gateup_w_bytes);
    DeviceBuffer device_h(h_bytes);
    DeviceBuffer device_h_scale(h_scale_bytes);
    DeviceBuffer device_down_w(down_w_bytes);
    DeviceBuffer device_c(c_bytes);
    DeviceBuffer device_a_scale(a_scale_bytes);
    DeviceBuffer device_gateup_table(gateup_table_bytes);
    DeviceBuffer device_down_table(down_table_bytes);
    DeviceBuffer device_scale_upgate(scale_upgate_bytes);
    DeviceBuffer device_scale_down(scale_down_bytes);
    DeviceBuffer device_is_expert_w4(mask_bytes);
    DeviceBuffer device_is_down_w4(mask_bytes);
    DeviceBuffer device_tiling(kMixedTilingBytes);
    RuntimeStream stream;

    CopyHostToDevice(device_group_list, host_group_list.data());
    CopyHostToDevice(device_a, host_a.data());
    CopyHostToDevice(device_a_scale, host_a_scale.data());
    CopyHostToDevice(device_gateup_w, host_gateup_w.data());
    CopyHostToDevice(device_gateup_table, host_gateup_table.data());
    CopyHostToDevice(device_down_w, host_down_w.data());
    CopyHostToDevice(device_down_table, host_down_table.data());
    CopyHostToDevice(device_scale_upgate, host_scale_upgate.data());
    CopyHostToDevice(device_scale_down, host_scale_down.data());
    CopyHostToDevice(device_is_expert_w4, host_is_expert_w4.data());
    CopyHostToDevice(device_is_down_w4, host_is_down_w4.data());
    CopyHostToDevice(device_h, host_h.data());
    CopyHostToDevice(device_h_scale, host_h_scale.data());
    CopyHostToDevice(device_c, host_c.data());
    CopyHostToDevice(device_tiling, host_tiling.data());

    for (uint64_t cycle = 0; cycle < warmup_cycles; ++cycle) {
        LaunchFusedPass(stream, group_count, k, n, d, device_group_list,
                        device_a, device_a_scale, device_gateup_w, device_gateup_table,
                        device_down_w, device_down_table, device_scale_upgate, device_scale_down, device_is_expert_w4,
                        device_is_down_w4, device_h, device_h_scale, device_c,
                        device_tiling);
    }
    stream.Synchronize("after warmup kernels");

    RuntimeEvent start_event;
    RuntimeEvent end_event;
    start_event.Record(stream);
    for (uint64_t cycle = 0; cycle < test_cycles; ++cycle) {
        CopyHostToDevice(device_a, host_a.data());
        CopyHostToDevice(device_a_scale, host_a_scale.data());
        CopyHostToDevice(device_gateup_w, host_gateup_w.data());
        CopyHostToDevice(device_gateup_table, host_gateup_table.data());
        CopyHostToDevice(device_down_w, host_down_w.data());
        CopyHostToDevice(device_down_table, host_down_table.data());
        CopyHostToDevice(device_scale_upgate, host_scale_upgate.data());
        CopyHostToDevice(device_scale_down, host_scale_down.data());
        LaunchFusedPass(stream, group_count, k, n, d, device_group_list,
                        device_a, device_a_scale, device_gateup_w, device_gateup_table,
                        device_down_w, device_down_table, device_scale_upgate, device_scale_down, device_is_expert_w4,
                        device_is_down_w4, device_h, device_h_scale, device_c,
                        device_tiling);
    }
    end_event.Record(stream);
    stream.Synchronize("after timed kernels");

    CheckRt(rtMemcpy(host_c.data(), c_bytes, device_c.data(), c_bytes,
                     RT_MEMCPY_DEVICE_TO_HOST), "rtMemcpy(D2H C)");
    if (mkdir("output", 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error("cannot create output directory: " + std::string(std::strerror(errno)));
    }
    if (!WriteFile("output/output_c.bin", host_c.data(), c_bytes)) {
        throw std::runtime_error("failed to write C output file");
    }

    const double elapsed_ms = static_cast<double>(start_event.ElapsedMillisecondsTo(end_event));
    const double pass_ms = elapsed_ms / static_cast<double>(test_cycles);
    const double operations = 2.0 * static_cast<double>(total_m) *
        (static_cast<double>(k) * static_cast<double>(n) +
         static_cast<double>(hidden) * static_cast<double>(d));
    const double throughput_tflops = operations / (pass_ms * 1.0e9);

    std::cout << "Mixed W2/W4 GateUp + SwiGLU/quant + Down run completed\n"
              << "  experts / active:     " << group_count << " / " << active_group_count << '\n'
              << "  routes 2+2/2+4/4+4:  " << mode_22 << "/" << mode_24 << "/" << mode_44 << '\n'
              << "  dimensions T,K,N,D:   " << total_m << "," << k << "," << n << "," << d << '\n'
              << "  hidden width:         " << hidden << '\n'
              << "  route-mask storage:   " << mask_bytes << " bytes each (1 bit/expert)\n"
              << "  active stage bitmap:  0x" << std::hex << stage_mask << std::dec << '\n'
              << "  Gate W/table bytes:   " << gateup_w_bytes << "/" << gateup_table_bytes << '\n'
              << "  Down W/table bytes:   " << down_w_bytes << "/" << down_table_bytes << '\n'
              << "  output C:             [" << total_m << "," << d << "] BF16, "
              << c_bytes << " bytes\n"
              << "  kernels per pass:     1\n"
              << "  warmup / timed:       " << warmup_cycles << " / " << test_cycles << '\n'
              << "  average pass:         " << pass_ms << " ms\n"
              << "  GEMM-equivalent rate: " << throughput_tflops
              << " TFLOP/s (SwiGLU/quant arithmetic excluded)\n";
}

uint64_t ParseUnsigned(const char *text, const char *name)
{
    try {
        size_t parsed = 0;
        const std::string value(text);
        if (value.empty() || value.front() == '-') {
            throw std::invalid_argument("negative or empty value");
        }
        const uint64_t result = std::stoull(value, &parsed);
        if (parsed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    }
}

void PrintUsage(const char *program)
{
    std::cerr << "Usage: " << program
              << " <G=groups,1..256> <K> <N=total Gate+Up width> <D=Down output width>"
              << " [device_id=0] [warmup_cycles=0] [test_cycles=1]\n";
}

}

int main(int argc, char *argv[])
{
    if (argc < 5 || argc > 8) {
        PrintUsage(argv[0]);
        return 2;
    }

    bool acl_initialized = false;
    bool device_selected = false;
    int32_t device_id = 0;

    try {
        const uint64_t group_count = ParseUnsigned(argv[1], "G");
        const uint64_t k = ParseUnsigned(argv[2], "K");
        const uint64_t n = ParseUnsigned(argv[3], "N");
        const uint64_t d = ParseUnsigned(argv[4], "D");
        const uint64_t raw_device_id = argc >= 6 ? ParseUnsigned(argv[5], "device_id") : 0;
        if (raw_device_id > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::invalid_argument("device_id is outside int32 range");
        }
        device_id = static_cast<int32_t>(raw_device_id);
        const uint64_t warmup_cycles = argc >= 7 ? ParseUnsigned(argv[6], "warmup_cycles") : 0;
        const uint64_t test_cycles = argc >= 8 ? ParseUnsigned(argv[7], "test_cycles") : 1;

        CheckAcl(aclInit(nullptr), "aclInit");
        acl_initialized = true;
        CheckAcl(aclrtSetDevice(device_id), "aclrtSetDevice");
        device_selected = true;

        RunTest(group_count, k, n, d, warmup_cycles, test_cycles);

        CheckAcl(aclrtResetDevice(device_id), "aclrtResetDevice");
        device_selected = false;
        CheckAcl(aclFinalize(), "aclFinalize");
        acl_initialized = false;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        if (device_selected) {
            (void)aclrtResetDevice(device_id);
        }
        if (acl_initialized) {
            (void)aclFinalize();
        }
        return 1;
    }
}
