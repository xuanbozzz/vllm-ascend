// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Diagnostic-only, single-launch reproduction of the staged Torch adapter.
// This file neither changes the imported kernel nor implements a model backend.

#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "../mixed_w2w4_moe_ffn/hacl_rt.h"
#include "../mixed_w2w4_moe_ffn/components/down_w2/down_w2_tiling_builder.h"
#include "../mixed_w2w4_moe_ffn/mixed_tiling.h"

namespace vq2a8_gs32_native {

constexpr char kKernelName[] = "mixed_w2w4_moe_ffn";
constexpr uint32_t kKernelCoreCount = 32;
constexpr uint64_t kMaxTokens = 128;
// Bounds for this diagnostic process, not claims about kernel shape support.
constexpr uint64_t kMaxDimension = 65536;
constexpr uint64_t kMaxCaseBytes = 1024ULL * 1024 * 1024;
constexpr uint64_t kMaxElfBytes = 256ULL * 1024 * 1024;
constexpr uintptr_t kRequiredAlignment = 32;
constexpr uint16_t kOutputSentinel = 0x7fc0;  // BF16 quiet NaN.

struct MatFP4Args {
  uint64_t group_count;
  uint64_t k;
  uint64_t n;
  uint64_t d;
  void* group_list;
  void* a;
  void* a_scale;
  void* gateup_w;
  void* gateup_table;
  void* down_w;
  void* down_table;
  void* scale_upgate;
  void* scale_down;
  void* is_expert_w4;
  void* is_down_w4;
  void* h;
  void* h_scale;
  void* c;
  void* tiling;
};

static_assert(sizeof(void*) == sizeof(uint64_t), "64-bit host ABI required");
static_assert(sizeof(MatFP4Args) == 152, "mixed kernel argument ABI mismatch");
static_assert(kMixedTilingBytes == 1600, "mixed tiling ABI mismatch");
static_assert(kMixedStageDownW2 == 4, "DownW2 stage bit mismatch");

[[noreturn]] void FatalDevice(const char* operation, int64_t code) {
  std::cerr << "GS32_NATIVE_EXECUTION=FAIL OP=" << operation << " CODE=" << code << '\n';
  std::cerr << "Device failure: exiting without rtFree, event recording, device reset, or ACL finalization.\n";
  std::cout.flush();
  std::cerr.flush();
  std::_Exit(EXIT_FAILURE);
}

void CheckRt(rtError_t code, const char* operation) {
  if (code != RT_ERROR_NONE) FatalDevice(operation, static_cast<int64_t>(code));
}

void CheckAcl(aclError code, const char* operation) {
  if (code != ACL_SUCCESS) FatalDevice(operation, static_cast<int64_t>(code));
}

void Stage(const char* name) {
  std::cout << "GS32_NATIVE_STAGE=" << name << std::endl;
}

uint64_t ParseUnsigned(const char* text, const char* name, uint64_t maximum) {
  const std::string value(text);
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument(std::string(name) + " must be an unsigned decimal integer");
  }
  size_t consumed = 0;
  const auto result = std::stoull(value, &consumed, 10);
  if (consumed != value.size() || result > maximum) {
    throw std::invalid_argument(std::string(name) + " exceeds diagnostic limit " + std::to_string(maximum));
  }
  return result;
}

uint64_t Product(uint64_t left, uint64_t right) {
  if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right) {
    throw std::overflow_error("diagnostic tensor size overflow");
  }
  return left * right;
}

void RequireAbsent(const std::string& path) {
  struct stat info {};
  if (lstat(path.c_str(), &info) == 0) {
    throw std::runtime_error("refusing to overwrite existing diagnostic artifact: " + path);
  }
  if (errno != ENOENT) throw std::runtime_error("cannot check diagnostic path: " + path);
}

std::vector<uint8_t> ReadExact(const std::string& path, uint64_t expected_bytes) {
  struct stat info {};
  if (stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0 ||
      static_cast<uint64_t>(info.st_size) != expected_bytes) {
    throw std::runtime_error("expected regular file of " + std::to_string(expected_bytes) + " bytes: " + path);
  }
  if (expected_bytes == 0 || expected_bytes > std::numeric_limits<size_t>::max() ||
      expected_bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("invalid input byte count: " + path);
  }
  std::ifstream file(path, std::ios::binary);
  std::vector<uint8_t> bytes(static_cast<size_t>(expected_bytes));
  file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!file || file.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("input file was unreadable or changed while reading: " + path);
  }
  return bytes;
}

std::vector<uint8_t> ReadElf(const std::string& path) {
  struct stat info {};
  if (stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 64 ||
      static_cast<uint64_t>(info.st_size) > kMaxElfBytes) {
    throw std::runtime_error("kernel must be a regular ELF file of 64 bytes to 256 MiB: " + path);
  }
  auto bytes = ReadExact(path, static_cast<uint64_t>(info.st_size));
  if (bytes[0] != 0x7f || bytes[1] != 'E' || bytes[2] != 'L' || bytes[3] != 'F' ||
      bytes[4] != 2 || bytes[5] != 1 || bytes[6] != 1) {
    throw std::runtime_error("kernel must be a 64-bit little-endian ELF object: " + path);
  }
  return bytes;
}

void WriteExclusive(const std::string& path, const void* data, size_t bytes) {
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (fd < 0) throw std::runtime_error("cannot exclusively create diagnostic artifact: " + path);
  const auto* source = static_cast<const uint8_t*>(data);
  size_t written = 0;
  while (written < bytes) {
    const auto count = write(fd, source + written, bytes - written);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      close(fd);
      throw std::runtime_error("partial diagnostic artifact retained after write failure: " + path);
    }
    written += static_cast<size_t>(count);
  }
  if (close(fd) != 0) throw std::runtime_error("failed to close diagnostic artifact: " + path);
}

std::vector<uint8_t> MakeTiling(uint64_t k, uint64_t n) {
  const auto blob = BuildDownW2TilingBlob(k, n);
  if (blob.size() != kMixedDownW2TilingBytes) throw std::runtime_error("DownW2 tiling size mismatch");
  std::vector<uint8_t> tiling(kMixedTilingBytes, 0);
  const MixedTilingHeader header{
      kMixedTilingMagic, kMixedTilingVersion, kMixedGateW2TilingOffset, kMixedGateW4TilingOffset,
      kMixedDownW2TilingOffset, kMixedDownW4TilingOffset, kMixedTilingBytes, kMixedStageDownW2};
  std::memcpy(tiling.data(), &header, sizeof(header));
  std::memcpy(tiling.data() + kMixedDownW2TilingOffset, blob.data(), blob.size());
  return tiling;
}

// Deliberately no destructor: after any runtime error only process exit is safe.
struct DeviceBuffer {
  const char* name;
  void* pointer;
  uint64_t bytes;
};

DeviceBuffer Upload(const char* name, const void* source, uint64_t bytes) {
  DeviceBuffer result{name, nullptr, bytes};
  CheckRt(rtMalloc(&result.pointer, bytes, RT_MEMORY_HBM), "rtMalloc");
  const auto begin = reinterpret_cast<uintptr_t>(result.pointer);
  if (begin == 0 || begin % kRequiredAlignment != 0 || bytes > std::numeric_limits<uintptr_t>::max() - begin) {
    FatalDevice("rtMalloc_invalid_pointer_range", -1);
  }
  CheckRt(rtMemcpy(result.pointer, bytes, source, bytes, RT_MEMCPY_HOST_TO_DEVICE), "rtMemcpy_H2D");
  return result;
}

void PrintRange(const DeviceBuffer& buffer) {
  const auto begin = reinterpret_cast<uintptr_t>(buffer.pointer);
  std::cout << "GS32_NATIVE_BUFFER=" << buffer.name << " BEGIN=0x" << std::hex << begin
            << " END_EXCLUSIVE=0x" << begin + buffer.bytes << std::dec << " BYTES=" << buffer.bytes << '\n';
}

void PrintArguments(const MatFP4Args& args, rtStream_t stream) {
  std::cout << "GS32_NATIVE_ARGS BYTES=" << sizeof(args) << " BLOCKS=" << kKernelCoreCount
            << " G=" << args.group_count << " K=" << args.k << " N=" << args.n << " D=" << args.d
            << " STAGE_MASK=" << kMixedStageDownW2 << " STREAM=" << stream << '\n';
  const char* names[] = {"group_list", "a", "a_scale", "gateup_w", "gateup_table", "down_w", "down_table",
                         "scale_upgate", "scale_down", "is_expert_w4", "is_down_w4", "h", "h_scale", "c", "tiling"};
  const void* pointers[] = {args.group_list, args.a, args.a_scale, args.gateup_w, args.gateup_table,
                            args.down_w, args.down_table, args.scale_upgate, args.scale_down, args.is_expert_w4,
                            args.is_down_w4, args.h, args.h_scale, args.c, args.tiling};
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
    std::cout << "GS32_NATIVE_ARG=" << names[i] << " OFFSET=" << (4 + i) * sizeof(uint64_t)
              << " POINTER=" << pointers[i] << '\n';
  }
  std::cout.flush();
}

int Run(int argc, char** argv) {
  if (argc != 7) {
    throw std::invalid_argument("usage: native_probe <kernel.o> <case_dir> <logical_device> <M> <K> <N>");
  }
  Stage("validate_inputs");
  const std::string kernel_path(argv[1]);
  const std::string directory(argv[2]);
  const auto device = static_cast<int32_t>(ParseUnsigned(argv[3], "logical_device", std::numeric_limits<int32_t>::max()));
  const uint64_t m = ParseUnsigned(argv[4], "M", kMaxTokens);
  const uint64_t k = ParseUnsigned(argv[5], "K", kMaxDimension);
  const uint64_t n = ParseUnsigned(argv[6], "N", kMaxDimension);
  if (m == 0 || k == 0 || k % 64 != 0 || n == 0 || n % 32 != 0) {
    throw std::invalid_argument("requires 1<=M<=128, positive K divisible by 64, positive N divisible by 32");
  }
  const uint16_t endian = 1;
  if (*reinterpret_cast<const uint8_t*>(&endian) != 1) throw std::runtime_error("little-endian host required");
  struct stat dir_info {};
  if (stat(directory.c_str(), &dir_info) != 0 || !S_ISDIR(dir_info.st_mode)) {
    throw std::runtime_error("case_dir must already exist and be a directory");
  }
  const std::string output_path = directory + "/output.bin";
  const std::string tiling_path = directory + "/tiling.bin";
  RequireAbsent(output_path);
  RequireAbsent(tiling_path);
  const uint64_t activation_bytes = Product(m, k);
  const uint64_t weight_bytes = Product(n, k) / 4;
  const uint64_t table_bytes = Product(n, k) / 32;
  const uint64_t row_scale_bytes = Product(n, sizeof(uint64_t));
  const uint64_t output_bytes = Product(Product(m, n), sizeof(uint16_t));
  const uint64_t case_bytes = activation_bytes + activation_bytes / 32 + weight_bytes + table_bytes + row_scale_bytes + output_bytes;
  if (case_bytes > kMaxCaseBytes) throw std::runtime_error("diagnostic tensors exceed 1 GiB budget");

  // Validate and retain every input before initializing any device context.
  const auto binary = ReadElf(kernel_path);
  const auto input = ReadExact(directory + "/input_bytes.bin", activation_bytes);
  const auto input_scale = ReadExact(directory + "/input_scale.bin", activation_bytes / 32);
  const auto weight = ReadExact(directory + "/packed_weight.bin", weight_bytes);
  const auto table = ReadExact(directory + "/pair_lut.bin", table_bytes);
  const auto row_scale = ReadExact(directory + "/row_scale.bin", row_scale_bytes);
  for (size_t offset = 0; offset < row_scale.size(); offset += sizeof(uint64_t)) {
    uint64_t bits = 0;
    std::memcpy(&bits, row_scale.data() + offset, sizeof(bits));
    if ((bits >> 32) != 0 || (bits & 0x7f800000ULL) == 0x7f800000ULL) {
      throw std::runtime_error("row_scale must contain finite FP32 bits in each uint64 low word, upper word zero");
    }
  }
  const auto tiling = MakeTiling(k, n);
  const int64_t group_list = static_cast<int64_t>(m);
  const int32_t route_mask = 0;
  std::vector<uint16_t> output(static_cast<size_t>(m * n), kOutputSentinel);
  WriteExclusive(tiling_path, tiling.data(), tiling.size());
  std::cout << "GS32_NATIVE_TILING=" << tiling_path << " BYTES=" << tiling.size() << std::endl;

  Stage("initialize_device");
  CheckAcl(aclInit(nullptr), "aclInit");
  CheckAcl(aclrtSetDevice(device), "aclrtSetDevice");
  const char* soc = aclrtGetSocName();
  if (soc == nullptr || std::strncmp(soc, "Ascend950", 9) != 0) FatalDevice("requires_Ascend950", -1);
  std::cout << "GS32_NATIVE_DEVICE=" << device << " SOC=" << soc << std::endl;
  rtStream_t stream = nullptr;
  CheckRt(rtStreamCreate(&stream, 0), "rtStreamCreate");

  Stage("register_kernel");
  rtDevBinary_t descriptor{};
  descriptor.magic = RT_DEV_BINARY_MAGIC_ELF;
  descriptor.version = 0;
  descriptor.data = binary.data();
  descriptor.length = binary.size();
  void* binary_handle = nullptr;
  CheckRt(rtDevBinaryRegister(&descriptor, &binary_handle), "rtDevBinaryRegister");
  if (binary_handle == nullptr) FatalDevice("null_binary_handle", -1);
  const std::string kernel_name(kKernelName);
  const void* stub = static_cast<const void*>(kernel_name.c_str());
  CheckRt(rtFunctionRegister(binary_handle, stub, kernel_name.c_str(), const_cast<void*>(stub), 0), "rtFunctionRegister");

  Stage("upload_exact_buffers");
  const auto d_input = Upload("input_bytes", input.data(), input.size());
  const auto d_input_scale = Upload("input_scale", input_scale.data(), input_scale.size());
  const auto d_weight = Upload("packed_weight", weight.data(), weight.size());
  const auto d_table = Upload("pair_lut", table.data(), table.size());
  const auto d_row_scale = Upload("row_scale", row_scale.data(), row_scale.size());
  const auto d_output = Upload("output", output.data(), output_bytes);
  const auto d_group = Upload("group_list", &group_list, sizeof(group_list));
  const auto d_mask = Upload("route_mask", &route_mask, sizeof(route_mask));
  const auto d_tiling = Upload("tiling", tiling.data(), tiling.size());
  const DeviceBuffer buffers[] = {d_input, d_input_scale, d_weight, d_table, d_row_scale, d_output, d_group, d_mask, d_tiling};
  for (const auto& buffer : buffers) PrintRange(buffer);

  // Match the Torch bridge exactly, including aliases for skipped Gate stages.
  MatFP4Args args{1, k, 2 * k, n, d_group.pointer, d_input.pointer, d_input_scale.pointer,
                 d_weight.pointer, d_table.pointer, d_weight.pointer, d_table.pointer,
                 d_row_scale.pointer, d_row_scale.pointer, d_mask.pointer, d_mask.pointer,
                 d_input.pointer, d_input_scale.pointer, d_output.pointer, d_tiling.pointer};
  PrintArguments(args, stream);
  Stage("launch_once");
  CheckRt(rtKernelLaunch(stub, kKernelCoreCount, &args, sizeof(args), nullptr, stream), "rtKernelLaunch");
  Stage("synchronize_once");
  CheckRt(rtStreamSynchronize(stream), "rtStreamSynchronize");
  Stage("copy_output");
  CheckRt(rtMemcpy(output.data(), output_bytes, d_output.pointer, output_bytes, RT_MEMCPY_DEVICE_TO_HOST), "rtMemcpy_D2H");
  WriteExclusive(output_path, output.data(), static_cast<size_t>(output_bytes));
  std::cout << "GS32_NATIVE_OUTPUT=" << output_path << " BYTES=" << output_bytes << std::endl;

  Stage("normal_cleanup");
  for (const auto& buffer : buffers) CheckRt(rtFree(buffer.pointer), "rtFree");
  CheckRt(rtStreamDestroy(stream), "rtStreamDestroy");
  // No device reset: this diagnostic must not reset another process's device.
  CheckAcl(aclFinalize(), "aclFinalize");
  std::cout << "GS32_NATIVE_NUMERICAL_VALIDATION=NOT_RUN\nGS32_NATIVE_EXECUTION=PASS" << std::endl;
  return EXIT_SUCCESS;
}

}  // namespace vq2a8_gs32_native

int main(int argc, char** argv) {
  try {
    return vq2a8_gs32_native::Run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "GS32_NATIVE_EXECUTION=FAIL ERROR=" << error.what() << '\n';
    std::cout.flush();
    std::cerr.flush();
    // Device-owning objects intentionally have no destructors. Exit also avoids
    // runtime finalizers if a host/file error occurred after device startup.
    std::_Exit(EXIT_FAILURE);
  }
}
