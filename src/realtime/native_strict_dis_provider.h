#pragma once
#include <memory>
#include <string>
#include <cstdint>

namespace xess_gpu {
class MotionProvider;
enum class StrictDisPreset { Fast, Medium };
enum class StrictDisInput { RawLuma8, Ffmpeg71RgbGrayLimited };
// Frozen CPU FAST/MEDIUM semantics, no caller-provided temporal initial flow.
// The caller owns producer waits, command submission, completion signaling,
// and source pool leases. NV12 Y and R8_UNORM crop views are supported.
std::unique_ptr<MotionProvider> make_strict_dis_provider(
    const std::string& shader_directory, StrictDisPreset preset=StrictDisPreset::Fast,
    StrictDisInput input=StrictDisInput::RawLuma8);
uint64_t strict_dis_shader_dispatches(const MotionProvider& provider) noexcept;
}
