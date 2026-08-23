// Op registration and input validation for the XeSS XPU correlation ops.
//
// The kernels live in gather_correlate.sycl (compiled by icx with -fsycl);
// this translation unit is plain C++ so it can be built by the MSVC host
// compiler that torch's BuildExtension drives on Windows.  Only the XPU
// dispatch key is implemented on purpose: calling these ops from any other
// backend raises a loud NotImplementedError instead of silently running a
// slower fallback path.

#include <torch/extension.h>

#include <vector>

at::Tensor xess_gather_correlate_forward(
    const at::Tensor& fmap1, const std::vector<at::Tensor>& fmap2_levels,
    const at::Tensor& coords, const at::Tensor& dilation,
    const std::vector<int64_t>& level_scales, int64_t radius);

at::Tensor xess_smoke_add_forward(const at::Tensor& input);

// Defined in gather_correlate.sycl next to the launcher that writes the
// counters; returns the cumulative {pure-staged, any-fallback} work-group
// counts used by tests and benchmarks (XESS_XPU_CORR_STATS=1).
at::Tensor xess_corr_stats_forward();
void xess_reset_corr_stats_forward();

namespace {

constexpr int64_t kMaxLevels = 8;

void check_xpu_f32_contiguous(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_xpu(), name, " must be an XPU tensor, got ",
                t.device().str());
    TORCH_CHECK(t.scalar_type() == at::kFloat, name,
                " must be float32, got ", t.scalar_type());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_gather_inputs(const at::Tensor& fmap1,
                         const std::vector<at::Tensor>& fmap2_levels,
                         const at::Tensor& coords,
                         const at::Tensor& dilation,
                         const std::vector<int64_t>& level_scales,
                         int64_t radius) {
    TORCH_CHECK(fmap1.dim() == 4, "fmap1 must be [B, C, H, W]");
    check_xpu_f32_contiguous(fmap1, "fmap1");

    const int64_t batch = fmap1.size(0);
    const int64_t channels = fmap1.size(1);
    const int64_t height = fmap1.size(2);
    const int64_t width = fmap1.size(3);

    TORCH_CHECK(radius >= 0, "radius must be non-negative, got ", radius);
    TORCH_CHECK(!fmap2_levels.empty(), "fmap2_levels must not be empty");
    TORCH_CHECK(fmap2_levels.size() == level_scales.size(),
                "fmap2_levels size (", fmap2_levels.size(),
                ") must match level_scales size (", level_scales.size(), ")");
    TORCH_CHECK(fmap2_levels.size() <= kMaxLevels, "at most ", kMaxLevels,
                " pyramid levels are supported, got ", fmap2_levels.size());

    for (size_t l = 0; l < fmap2_levels.size(); ++l) {
        const auto& level = fmap2_levels[l];
        TORCH_CHECK(level.dim() == 4, "fmap2_levels[", l,
                    "] must be [B, C, H2, W2]");
        check_xpu_f32_contiguous(level, "fmap2_levels entry");
        TORCH_CHECK(level.size(0) == batch && level.size(1) == channels,
                    "fmap2_levels[", l, "] shape (", level.size(0), ", ",
                    level.size(1), ") must match fmap1 batch/channels (",
                    batch, ", ", channels, ")");
        // build_fmap2_pyramid zero-pads every level to at least 2x2 because
        // align_corners normalisation divides by (size - 1); keep that
        // contract here so the kernel never divides by zero.
        TORCH_CHECK(level.size(2) >= 2 && level.size(3) >= 2, "fmap2_levels[",
                    l, "] must be at least 2x2 after padding, got (",
                    level.size(2), ", ", level.size(3), ")");
        TORCH_CHECK(level_scales[l] > 0, "level_scales[", l,
                    "] must be positive, got ", level_scales[l]);
    }

    TORCH_CHECK(coords.dim() == 4 && coords.size(0) == batch &&
                    coords.size(1) == 2 && coords.size(2) == height &&
                    coords.size(3) == width,
                "coords must be [B, 2, H, W] with B/H/W from fmap1");
    check_xpu_f32_contiguous(coords, "coords");

    TORCH_CHECK(dilation.dim() == 4 && dilation.size(0) == batch &&
                    dilation.size(1) == 1 && dilation.size(2) == height &&
                    dilation.size(3) == width,
                "dilation must be [B, 1, H, W] with B/H/W from fmap1");
    check_xpu_f32_contiguous(dilation, "dilation");
}

}  // namespace

static at::Tensor gather_correlate_pyramid_op(
    const at::Tensor& fmap1, const std::vector<at::Tensor>& fmap2_levels,
    const at::Tensor& coords, const at::Tensor& dilation,
    const std::vector<int64_t>& level_scales, int64_t radius) {
    check_gather_inputs(fmap1, fmap2_levels, coords, dilation, level_scales,
                        radius);
    return xess_gather_correlate_forward(fmap1, fmap2_levels, coords,
                                         dilation, level_scales, radius);
}

static at::Tensor smoke_add_op(const at::Tensor& input) {
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    check_xpu_f32_contiguous(input, "input");
    return xess_smoke_add_forward(input);
}

TORCH_LIBRARY(xess_xpu, m) {
    m.def("gather_correlate_pyramid(Tensor fmap1, Tensor[] fmap2_levels, "
          "Tensor coords, Tensor dilation, int[] level_scales, int radius)"
          " -> Tensor");
    m.def("smoke_add(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(xess_xpu, XPU, m) {
    m.impl("gather_correlate_pyramid", &gather_correlate_pyramid_op);
    m.impl("smoke_add", &smoke_add_op);
}

// The stats helpers take no tensor argument, so the PyTorch dispatcher could
// not route them to the XPU-only registration anyway; expose them straight
// through pybind as process-wide test/bench utilities.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("corr_stats",
          []() {
              const at::Tensor counts = xess_corr_stats_forward().cpu();
              return std::vector<int64_t>{
                  counts[0].item<int64_t>(), counts[1].item<int64_t>()};
          });
    m.def("reset_corr_stats",
          []() { xess_reset_corr_stats_forward(); });
}
