"""Build script for the XeSS XPU fused gather-correlate extension.

Produces ``xess_xpu_corr`` (``xess_xpu_corr.pyd`` on Windows) in this
directory when run with::

    python setup.py build_ext --inplace

Prerequisites (see ``build.cmd`` next to this file for environment setup):
  * Visual Studio Build Tools with the MSVC toolset (host compiler),
  * Intel oneAPI DPC++ Compiler providing ``icx.exe`` on PATH,
  * Ninja (BuildExtension refuses to build SYCL extensions without it),
  * the PyTorch-XPU environment that will load the binary at runtime.

By default the SYCL device image targets every architecture reported by
``torch.xpu.get_arch_list()`` (AOT via spir64_gen) plus generic ``spir64``
JIT.  Set ``TORCH_XPU_ARCH_LIST`` (comma separated, e.g. ``bmg,dg2``) to
override; leaving it unset keeps the shipped binaries portable across Arc
generations.
"""

from setuptools import setup

import torch.utils.cpp_extension as _cpp_ext

# Workaround for a Windows bug in torch's SYCL host-flag wrapping:
# _wrap_sycl_host_flags converts "-I"/"-D" with str.replace(), which also
# rewrites those substrings *inside* paths.  Any environment whose include
# directories contain "-I" (e.g. ".../ComfyUI-aki-v3-IntelArc/...") gets its
# host-compiler include paths corrupted and the build dies with C1083.  Patch
# it to translate only leading prefixes, preserving upstream quoting exactly.
if _cpp_ext.IS_WINDOWS:
    def _wrap_sycl_host_flags_fixed(cflags):
        host_cxx = _cpp_ext.get_cxx_compiler()
        host_cflags = []
        for flag in cflags:
            if flag.startswith("-I"):
                flag = "/I" + flag.replace("\\", "\\\\")[2:]
            elif flag.startswith("-D"):
                flag = "/D" + flag[2:]
            flag = flag.replace('"', '\\"')
            host_cflags.append(flag)
        joined_host_cflags = " ".join(host_cflags)
        external_include = _cpp_ext._join_sycl_home("include").replace(
            "\\", "\\\\")
        return [
            f"-fsycl-host-compiler={host_cxx}",
            f'-fsycl-host-compiler-options="\\"/external:I{external_include}'
            f'\\" /external:W0 {joined_host_cflags}"',
        ]

    _cpp_ext._wrap_sycl_host_flags = _wrap_sycl_host_flags_fixed


from torch.utils.cpp_extension import BuildExtension, SyclExtension  # noqa: E402

# BuildExtension requires both keys when a dict is passed; the 'sycl' entry
# receives the automatic -fsycl-targets/-Xs arch flags on top of these.
EXTRA_COMPILE_ARGS = {
    "cxx": [],
    "sycl": ["-O2"],
}

setup(
    name="xess_xpu_corr",
    version="0.1.0",
    description="Fused SEA-RAFT gather-correlate forward kernel for PyTorch XPU",
    ext_modules=[
        SyclExtension(
            name="xess_xpu_corr",
            sources=["register_ops.cpp", "gather_correlate.sycl"],
            extra_compile_args=EXTRA_COMPILE_ARGS,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=[],
)
