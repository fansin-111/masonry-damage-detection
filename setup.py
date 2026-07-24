"""Build the CUDA-accelerated PointNet++ operators in place."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "pointnet2" / "_ext_src"
SOURCES = [str(path) for path in (SOURCE_ROOT / "src").glob("*.cpp")]
SOURCES += [str(path) for path in (SOURCE_ROOT / "src").glob("*.cu")]


setup(
    name="masonry-damage-pointnet2",
    version="1.0.0",
    packages=["pointnet2"],
    ext_modules=[
        CUDAExtension(
            name="pointnet2._ext",
            sources=SOURCES,
            include_dirs=[str(SOURCE_ROOT / "include")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
