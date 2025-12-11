"""
Install DGL from source for platforms where wheels are not available.
This script is called automatically during the build process.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DGL_VERSION = "v2.2.1"
DGL_REPO = "https://github.com/dmlc/dgl.git"


def check_dgl_installed():
    """Check if DGL is already installed."""
    try:
        import dgl  # type: ignore[import-untyped]

        if dgl.__version__.startswith(DGL_VERSION.replace("v", "")):
            print(f"DGL {dgl.__version__} is already installed. Skipping build.")
            return True
    except ImportError:
        pass
    return False


def check_build_requirements():
    """Check if required build tools are available."""
    if not shutil.which("cmake"):
        print("Error: cmake is required but not installed.")
        print("Install it with: brew install cmake")
        return False

    # Check for OpenMP on macOS
    if sys.platform == "darwin":
        # Check if libomp is installed via Homebrew
        libomp_paths = [
            Path("/opt/homebrew/opt/libomp"),  # Apple Silicon
            Path("/usr/local/opt/libomp"),  # Intel Mac
        ]

        libomp_found = False
        for libomp_path in libomp_paths:
            if libomp_path.exists():
                print(f"Found OpenMP at: {libomp_path}")
                libomp_found = True
                break

        if not libomp_found:
            print("Warning: OpenMP (libomp) is required for building DGL on macOS.")
            print("Install it with: brew install libomp")
            print("The script will check again during the build process.")

    return True


def build_dgl_from_source():
    """Build and install DGL from source."""
    # Check if already installed
    if check_dgl_installed():
        return 0

    # Check build requirements
    if not check_build_requirements():
        return 1

    # Determine build directory
    build_dir = os.environ.get("DGL_BUILD_DIR", tempfile.mkdtemp(prefix="dgl-build-"))
    build_path = Path(build_dir) / "dgl"

    print(f"Building DGL {DGL_VERSION} from source...")
    print(f"Build directory: {build_dir}")

    # Clone repository if needed
    if not build_path.exists():
        print("Cloning DGL repository...")
        subprocess.run(["git", "clone", "--recursive", "--branch", DGL_VERSION, DGL_REPO, str(build_path)], check=True)
    else:
        print("DGL repository already cloned. Updating...")
        subprocess.run(["git", "fetch"], cwd=build_path, check=True)
        subprocess.run(["git", "checkout", DGL_VERSION], cwd=build_path, check=True)
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=build_path, check=True)

    # Apply local patch for AppleClang std::min issue in parallel_for.h
    # See https://github.com/dmlc/dgl/issues/3631
    pf = build_path / "include" / "dgl" / "runtime" / "parallel_for.h"
    if pf.exists():
        text = pf.read_text()
        old = "auto end_tid = std::min(end, chunk_size + begin_tid);"
        new = "auto end_tid = std::min(end, static_cast<size_t>(chunk_size + begin_tid));"
        if old in text:
            print("Patching parallel_for.h for AppleClang std::min type mismatch...")
            pf.write_text(text.replace(old, new))
        else:
            print("Warning: expected std::min line not found in parallel_for.h; patch not applied.")
    else:
        print(f"Warning: parallel_for.h not found at {pf}; patch not applied.")

    # Patch dmlc-core CMakeLists.txt to fix CMake version compatibility issue
    # Modern CMake versions have removed compatibility with versions < 3.5
    dmlc_cmake = build_path / "third_party" / "dmlc-core" / "CMakeLists.txt"
    if dmlc_cmake.exists():
        text = dmlc_cmake.read_text()
        # Pattern to match cmake_minimum_required with old version
        pattern = r"cmake_minimum_required\s*\(\s*VERSION\s+[0-9]+\.[0-9]+(?:\.[0-9]+)?\s*\)"
        match = re.search(pattern, text)
        if match:
            # Check if version is less than 3.5
            version_match = re.search(r"VERSION\s+([0-9]+\.[0-9]+)", match.group())
            if version_match:
                version = float(version_match.group(1))
                if version < 3.5:
                    print("Patching dmlc-core CMakeLists.txt for CMake version compatibility...")
                    text = re.sub(pattern, "cmake_minimum_required(VERSION 3.5)", text)
                    dmlc_cmake.write_text(text)
                else:
                    print(f"dmlc-core CMakeLists.txt already requires CMake >= {version}")
        else:
            print("Warning: Could not find cmake_minimum_required in dmlc-core CMakeLists.txt")
    else:
        print(f"Warning: dmlc-core CMakeLists.txt not found at {dmlc_cmake}; patch not applied.")

    # Build C++ library
    print("Building DGL C++ library...")
    build_cpp_dir = build_path / "build"
    build_cpp_dir.mkdir(exist_ok=True)

    # Configure with CMake
    cmake_args = ["cmake", "..", "-DUSE_CUDA=OFF", "-DCMAKE_BUILD_TYPE=Release"]

    # Add OpenMP configuration for macOS
    env = os.environ.copy()
    if sys.platform == "darwin":
        # Try to find libomp from Homebrew
        libomp_paths = [
            Path("/opt/homebrew/opt/libomp"),  # Apple Silicon
            Path("/usr/local/opt/libomp"),  # Intel Mac
        ]

        libomp_path = None
        for path in libomp_paths:
            if path.exists():
                libomp_path = path
                break

        if libomp_path:
            print(f"Using OpenMP from: {libomp_path}")
            omp_include = f"-Xpreprocessor -fopenmp -I{libomp_path}/include"
            omp_lib = str(libomp_path / "lib" / "libomp.dylib")

            # Set CMake variables for OpenMP
            cmake_args.extend([
                f"-DOpenMP_C_FLAGS={omp_include}",
                "-DOpenMP_C_LIB_NAMES=omp",
                f"-DOpenMP_omp_LIBRARY={omp_lib}",
                f"-DOpenMP_CXX_FLAGS={omp_include}",
                "-DOpenMP_CXX_LIB_NAMES=omp",
                f"-DOpenMP_omp_CXX_LIBRARY={omp_lib}",
            ])

            # Set compiler flags in environment
            env["CFLAGS"] = f"{omp_include} {env.get('CFLAGS', '')}"
            env["CXXFLAGS"] = f"{omp_include} {env.get('CXXFLAGS', '')}"
            env["LDFLAGS"] = f"-L{libomp_path}/lib -lomp {env.get('LDFLAGS', '')}"
        else:
            print("Error: OpenMP (libomp) is required but not found.")
            print("Please install it with: brew install libomp")
            print("Then run this script again.")
            return 1

    subprocess.run(cmake_args, cwd=build_cpp_dir, check=True, env=env)

    # Build
    cpu_count = os.cpu_count() or 4
    # Use the same environment for the build step
    build_env = env if sys.platform == "darwin" else None
    subprocess.run(["cmake", "--build", ".", "-j", str(cpu_count)], cwd=build_cpp_dir, check=True, env=build_env)

    # Install Python package
    print("Installing DGL Python package...")
    python_dir = build_path / "python"

    # Use uv pip if available, otherwise try pip
    try:
        # Try using uv pip (preferred for uv environments)
        subprocess.run(
            ["uv", "pip", "install", "-e", ".", "--no-build-isolation", "--no-deps"],
            cwd=python_dir,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to python -m pip
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation", "--no-deps"],
            cwd=python_dir,
            check=True,
        )

    print(f"DGL {DGL_VERSION} has been successfully built and installed!")
    return 0


def main():
    """Main entry point for the install-dgl script."""
    sys.exit(build_dgl_from_source())


if __name__ == "__main__":
    main()
