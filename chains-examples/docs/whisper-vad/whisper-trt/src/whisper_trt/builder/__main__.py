from .jit_build import build_engine

if __name__ == "__main__":
    # Call the build function from jit_build module
    new_engine = build_engine("large-v2")
    print(f"Built engine at: {new_engine}")
