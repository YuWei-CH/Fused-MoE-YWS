#pragma once

namespace tile_config
{

    // Current best-known baseline after the b_tile shared-memory layout rewrite.
    inline constexpr int kGemm1TileM = 16;
    inline constexpr int kGemm1TileN = 32;
    inline constexpr int kGemm1TileK = 64;

    inline constexpr int kGemm2TileM = 16;
    inline constexpr int kGemm2TileN = 32;
    inline constexpr int kGemm2TileK = 64;

    inline constexpr int kLargeGemm1TensorCoreThreshold = 780;
    inline constexpr int kLargeGemm2TensorCoreThreshold = 780;

} // namespace tile_config
