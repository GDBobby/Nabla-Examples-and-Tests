#pragma shader_stage(compute)

#include <nbl/builtin/hlsl/workgroup/basic.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/workgroup/arithmetic.hlsl>
#include <nbl/builtin/hlsl/workgroup/scratch_size.hlsl>
#include "nbl/builtin/hlsl/central_limit_blur/common.hlsl"
// #include "descriptors.hlsl"

// #include "nbl/builtin/hlsl/central_limit_blur/box_blur.hlsl"

groupshared uint32_t3 tile[4][128]; // SUS ORDER

[[vk::binding( 0, 0 )]] Texture2D<nbl::hlsl::float32_t4> input;   // TODO: ALI
[[vk::binding( 1, 0 )]] RWTexture2D<nbl::hlsl::float32_t4> output;// TODO: ALI

[[vk::push_constant]]
nbl::hlsl::central_limit_blur::BoxBlurParams params;

[numthreads( 32, 1, 1 )]
void main(uint32_t3 GroupID: SV_GroupID, uint32_t3 GroupThreadID: SV_GroupThreadID)
{
  uint32_t filterOffset = (params.filterDim - 1) / 2;
  uint32_t3 texSize;
  input.GetDimensions( 0, texSize.x, texSize.y, texSize.z );
  uint32_t2 dims = texSize.xy; 
   
  uint32_t2 baseIndex = (GroupID.xy * uint32_t2(params.blockDim, 4) +
                            GroupThreadID.xy * uint32_t2(4, 1))
                  - uint32_t2(filterOffset, 0);

  for (uint32_t r = 0; r < 4; r++) {
    for (uint32_t c = 0; c < 4; c++) {
      uint32_t2 loadIndex = baseIndex + uint32_t2(c, r);
      if (params.flip != 0u) {
        loadIndex = loadIndex.yx;
      }
      
      //uint32_t2 loadIndexU = uint32_t2((float32_t2(loadIndex) + float32_t2(0.25, 0.25)) / float32_t2(dims));
      tile[r][4 * GroupThreadID.x + c] = input[loadIndex].rgb;
    }
  }
    
  AllMemoryBarrierWithGroupSync();

  for (uint32_t r = 0; r < 4; r++) {
    for (uint32_t c = 0; c < 4; c++) {
      uint32_t2 writeIndex = baseIndex + uint32_t2(c, r);
      if (params.flip != 0) {
        writeIndex = writeIndex.yx;
      }

      uint32_t center = 4 * GroupThreadID.x + c;
      if (center >= filterOffset &&
          center < 128 - filterOffset &&
          all(writeIndex < dims)) {
        float32_t3 acc = 0;
        for (uint32_t f = 0; f < params.filterDim; f++) {
          uint32_t i = center + f - filterOffset;
          acc = acc + (1.0 / float32_t(params.filterDim)) * tile[r][i];
          //acc = tile[r][i];
        }
        
        output[writeIndex] = float32_t4(acc, 1.0);
      }
    }
  }
}