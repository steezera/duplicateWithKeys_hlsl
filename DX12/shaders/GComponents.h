#pragma once
#include "Components.h"
#include "Common/Backend/GBackendDevice.h"

namespace vz
{
	struct CORE_EXPORT GGeometryComponent : GeometryComponent
	{
		GGeometryComponent(const Entity entity, const VUID vuid = 0) : GeometryComponent(entity, vuid) {}

		struct GaussianSplattingBuffers
		{
			// gaussian kernels
			graphics::GPUBuffer gaussianSHs;
			graphics::GPUBuffer gaussianScale_Opacities;
			graphics::GPUBuffer gaussianQuaterinions;

			// inter-processing buffers
			graphics::GPUBuffer touchedTiles_0;
			graphics::GPUBuffer offsetTiles_0;
			graphics::GPUBuffer offsetTilesPing; // Ping buffer

			graphics::GPUBuffer duplicatedDepthGaussians; // new version
			//graphics::GPUBuffer duplicatedTileDepthGaussians_0;
			graphics::GPUBuffer duplicatedIdGaussians;

			bool IsValid() const { return gaussianScale_Opacities.IsValid(); }
		};
		struct BVHBuffers
		{
			// https://github.com/ToruNiina/lbvh
			// Scene BVH intersection resources:
			graphics::GPUBuffer bvhNodeBuffer;
			graphics::GPUBuffer bvhParentBuffer;
			graphics::GPUBuffer bvhFlagBuffer;
			graphics::GPUBuffer primitiveCounterBuffer;
			graphics::GPUBuffer primitiveIDBuffer;
			graphics::GPUBuffer primitiveBuffer;
			graphics::GPUBuffer primitiveMortonBuffer;
			uint32_t primitiveCapacity = 0;
			bool IsValid() const { return primitiveCounterBuffer.IsValid(); }
		};
		struct GPrimBuffers
		{
			uint32_t slot = 0;

			graphics::GPUBuffer generalBuffer; // index buffer + all static vertex buffers
			graphics::GPUBuffer streamoutBuffer; // all dynamic vertex buffers

			BufferView ib;
			BufferView vbPosW;
			BufferView vbNormal;
			BufferView vbTangent;
			BufferView vbUVs;
			BufferView vbColor;

			// 'so' refers to Stream-Output:
			//		useful when the mesh undergoes dynamic changes, 
			//		such as in real-time physics simulations, deformations, or 
			//		when the normals are affected by geometry shaders or other GPU-side processes.
			BufferView soPosW;
			BufferView soNormal;
			BufferView soTangent;
			BufferView soPre;

			BVHBuffers bvhBuffers;
			GaussianSplattingBuffers gaussianSplattingBuffers;

			void Destroy()
			{
				generalBuffer = {};
				streamoutBuffer = {};

				bvhBuffers = {};
				gaussianSplattingBuffers = {};

				// buffer views
				ib = {};
				vbPosW = {};
				vbTangent = {};
				vbNormal = {};
				vbUVs = {};
				vbColor = {};

				soPosW = {};
				soNormal = {};
				soTangent = {};
				soPre = {};
			}
		};
    }
}