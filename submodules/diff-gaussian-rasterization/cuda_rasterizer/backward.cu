/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 *means, glm::vec3 campos, const float *shs, const bool *clamped, const glm::vec3 *dL_dcolor, glm::vec3 *dL_dmeans, glm::vec3 *dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3 *dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy +
						   SH_C3[1] * sh[10] * yz +
						   SH_C3[2] * sh[11] * -2.f * xy +
						   SH_C3[3] * sh[12] * -3.f * 2.f * xz +
						   SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
						   SH_C3[5] * sh[14] * 2.f * xz +
						   SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (SH_C3[0] * sh[9] * 3.f * (xx - yy) +
						   SH_C3[1] * sh[10] * xz +
						   SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
						   SH_C3[3] * sh[12] * -3.f * 2.f * yz +
						   SH_C3[4] * sh[13] * -2.f * xy +
						   SH_C3[5] * sh[14] * -2.f * yz +
						   SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (SH_C3[1] * sh[10] * xy +
						   SH_C3[2] * sh[11] * 4.f * 2.f * yz +
						   SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
						   SH_C3[4] * sh[13] * 4.f * 2.f * xz +
						   SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z}, float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

__device__ void computeASTuv(const glm::vec3 scale,
							 const float mod,
							 const glm::vec4 rot,
							 const float3 p_k,
							 const float *view_matrix,
							 const float *dL_dA,
							 glm::vec3 *dL_dmeans,
							 glm::vec3 *dL_dscales,
							 glm::vec4 *dL_drots)
{
	glm::vec4 q = rot; // / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 tuvw = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

	glm::mat3 R = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	float dL_dsu = 0;
	float dL_dsv = 0;
	float dL_dtu0 = 0;
	float dL_dtu1 = 0;
	float dL_dtu2 = 0;
	float dL_dtv0 = 0;
	float dL_dtv1 = 0;
	float dL_dtv2 = 0;

	// const float dA0_dsu = R[0][0] * tuvw[0][0] + R[0][1] * tuvw[1][0] + R[0][2] * tuvw[2][0];
	// const float dA0_dtu0 = R[0][0] * scale.x;
	// const float dA0_dtu1 = R[0][1] * scale.x;
	// const float dA0_dtu2 = R[0][2] * scale.x;

	float dAi_dsu = 0;
	float dAi_dtu0 = 0;
	float dAi_dtu1 = 0;
	float dAi_dtu2 = 0;

	for (int i = 0; i < 3; i++)
	{
		dAi_dsu = R[i][0] * tuvw[0][0] + R[i][1] * tuvw[1][0] + R[i][2] * tuvw[2][0];
		dAi_dsu *= mod;
		dAi_dtu0 = R[i][0] * scale.x * mod;
		dAi_dtu1 = R[i][1] * scale.x * mod;
		dAi_dtu2 = R[i][2] * scale.x * mod;

		dL_dsu += dAi_dsu * dL_dA[i * 3];
		dL_dtu0 += dAi_dtu0 * dL_dA[i * 3];
		dL_dtu1 += dAi_dtu1 * dL_dA[i * 3];
		dL_dtu2 += dAi_dtu2 * dL_dA[i * 3];
	}

	float dAi_dsv = 0;
	float dAi_dtv0 = 0;
	float dAi_dtv1 = 0;
	float dAi_dtv2 = 0;

	for (int i = 0; i < 3; i++)
	{
		dAi_dsv = R[i][0] * tuvw[0][1] + R[i][1] * tuvw[1][1] + R[i][2] * tuvw[2][1];
		dAi_dsv *= mod;
		dAi_dtv0 = R[i][0] * scale.y * mod;
		dAi_dtv1 = R[i][1] * scale.y * mod;
		dAi_dtv2 = R[i][2] * scale.y * mod;

		dL_dsv += dAi_dsv * dL_dA[i * 3 + 1];
		dL_dtv0 += dAi_dtv0 * dL_dA[i * 3 + 1];
		dL_dtv1 += dAi_dtv1 * dL_dA[i * 3 + 1];
		dL_dtv2 += dAi_dtv2 * dL_dA[i * 3 + 1];
	}

	float dL_dp0 = dL_dA[2] * R[0][0] + dL_dA[5] * R[1][0] + dL_dA[8] * R[2][0];
	float dL_dp1 = dL_dA[2] * R[0][1] + dL_dA[5] * R[1][1] + dL_dA[8] * R[2][1];
	float dL_dp2 = dL_dA[2] * R[0][2] + dL_dA[5] * R[1][2] + dL_dA[8] * R[2][2];

	// compute gradient through quaternion
	const float dL_dr = 2.0f * (z * (dL_dtu1 - dL_dtv0) - y * dL_dtu2 + x * dL_dtv2);
	const float dL_dx = 2.0f * (y * (dL_dtu1 + dL_dtv0) + z * dL_dtu2 + r * dL_dtv2 - 2.0f * x * dL_dtv1);
	const float dL_dy = 2.0f * (x * (dL_dtu1 + dL_dtv0) - r * dL_dtu2 + z * dL_dtv2 - 2.0f * y * dL_dtu0);
	const float dL_dz = 2.0f * (r * (dL_dtu1 - dL_dtv0) + x * dL_dtu2 + y * dL_dtv2 - 2.0f * z * (dL_dtu0 + dL_dtv1));

	dL_dmeans->x += dL_dp0;
	dL_dmeans->y += dL_dp1;
	dL_dmeans->z += dL_dp2;

	dL_dscales->x += dL_dsu;
	dL_dscales->y += dL_dsv;
	// dL_dscales->z += 0;

	dL_drots->x += dL_dr;
	dL_drots->y += dL_dx;
	dL_drots->z += dL_dy;
	dL_drots->w += dL_dz;
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template <int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3 *means,
	const int *radii,
	const float *shs,
	const bool *clamped,
	const glm::vec3 *scales,
	const glm::vec4 *rotations,
	const float scale_modifier,
	const float *view,
	const float *proj,
	const glm::vec3 *campos,
	const float *dL_dA,
	const float2 *dL_dc_margin,
	const float3 *dL_dmean2D,
	glm::vec3 *dL_dmeans,
	float *dL_dcolor,
	float *dL_ddepth,
	float *dL_dcov3D,
	float *dL_dsh,
	glm::vec3 *dL_dscale,
	glm::vec4 *dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	glm::vec3 dL_dmean;
	const float px = view[0] * m.x + view[4] * m.y + view[8] * m.z + view[12];
	const float py = view[1] * m.x + view[5] * m.y + view[9] * m.z + view[13];
	const float pz = view[2] * m.x + view[6] * m.y + view[10] * m.z + view[14];
	const float dpx_dPx = view[0] / pz - (px * view[2]) / (pz * pz);
	const float dpx_dPy = view[4] / pz - (px * view[6]) / (pz * pz);
	const float dpx_dPz = view[8] / pz - (px * view[10]) / (pz * pz);
	const float dpy_dPx = view[1] / pz - (py * view[2]) / (pz * pz);
	const float dpy_dPy = view[5] / pz - (py * view[6]) / (pz * pz);
	const float dpy_dPz = view[9] / pz - (py * view[10]) / (pz * pz);

	dL_dmean.x = dpx_dPx * dL_dc_margin[idx].x + dpy_dPx * dL_dc_margin[idx].y;
	dL_dmean.y = dpx_dPy * dL_dc_margin[idx].x + dpy_dPy * dL_dc_margin[idx].y;
	dL_dmean.z = dpx_dPz * dL_dc_margin[idx].x + dpy_dPz * dL_dc_margin[idx].y;

	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3 *)means, *campos, shs, clamped, (glm::vec3 *)dL_dcolor, (glm::vec3 *)dL_dmeans, (glm::vec3 *)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeASTuv(scales[idx], scale_modifier, rotations[idx], m, view, dL_dA + idx * 9, dL_dmeans + idx, dL_dscale + idx, dL_drot + idx);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderCUDA(
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		const float *__restrict__ orig_points,
		const float *viewmatrix,
		int W, int H,
		const float focal_x, const float focal_y,
		const float *__restrict__ bg_color,
		const float2 *__restrict__ points_xy_image,
		const float4 *__restrict__ conic_opacity,
		const float *__restrict__ colors,
		const float *__restrict__ depths,
		const float *__restrict__ alphas,
		const float *__restrict__ STuv,
		const glm::vec3 *__restrict__ scale,
		const glm::vec4 *__restrict__ rotation,
		const float *__restrict__ A,
		const float *__restrict__ ray_R,
		const float *__restrict__ ray_S,
		const uint32_t *__restrict__ n_contrib,
		const float *__restrict__ dL_dpixels,
		const float *__restrict__ dL_dpixel_depths,
		const float *__restrict__ dL_dalphas,
		float3 *__restrict__ dL_dmean2D,
		float4 *__restrict__ dL_dconic2D,
		float *__restrict__ dL_dopacity,
		float *__restrict__ dL_dcolors,
		float *__restrict__ dL_ddepths,
		float *__restrict__ dL_dA,
		float2 *__restrict__ dL_dc_margin,
		glm::vec3 *__restrict__ dL_dmeans3D,
		glm::vec3 *__restrict__ dL_dscale,
		glm::vec4 *__restrict__ dL_drot,
		float *__restrict__ Ld_value)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	float2 pix_cam = {(pixf.x - (W - 1) * 0.5f) / focal_x, (pixf.y - (H - 1) * 0.5f) / focal_y};

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ uint32_t collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float collected_A[BLOCK_SIZE * 9];
	__shared__ float collected_STuv[BLOCK_SIZE * 6];
	__shared__ float collected_origin[BLOCK_SIZE * 3];
	__shared__ glm::vec3 collected_scale[BLOCK_SIZE];
	__shared__ glm::vec4 collected_rotation[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors.
	const float T_final = inside ? (1 - alphas[pix_id]) : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = {0};
	float dL_dpixel[C];
	float accum_depth_rec = 0;
	float dL_dpixel_depth;
	float accum_alpha_rec = 0;
	float dL_dalpha;

	float3 intersect_w;
	float3 intersect_c;
	
	float dL_dz;

	if (inside)
	{
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_dpixel_depth = dL_dpixel_depths[pix_id];
		dL_dalpha = dL_dalphas[pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = {0};
	float last_depth = 0;

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	float P_acc = 0.0f;
	float Q_acc = 0.0f;
	float R_acc = ray_R[pix_id];
	float S_acc = ray_S[pix_id];

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const uint32_t coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			collected_depths[block.thread_rank()] = depths[coll_id];
			for (int j = 0; j < 9; j++)
				collected_A[block.thread_rank() * 9 + j] = A[coll_id * 9 + j];
			for (int j = 0; j < 6; j++)
				collected_STuv[block.thread_rank() * 6 + j] = STuv[coll_id * 6 + j];
			for (int j = 0; j < 3; j++)
				collected_origin[block.thread_rank() * 3 + j] = orig_points[coll_id * 3 + j];
			collected_scale[block.thread_rank()] = scale[coll_id];
			collected_rotation[block.thread_rank()] = rotation[coll_id];
			
		}
		block.sync();

		// Iterate over Gaussians, current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			const float4 con_o = collected_conic_opacity[j];
			

			float hu_1 = -1.0f * collected_A[j * 9 + 0] + pix_cam.x * collected_A[j * 9 + 6];
			float hu_2 = -1.0f * collected_A[j * 9 + 1] + pix_cam.x * collected_A[j * 9 + 7];
			float hu_4 = -1.0f * collected_A[j * 9 + 2] + pix_cam.x * collected_A[j * 9 + 8];

			float hv_1 = -1.0f * collected_A[j * 9 + 3] + pix_cam.y * collected_A[j * 9 + 6];
			float hv_2 = -1.0f * collected_A[j * 9 + 4] + pix_cam.y * collected_A[j * 9 + 7];
			float hv_4 = -1.0f * collected_A[j * 9 + 5] + pix_cam.y * collected_A[j * 9 + 8];

			const float Denom = hu_1 * hv_2 - hu_2 * hv_1;

			float u = (hu_2 * hv_4 - hu_4 * hv_2) / Denom;
			float v = (hu_4 * hv_1 - hu_1 * hv_4) / Denom;

			float G_u = exp(-0.5f * (u * u + v * v));

			float G_xc = exp(-0.5f * (d.x * d.x + d.y * d.y) * 2.0f);

			float G_hat = max(G_u, G_xc);
			// G_u = G_xc;

			intersect_w = {collected_origin[j * 3 + 0] + collected_STuv[j * 6 + 0] * u + collected_STuv[j * 6 + 3] * v,
						   collected_origin[j * 3 + 1] + collected_STuv[j * 6 + 1] * u + collected_STuv[j * 6 + 4] * v,
						   collected_origin[j * 3 + 2] + collected_STuv[j * 6 + 2] * u + collected_STuv[j * 6 + 5] * v};
			intersect_c = transformPoint4x3(intersect_w, viewmatrix);

			float alpha = min(0.99f, con_o.w * G_hat);

			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float dpixel_depth_ddepth = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dopa = 0.0f;
			const uint32_t global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dopa += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian.
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			// Propagate gradients from pixel depth to opacity
			const float c_d = collected_depths[j];
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dopa += (c_d - accum_depth_rec) * dL_dpixel_depth;
			atomicAdd(&(dL_ddepths[global_id]), dpixel_depth_ddepth * dL_dpixel_depth);

			// Propagate gradients from pixel alpha (weights_sum) to opacity
			accum_alpha_rec = last_alpha + (1.f - last_alpha) * accum_alpha_rec;
			dL_dopa += (1 - accum_alpha_rec) * dL_dalpha; //- (alpha - accum_alpha_rec) * dL_dalpha;

			dL_dopa *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dopa += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			const float dG_du = -u * G_hat;
			const float dG_dv = -v * G_hat;

			const float du_dhu1 = -u * hv_2 / Denom;
			const float du_dhu2 = -v * hv_2 / Denom;
			const float du_dhu3 = -hv_2 / Denom;

			const float du_dhv1 = u * hu_2 / Denom;
			const float du_dhv2 = v * hu_2 / Denom;
			const float du_dhv3 = hu_2 / Denom;

			const float dv_dhu1 = u * hv_1 / Denom;
			const float dv_dhu2 = v * hv_1 / Denom;
			const float dv_dhu3 = hv_1 / Denom;

			const float dv_dhv1 = -u * hu_1 / Denom;
			const float dv_dhv2 = -v * hu_1 / Denom;
			const float dv_dhv3 = -hu_1 / Denom;

			// A is AT here
			glm::mat3 dL_dA_local = glm::mat3(0.0f);


#ifdef Ld
			// weight: macro Wd

			//dL/domega = dL/dopa
			const float dLd_domega = Wd * (P_acc - Q_acc * intersect_c.z + S_acc * intersect_c.z - R_acc);
			dL_dopa +=  dLd_domega;
			// atomicAdd(Ld_value, dLd_domega);


			// dz/dp

			dL_dz = Wd * alpha * (S_acc - Q_acc);

			// dL/dp
			// viewmatrix[2, 0], coloumn major
			const float dz_dp0 = viewmatrix[8];
			const float dz_dp1 = viewmatrix[9];
			const float dz_dp2 = viewmatrix[10];

			atomicAdd(&dL_dmeans3D[global_id].x, dL_dz * dz_dp0);
			atomicAdd(&dL_dmeans3D[global_id].y, dL_dz * dz_dp1);
			atomicAdd(&dL_dmeans3D[global_id].z, dL_dz * dz_dp2);

			// dz/ds

			float dz_dsu = 0; // ti =sti / s
			float dz_dsv = 0;
			for (int k = 0; k < 3; k++)
			{
				dz_dsu += viewmatrix[8+k] * collected_STuv[j * 6 + k] * u / collected_scale[j].x;
				dz_dsv += viewmatrix[8+k] * collected_STuv[j * 6 + k + 3] * v / collected_scale[j].y;
			}

			atomicAdd(&dL_dscale[global_id].x, dL_dz * dz_dsu);
			atomicAdd(&dL_dscale[global_id].y, dL_dz * dz_dsv);

			// dz/dt

			float dL_dtu0 = dL_dz * viewmatrix[8] * collected_scale[j].x * u;
			float dL_dtu1 = dL_dz * viewmatrix[9] * collected_scale[j].x * u;
			float dL_dtu2 = dL_dz * viewmatrix[10] * collected_scale[j].x * u;
			float dL_dtv0 = dL_dz * viewmatrix[8] * collected_scale[j].y * v;
			float dL_dtv1 = dL_dz * viewmatrix[9] * collected_scale[j].y * v;
			float dL_dtv2 = dL_dz * viewmatrix[10] * collected_scale[j].y * v;

			// compute gradient through quaternion
			glm::vec4 q = collected_rotation[j]; // / glm::length(rot);
			float r = q.x;
			float x = q.y;
			float y = q.z;
			float z = q.w;
			const float dLd_dr = 2.0f * (z * (dL_dtu1 - dL_dtv0) - y * dL_dtu2 + x * dL_dtv2);
			const float dLd_dx = 2.0f * (y * (dL_dtu1 + dL_dtv0) + z * dL_dtu2 + r * dL_dtv2 - 2.0f * x * dL_dtv1);
			const float dLd_dy = 2.0f * (x * (dL_dtu1 + dL_dtv0) - r * dL_dtu2 + z * dL_dtv2 - 2.0f * y * dL_dtu0);
			const float dLd_dz = 2.0f * (r * (dL_dtu1 - dL_dtv0) + x * dL_dtu2 + y * dL_dtv2 - 2.0f * z * (dL_dtu0 + dL_dtv1));
			
			atomicAdd(&dL_drot[global_id].x, dLd_dr);
			atomicAdd(&dL_drot[global_id].y, dLd_dx);
			atomicAdd(&dL_drot[global_id].z, dLd_dy);
			atomicAdd(&dL_drot[global_id].w, dLd_dz);

			// dL/dA

			float dz_du = collected_STuv[j * 6 + 0] + collected_STuv[j * 6 + 1] + collected_STuv[j * 6 + 2];
			float dz_dv = collected_STuv[j * 6 + 3] + collected_STuv[j * 6 + 4] + collected_STuv[j * 6 + 5];

			// same as the original but replace 
			// 							dL/dG with dL/dz
			//							dG/du with dz/du
			// 							dG/dv with dz/dv
			dL_dA_local[0][0] += dL_dz * (dz_du * (-du_dhu1) + dz_dv * (-dv_dhu1));
			dL_dA_local[0][1] += dL_dz * (dz_du * (-du_dhv1) + dz_dv * (-dv_dhv1));
			dL_dA_local[0][2] += dL_dz * (dz_du * (du_dhu1 * pix_cam.x + du_dhv1 * pix_cam.y) + dz_dv * (dv_dhu1 * pix_cam.x + dv_dhv1 * pix_cam.y));
			dL_dA_local[1][0] += dL_dz * (dz_du * (-du_dhu2) + dz_dv * (-dv_dhu2));
			dL_dA_local[1][1] += dL_dz * (dz_du * (-du_dhv2) + dz_dv * (-dv_dhv2));
			dL_dA_local[1][2] += dL_dz * (dz_du * (du_dhu2 * pix_cam.x + du_dhv2 * pix_cam.y) + dz_dv * (dv_dhu2 * pix_cam.x + dv_dhv2 * pix_cam.y));
			dL_dA_local[2][0] += dL_dz * (dz_du * (-du_dhu3) + dz_dv * (-dv_dhu3));
			dL_dA_local[2][1] += dL_dz * (dz_du * (-du_dhv3) + dz_dv * (-dv_dhv3));
			dL_dA_local[2][2] += dL_dz * (dz_du * (du_dhu3 * pix_cam.x + du_dhv3 * pix_cam.y) + dz_dv * (dv_dhu3 * pix_cam.x + dv_dhv3 * pix_cam.y));


			// update PQRS
			P_acc += alpha * intersect_c.z;
			Q_acc += alpha;
			R_acc -= alpha * intersect_c.z;
			S_acc -= alpha;

#endif


			// Margin cases
			// dL_dpx = dL_dG * dG_dpx, dL_dG = con_o.w * dL_dopa
			const float dL_dpx_margin = G_xc > G_u ? (con_o.w * dL_dopa * 2 * G_hat * d.x * focal_x) : 0.f;
			const float dL_dpy_margin = G_xc > G_u ? (con_o.w * dL_dopa * 2 * G_hat * d.y * focal_y) : 0.f;

			// Helpful reusable temporary variables
			const float dL_dG = G_u >= G_xc ? con_o.w * dL_dopa : 0.f;

			
			// WARNING: transpose
			dL_dA_local[0][0] += dL_dG * (dG_du * (-du_dhu1) + dG_dv * (-dv_dhu1));
			dL_dA_local[0][1] += dL_dG * (dG_du * (-du_dhv1) + dG_dv * (-dv_dhv1));
			dL_dA_local[0][2] += dL_dG * (dG_du * (du_dhu1 * pix_cam.x + du_dhv1 * pix_cam.y) + dG_dv * (dv_dhu1 * pix_cam.x + dv_dhv1 * pix_cam.y));
			dL_dA_local[1][0] += dL_dG * (dG_du * (-du_dhu2) + dG_dv * (-dv_dhu2));
			dL_dA_local[1][1] += dL_dG * (dG_du * (-du_dhv2) + dG_dv * (-dv_dhv2));
			dL_dA_local[1][2] += dL_dG * (dG_du * (du_dhu2 * pix_cam.x + du_dhv2 * pix_cam.y) + dG_dv * (dv_dhu2 * pix_cam.x + dv_dhv2 * pix_cam.y));
			dL_dA_local[2][0] += dL_dG * (dG_du * (-du_dhu3) + dG_dv * (-dv_dhu3));
			dL_dA_local[2][1] += dL_dG * (dG_du * (-du_dhv3) + dG_dv * (-dv_dhv3));
			dL_dA_local[2][2] += dL_dG * (dG_du * (du_dhu3 * pix_cam.x + du_dhv3 * pix_cam.y) + dG_dv * (dv_dhu3 * pix_cam.x + dv_dhv3 * pix_cam.y));

			dL_dA_local = glm::transpose(dL_dA_local);

			atomicAdd(&dL_dc_margin[global_id].x, dL_dpx_margin);
			atomicAdd(&dL_dc_margin[global_id].y, dL_dpy_margin);
			const float du_dx = (du_dhu1 * collected_A[j * 9 + 6] + du_dhu2 * collected_A[j * 9 + 7] + du_dhu3 * collected_A[j * 9 + 8]);
			const float dv_dx = (dv_dhu1 * collected_A[j * 9 + 6] + dv_dhu2 * collected_A[j * 9 + 7] + dv_dhu3 * collected_A[j * 9 + 8]);
			const float du_dy = (du_dhv1 * collected_A[j * 9 + 6] + du_dhv2 * collected_A[j * 9 + 7] + du_dhv3 * collected_A[j * 9 + 8]);
			const float dv_dy = (dv_dhv1 * collected_A[j * 9 + 6] + dv_dhv2 * collected_A[j * 9 + 7] + dv_dhv3 * collected_A[j * 9 + 8]);
			const float dL_dx = 0.5 * W * (1 / focal_x) * dL_dG * (dG_du * du_dx + dG_dv * dv_dx);
			const float dL_dy = 0.5 * H * (1 / focal_y) * dL_dG * (dG_du * du_dy + dG_dv * dv_dy);
			atomicAdd(&dL_dmean2D[global_id].x, dL_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dy);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G_hat * dL_dopa);

			// dL_dA: N*9 array
			atomicAdd(&(dL_dA[global_id * 9 + 0]), dL_dA_local[0][0]);
			atomicAdd(&(dL_dA[global_id * 9 + 1]), dL_dA_local[0][1]);
			atomicAdd(&(dL_dA[global_id * 9 + 2]), dL_dA_local[0][2]);
			atomicAdd(&(dL_dA[global_id * 9 + 3]), dL_dA_local[1][0]);
			atomicAdd(&(dL_dA[global_id * 9 + 4]), dL_dA_local[1][1]);
			atomicAdd(&(dL_dA[global_id * 9 + 5]), dL_dA_local[1][2]);
			atomicAdd(&(dL_dA[global_id * 9 + 6]), dL_dA_local[2][0]);
			atomicAdd(&(dL_dA[global_id * 9 + 7]), dL_dA_local[2][1]);
			atomicAdd(&(dL_dA[global_id * 9 + 8]), dL_dA_local[2][2]);

		}
	}
	float diff = R_acc - 0.0f;
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3 *means3D,
	const int *radii,
	const float *shs,
	const bool *clamped,
	const glm::vec3 *scales,
	const glm::vec4 *rotations,
	const float scale_modifier,
	const float *cov3Ds,
	const float *viewmatrix,
	const float *projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3 *campos,
	const float *dL_dA,
	const float2 *dL_dc_margin,
	const float3 *dL_dmean2D,
	const float *dL_dconic,
	glm::vec3 *dL_dmean3D,
	float *dL_dcolor,
	float *dL_ddepth,
	float *dL_dcov3D,
	float *dL_dsh,
	glm::vec3 *dL_dscale,
	glm::vec4 *dL_drot)
{

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M,
		(float3 *)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3 *)scales,
		(glm::vec4 *)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		campos,
		dL_dA,
		dL_dc_margin,
		(float3 *)dL_dmean2D,
		(glm::vec3 *)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2 *ranges,
	const uint32_t *point_list,
	const float *orig_points,
	const float *viewmatrix,
	int W, int H,
	const float focal_x, const float focal_y,
	const float *bg_color,
	const float2 *means2D,
	const float4 *conic_opacity,
	const float *colors,
	const float *depths,
	const float *alphas,
	const float *STuv,
	const glm::vec3 *scale,
	const glm::vec4 *rotation,
	const float *A,
	const float *ray_R,
	const float *ray_S,
	const uint32_t *n_contrib,
	const float *dL_dpixels,
	const float *dL_dpixel_depths,
	const float *dL_dalphas,
	float3 *dL_dmean2D,
	float4 *dL_dconic2D,
	float *dL_dopacity,
	float *dL_dcolors,
	float *dL_ddepths,
	float *dL_dA,
	float2 *dL_dc_margin,
	glm::vec3 *dL_dmean3D,
	glm::vec3 *dL_dscale,
	glm::vec4 *dL_drot)
{
	float Ld_value = 0.0f;
	float *Ld_value_d;
	cudaMalloc(&Ld_value_d, sizeof(float));
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		orig_points,
		viewmatrix,
		W, H,
		focal_x, focal_y,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		alphas,
		STuv,
		scale,
		rotation,
		A,
		ray_R,
		ray_S,
		n_contrib,
		dL_dpixels,
		dL_dpixel_depths,
		dL_dalphas,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths,
		dL_dA,
		dL_dc_margin,
		dL_dmean3D,
		dL_dscale,
		dL_drot,
		Ld_value_d);
	cudaMemcpy(&Ld_value, Ld_value_d, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(Ld_value_d);
		
}