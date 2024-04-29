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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

#define Ld
#define Wd 100.0f
#define far 1000.0f
#define near 0.2f

#define Ln
#define Wn 0.05f

// define an macro that represent Ld || Ln, used to compute shared values in backwards
#ifdef Ld
#define LdOrLn
#else
#ifdef Ln
#define LdOrLn
#endif
#endif

#endif