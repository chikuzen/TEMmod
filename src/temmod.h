/*
  temmod.h

  This file is part of TEMmod

  Copyright (C) 2013 Oka Motofumi

  Authors: Oka Motofumi (chikuzen.mo at gmail dot com)

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
*/


#ifndef TEMMOD_H
#define TEMMOD_H

#include <stdint.h>
#include <windows.h>
#include "avisynth.h"

#define TEMM_VERSION "0.1.0"


typedef void (__stdcall *calc_map_func)(const uint8_t* srcp, uint8_t* dstp,
                                        uint8_t* buff, int src_pitch,
                                        int dst_pitch, int buff_pitch,
                                        int width, int height,
                                        int threshold);

typedef void (__stdcall *link_planes_func)(PVideoFrame& dst);

extern const calc_map_func calc_1;
extern const calc_map_func calc_1t;
extern const calc_map_func calc_2;
extern const calc_map_func calc_2t;
extern const calc_map_func calc_3;
extern const calc_map_func calc_4;

extern const link_planes_func link_y_to_uv[];

extern const link_planes_func link_all[];

#endif
