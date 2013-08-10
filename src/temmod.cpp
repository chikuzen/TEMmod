/*
  temmod.cpp

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


#include <stdint.h>
#include <malloc.h>
#include "temmod.h"


const AVS_Linkage* AVS_linkage = 0;


static void __stdcall
invert_plane(uint8_t* d, int dst_pitch, int height)
{
    unsigned* dstp = (unsigned*)d;
    size_t size = height * dst_pitch / sizeof(unsigned);

    while(size--) {
        *dstp = ~*dstp++;
    }
}


class TEMmod : public GenericVideoFilter {
    int process[3];
    int threshold[3];
    int link;
    bool invert;
    uint8_t* buff;
    int buff_pitch;
    int type;
    float scale;

    calc_map_func calc_map;
    link_planes_func link_planes;

public:
    TEMmod(PClip c, float thy, float thc, int type, int chroma, int link,
           bool invert, float scale, IScriptEnvironment* env);
    ~TEMmod();
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
};


TEMmod::TEMmod(PClip c, float thy, float thc, int tp, int chroma, int lnk,
               bool inv, float sc, IScriptEnvironment* env)
    : GenericVideoFilter(c), link(lnk), invert(inv), type(tp), scale(sc)
{
    if (!vi.IsPlanar()) {
        env->ThrowError("TEMmod: Planar format only.");
    }
    if (vi.IsY8()) {
        link = 0;
        chroma = 0;
    }

    process[0] = 1; process[1] = chroma; process[2] = chroma;
    float th[3] = {thy, thc, thc};

    for (int i = 0; i < 3; i++) {
        if (type == 1) {
            threshold[i] = (int)(th[i] * th[i] * 4 + 0.5);
        } else if (type == 2) {
            threshold[i] = (int)(th[i] * th[i] * 10000 + 0.5);
        } else if (type == 3) {
            threshold[i] = (int)(th[i] * 2 + 0.5);
        } else if (type == 4) {
            threshold[i] = (int)(th[i] * 100 / 3.0 + 0.5);
        } else {
            threshold[i] = (int)(th[i] * 4 + 0.5);
        }
    }

    if (threshold[0] == 0 || threshold[1] == 0) {
        link = 0;
    }

    if (type == 1) {
        calc_map = calc_maps[threshold[0] > 0 ? 1 : 0];
    } else if (type == 2) {
        calc_map = calc_maps[2 + (threshold[0] > 0 ? 1 : 0)];
    } else {
        calc_map = calc_maps[type + 1];
    }

    const link_planes_func* links = link == 1 ? link_y_to_uv : link_all;
    if (vi.IsYV24()) {
        link_planes = links[0];
    } else if (vi.IsYV16()) {
        link_planes = links[1];
    } else if (vi.IsYV12()) {
        link_planes = links[2];
    } else {
        link_planes = links[3];
    }

    buff_pitch = ((vi.width + 47) / 16) * 16;
    buff = (uint8_t*)_aligned_malloc(buff_pitch * (type * 2 + 1), 16);
    if (!buff) {
        env->ThrowError("TEMmod: failed to allocate buffer.");
    }
}


TEMmod::~TEMmod()
{
    _aligned_free(buff);
}


PVideoFrame __stdcall TEMmod::GetFrame(int n, IScriptEnvironment* env)
{
    const int planes[3] = {PLANAR_Y, PLANAR_U, PLANAR_V};

    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    for (int i = 0; i < 3; i++) {
        if (process[i] == 0) {
            break;
        }
        int p = planes[i];
        int dst_pitch = dst->GetPitch(p);
        uint8_t* dstp = dst->GetWritePtr(p);
        int height = src->GetHeight(p);

        if (process[i] == 2) {
            memset(dstp, 0, dst_pitch * height);
            continue;
        }
        int src_pitch = src->GetPitch(p);
        int width = src->GetRowSize(p);
        const uint8_t* srcp = src->GetReadPtr(p);

        if (((intptr_t)srcp & 15) && type > 2) {
            env->ThrowError("TEMmod: invalid memory alignment found!");
        }

        calc_map(srcp, dstp, buff, src_pitch, dst_pitch, buff_pitch,
                 width, height, threshold[i], scale);

    }

    if (link > 0) {
        link_planes(dst);
    }

    if (!invert) {
        return dst;
    }

    for (int i = 0; i < 3; i++) {
        if (process[i] == 0) {
            break;
        }
        invert_plane(dst->GetWritePtr(planes[i]), dst->GetPitch(planes[i]),
                     dst->GetHeight(planes[i]));
    }

    return dst;
}


AVSValue __cdecl
create_temmod(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    float thy = args[1].AsFloat(8.0);
    if (thy < 0) {
        env->ThrowError("TEMmod: threshY must be higher than zero.");
    }

    int chroma = args[5].AsInt(1);
    if (chroma < 0 || chroma > 2) {
        env->ThrowError("TEMmod: chroma must be set to 0, 1, or 2.");
    }
    float thc = args[2].AsFloat(8.0);
    if (chroma == 1 && thc < 0) {
        env->ThrowError("TEMmod: threshC must be higher than zero.");
    }

    int type = args[3].AsInt(4);
    if (type < 1 && type > 5) {
        env->ThrowError("TEMmod: type must be between 1 and 5.");
    }

    int link = args[4].AsInt(1);
    if (chroma > 0 && link < 0 || link > 2) {
        env->ThrowError("TEMmod: link must be set to 0, 1 or 2.");
    }

    bool invert = args[6].AsBool(false);

    PClip clip = args[0].AsClip();
    if (args[7].AsBool(false)) {
        try {
            AVSValue blur[2] = {clip, 1.0};
            clip = env->Invoke("Blur", AVSValue(blur, 2)).AsClip();
        } catch (IScriptEnvironment::NotFound) {
            env->ThrowError("TEMmod: failed to invoke Blur().");
        }
    }

    float scale = args[8].AsFloat(0);
    if (scale < 0) {
        env->ThrowError("TEMmod: scale must be higher than zero.");
    }

    return new TEMmod(clip, thy, thc, type, chroma, link, invert, scale, env);
}
extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("TEMmod",
                    "c[threshY]f[threshC]f[type]i[link]i[chroma]i"
                    "[invert]b[preblur]b[scale]f", create_temmod, 0);
    return "TEdgeMask_modified for aisynth2.6 ver." TEMM_VERSION;
}
