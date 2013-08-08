TEMmod - TEdgeMask modified for Avisynth2.6x

TEMmod is an Avisynth plugin which based on TEdgeMask written by Kevin Stone
(a.k.a. tritical) and was written from scratch.


Info:
    Creates an edgemask using gradient vector magnitude.
    Only planar formats are supported.


Requirement:
    Avisynth 2.6 alpha 4 or later.
    Microsoft Visual C++ 2010 Redistributable Package
    SSE2 capable CPU


Syntax:

    TEMmod(clip c, float "threshY", float "threshC", int "type", int "link",
           int "chroma", bool "invert", bool "preblur")


    threshY:
        Set the magnitude thresholds for Y-plane.
        If over this value then a sample will be considered an edge, and the
        output mask will be binarizeto by 0 and 255.
        Set this to 0 means output a magnitude mask instead of binary mask.

        default: 8.0 (float)

    threshC:
        Set the magnitude thresholds for U-plane and V-plane.
        Others are same as threshY.

        default: 8.0 (float)

    type:
        Sets the type of first order partial derivative approximation that is used.
        possible values:

            1 - 2 pixel
            2 - 4 pixel

        default: 2 (int)

    link:
        Specifies whether luma to chroma linking, no linking, or linking of every
        plane to every other plane is used.

            0 - no linking
            1 - luma to chroma linking
            2 - every plane to every other plane

        default: 1 (int)

    chroma:
        This control how chroma(U-plane and V-plane) is processed.
        Set to 0 to not process and never touch.
        Set to 1 to process chroma.
        Set to 2 to not process and all samples are fill to zero.

        default: 1 (int)

    invert:
        If this set to True, the output mask will be inverted.

        default: False (bool)

    preblur:
        Indicates whether to apply a 3x3 guassian blur to the input image
        prior to generating the edge map.

        default: False (bool)


Note:
    Although the output of this plugin is almost the same as original, it is
    not completely in agreement since some approximations are used.


Source code:
    https://github.com/chikuzen/TEMmod/


Author:
    Oka Motofumi (chikuzen.mo at gmail dot com)
