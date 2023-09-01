import collections
import cv2
import gc
import itertools
import logging
import numpy as np
from numpy.fft import ifftshift
import os
import sys
import pylab as pl
import tifffile
from typing import List, Optional, Tuple
from skimage.transform import resize as resize_sk
from skimage.transform import warp as warp_sk
from cv2 import dft as fftn
from cv2 import idft as ifftn

opencv = True
def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be zero if images are non-negative).

    Args:
        cross_correlation_max : complex
            The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)

def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None):
    """
    adapted from SIMA (https://github.com/losonczylab) and the scikit-image (http://scikit-image.org/) package.

    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Args:
        data : 2D ndarray
            The input data array (DFT of original data) to upsample.

        upsampled_region_size : integer or tuple of integers, optional
            The size of the region to be sampled.  If one integer is provided, it
            is duplicated up to the dimensionality of ``data``.

        upsample_factor : integer, optional
            The upsampling factor.  Defaults to 1.

        axis_offsets : tuple of integers, optional
            The offsets of the region to be sampled.  Defaults to None (uses
            image center)

    Returns:
        output : 2D ndarray
                The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    col_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * upsample_factor)) *
        (ifftshift(np.arange(data.shape[1]))[:, None] -
         np.floor(data.shape[1] // 2)).dot(
             np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1])
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor)) *
        (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            ifftshift(np.arange(data.shape[0]))[None, :] -
            np.floor(data.shape[0] // 2))
    )

    if data.ndim > 2:
        pln_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[2] * upsample_factor)) *
        (np.arange(upsampled_region_size[2])[:, None] - axis_offsets[2]).dot(
                ifftshift(np.arange(data.shape[2]))[None, :] -
                np.floor(data.shape[2] // 2)))

    # output = np.tensordot(np.tensordot(row_kernel,data,axes=[1,0]),col_kernel,axes=[1,0])
    output = np.tensordot(row_kernel, data, axes = [1,0])
    output = np.tensordot(output, col_kernel, axes = [1,0])

    if data.ndim > 2:
        output = np.tensordot(output, pln_kernel, axes = [1,1])
    #output = row_kernel.dot(data).dot(col_kernel)
    return output
def apply_shifts_dft(src_freq, shifts, diffphase, is_freq=True, border_nan=True):
    """
    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Args:
        apply shifts using inverse dft
        src_freq: ndarray
            if is_freq it is fourier transform image else original image
        shifts: shifts to apply
        diffphase: comes from the register_translation output
    """

    is3D = len(src_freq.shape) == 3
    if not is_freq:
        if is3D:
            src_freq = np.fft.fftn(src_freq)
        else:
            src_freq = np.dstack([np.real(src_freq), np.imag(src_freq)])
            src_freq = fftn(src_freq, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            src_freq = src_freq[:, :, 0] + 1j * src_freq[:, :, 1]
            src_freq = np.array(src_freq, dtype=np.complex128, copy=False)

    if not is3D:
        nr, nc = np.shape(src_freq)
        Nr = ifftshift(np.arange(-np.fix(nr/2.), np.ceil(nr/2.)))
        Nc = ifftshift(np.arange(-np.fix(nc/2.), np.ceil(nc/2.)))
        Nc, Nr = np.meshgrid(Nc, Nr)
        Greg = src_freq * np.exp(1j * 2 * np.pi *
                                 (-shifts[0] * Nr / nr - shifts[1] * Nc / nc))
    else:
        nr, nc, nd = np.array(np.shape(src_freq), dtype=float)
        Nr = ifftshift(np.arange(-np.fix(nr / 2.), np.ceil(nr / 2.)))
        Nc = ifftshift(np.arange(-np.fix(nc / 2.), np.ceil(nc / 2.)))
        Nd = ifftshift(np.arange(-np.fix(nd / 2.), np.ceil(nd / 2.)))
        Nc, Nr, Nd = np.meshgrid(Nc, Nr, Nd)
        Greg = src_freq * np.exp(1j * 2 * np.pi *
                                 (-shifts[0] * Nr / nr - shifts[1] * Nc / nc -
                                  shifts[2] * Nd / nd))

    Greg = Greg.dot(np.exp(1j * diffphase))
    if is3D:
        new_img = np.real(np.fft.ifftn(Greg))
    else:
        Greg = np.dstack([np.real(Greg), np.imag(Greg)])
        new_img = ifftn(Greg)[:, :, 0]

    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shifts[:2])).astype(int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shifts[:2])).astype(int)
        if is3D:
            max_d = np.ceil(np.maximum(0, shifts[2])).astype(int)
            min_d = np.floor(np.minimum(0, shifts[2])).astype(int)
        if border_nan is True:
            new_img[:max_h, :] = np.nan
            if min_h < 0:
                new_img[min_h:, :] = np.nan
            new_img[:, :max_w] = np.nan
            if min_w < 0:
                new_img[:, min_w:] = np.nan
            if is3D:
                new_img[:, :, :max_d] = np.nan
                if min_d < 0:
                    new_img[:, :, min_d:] = np.nan
        elif border_nan == 'min':
            min_ = np.nanmin(new_img)
            new_img[:max_h, :] = min_
            if min_h < 0:
                new_img[min_h:, :] = min_
            new_img[:, :max_w] = min_
            if min_w < 0:
                new_img[:, min_w:] = min_
            if is3D:
                new_img[:, :, :max_d] = min_
                if min_d < 0:
                    new_img[:, :, min_d:] = min_
        elif border_nan == 'copy':
            new_img[:max_h] = new_img[max_h]
            if min_h < 0:
                new_img[min_h:] = new_img[min_h-1]
            if max_w > 0:
                new_img[:, :max_w] = new_img[:, max_w, np.newaxis]
            if min_w < 0:
                new_img[:, min_w:] = new_img[:, min_w-1, np.newaxis]
            if is3D:
                if max_d > 0:
                    new_img[:, :, :max_d] = new_img[:, :, max_d, np.newaxis]
                if min_d < 0:
                    new_img[:, :, min_d:] = new_img[:, :, min_d-1, np.newaxis]

    return new_img


def apply_shift_iteration(img, shift, border_nan:bool=False, border_type=cv2.BORDER_REFLECT):
    # todo todocument

    sh_x_n, sh_y_n = shift
    w_i, h_i = img.shape
    M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
    min_, max_ = np.nanmin(img), np.nanmax(img)
    img = np.clip(cv2.warpAffine(img, M, (h_i, w_i),
                                 flags=cv2.INTER_CUBIC, borderMode=border_type), min_, max_)
    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shift)).astype(int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shift)).astype(int)
        if border_nan is True:
            img[:max_h, :] = np.nan
            if min_h < 0:
                img[min_h:, :] = np.nan
            img[:, :max_w] = np.nan
            if min_w < 0:
                img[:, min_w:] = np.nan
        elif border_nan == 'min':
            img[:max_h, :] = min_
            if min_h < 0:
                img[min_h:, :] = min_
            img[:, :max_w] = min_
            if min_w < 0:
                img[:, min_w:] = min_
        elif border_nan == 'copy':
            if max_h > 0:
                img[:max_h] = img[max_h]
            if min_h < 0:
                img[min_h:] = img[min_h-1]
            if max_w > 0:
                img[:, :max_w] = img[:, max_w, np.newaxis]
            if min_w < 0:
                img[:, min_w:] = img[:, min_w-1, np.newaxis]

    return img


def register_translation(src_image, target_image, upsample_factor=1,
                         space="real", shifts_lb=None, shifts_ub=None, max_shifts=(10, 10),
                         use_cuda=False):
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must really be same size for "
                         "register_translation")

    # only 2D data makes sense right now
    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D images")
    elif space.lower() == 'real':

        if opencv:
            src_freq_1 = fftn(
                src_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            src_freq = src_freq_1[:, :, 0] + 1j * src_freq_1[:, :, 1]
            src_freq = np.array(src_freq, dtype=np.complex128, copy=False)
            target_freq_1 = fftn(
                target_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            target_freq = target_freq_1[:, :, 0] + 1j * target_freq_1[:, :, 1]
            target_freq = np.array(
                target_freq, dtype=np.complex128, copy=False)
        else:
            src_image_cpx = np.array(
                src_image, dtype=np.complex128, copy=False)
            target_image_cpx = np.array(
                target_image, dtype=np.complex128, copy=False)
            src_freq = np.fft.fftn(src_image_cpx)
            target_freq = np.fft.fftn(target_image_cpx)

    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

        # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if opencv:

        image_product_cv = np.dstack(
            [np.real(image_product), np.imag(image_product)])
        cross_correlation = fftn(
            image_product_cv, flags=cv2.DFT_INVERSE + cv2.DFT_SCALE)
        cross_correlation = cross_correlation[:,
                            :, 0] + 1j * cross_correlation[:, :, 1]

    else:
        cross_correlation = ifftn(image_product)
    # Locate maximum
    new_cross_corr = np.abs(cross_correlation)
    if (shifts_lb is not None) or (shifts_ub is not None):

        if (shifts_lb[0] < 0) and (shifts_ub[0] >= 0):
            new_cross_corr[shifts_ub[0]:shifts_lb[0], :] = 0
        else:
            new_cross_corr[:shifts_lb[0], :] = 0
            new_cross_corr[shifts_ub[0]:, :] = 0

        if (shifts_lb[1] < 0) and (shifts_ub[1] >= 0):
            new_cross_corr[:, shifts_ub[1]:shifts_lb[1]] = 0
        else:
            new_cross_corr[:, :shifts_lb[1]] = 0
            new_cross_corr[:, shifts_ub[1]:] = 0
    else:

        new_cross_corr[max_shifts[0]:-max_shifts[0], :] = 0

        new_cross_corr[:, max_shifts[1]:-max_shifts[1]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size//2)
                          for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
    if upsample_factor == 1:

        src_amp = np.sum(np.abs(src_freq) ** 2) / src_freq.size
        target_amp = np.sum(np.abs(target_freq) ** 2) / target_freq.size
        CCmax = cross_correlation.max()
        # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor

        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(cross_correlation)),
            cross_correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + (maxima / upsample_factor)
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

        # If its only one row or column the shift along that dimension has no
        # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts, src_freq, _compute_phasediff(CCmax)

def create_weight_matrix_for_blending(img, overlaps, strides):
    """ create a matrix that is used to normalize the intersection of the stiched patches

    Args:
        img: original image, ndarray

        shapes, overlaps, strides:  tuples
            shapes, overlaps and strides of the patches

    Returns:
        weight_mat: normalizing weight matrix
    """
    shapes = np.add(strides, overlaps)

    max_grid_1, max_grid_2 = np.max(
        np.array([it[:2] for it in sliding_window(img, overlaps, strides)]), 0)

    for grid_1, grid_2, _, _, _ in sliding_window(img, overlaps, strides):

        weight_mat = np.ones(shapes)

        if grid_1 > 0:
            weight_mat[:overlaps[0], :] = np.linspace(
                0, 1, overlaps[0])[:, None]
        if grid_1 < max_grid_1:
            weight_mat[-overlaps[0]:,
                       :] = np.linspace(1, 0, overlaps[0])[:, None]
        if grid_2 > 0:
            weight_mat[:, :overlaps[1]] = weight_mat[:, :overlaps[1]
                                                     ] * np.linspace(0, 1, overlaps[1])[None, :]
        if grid_2 < max_grid_2:
            weight_mat[:, -overlaps[1]:] = weight_mat[:, -
                                                      overlaps[1]:] * np.linspace(1, 0, overlaps[1])[None, :]

        yield weight_mat

def sliding_window(image, overlaps, strides):
    """ efficiently and lazily slides a window across the image

    Args:
        img:ndarray 2D
            image that needs to be slices

        windowSize: tuple
            dimension of the patch

        strides: tuple
            stride in each dimension

     Returns:
         iterator containing five items
              dim_1, dim_2 coordinates in the patch grid
              x, y: bottom border of the patch in the original matrix

              patch: the patch
     """
    windowSize = np.add(overlaps, strides)
    range_1 = list(range(
        0, image.shape[0] - windowSize[0], strides[0])) + [image.shape[0] - windowSize[0]]
    range_2 = list(range(
        0, image.shape[1] - windowSize[1], strides[1])) + [image.shape[1] - windowSize[1]]
    for dim_1, x in enumerate(range_1):
        for dim_2, y in enumerate(range_2):
            # yield the current window
            yield (dim_1, dim_2, x, y, image[x:x + windowSize[0], y:y + windowSize[1]])


def tile_and_correct(img, template, strides, overlaps, max_shifts, newoverlaps=None, newstrides=None, upsample_factor_grid=4,
                     upsample_factor_fft=10, show_movie=False, max_deviation_rigid=2, add_to_movie=0, shifts_opencv=True, gSig_filt=None,
                     use_cuda=False, border_nan=True):
    print(img)
    img = img.astype(np.float64).copy()
    print(img)
    cv2.imshow('000',img)
    template = template.astype(np.float64).copy()

    if gSig_filt is not None:

        img_orig = img.copy()
        # img = high_pass_filter_space(img_orig, gSig_filt)

    img = img + add_to_movie
    cv2.imshow('111',img)
    template = template + add_to_movie
    # compute rigid shifts
    rigid_shts, sfr_freq, diffphase = register_translation(
        img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts, use_cuda=use_cuda)
    if max_deviation_rigid == 0:

        if shifts_opencv:
            if gSig_filt is not None:
                img = img_orig

            new_img = apply_shift_iteration(
                img, (-rigid_shts[0], -rigid_shts[1]), border_nan=border_nan)

        return new_img - add_to_movie, (-rigid_shts[0], -rigid_shts[1]), None, None
    else:
        # extract patches
        templates = [
            it[-1] for it in sliding_window(template, overlaps=overlaps, strides=strides)]
        xy_grid = [(it[0], it[1]) for it in sliding_window(
            template, overlaps=overlaps, strides=strides)]
        num_tiles = np.prod(np.add(xy_grid[-1], 1))
        imgs = [it[-1]
                for it in sliding_window(img, overlaps=overlaps, strides=strides)]
        dim_grid = tuple(np.add(xy_grid[-1], 1))

        if max_deviation_rigid is not None:

            lb_shifts = np.ceil(np.subtract(
                rigid_shts, max_deviation_rigid)).astype(int)
            ub_shifts = np.floor(
                np.add(rigid_shts, max_deviation_rigid)).astype(int)

        else:

            lb_shifts = None
            ub_shifts = None

        # extract shifts for each patch
        shfts_et_all = [register_translation(
            a, b, c, shifts_lb=lb_shifts, shifts_ub=ub_shifts, max_shifts=max_shifts, use_cuda=use_cuda) for a, b, c in
            zip(
                imgs, templates, [upsample_factor_fft] * num_tiles)]
        shfts = [sshh[0] for sshh in shfts_et_all]
        diffs_phase = [sshh[2] for sshh in shfts_et_all]
        # create a vector field
        shift_img_x = np.reshape(np.array(shfts)[:, 0], dim_grid)
        shift_img_y = np.reshape(np.array(shfts)[:, 1], dim_grid)
        diffs_phase_grid = np.reshape(np.array(diffs_phase), dim_grid)

    if shifts_opencv:
        if gSig_filt is not None:
            img = img_orig

        dims = img.shape
        x_grid, y_grid = np.meshgrid(np.arange(0., dims[1]).astype(
            np.float32), np.arange(0., dims[0]).astype(np.float32))
        cv2.imshow("222",img)
        m_reg = cv2.remap(img, cv2.resize(shift_img_y.astype(np.float32), dims[::-1]) + x_grid,
                          cv2.resize(shift_img_x.astype(np.float32), dims[::-1]) + y_grid,
                          cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # borderValue=add_to_movie)
        total_shifts = [
            (-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles))]
        return m_reg - add_to_movie, total_shifts, None, None

        # create automatically upsample parameters if not passed
    if newoverlaps is None:
        newoverlaps = overlaps
    if newstrides is None:
        newstrides = tuple(
            np.round(np.divide(strides, upsample_factor_grid)).astype(int))

    newshapes = np.add(newstrides, newoverlaps)

    imgs = [it[-1]
            for it in sliding_window(img, overlaps=newoverlaps, strides=newstrides)]

    xy_grid = [(it[0], it[1]) for it in sliding_window(
        img, overlaps=newoverlaps, strides=newstrides)]

    start_step = [(it[2], it[3]) for it in sliding_window(
        img, overlaps=newoverlaps, strides=newstrides)]

    dim_new_grid = tuple(np.add(xy_grid[-1], 1))

    shift_img_x = cv2.resize(
        shift_img_x, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)
    shift_img_y = cv2.resize(
        shift_img_y, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)
    diffs_phase_grid_us = cv2.resize(
        diffs_phase_grid, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)

    num_tiles = np.prod(dim_new_grid)

    max_shear = np.percentile(
        [np.max(np.abs(np.diff(ssshh, axis=xxsss))) for ssshh, xxsss in itertools.product(
            [shift_img_x, shift_img_y], [0, 1])], 75)

    total_shifts = [
        (-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles))]
    total_diffs_phase = [
        dfs for dfs in diffs_phase_grid_us.reshape(num_tiles)]

    if gSig_filt is not None:
        raise Exception(
            'The use of FFT and filtering options have not been tested. Set opencv=True')

    imgs = [apply_shifts_dft(im, (
        sh[0], sh[1]), dffphs, is_freq=False, border_nan=border_nan) for im, sh, dffphs in zip(
        imgs, total_shifts, total_diffs_phase)]

    normalizer = np.zeros_like(img) * np.nan
    new_img = np.zeros_like(img) * np.nan

    weight_matrix = create_weight_matrix_for_blending(
        img, newoverlaps, newstrides)

    if max_shear < 0.5:
        for (x, y), (_, _), im, (_, _), weight_mat in zip(start_step, xy_grid, imgs, total_shifts, weight_matrix):
            prev_val_1 = normalizer[x:x + newshapes[0], y:y + newshapes[1]]

            normalizer[x:x + newshapes[0], y:y + newshapes[1]] = np.nansum(
                np.dstack([~np.isnan(im) * 1 * weight_mat, prev_val_1]), -1)
            prev_val = new_img[x:x + newshapes[0], y:y + newshapes[1]]
            new_img[x:x + newshapes[0], y:y + newshapes[1]
            ] = np.nansum(np.dstack([im * weight_mat, prev_val]), -1)

        new_img = new_img / normalizer

    else:  # in case the difference in shift between neighboring patches is larger than 0.5 pixels we do not interpolate in the overlaping area
        half_overlap_x = int(newoverlaps[0] / 2)
        half_overlap_y = int(newoverlaps[1] / 2)
        for (x, y), (idx_0, idx_1), im, (_, _), weight_mat in zip(start_step, xy_grid, imgs, total_shifts,
                                                                  weight_matrix):

            if idx_0 == 0:
                x_start = x
            else:
                x_start = x + half_overlap_x

            if idx_1 == 0:
                y_start = y
            else:
                y_start = y + half_overlap_y

            x_end = x + newshapes[0]
            y_end = y + newshapes[1]
            new_img[x_start:x_end,
            y_start:y_end] = im[x_start - x:, y_start - y:]

    if show_movie:
        img = apply_shifts_dft(
            sfr_freq, (-rigid_shts[0], -rigid_shts[1]), diffphase, border_nan=border_nan)
        img_show = np.vstack([new_img, img])

        img_show = cv2.resize(img_show, None, fx=1, fy=1)

        cv2.imshow('frame', img_show / np.percentile(template, 99))
        cv2.waitKey(int(1. / 500 * 1000))

    else:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    return new_img - add_to_movie, total_shifts, start_step, xy_grid

import cv2
import numpy as np

def B_spline(u, flag):
    if flag == 0:
        return (1 - u**3 + 3 * u**2 - 3 * u) / 6.0
    elif flag == 1:
        return (4 + 3 * u**3 - 6 * u**2) / 6.0
    elif flag == 2:
        return (1 - 3 * u**3 + 3 * u**2 + 3 * u) / 6.0
    elif flag == 3:
        return u**3 / 6.0
    else:
        return 0.0

def B_spline_form(srcimg, delta_x=32, delta_y=32):
    grid_rows = (srcimg.shape[0] // delta_x) + 1 + 3
    grid_cols = (srcimg.shape[1] // delta_y) + 1 + 3
    noiseMat = np.zeros((grid_rows, grid_cols, 2), dtype=np.float32)
    # uniform_distribution = random.uniform(-20, 20)

    for row in range(grid_rows):
        for col in range(grid_cols):
            noiseMat[row, col, 0] = np.random.uniform(-10, 10)
            noiseMat[row, col, 1] = np.random.uniform(-10, 10)

    dstimg = np.zeros_like(srcimg, dtype=np.uint8)
    offset = np.zeros((srcimg.shape[0], srcimg.shape[1], 2), dtype=np.float32)

    for x in range(srcimg.shape[0]):
        for y in range(srcimg.shape[1]):
            i = int(x / delta_x)
            j = int(y / delta_y)
            u = float(x / delta_x) - i
            v = float(y / delta_y) - j

            pX = [B_spline(u, k) for k in range(4)]
            pY = [B_spline(v, k) for k in range(4)]

            Tx, Ty = 0.0, 0.0
            for m in range(4):
                for n in range(4):
                    control_point_x = i + m
                    control_point_y = j + n
                    temp = pY[n] * pX[m]
                    Tx += temp * (noiseMat[control_point_x, control_point_y, 0])
                    Ty += temp * (noiseMat[control_point_x, control_point_y, 1])

            offset[x, y, 0] = Tx
            offset[x, y, 1] = Ty

    for row in range(dstimg.shape[0]):
        for col in range(dstimg.shape[1]):
            src_x = row + offset[row, col, 0]
            src_y = col + offset[row, col, 1]
            x1, y1 = int(src_x), int(src_y)
            x2, y2 = x1 + 1, y1 + 1

            if x1 < 0 or x1 > (srcimg.shape[0] - 2) or y1 < 0 or y1 > (srcimg.shape[1] - 2):
                dstimg[row, col] = [0, 0, 0]
            else:
                pointa = srcimg[x1, y1]
                pointb = srcimg[x2, y1]
                pointc = srcimg[x1, y2]
                pointd = srcimg[x2, y2]

                B = (x2 - src_x) * (y2 - src_y) * pointa[0] - (x1 - src_x) * (y2 - src_y) * pointb[0] - \
                    (x2 - src_x) * (y1 - src_y) * pointc[0] + (x1 - src_x) * (y1 - src_y) * pointd[0]
                G = (x2 - src_x) * (y2 - src_y) * pointa[1] - (x1 - src_x) * (y2 - src_y) * pointb[1] - \
                    (x2 - src_x) * (y1 - src_y) * pointc[1] + (x1 - src_x) * (y1 - src_y) * pointd[1]
                R = (x2 - src_x) * (y2 - src_y) * pointa[2] - (x1 - src_x) * (y2 - src_y) * pointb[2] - \
                    (x2 - src_x) * (y1 - src_y) * pointc[2] + (x1 - src_x) * (y1 - src_y) * pointd[2]

                dstimg[row, col] = [B, G, R]

    return dstimg

# if __name__ == "__main__":
#     srcimg = cv2.imread("D:\Renyi\IsoNet\\n01983481_171.JPEG")
#     dstimg = B_spline_form(srcimg)
#
#
#     diff = cv2.subtract(dstimg, srcimg)
#     # cv2.imshow("Original Image", srcimg)
#     # cv2.imshow("B_spline_form", dstimg)
#     # cv2.imshow("Difference", diff)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     img =dstimg[:, :, 1]
#     template =srcimg[:,:,1]
#     strides =[10,10]
#     overlaps =[2,2]
#     max_shifts =[50,50]
#     re, total_shifts, _, _ = tile_and_correct(img,template,strides,overlaps,max_shifts,add_to_movie=0,shifts_opencv=True)
#     cv2.imshow("Original Image", template)
#     cv2.imshow("B_spline_form", img)
#     re = (re-np.min(re))/(np.max(re)-np.min(re))
#     cv2.imshow("correct",re)
#     cv2.waitKey(0)



# if __name__ == '__main__':
#     img =
#     template =
#     strides =
#     overlaps =
#     max_shifts =
