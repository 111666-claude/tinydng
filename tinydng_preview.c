/*
 * tinydng_preview.c - configurable linear preview.
 *
 * Post-processing on top of the unchanged raw decode path: raw samples are
 * normalized with DNG BlackLevel/WhiteLevel (and the optional
 * LinearizationTable), then emitted as grayscale or bilinear-demosaiced RGB
 * at the full raw window size. All arithmetic is integer, saturated, and
 * overflow-checked.
 * SPDX-License-Identifier: MIT
 */
#include "td_internal.h"

#include <inttypes.h>

/* ------------------------------------------------------------------ */
/* Decoded-sample access (host order, unpacked 8/16/32-bit elements)   */
/* ------------------------------------------------------------------ */

static uint32_t td_read_sample(const tinydng_pixels *px, size_t i) {
  switch (px->bits_per_sample) {
    case 8:
      return (uint32_t)px->data[i];
    case 16:
      return (uint32_t)((const uint16_t *)(const void *)px->data)[i];
    case 32:
      return ((const uint32_t *)(const void *)px->data)[i];
    default:
      return 0u;
  }
}

/* ------------------------------------------------------------------ */
/* Level handling                                                      */
/* ------------------------------------------------------------------ */

/* Default WhiteLevel when the tag is absent: the stored-depth max code. */
static uint32_t td_default_white(uint16_t bps) {
  if (bps >= 32u) {
    return UINT32_MAX;
  }
  if (bps == 0u) {
    return 0u;
  }
  return (uint32_t)((1u << bps) - 1u);
}

/* Index into the 4-element BlackLevel/WhiteLevel arrays for raw pixel
 * (x, y). A single level applies everywhere; four levels use the DNG 2x2
 * repeat-pattern ordering (row-major, aligned with the Bayer quadrant).
 * Any other count deterministically falls back to the first level. */
static int td_cfa_level_index(const tinydng_image_info *img, uint32_t x,
                              uint32_t y, uint8_t count) {
  const tinydng_cfa *cfa = &img->cfa;
  if (count <= 1u) {
    return 0;
  }
  if (count >= 4u && cfa->present && cfa->pattern_dim[0] == 2u &&
      cfa->pattern_dim[1] == 2u) {
    return (int)(((y & 1u) << 1) | (x & 1u));
  }
  return 0;
}

/* Validate and resolve the per-plane levels. A tag carrying one element is
 * broadcast to every plane; four elements index planes directly (DNG 2x2
 * BlackLevel ordering); any other count clamps the plane index to its last
 * element. Returns 0 on success; on an inverted/degenerate level range sets
 * *err and returns non-zero. */
static int td_resolve_levels(const tinydng_image_info *img,
                             int64_t black[4], int64_t white[4],
                             tinydng_error *err) {
  const tinydng_raw_info *raw = &img->raw;
  uint32_t def_white = td_default_white(img->bits_per_sample);
  int k;
  for (k = 0; k < 4; k++) {
    if (raw->black_level_present && raw->black_level_count > 0u) {
      int bi = (k < (int)raw->black_level_count)
                   ? k
                   : (int)raw->black_level_count - 1;
      black[k] = (int64_t)raw->black_level[bi];
    } else {
      black[k] = 0;
    }
    if (raw->white_level_present && raw->white_level_count > 0u) {
      int wi = (k < (int)raw->white_level_count)
                   ? k
                   : (int)raw->white_level_count - 1;
      white[k] = (int64_t)raw->white_level[wi];
    } else {
      white[k] = (int64_t)def_white;
    }
    if (white[k] <= black[k]) {
      td_set_error(err, TINYDNG_E_INVALID_ARG, TINYDNG_STAGE_METADATA, 0, 0,
                   0,
                   "invalid level range: white=%" PRId64 " <= black=%" PRId64
                   " (plane %d)",
                   white[k], black[k], k);
      return 1;
    }
  }
  return 0;
}

/* Apply linearization (clamped index) + level normalization in one saturated
 * integer step: clamp(s - b, 0, w-b) * maxcode / (w - b). */
static uint32_t td_normalize(const tinydng_image_info *img, uint32_t s,
                             int level_idx, const int64_t black[4],
                             const int64_t white[4], uint32_t maxcode) {
  const tinydng_raw_info *raw = &img->raw;
  int64_t b, w, diff, range;
  uint64_t v = (uint64_t)s;

  if (raw->linearization_table && raw->linearization_table_count > 0u) {
    size_t idx = (size_t)s;
    if (idx >= raw->linearization_table_count) {
      idx = raw->linearization_table_count - 1u;
    }
    v = (uint64_t)raw->linearization_table[idx];
  }

  b = black[level_idx];
  w = white[level_idx];
  diff = (int64_t)v - b;
  range = w - b; /* guaranteed > 0 by td_resolve_levels */
  if (diff <= 0) {
    return 0u;
  }
  if (diff >= range) {
    return maxcode;
  }
  return (uint32_t)(((uint64_t)diff * (uint64_t)maxcode) / (uint64_t)range);
}

/* ------------------------------------------------------------------ */
/* CFA validation                                                      */
/* ------------------------------------------------------------------ */

/* Returns 1 when the CFA is a usable 2x2 Bayer-style repeat pattern. */
static int td_cfa_usable(const tinydng_image_info *img, tinydng_error *err) {
  const tinydng_cfa *cfa = &img->cfa;
  int i;
  if (!cfa->present) {
    td_set_error(err, TINYDNG_E_INVALID_ARG, TINYDNG_STAGE_METADATA, 0, 0, 0,
                 "RGB preview requires a CFA pattern or a native 3-sample "
                 "image");
    return 0;
  }
  /* CFALayout 2 = rectangular (non-square) color filter array; the bilinear
     demosaic below assumes the square 2x2 repeat of layouts 0/1. */
  if (cfa->layout == 2u) {
    td_set_error(err, TINYDNG_E_UNSUPPORTED, TINYDNG_STAGE_METADATA, 0, 0, 0,
                 "rectangular CFA layout (CFALayout=2) not supported");
    return 0;
  }
  if (cfa->pattern_dim[0] != 2u || cfa->pattern_dim[1] != 2u ||
      cfa->pattern_size != 4u) {
    td_set_error(err, TINYDNG_E_INVALID_ARG, TINYDNG_STAGE_METADATA, 0, 0, 0,
                 "CFA repeat pattern must be 2x2 (got %ux%u, size=%u)",
                 (unsigned)cfa->pattern_dim[0],
                 (unsigned)cfa->pattern_dim[1], (unsigned)cfa->pattern_size);
    return 0;
  }
  for (i = 0; i < 4; i++) {
    uint8_t plane = cfa->pattern[i];
    if (plane >= 4u ||
        (cfa->plane_color_count > 0u &&
         plane >= cfa->plane_color_count) ||
        (cfa->plane_color_count == 0u && plane > 2u)) {
      td_set_error(err, TINYDNG_E_INVALID_ARG, TINYDNG_STAGE_METADATA, 0, 0, 0,
                   "CFA pattern plane index %u out of range",
                   (unsigned)plane);
      return 0;
    }
  }
  return 1;
}

static inline uint8_t td_cfa_plane_at(const tinydng_cfa *cfa, uint32_t x,
                                      uint32_t y) {
  return cfa->pattern[((y & 1u) << 1) | (x & 1u)];
}

/* ------------------------------------------------------------------ */
/* Output sinks                                                        */
/* ------------------------------------------------------------------ */

static void td_store_out(uint8_t *dst, size_t i, uint32_t v, int bps16) {
  if (bps16) {
    ((uint16_t *)(void *)dst)[i] = (uint16_t)v;
  } else {
    dst[i] = (uint8_t)v;
  }
}

/* Normalize every raw pixel once, using that pixel's own CFA-plane levels.
 * The temp element width matches the output depth. */
static tinydng_status td_build_norm(tinydng_context *ctx,
                                    const tinydng_image_info *img,
                                    const tinydng_pixels *raw,
                                    const int64_t black[4],
                                    const int64_t white[4], uint32_t maxcode,
                                    int bps16, void **tmp_out,
                                    tinydng_error *err) {
  size_t npix = (size_t)img->width * (size_t)img->height;
  size_t esize = bps16 ? sizeof(uint16_t) : sizeof(uint8_t);
  size_t bytes;
  void *tmp;
  size_t i;
  if (!td_safe_mul_size(npix, esize, &bytes)) {
    td_set_error(err, TINYDNG_E_BOUNDS, TINYDNG_STAGE_DECODE, 0, 0, 0,
                 "preview normalization buffer overflow");
    return TINYDNG_E_BOUNDS;
  }
  tmp = td_ctx_alloc(ctx, bytes, err);
  if (!tmp) {
    return TINYDNG_E_OOM;
  }
  for (i = 0; i < npix; i++) {
    uint32_t x = (uint32_t)(i % img->width);
    uint32_t y = (uint32_t)(i / img->width);
    uint32_t s = td_read_sample(raw, i);
    int li;
    uint32_t v;
    if (img->cfa.present) {
      uint8_t bcount = img->raw.black_level_present
                           ? img->raw.black_level_count
                           : 1u;
      li = td_cfa_level_index(img, x, y, bcount);
      if (bcount > 0u && li >= (int)bcount) {
        li = (int)bcount - 1;
      }
    } else {
      li = 0;
    }
    v = td_normalize(img, s, li, black, white, maxcode);
    if (bps16) {
      ((uint16_t *)tmp)[i] = (uint16_t)v;
    } else {
      ((uint8_t *)tmp)[i] = (uint8_t)v;
    }
  }
  *tmp_out = tmp;
  return TINYDNG_OK;
}

static inline uint32_t td_norm_get(const void *tmp, size_t i, int bps16) {
  return bps16 ? (uint32_t)((const uint16_t *)tmp)[i]
               : (uint32_t)((const uint8_t *)tmp)[i];
}

/* Bilinear Bayer demosaic: for output channel c average the normalized
 * same-plane raw samples inside the (clipped) 3x3 neighborhood. Border
 * pixels therefore reuse in-range neighbors instead of reading out of
 * bounds, and the output window keeps the full raw dimensions. */
static void td_demosaic(const tinydng_image_info *img, const void *norm,
                        uint8_t *dst, int bps16) {
  const tinydng_cfa *cfa = &img->cfa;
  uint32_t W = img->width, H = img->height;
  uint32_t y, x, c;
  for (y = 0; y < H; y++) {
    uint32_t y0 = (y > 0u) ? (y - 1u) : 0u;
    uint32_t y1 = (y + 1u < H) ? (y + 1u) : y;
    for (x = 0; x < W; x++) {
      uint32_t x0 = (x > 0u) ? (x - 1u) : 0u;
      uint32_t x1 = (x + 1u < W) ? (x + 1u) : x;
      size_t out_base = ((size_t)y * W + x) * 3u;
      uint32_t yy, xx;
      for (c = 0; c < 3u; c++) {
        uint32_t sum = 0u, cnt = 0u;
        for (yy = y0; yy <= y1; yy++) {
          for (xx = x0; xx <= x1; xx++) {
            if (td_cfa_plane_at(cfa, xx, yy) == (uint8_t)c) {
              sum += td_norm_get(norm, (size_t)yy * W + xx, bps16);
              cnt++;
            }
          }
        }
        if (cnt == 0u) { /* no same-plane neighbor: fall back to this pixel */
          sum = td_norm_get(norm, (size_t)y * W + x, bps16);
          cnt = 1u;
        }
        td_store_out(dst, out_base + c, sum / cnt, bps16);
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                  */
/* ------------------------------------------------------------------ */

tinydng_status tinydng_preview_linear(tinydng_context *ctx,
                                      const tinydng_document *doc,
                                      size_t image_idx,
                                      const tinydng_preview_options *opts,
                                      tinydng_pixels *out, tinydng_error *err) {
  const tinydng_image_info *img;
  tinydng_preview_mode mode;
  int bps16;
  uint32_t maxcode;
  uint32_t out_spp;
  size_t out_esize, out_row, out_size;
  int64_t black[4], white[4];
  tinydng_decode_options dopts;
  tinydng_pixels raw;
  uint8_t *dst;
  uint8_t owns = 0;
  void *norm = NULL;
  tinydng_status st;
  int cfa_image;

  tinydng_error_clear(err);
  if (!ctx || !doc || !out) {
    return TINYDNG_E_INVALID_ARG;
  }
  memset(out, 0, sizeof(*out));
  img = tinydng_image_get(doc, image_idx);
  if (!img) {
    td_set_error(err, TINYDNG_E_INVALID_ARG, TINYDNG_STAGE_DECODE, 0, 0, 0,
                 "image index out of range");
    return TINYDNG_E_INVALID_ARG;
  }

  mode = opts ? opts->mode : TINYDNG_PREVIEW_GRAY;
  bps16 = (opts && opts->output_bps == TINYDNG_PREVIEW_16) ? 1 : 0;
  maxcode = bps16 ? 65535u : 255u;
  if (mode != TINYDNG_PREVIEW_GRAY && mode != TINYDNG_PREVIEW_RGB) {
    td_set_error(err, TINYDNG_E_INVALID_ARG, TINYDNG_STAGE_DECODE, 0, 0, 0,
                 "unknown preview mode %d", (int)mode);
    return TINYDNG_E_INVALID_ARG;
  }
  /* A zeroed options struct defaults to 8-bit output; only an explicit
     non-zero value other than 8/16 is invalid. */
  if (opts && opts->output_bps != 0 &&
      opts->output_bps != TINYDNG_PREVIEW_8 &&
      opts->output_bps != TINYDNG_PREVIEW_16) {
    td_set_error(err, TINYDNG_E_INVALID_ARG, TINYDNG_STAGE_DECODE, 0, 0, 0,
                 "output_bps must be 8 or 16");
    return TINYDNG_E_INVALID_ARG;
  }
  if (img->width == 0u || img->height == 0u) {
    td_set_error(err, TINYDNG_E_INVALID_ARG, TINYDNG_STAGE_GEOMETRY, 0, 0, 0,
                 "image has zero dimensions");
    return TINYDNG_E_INVALID_ARG;
  }
  if (img->sample_format != TINYDNG_SAMPLEFORMAT_UINT) {
    td_set_error(err, TINYDNG_E_UNSUPPORTED, TINYDNG_STAGE_DECODE, 0, 0, 0,
                 "linear preview requires unsigned integer samples");
    return TINYDNG_E_UNSUPPORTED;
  }
  if (img->bits_per_sample == 0u || img->bits_per_sample > 32u) {
    td_set_error(err, TINYDNG_E_UNSUPPORTED, TINYDNG_STAGE_DECODE, 0, 0, 0,
                 "stored bit depth %u outside 1..32",
                 (unsigned)img->bits_per_sample);
    return TINYDNG_E_UNSUPPORTED;
  }

  cfa_image = (img->samples_per_pixel == 1u && img->cfa.present) ? 1 : 0;
  if (mode == TINYDNG_PREVIEW_RGB) {
    if (cfa_image) {
      /* Rectangular (non-square) CFALayout 2 is rejected before the generic
         CFA validation so it keeps its E_UNSUPPORTED status. */
      if (img->cfa.layout == 2u) {
        td_set_error(err, TINYDNG_E_UNSUPPORTED, TINYDNG_STAGE_METADATA, 0, 0,
                     0, "rectangular CFA layout (CFALayout=2) not supported");
        return TINYDNG_E_UNSUPPORTED;
      }
      if (!td_cfa_usable(img, err)) {
        return TINYDNG_E_INVALID_ARG;
      }
    } else if (img->samples_per_pixel < 3u) {
      td_set_error(err, TINYDNG_E_INVALID_ARG, TINYDNG_STAGE_METADATA, 0, 0, 0,
                   "RGB preview on a non-CFA image requires >= 3 samples per "
                   "pixel (got %u)",
                   (unsigned)img->samples_per_pixel);
      return TINYDNG_E_INVALID_ARG;
    } else if (img->samples_per_pixel > 3u) {
      /* The demosaic/output contract is exactly 3 channels; silently
         dropping extra planes (e.g. alpha) would be surprising. */
      td_set_error(err, TINYDNG_E_UNSUPPORTED, TINYDNG_STAGE_METADATA, 0, 0, 0,
                   "RGB preview supports 3-sample images (got %u)",
                   (unsigned)img->samples_per_pixel);
      return TINYDNG_E_UNSUPPORTED;
    }
    out_spp = 3u;
  } else {
    out_spp = 1u;
  }

  if (td_resolve_levels(img, black, white, err) != 0) {
    return TINYDNG_E_INVALID_ARG;
  }

  /* Output geometry: full raw window, overflow-checked and capped by the
     context pixel budget. */
  if (ctx->max_image_pixels &&
      (uint64_t)img->width * (uint64_t)img->height >
          ctx->max_image_pixels) {
    td_set_error(err, TINYDNG_E_BOUNDS, TINYDNG_STAGE_GEOMETRY, 0, 0, 0,
                 "preview size %ux%u exceeds max_image_pixels",
                 (unsigned)img->width, (unsigned)img->height);
    return TINYDNG_E_BOUNDS;
  }
  out_esize = bps16 ? sizeof(uint16_t) : sizeof(uint8_t);
  if (!td_safe_mul_size((size_t)img->width, out_esize * out_spp, &out_row) ||
      !td_safe_mul_size(out_row, (size_t)img->height, &out_size)) {
    td_set_error(err, TINYDNG_E_BOUNDS, TINYDNG_STAGE_GEOMETRY, 0, 0, 0,
                 "preview output size overflow");
    return TINYDNG_E_BOUNDS;
  }
  if (opts && opts->dst) {
    if (opts->dst_capacity < out_size) {
      td_set_error(err, TINYDNG_E_INVALID_ARG, TINYDNG_STAGE_DECODE, 0, 0, 0,
                   "dst capacity %zu < required %zu", opts->dst_capacity,
                   out_size);
      return TINYDNG_E_INVALID_ARG;
    }
    dst = (uint8_t *)opts->dst;
    owns = 0u;
  } else {
    dst = (uint8_t *)td_ctx_alloc(ctx, out_size, err);
    if (!dst) {
      return TINYDNG_E_OOM;
    }
    owns = 1u;
  }

  /* The raw decode itself goes through the existing, unchanged API. */
  memset(&dopts, 0, sizeof(dopts));
  if (opts) {
    dopts.num_threads = opts->num_threads;
  }
  st = tinydng_decode_image(ctx, doc, image_idx, &dopts, &raw, err);
  if (st != TINYDNG_OK) {
    goto fail;
  }
  if (raw.width != img->width || raw.height != img->height) {
    td_set_error(err, TINYDNG_E_INTERNAL, TINYDNG_STAGE_DECODE, 0, 0, 0,
                 "decoded dimensions differ from image window");
    st = TINYDNG_E_INTERNAL;
    goto fail_decode;
  }

  if (cfa_image) {
    /* Mono CFA: normalize once per raw pixel (per-plane levels), then either
       emit the mosaic directly (GRAY) or demosaic (RGB). */
    st = td_build_norm(ctx, img, &raw, black, white, maxcode, bps16, &norm,
                       err);
    if (st != TINYDNG_OK) {
      goto fail_decode;
    }
    if (mode == TINYDNG_PREVIEW_GRAY) {
      memcpy(dst, norm, out_size);
    } else {
      td_demosaic(img, norm, dst, bps16);
    }
  } else if (img->samples_per_pixel == 1u) {
    /* Plain mono (no CFA). */
    size_t npix = (size_t)img->width * (size_t)img->height;
    size_t i;
    for (i = 0; i < npix; i++) {
      uint32_t v = td_normalize(img, td_read_sample(&raw, i), 0, black, white,
                                maxcode);
      td_store_out(dst, i, v, bps16);
    }
  } else {
    /* Native multi-sample image: normalize each channel independently. */
    uint16_t spp = img->samples_per_pixel;
    uint32_t y, x, c;
    for (y = 0; y < img->height; y++) {
      for (x = 0; x < img->width; x++) {
        size_t raw_base = ((size_t)y * img->width + x) * spp;
        size_t out_base = ((size_t)y * img->width + x) * out_spp;
        if (mode == TINYDNG_PREVIEW_GRAY) {
          /* uint64: up to 4 channels at 65535 would overflow a uint32 sum. */
          uint64_t sum = 0u;
          for (c = 0; c < spp; c++) {
            int li = (img->raw.black_level_count > 1u) ? (int)c : 0;
            sum += td_normalize(img, td_read_sample(&raw, raw_base + c), li,
                                black, white, maxcode);
          }
          td_store_out(dst, out_base, (uint32_t)(sum / spp), bps16);
        } else {
          for (c = 0; c < 3u; c++) {
            int li = (img->raw.black_level_count > 1u) ? (int)c : 0;
            uint32_t v = td_normalize(img, td_read_sample(&raw, raw_base + c),
                                      li, black, white, maxcode);
            td_store_out(dst, out_base + c, v, bps16);
          }
        }
      }
    }
  }

  out->data = dst;
  out->size = out_size;
  out->width = img->width;
  out->height = img->height;
  out->samples_per_pixel = out_spp;
  out->bits_per_sample = bps16 ? 16u : 8u;
  out->sample_format = TINYDNG_SAMPLEFORMAT_UINT;
  out->owns_memory = owns;

  td_ctx_free(ctx, norm);
  tinydng_pixels_free(ctx, &raw);
  return TINYDNG_OK;

fail_decode:
  tinydng_pixels_free(ctx, &raw);
fail:
  td_ctx_free(ctx, norm);
  if (owns) {
    td_ctx_free(ctx, dst);
  }
  return st;
}
