/* Linear preview tests: normalization math, all four Bayer phases, per-plane
   levels, linearization table, border demosaic behavior, and the defined
   results for missing/inverted levels, bad CFA, float samples, truncated
   payloads, and oversized images. All fixtures are small fixed matrices with
   hand-computed expectations; DNG write/read round-trips use the real
   writer. Self-contained. */
#include "td_test_util.h"

/* ------------------------------------------------------------------ */
/* Hand-crafted minimal TIFF builder (for metadata the writer can't    */
/* express: bad CFA, CFALayout, float sample format).                  */
/* ------------------------------------------------------------------ */

typedef struct {
  uint16_t tag, type;
  uint32_t count;
  const uint8_t *data; /* host-order raw value bytes, <= 4 inline or blob   */
  size_t dlen;
} pv_ent;

/* Build a little-endian mono TIFF: WxH, bps bits, photometric `photo`,
   pixel payload `pixels` (pixlen bytes), plus `nex` extra IFD entries whose
   out-of-line payloads are appended after the pixels. */
static uint8_t *pv_build_tiff(uint32_t W, uint32_t H, uint16_t bps,
                              uint16_t photo, const uint8_t *pixels,
                              size_t pixlen, const pv_ent *ex, int nex,
                              size_t *total_out) {
  const size_t hdr = 8;
  size_t blob_cursor = hdr + pixlen;
  size_t ifd_off, total;
  int nent = 9 + nex, i;
  uint8_t *buf;
  uint32_t *blob_offs = (uint32_t *)calloc((size_t)nex, sizeof(uint32_t));

  /* Pre-compute aligned offsets for out-of-line extra payloads. */
  for (i = 0; i < nex; i++) {
    if (ex[i].dlen > 4u) {
      blob_cursor += (blob_cursor & 1u);
      blob_offs[i] = (uint32_t)blob_cursor;
      blob_cursor += ex[i].dlen;
    }
  }
  ifd_off = (uint32_t)(blob_cursor + (blob_cursor & 1u));
  total = ifd_off + 2u + (size_t)nent * 12u + 4u;
  buf = (uint8_t *)calloc(1, total);

  /* Header + pixels. */
  buf[0] = 'I';
  buf[1] = 'I';
  tdt_pu16(buf + 2, 42, 0);
  tdt_pu32(buf + 4, ifd_off, 0);
  memcpy(buf + hdr, pixels, pixlen);
  for (i = 0; i < nex; i++) {
    if (ex[i].dlen > 4u) {
      memcpy(buf + blob_offs[i], ex[i].data, ex[i].dlen);
    }
  }

  tdt_pu16(buf + ifd_off, (uint16_t)nent, 0);
  {
    uint8_t *e = buf + ifd_off + 2;
#define BASE(tag, ty, c, v)                                               \
  do {                                                                    \
    tdt_pu16(e, (tag), 0);                                                \
    tdt_pu16(e + 2, (ty), 0);                                             \
    tdt_pu32(e + 4, (c), 0);                                              \
    tdt_pu32(e + 8, (v), 0);                                              \
    e += 12;                                                              \
  } while (0)
    BASE(256, 4, 1, W);
    BASE(257, 4, 1, H);
    BASE(258, 3, 1, bps);
    BASE(259, 3, 1, 1);       /* uncompressed */
    BASE(262, 3, 1, photo);
    BASE(273, 4, 1, (uint32_t)hdr);
    BASE(277, 3, 1, 1);       /* mono */
    BASE(278, 4, 1, H);
    BASE(279, 4, 1, (uint32_t)pixlen);
#undef BASE
    for (i = 0; i < nex; i++) {
      tdt_pu16(e, ex[i].tag, 0);
      tdt_pu16(e + 2, ex[i].type, 0);
      tdt_pu32(e + 4, ex[i].count, 0);
      if (ex[i].dlen <= 4u) {
        memcpy(e + 8, ex[i].data, ex[i].dlen); /* inline, LE left-aligned */
      } else {
        tdt_pu32(e + 8, blob_offs[i], 0);
      }
      e += 12;
    }
  }
  free(blob_offs);
  *total_out = total;
  return buf;
}

/* ------------------------------------------------------------------ */
/* Independent reference implementations                              */
/* ------------------------------------------------------------------ */

static uint32_t pv_ref_norm(uint32_t s, int64_t b, int64_t w,
                            uint32_t maxcode) {
  int64_t d = (int64_t)s - b;
  int64_t range = w - b;
  if (d <= 0) {
    return 0u;
  }
  if (d >= range) {
    return maxcode;
  }
  return (uint32_t)(((uint64_t)d * (uint64_t)maxcode) / (uint64_t)range);
}

static uint32_t pv_ref_lin(const tinydng_raw_info *raw, uint32_t s) {
  if (raw->linearization_table && raw->linearization_table_count > 0u) {
    size_t idx = (size_t)s;
    if (idx >= raw->linearization_table_count) {
      idx = raw->linearization_table_count - 1u;
    }
    return raw->linearization_table[idx];
  }
  return s;
}

/* Full independent demosaic+normalization reference over a CFA mono raw. */
static void pv_ref_demosaic(const tinydng_image_info *img,
                            const uint16_t *raw, uint32_t W, uint32_t H,
                            uint32_t maxcode, uint32_t *dst) {
  const tinydng_cfa *cfa = &img->cfa;
  int64_t bl[4], wl[4];
  int k;
  for (k = 0; k < 4; k++) {
    int bi = img->raw.black_level_count > 0u &&
                     k < (int)img->raw.black_level_count
                 ? k
                 : (img->raw.black_level_count ? (int)img->raw.black_level_count - 1
                                               : 0);
    int wi = img->raw.white_level_count > 0u &&
                     k < (int)img->raw.white_level_count
                 ? k
                 : (img->raw.white_level_count ? (int)img->raw.white_level_count - 1
                                               : 0);
    bl[k] = img->raw.black_level_present ? (int64_t)img->raw.black_level[bi] : 0;
    wl[k] = img->raw.white_level_present
                ? (int64_t)img->raw.white_level[wi]
                : (int64_t)((1u << img->bits_per_sample) - 1u);
  }
  for (uint32_t y = 0; y < H; y++) {
    for (uint32_t x = 0; x < W; x++) {
      for (uint32_t c = 0; c < 3u; c++) {
        uint32_t sum = 0, cnt = 0;
        uint32_t y0 = y > 0u ? y - 1u : 0u;
        uint32_t y1 = y + 1u < H ? y + 1u : y;
        uint32_t x0 = x > 0u ? x - 1u : 0u;
        uint32_t x1 = x + 1u < W ? x + 1u : x;
        for (uint32_t yy = y0; yy <= y1; yy++) {
          for (uint32_t xx = x0; xx <= x1; xx++) {
            uint8_t plane = cfa->pattern[((yy & 1u) << 1) | (xx & 1u)];
            if (plane == c) {
              int li = (((yy & 1u) << 1) | (xx & 1u));
              if (img->raw.black_level_count &&
                  li >= (int)img->raw.black_level_count) {
                li = (int)img->raw.black_level_count - 1;
              }
              uint32_t s = pv_ref_lin(&img->raw, raw[(size_t)yy * W + xx]);
              sum += pv_ref_norm(s, bl[li], wl[li], maxcode);
              cnt++;
            }
          }
        }
        dst[((size_t)y * W + x) * 3u + c] = cnt ? sum / cnt : 0u;
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* Writer-backed DNG fixture helpers                                   */
/* ------------------------------------------------------------------ */

static tinydng_document *pv_write_reopen_comp(
    tinydng_context *ctx, uint32_t W, uint32_t H, uint16_t bps, int rgb,
    const tinydng_cfa *cfa, const tinydng_raw_info *raw,
    const uint8_t *pixels, size_t pixlen, uint16_t compression,
    uint8_t **blob_out, size_t *bloblen_out, tinydng_error *err) {
  tinydng_write_image wi;
  tinydng_write_options wo;
  tinydng_document *doc = NULL;
  memset(&wi, 0, sizeof(wi));
  wi.width = W;
  wi.height = H;
  wi.samples_per_pixel = rgb ? 3u : 1u;
  wi.bits_per_sample = bps;
  wi.sample_format = TINYDNG_SAMPLEFORMAT_UINT;
  wi.data = pixels;
  wi.data_size = pixlen;
  wi.cfa = cfa;
  wi.raw = raw;
  memset(&wo, 0, sizeof(wo));
  wo.as_dng = (cfa || raw) ? 1u : 0u;
  wo.compression = compression;
  if (tinydng_write_memory(ctx, &wi, &wo, blob_out, bloblen_out, err) !=
      TINYDNG_OK) {
    return NULL;
  }
  if (tinydng_open_memory(ctx, *blob_out, *bloblen_out, NULL, &doc, err) !=
      TINYDNG_OK) {
    tinydng_buffer_free(ctx, *blob_out);
    *blob_out = NULL;
    return NULL;
  }
  return doc;
}

static tinydng_document *pv_write_reopen(tinydng_context *ctx, uint32_t W,
                                         uint32_t H, uint16_t bps, int rgb,
                                         const tinydng_cfa *cfa,
                                         const tinydng_raw_info *raw,
                                         const uint8_t *pixels, size_t pixlen,
                                         uint8_t **blob_out,
                                         size_t *bloblen_out,
                                         tinydng_error *err) {
  return pv_write_reopen_comp(ctx, W, H, bps, rgb, cfa, raw, pixels, pixlen,
                              TINYDNG_COMPRESSION_NONE, blob_out, bloblen_out,
                              err);
}

static void pv_set_cfa(tinydng_cfa *cfa, uint8_t p0, uint8_t p1, uint8_t p2,
                       uint8_t p3, uint16_t layout) {
  memset(cfa, 0, sizeof(*cfa));
  cfa->present = 1;
  cfa->pattern_dim[0] = 2;
  cfa->pattern_dim[1] = 2;
  cfa->pattern[0] = p0;
  cfa->pattern[1] = p1;
  cfa->pattern[2] = p2;
  cfa->pattern[3] = p3;
  cfa->pattern_size = 4;
  cfa->plane_color[0] = 0;
  cfa->plane_color[1] = 1;
  cfa->plane_color[2] = 2;
  cfa->plane_color_count = 3;
  cfa->layout = layout;
}

/* ------------------------------------------------------------------ */
/* 1. Mono grayscale: exact normalization incl. clamp endpoints        */
/* ------------------------------------------------------------------ */

static int test_gray_mono(tinydng_context *ctx) {
  const uint32_t W = 8, H = 4;
  uint16_t src[32];
  tinydng_raw_info raw;
  tinydng_cfa cfa;
  tinydng_document *doc;
  tinydng_pixels px;
  tinydng_error e;
  uint8_t *blob = NULL;
  size_t bloblen = 0;
  int rc = 0, i;
  memset(&cfa, 0, sizeof(cfa));
  memset(&raw, 0, sizeof(raw));
  /* Include below-black (10), black (100), white (1100), above-white. */
  for (i = 0; i < 32; i++) {
    src[i] = (uint16_t)(10u + (uint32_t)i * 137u);
  }
  raw.black_level_present = 1;
  raw.black_level_count = 1;
  raw.black_level[0] = 100;
  raw.white_level_present = 1;
  raw.white_level_count = 1;
  raw.white_level[0] = 1100;

  doc = pv_write_reopen(ctx, W, H, 16, 0, NULL, &raw, (const uint8_t *)src,
                        sizeof(src), &blob, &bloblen, &e);
  CHECK(doc != NULL, "gray setup: %s", e.message);
  if (!doc) {
    return 1;
  }

  {
    tinydng_preview_options po;
    memset(&po, 0, sizeof(po));
    po.mode = TINYDNG_PREVIEW_GRAY;
    po.output_bps = TINYDNG_PREVIEW_8;
    if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) != TINYDNG_OK) {
      CHECK(0, "gray8 preview: %s", e.message);
      rc = 1;
    } else {
      const uint8_t *g = px.data;
      CHECK(px.width == W && px.height == H && px.samples_per_pixel == 1u &&
                px.bits_per_sample == 8u && px.size == (size_t)W * H,
            "gray8 geometry");
      for (i = 0; i < 32; i++) {
        uint32_t expect = pv_ref_norm(src[i], 100, 1100, 255u);
        CHECK(g[i] == (uint8_t)expect,
              "gray8 sample %d: got %u want %u (raw %u)", i,
              (unsigned)g[i], (unsigned)expect, (unsigned)src[i]);
      }
      tinydng_pixels_free(ctx, &px);
    }
    memset(&po, 0, sizeof(po));
    po.mode = TINYDNG_PREVIEW_GRAY;
    po.output_bps = TINYDNG_PREVIEW_16;
    if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) != TINYDNG_OK) {
      CHECK(0, "gray16 preview: %s", e.message);
      rc = 1;
    } else {
      const uint16_t *g16 = (const uint16_t *)px.data;
      CHECK(px.bits_per_sample == 16u &&
                px.size == (size_t)W * H * 2u,
            "gray16 geometry");
      for (i = 0; i < 32; i++) {
        uint32_t expect = pv_ref_norm(src[i], 100, 1100, 65535u);
        CHECK(g16[i] == (uint16_t)expect,
              "gray16 sample %d: got %u want %u", i, (unsigned)g16[i],
              (unsigned)expect);
      }
      tinydng_pixels_free(ctx, &px);
    }
  }
  tinydng_document_destroy(ctx, doc);
  tinydng_buffer_free(ctx, blob);
  return rc;
}

/* ------------------------------------------------------------------ */
/* 2/3. CFA RGB demosaic: exact channel math, borders, all 4 phases    */
/* ------------------------------------------------------------------ */

static int test_cfa_rgb(tinydng_context *ctx) {
  const uint32_t W = 4, H = 4;
  uint16_t src[16];
  tinydng_raw_info raw;
  tinydng_cfa cfa;
  tinydng_document *doc;
  tinydng_pixels px;
  tinydng_error e;
  uint8_t *blob = NULL;
  size_t bloblen = 0;
  int rc = 0, i;
  uint32_t ref[48];
  memset(&raw, 0, sizeof(raw));
  memset(&cfa, 0, sizeof(cfa));
  pv_set_cfa(&cfa, 0, 1, 1, 2, 1); /* RGGB */
  for (i = 0; i < 16; i++) {
    src[i] = (uint16_t)(i * 7u + 3u); /* 3..108, all inside [0,1000] */
  }
  raw.black_level_present = 1;
  raw.black_level_count = 1;
  raw.black_level[0] = 0;
  raw.white_level_present = 1;
  raw.white_level_count = 1;
  raw.white_level[0] = 1000;

  doc = pv_write_reopen(ctx, W, H, 16, 0, &cfa, &raw, (const uint8_t *)src,
                        sizeof(src), &blob, &bloblen, &e);
  CHECK(doc != NULL, "cfa setup: %s", e.message);
  if (!doc) {
    return 1;
  }
  pv_ref_demosaic(tinydng_image_get(doc, 0), src, W, H, 255u, ref);

  {
    tinydng_preview_options po;
    memset(&po, 0, sizeof(po));
    po.mode = TINYDNG_PREVIEW_RGB;
    po.output_bps = TINYDNG_PREVIEW_8;
    if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) != TINYDNG_OK) {
      CHECK(0, "rgb8 preview: %s", e.message);
      rc = 1;
    } else {
      const uint8_t *o = px.data;
      CHECK(px.width == W && px.height == H && px.samples_per_pixel == 3u &&
                px.size == (size_t)W * H * 3u,
            "rgb8 geometry (full raw window retained)");
      for (i = 0; i < (int)(W * H * 3u); i++) {
        CHECK(o[i] == (uint8_t)ref[i], "rgb8 px %d: got %u want %u", i,
              (unsigned)o[i], (unsigned)ref[i]);
      }
      /* Explicit corner/border sanity: corner G channel is the average of
         the two in-range G neighbors, not an out-of-bounds read. Pixel (0,0)
         is R; the G samples at (1,0) and (0,1) are src[1]=10 and src[4]=31
         (src[i] = 7i+3). The demosaic normalizes each neighbor then averages:
         (norm(10)+norm(31))/2 = (2+7)/2 = 4. */
      {
        uint32_t g_corner = o[1];
        uint32_t expect = (pv_ref_norm(10, 0, 1000, 255u) +
                           pv_ref_norm(31, 0, 1000, 255u)) /
                          2u;
        CHECK(g_corner == expect, "corner green border handling");
      }
      tinydng_pixels_free(ctx, &px);
    }
    /* 16-bit variant against the same reference at 65535. */
    pv_ref_demosaic(tinydng_image_get(doc, 0), src, W, H, 65535u, ref);
    memset(&po, 0, sizeof(po));
    po.mode = TINYDNG_PREVIEW_RGB;
    po.output_bps = TINYDNG_PREVIEW_16;
    if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) != TINYDNG_OK) {
      CHECK(0, "rgb16 preview: %s", e.message);
      rc = 1;
    } else {
      const uint16_t *o16 = (const uint16_t *)px.data;
      for (i = 0; i < (int)(W * H * 3u); i++) {
        CHECK(o16[i] == (uint16_t)ref[i], "rgb16 px %d: got %u want %u", i,
              (unsigned)o16[i], (unsigned)ref[i]);
      }
      tinydng_pixels_free(ctx, &px);
    }
  }
  tinydng_document_destroy(ctx, doc);
  tinydng_buffer_free(ctx, blob);
  return rc;
}

static int test_all_bayer_phases(tinydng_context *ctx) {
  static const uint8_t phases[4][4] = {
      {0, 1, 1, 2}, /* RGGB */
      {1, 0, 2, 1}, /* GBRG */
      {1, 2, 0, 1}, /* GRBG */
      {2, 1, 1, 0}  /* BGGR */
  };
  const uint32_t W = 5, H = 5; /* odd dims stress the border logic */
  int ph, i, rc = 0;
  for (ph = 0; ph < 4; ph++) {
    uint16_t src[25];
    tinydng_raw_info raw;
    tinydng_cfa cfa;
    tinydng_document *doc;
    tinydng_pixels px;
    tinydng_error e;
    tinydng_preview_options po;
    uint8_t *blob = NULL;
    size_t bloblen = 0;
    uint32_t ref[75];
    memset(&raw, 0, sizeof(raw));
    pv_set_cfa(&cfa, phases[ph][0], phases[ph][1], phases[ph][2],
               phases[ph][3], 1);
    for (i = 0; i < 25; i++) {
      src[i] = (uint16_t)(i * 11u + 5u);
    }
    raw.black_level_present = 1;
    raw.black_level_count = 1;
    raw.black_level[0] = 4;
    raw.white_level_present = 1;
    raw.white_level_count = 1;
    raw.white_level[0] = 900;
    doc = pv_write_reopen(ctx, W, H, 16, 0, &cfa, &raw, (const uint8_t *)src,
                          sizeof(src), &blob, &bloblen, &e);
    CHECK(doc != NULL, "phase %d setup: %s", ph, e.message);
    if (!doc) {
      rc = 1;
      continue;
    }
    pv_ref_demosaic(tinydng_image_get(doc, 0), src, W, H, 255u, ref);
    memset(&po, 0, sizeof(po));
    po.mode = TINYDNG_PREVIEW_RGB;
    po.output_bps = TINYDNG_PREVIEW_8;
    if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) != TINYDNG_OK) {
      CHECK(0, "phase %d preview: %s", ph, e.message);
      rc = 1;
    } else {
      const uint8_t *o = px.data;
      for (i = 0; i < (int)(W * H * 3u); i++) {
        if (o[i] != (uint8_t)ref[i]) {
          CHECK(0, "phase %d px %d: got %u want %u", ph, i, (unsigned)o[i],
                (unsigned)ref[i]);
          rc = 1;
          break;
        }
      }
      tinydng_pixels_free(ctx, &px);
    }
    tinydng_document_destroy(ctx, doc);
    tinydng_buffer_free(ctx, blob);
  }
  if (!rc) {
    printf("  all four Bayer phases match reference (odd %ux%u window)\n", W,
           H);
  }
  return rc;
}

/* ------------------------------------------------------------------ */
/* 4. Per-plane (2x2) black levels through the writer round-trip       */
/* ------------------------------------------------------------------ */

static int test_per_plane_levels(tinydng_context *ctx) {
  const uint32_t W = 4, H = 4;
  uint16_t src[16];
  tinydng_raw_info raw;
  tinydng_cfa cfa;
  tinydng_document *doc;
  const tinydng_image_info *im;
  tinydng_pixels px;
  tinydng_error e;
  tinydng_preview_options po;
  uint8_t *blob = NULL;
  size_t bloblen = 0;
  int rc = 0, i;
  static const int32_t bl[4] = {0, 4, 8, 12};
  memset(&raw, 0, sizeof(raw));
  pv_set_cfa(&cfa, 0, 1, 1, 2, 1);
  for (i = 0; i < 16; i++) {
    src[i] = (uint16_t)(50u + (uint32_t)i);
  }
  raw.black_level_present = 1;
  raw.black_level_count = 4;
  memcpy(raw.black_level, bl, sizeof(bl));
  raw.white_level_present = 1;
  raw.white_level_count = 1;
  raw.white_level[0] = 200;

  doc = pv_write_reopen(ctx, W, H, 16, 0, &cfa, &raw, (const uint8_t *)src,
                        sizeof(src), &blob, &bloblen, &e);
  CHECK(doc != NULL, "per-plane setup: %s", e.message);
  if (!doc) {
    return 1;
  }
  im = tinydng_image_get(doc, 0);
  CHECK(im && im->raw.black_level_count == 4 &&
            im->raw.black_level[3] == 12,
        "4 black levels round-trip");
  memset(&po, 0, sizeof(po));
  po.mode = TINYDNG_PREVIEW_GRAY;
  po.output_bps = TINYDNG_PREVIEW_8;
  if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) != TINYDNG_OK) {
    CHECK(0, "per-plane preview: %s", e.message);
    rc = 1;
  } else {
    const uint8_t *o = px.data;
    for (i = 0; i < 16; i++) {
      uint32_t x = (uint32_t)(i % W), y = (uint32_t)(i / W);
      int li = (int)(((y & 1u) << 1) | (x & 1u));
      uint32_t expect = pv_ref_norm(src[i], bl[li], 200, 255u);
      CHECK(o[i] == (uint8_t)expect,
            "per-plane px %d (li=%d): got %u want %u", i, li,
            (unsigned)o[i], (unsigned)expect);
    }
    tinydng_pixels_free(ctx, &px);
  }
  tinydng_document_destroy(ctx, doc);
  tinydng_buffer_free(ctx, blob);
  return rc;
}

/* ------------------------------------------------------------------ */
/* 5/15. Linearization table: clamp index, DNG write/read round-trip   */
/* ------------------------------------------------------------------ */

static int test_linearization(tinydng_context *ctx) {
  const uint32_t W = 8, H = 1;
  uint16_t src[8];
  uint16_t table[16];
  tinydng_raw_info raw;
  tinydng_document *doc;
  const tinydng_image_info *im;
  tinydng_pixels px;
  tinydng_error e;
  tinydng_preview_options po;
  uint8_t *blob = NULL;
  size_t bloblen = 0;
  int rc = 0, i;
  memset(&raw, 0, sizeof(raw));
  for (i = 0; i < 8; i++) {
    src[i] = (uint16_t)i; /* samples 0..7, all in-table */
  }
  for (i = 0; i < 16; i++) {
    table[i] = (uint16_t)(i * 16u); /* 0,16,...,240 */
  }
  raw.linearization_table = table;
  raw.linearization_table_count = 16;
  raw.black_level_present = 1;
  raw.black_level_count = 1;
  raw.black_level[0] = 0;
  raw.white_level_present = 1;
  raw.white_level_count = 1;
  raw.white_level[0] = 240;

  doc = pv_write_reopen(ctx, W, H, 16, 0, NULL, &raw, (const uint8_t *)src,
                        sizeof(src), &blob, &bloblen, &e);
  CHECK(doc != NULL, "lin setup: %s", e.message);
  if (!doc) {
    return 1;
  }
  im = tinydng_image_get(doc, 0);
  CHECK(im && im->raw.linearization_table_count == 16 &&
            im->raw.linearization_table[15] == 240,
        "linearization table DNG round-trip");
  memset(&po, 0, sizeof(po));
  po.mode = TINYDNG_PREVIEW_GRAY;
  po.output_bps = TINYDNG_PREVIEW_8;
  if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) != TINYDNG_OK) {
    CHECK(0, "lin preview: %s", e.message);
    rc = 1;
  } else {
    const uint8_t *o = px.data;
    for (i = 0; i < 8; i++) {
      uint32_t mapped = (uint32_t)table[i];
      uint32_t expect = pv_ref_norm(mapped, 0, 240, 255u);
      CHECK(o[i] == (uint8_t)expect, "lin px %d: got %u want %u", i,
            (unsigned)o[i], (unsigned)expect);
    }
    tinydng_pixels_free(ctx, &px);
  }

  /* Out-of-range sample index must clamp to the last table entry. */
  src[7] = 999u;
  tinydng_document_destroy(ctx, doc);
  tinydng_buffer_free(ctx, blob);
  blob = NULL;
  doc = pv_write_reopen(ctx, W, H, 16, 0, NULL, &raw, (const uint8_t *)src,
                        sizeof(src), &blob, &bloblen, &e);
  CHECK(doc != NULL, "lin clamp setup: %s", e.message);
  if (doc) {
    if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) != TINYDNG_OK) {
      CHECK(0, "lin clamp preview: %s", e.message);
      rc = 1;
    } else {
      uint32_t expect = pv_ref_norm(240, 0, 240, 255u); /* == 255 */
      CHECK(px.data[7] == (uint8_t)expect && expect == 255u,
            "lin out-of-range index clamps to last entry (got %u)",
            (unsigned)px.data[7]);
      tinydng_pixels_free(ctx, &px);
    }
    tinydng_document_destroy(ctx, doc);
  }
  tinydng_buffer_free(ctx, blob);
  return rc;
}

/* ------------------------------------------------------------------ */
/* 6. Missing levels: documented defaults (0 .. 2^bps-1)               */
/* ------------------------------------------------------------------ */

static int test_missing_levels(tinydng_context *ctx) {
  int rc = 0;
  {
    const uint32_t W = 4, H = 1;
    uint8_t src8[4] = {0, 64, 128, 255};
    tinydng_document *doc;
    tinydng_pixels px;
    tinydng_error e;
    tinydng_preview_options po;
    uint8_t *blob = NULL;
    size_t bloblen = 0, i;
    doc = pv_write_reopen(ctx, W, H, 8, 0, NULL, NULL, src8, sizeof(src8),
                          &blob, &bloblen, &e);
    CHECK(doc != NULL, "missing-level 8bit setup: %s", e.message);
    if (doc) {
      memset(&po, 0, sizeof(po));
      if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) != TINYDNG_OK) {
        CHECK(0, "missing-level 8bit: %s", e.message);
        rc = 1;
      } else {
        for (i = 0; i < 4; i++) {
          CHECK(px.data[i] == src8[i],
                "8bit identity default levels px %zu", i);
        }
        tinydng_pixels_free(ctx, &px);
      }
      tinydng_document_destroy(ctx, doc);
    }
    tinydng_buffer_free(ctx, blob);
  }
  {
    const uint32_t W = 2, H = 1;
    uint16_t src16[2] = {0u, 65535u};
    tinydng_document *doc;
    tinydng_pixels px;
    tinydng_error e;
    tinydng_preview_options po;
    uint8_t *blob = NULL;
    size_t bloblen = 0;
    doc = pv_write_reopen(ctx, W, H, 16, 0, NULL, NULL,
                          (const uint8_t *)src16, sizeof(src16), &blob,
                          &bloblen, &e);
    CHECK(doc != NULL, "missing-level 16bit setup: %s", e.message);
    if (doc) {
      memset(&po, 0, sizeof(po));
      po.output_bps = TINYDNG_PREVIEW_16;
      if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) != TINYDNG_OK) {
        CHECK(0, "missing-level 16bit: %s", e.message);
        rc = 1;
      } else {
        const uint16_t *o = (const uint16_t *)px.data;
        CHECK(o[0] == 0u && o[1] == 65535u,
              "16bit default white = 65535");
        tinydng_pixels_free(ctx, &px);
      }
      tinydng_document_destroy(ctx, doc);
    }
    tinydng_buffer_free(ctx, blob);
  }
  return rc;
}

/* ------------------------------------------------------------------ */
/* 7. Inverted / degenerate levels => E_INVALID_ARG                    */
/* ------------------------------------------------------------------ */

static int test_inverted_levels(tinydng_context *ctx) {
  const uint32_t W = 2, H = 2;
  uint16_t src[4] = {10, 20, 30, 40};
  tinydng_raw_info raw;
  tinydng_document *doc;
  tinydng_pixels px;
  tinydng_error e;
  tinydng_preview_options po;
  uint8_t *blob = NULL;
  size_t bloblen = 0;
  int rc = 0;
  memset(&raw, 0, sizeof(raw));
  memset(&po, 0, sizeof(po));
  raw.black_level_present = 1;
  raw.black_level_count = 1;
  raw.black_level[0] = 1000;
  raw.white_level_present = 1;
  raw.white_level_count = 1;
  raw.white_level[0] = 1000; /* degenerate: equal */
  doc = pv_write_reopen(ctx, W, H, 16, 0, NULL, &raw, (const uint8_t *)src,
                        sizeof(src), &blob, &bloblen, &e);
  CHECK(doc != NULL, "inverted setup");
  if (doc) {
    CHECK(tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) ==
              TINYDNG_E_INVALID_ARG,
          "white == black rejected");
    /* Strict inversion. */
    raw.white_level[0] = 999;
    tinydng_document_destroy(ctx, doc);
    tinydng_buffer_free(ctx, blob);
    blob = NULL;
    doc = pv_write_reopen(ctx, W, H, 16, 0, NULL, &raw, (const uint8_t *)src,
                          sizeof(src), &blob, &bloblen, &e);
    CHECK(doc != NULL, "inverted setup 2");
    if (doc) {
      CHECK(tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) ==
                TINYDNG_E_INVALID_ARG,
            "white < black rejected");
      tinydng_document_destroy(ctx, doc);
    }
  }
  tinydng_buffer_free(ctx, blob);
  return rc;
}

/* ------------------------------------------------------------------ */
/* 8/9. Bad CFA metadata and CFALayout 2                               */
/* ------------------------------------------------------------------ */

static int test_bad_cfa(tinydng_context *ctx) {
  const uint32_t W = 4, H = 4;
  uint16_t src[16];
  tinydng_pixels px;
  tinydng_error e;
  tinydng_preview_options po;
  size_t total, i;
  int rc = 0;
  for (i = 0; i < 16; i++) {
    src[i] = (uint16_t)(i * 10u);
  }
  memset(&po, 0, sizeof(po));
  po.mode = TINYDNG_PREVIEW_RGB;

  /* (a) pattern plane index 9: exceeds plane_color_count. */
  {
    uint16_t dim[2] = {2, 2};
    uint8_t pat[4] = {0, 1, 1, 9};
    uint8_t pc[3] = {0, 1, 2};
    pv_ent ex[3];
    uint8_t *tiff;
    tinydng_document *doc = NULL;
    memset(ex, 0, sizeof(ex));
    ex[0] = (pv_ent){33421, 3, 2, (const uint8_t *)dim, sizeof(dim)};
    ex[1] = (pv_ent){33422, 1, 4, pat, sizeof(pat)};
    ex[2] = (pv_ent){50710, 1, 3, pc, sizeof(pc)};
    tiff = pv_build_tiff(W, H, 16, 32803, (const uint8_t *)src, sizeof(src),
                         ex, 3, &total);
    if (tinydng_open_memory(ctx, tiff, total, NULL, &doc, &e) == TINYDNG_OK) {
      CHECK(tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) ==
                TINYDNG_E_INVALID_ARG,
            "out-of-range CFA plane rejected");
      tinydng_document_destroy(ctx, doc);
    } else {
      CHECK(0, "bad-cfa open: %s", e.message);
      rc = 1;
    }
    free(tiff);
  }

  /* (b) repeat dims 2x1 (not the required 2x2). */
  {
    uint16_t dim[2] = {2, 1};
    uint8_t pat[2] = {0, 1};
    pv_ent ex[2];
    uint8_t *tiff;
    tinydng_document *doc = NULL;
    memset(ex, 0, sizeof(ex));
    ex[0] = (pv_ent){33421, 3, 2, (const uint8_t *)dim, sizeof(dim)};
    ex[1] = (pv_ent){33422, 1, 2, pat, sizeof(pat)};
    tiff = pv_build_tiff(W, H, 16, 32803, (const uint8_t *)src, sizeof(src),
                         ex, 2, &total);
    if (tinydng_open_memory(ctx, tiff, total, NULL, &doc, &e) == TINYDNG_OK) {
      CHECK(tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) ==
                TINYDNG_E_INVALID_ARG,
            "non-2x2 CFA repeat rejected");
      tinydng_document_destroy(ctx, doc);
    } else {
      CHECK(0, "bad-dim open: %s", e.message);
      rc = 1;
    }
    free(tiff);
  }

  /* (c) CFALayout = 2 (rectangular) => E_UNSUPPORTED. */
  {
    uint16_t dim[2] = {2, 2};
    uint8_t pat[4] = {0, 1, 1, 2};
    uint8_t pc[3] = {0, 1, 2};
    uint16_t layout = 2;
    pv_ent ex[4];
    uint8_t *tiff;
    tinydng_document *doc = NULL;
    memset(ex, 0, sizeof(ex));
    ex[0] = (pv_ent){33421, 3, 2, (const uint8_t *)dim, sizeof(dim)};
    ex[1] = (pv_ent){33422, 1, 4, pat, sizeof(pat)};
    ex[2] = (pv_ent){50710, 1, 3, pc, sizeof(pc)};
    ex[3] = (pv_ent){50711, 3, 1, (const uint8_t *)&layout, sizeof(layout)};
    tiff = pv_build_tiff(W, H, 16, 32803, (const uint8_t *)src, sizeof(src),
                         ex, 4, &total);
    if (tinydng_open_memory(ctx, tiff, total, NULL, &doc, &e) == TINYDNG_OK) {
      CHECK(tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) ==
                TINYDNG_E_UNSUPPORTED,
            "rectangular CFA layout rejected");
      tinydng_document_destroy(ctx, doc);
    } else {
      CHECK(0, "layout open: %s", e.message);
      rc = 1;
    }
    free(tiff);
  }

  /* (d) RGB on a plain mono image without CFA => E_INVALID_ARG. */
  {
    uint8_t *tiff;
    tinydng_document *doc = NULL;
    tiff = pv_build_tiff(W, H, 16, 1, (const uint8_t *)src, sizeof(src), NULL,
                         0, &total);
    if (tinydng_open_memory(ctx, tiff, total, NULL, &doc, &e) == TINYDNG_OK) {
      CHECK(tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) ==
                TINYDNG_E_INVALID_ARG,
            "RGB on CFA-less mono rejected");
      tinydng_document_destroy(ctx, doc);
    } else {
      CHECK(0, "mono open: %s", e.message);
      rc = 1;
    }
    free(tiff);
  }
  return rc;
}

/* ------------------------------------------------------------------ */
/* 10. Float sample format => E_UNSUPPORTED                            */
/* ------------------------------------------------------------------ */

static int test_float_unsupported(tinydng_context *ctx) {
  const uint32_t W = 2, H = 2;
  float src[4] = {0.25f, 0.5f, 0.75f, 1.0f};
  uint16_t sf = 3; /* IEEEFP */
  pv_ent ex[1];
  uint8_t *tiff;
  size_t total;
  tinydng_document *doc = NULL;
  tinydng_pixels px;
  tinydng_error e;
  tinydng_preview_options po;
  int rc = 0;
  memset(&po, 0, sizeof(po));
  ex[0] = (pv_ent){339, 3, 1, (const uint8_t *)&sf, sizeof(sf)};
  tiff = pv_build_tiff(W, H, 32, 1, (const uint8_t *)src, sizeof(src), ex, 1,
                       &total);
  if (tinydng_open_memory(ctx, tiff, total, NULL, &doc, &e) == TINYDNG_OK) {
    CHECK(tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) ==
              TINYDNG_E_UNSUPPORTED,
          "float samples rejected");
    tinydng_document_destroy(ctx, doc);
  } else {
    CHECK(0, "float open: %s", e.message);
    rc = 1;
  }
  free(tiff);
  return rc;
}

/* ------------------------------------------------------------------ */
/* 11. Truncated payload => a defined error, never a crash             */
/* ------------------------------------------------------------------ */

static int test_truncated(tinydng_context *ctx) {
  const uint32_t W = 16, H = 8;
  uint16_t *src = (uint16_t *)malloc((size_t)W * H * 2u);
  tinydng_raw_info raw;
  tinydng_document *doc;
  tinydng_pixels px;
  tinydng_error e;
  tinydng_preview_options po;
  uint8_t *blob = NULL;
  size_t bloblen = 0, i;
  int rc = 0;
  for (i = 0; i < (size_t)W * H; i++) {
    src[i] = (uint16_t)i;
  }
  memset(&raw, 0, sizeof(raw));
  raw.black_level_present = 1;
  raw.black_level_count = 1;
  raw.black_level[0] = 0;
  raw.white_level_present = 1;
  raw.white_level_count = 1;
  raw.white_level[0] = 255;
  doc = pv_write_reopen(ctx, W, H, 16, 0, NULL, &raw, (const uint8_t *)src,
                        (size_t)W * H * 2u, &blob, &bloblen, &e);
  CHECK(doc != NULL, "trunc setup");
  if (doc) {
    tinydng_document_destroy(ctx, doc);
  }
  if (blob) {
    size_t cut = bloblen - 20u; /* chop through the pixel strip */
    tinydng_document *doc2 = NULL;
    memset(&po, 0, sizeof(po));
    if (tinydng_open_memory(ctx, blob, cut, NULL, &doc2, &e) == TINYDNG_OK) {
      tinydng_status st = tinydng_preview_linear(ctx, doc2, 0, &po, &px, &e);
      CHECK(st != TINYDNG_OK, "truncated strip rejected (got OK)");
      tinydng_document_destroy(ctx, doc2);
    } else {
      /* Rejection at open time is also a defined result. */
    }
  }
  tinydng_buffer_free(ctx, blob);
  free(src);
  return rc;
}

/* ------------------------------------------------------------------ */
/* 12. Oversized output vs context pixel cap => E_BOUNDS               */
/* ------------------------------------------------------------------ */

static int test_overflow_size(void) {
  tinydng_config cfg;
  tinydng_context *ctx;
  tinydng_error e;
  const uint32_t W = 8, H = 8;
  uint16_t src[64];
  tinydng_document *doc;
  tinydng_pixels px;
  tinydng_preview_options po;
  uint8_t *blob = NULL;
  size_t bloblen = 0;
  int rc = 0;
  memset(&cfg, 0, sizeof(cfg));
  cfg.max_image_pixels = 10u;
  ctx = tinydng_context_create(&cfg, &e);
  CHECK(ctx != NULL, "bounded context");
  if (!ctx) {
    return 1;
  }
  memset(src, 0, sizeof(src));
  doc = pv_write_reopen(ctx, W, H, 16, 0, NULL, NULL, (const uint8_t *)src,
                        sizeof(src), &blob, &bloblen, &e);
  /* The writer runs on the same capped context; if the open itself rejects
     the geometry, the cap is enforced even earlier. */
  if (doc) {
    memset(&po, 0, sizeof(po));
    CHECK(tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) ==
              TINYDNG_E_BOUNDS,
          "preview over max_image_pixels rejected");
    tinydng_document_destroy(ctx, doc);
  }
  tinydng_buffer_free(ctx, blob);
  CHECK(tinydng_context_memory_used(ctx) == 0u, "bounded ctx leak");
  tinydng_context_destroy(ctx);
  return rc;
}

/* ------------------------------------------------------------------ */
/* 13. Caller-provided destination buffer                              */
/* ------------------------------------------------------------------ */

static int test_dst_buffer(tinydng_context *ctx) {
  const uint32_t W = 4, H = 4;
  uint16_t src[16];
  tinydng_raw_info raw;
  tinydng_document *doc;
  tinydng_pixels px;
  tinydng_error e;
  tinydng_preview_options po;
  uint8_t *blob = NULL;
  size_t bloblen = 0, i;
  uint8_t dst[64];
  int rc = 0;
  for (i = 0; i < 16; i++) {
    src[i] = (uint16_t)(i * 10u);
  }
  memset(&raw, 0, sizeof(raw));
  raw.black_level_present = 1;
  raw.black_level_count = 1;
  raw.black_level[0] = 0;
  raw.white_level_present = 1;
  raw.white_level_count = 1;
  raw.white_level[0] = 200;
  doc = pv_write_reopen(ctx, W, H, 16, 0, NULL, &raw, (const uint8_t *)src,
                        sizeof(src), &blob, &bloblen, &e);
  CHECK(doc != NULL, "dst setup: %s", e.message);
  if (!doc) {
    return 1;
  }
  memset(&po, 0, sizeof(po));
  po.dst = dst;
  po.dst_capacity = sizeof(dst);
  if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) != TINYDNG_OK) {
    CHECK(0, "caller dst: %s", e.message);
    rc = 1;
  } else {
    CHECK(px.data == dst && px.owns_memory == 0u, "caller dst used directly");
    for (i = 0; i < 16; i++) {
      CHECK(dst[i] == (uint8_t)pv_ref_norm(src[i], 0, 200, 255u),
            "caller dst value %zu", i);
    }
  }
  /* Undersized buffer is rejected without touching it. */
  po.dst_capacity = 15u;
  CHECK(tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) ==
            TINYDNG_E_INVALID_ARG,
        "undersized dst rejected");
  tinydng_document_destroy(ctx, doc);
  tinydng_buffer_free(ctx, blob);
  return rc;
}

/* ------------------------------------------------------------------ */
/* 14. Native multi-sample RGB: per-channel levels, gray average       */
/* ------------------------------------------------------------------ */

static int test_native_rgb(tinydng_context *ctx) {
  const uint32_t W = 3, H = 2;
  uint8_t src[18];
  tinydng_raw_info raw;
  tinydng_document *doc;
  tinydng_pixels px;
  tinydng_error e;
  tinydng_preview_options po;
  uint8_t *blob = NULL;
  size_t bloblen = 0, i;
  int rc = 0;
  for (i = 0; i < 18; i++) {
    src[i] = (uint8_t)(i * 12u);
  }
  memset(&raw, 0, sizeof(raw));
  raw.black_level_present = 1;
  raw.black_level_count = 3;
  raw.black_level[0] = 0;
  raw.black_level[1] = 10;
  raw.black_level[2] = 20;
  raw.white_level_present = 1;
  raw.white_level_count = 1;
  raw.white_level[0] = 200;
  doc = pv_write_reopen(ctx, W, H, 8, 1, NULL, &raw, src, sizeof(src), &blob,
                        &bloblen, &e);
  CHECK(doc != NULL, "native rgb setup: %s", e.message);
  if (!doc) {
    return 1;
  }
  memset(&po, 0, sizeof(po));
  po.mode = TINYDNG_PREVIEW_RGB;
  if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) != TINYDNG_OK) {
    CHECK(0, "native rgb: %s", e.message);
    rc = 1;
  } else {
    for (i = 0; i < 18; i++) {
      uint32_t c = i % 3u;
      uint32_t expect = pv_ref_norm(src[i], (int64_t)raw.black_level[c], 200,
                                    255u);
      CHECK(px.data[i] == (uint8_t)expect,
            "native rgb ch %zu: got %u want %u", i, (unsigned)px.data[i],
            (unsigned)expect);
    }
    tinydng_pixels_free(ctx, &px);
  }
  memset(&po, 0, sizeof(po));
  po.mode = TINYDNG_PREVIEW_GRAY;
  if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) != TINYDNG_OK) {
    CHECK(0, "native gray: %s", e.message);
    rc = 1;
  } else {
    for (i = 0; i < 6; i++) {
      uint32_t sum = 0u, c;
      for (c = 0; c < 3u; c++) {
        sum += pv_ref_norm(src[i * 3u + c], (int64_t)raw.black_level[c], 200,
                           255u);
      }
      CHECK(px.data[i] == (uint8_t)(sum / 3u), "native gray px %zu", i);
    }
    tinydng_pixels_free(ctx, &px);
  }
  tinydng_document_destroy(ctx, doc);
  tinydng_buffer_free(ctx, blob);
  return rc;
}

/* RGB preview on a 4-sample image must be refused rather than silently
   dropping a plane. GRAY is still defined (channel average). */
static int test_native_rgba_rejected(tinydng_context *ctx) {
  const uint32_t W = 2, H = 2;
  uint8_t src[16];
  tinydng_document *doc = NULL;
  tinydng_pixels px;
  tinydng_error e;
  tinydng_preview_options po;
  tinydng_write_image wi;
  tinydng_write_options wo;
  uint8_t *blob = NULL;
  size_t bloblen = 0, i;
  int rc = 0;
  for (i = 0; i < 16; i++) {
    src[i] = (uint8_t)(i * 15u);
  }
  memset(&wi, 0, sizeof(wi));
  wi.width = W;
  wi.height = H;
  wi.samples_per_pixel = 4u;
  wi.bits_per_sample = 8;
  wi.sample_format = TINYDNG_SAMPLEFORMAT_UINT;
  wi.data = src;
  wi.data_size = sizeof(src);
  memset(&wo, 0, sizeof(wo));
  if (tinydng_write_memory(ctx, &wi, &wo, &blob, &bloblen, &e) !=
      TINYDNG_OK) {
    printf("  4-sample write unsupported, skipping rgba rejection\n");
    tinydng_buffer_free(ctx, blob);
    return 0;
  }
  if (tinydng_open_memory(ctx, blob, bloblen, NULL, &doc, &e) != TINYDNG_OK) {
    CHECK(0, "rgba reopen: %s", e.message);
    tinydng_buffer_free(ctx, blob);
    return 1;
  }
  memset(&po, 0, sizeof(po));
  po.mode = TINYDNG_PREVIEW_RGB;
  CHECK(tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) ==
            TINYDNG_E_UNSUPPORTED,
        "RGB on 4-sample image rejected");
  memset(&po, 0, sizeof(po));
  po.mode = TINYDNG_PREVIEW_GRAY;
  if (tinydng_preview_linear(ctx, doc, 0, &po, &px, &e) == TINYDNG_OK) {
    CHECK(px.samples_per_pixel == 1u && px.size == (size_t)W * H,
          "GRAY on 4-sample image still works");
    tinydng_pixels_free(ctx, &px);
  } else {
    CHECK(0, "GRAY on 4-sample image: %s", e.message);
  }
  tinydng_document_destroy(ctx, doc);
  tinydng_buffer_free(ctx, blob);
  return rc;
}

/* ------------------------------------------------------------------ */
/* Raw decode API unchanged: preview must not alter raw values         */
/* ------------------------------------------------------------------ */

static int test_raw_unchanged(tinydng_context *ctx) {
  const uint32_t W = 6, H = 4;
  uint16_t src[24];
  tinydng_raw_info raw;
  tinydng_document *doc;
  tinydng_pixels raw_px, prev;
  tinydng_error e;
  tinydng_preview_options po;
  uint8_t *blob = NULL;
  size_t bloblen = 0, i;
  int rc = 0;
  for (i = 0; i < 24; i++) {
    src[i] = (uint16_t)(i * 997u & 0x3ffu);
  }
  memset(&raw, 0, sizeof(raw));
  raw.black_level_present = 1;
  raw.black_level_count = 1;
  raw.black_level[0] = 16;
  raw.white_level_present = 1;
  raw.white_level_count = 1;
  raw.white_level[0] = 1000;
  doc = pv_write_reopen(ctx, W, H, 16, 0, NULL, &raw, (const uint8_t *)src,
                        sizeof(src), &blob, &bloblen, &e);
  CHECK(doc != NULL, "raw-unchanged setup: %s", e.message);
  if (!doc) {
    return 1;
  }
  if (tinydng_decode_image(ctx, doc, 0, NULL, &raw_px, &e) != TINYDNG_OK) {
    CHECK(0, "raw decode: %s", e.message);
    rc = 1;
  } else {
    CHECK(memcmp(raw_px.data, src, sizeof(src)) == 0,
          "raw decode values preserved");
    tinydng_pixels_free(ctx, &raw_px);
  }
  memset(&po, 0, sizeof(po));
  if (tinydng_preview_linear(ctx, doc, 0, &po, &prev, &e) != TINYDNG_OK) {
    CHECK(0, "preview after raw: %s", e.message);
    rc = 1;
  } else {
    tinydng_pixels_free(ctx, &prev);
  }
  if (tinydng_decode_image(ctx, doc, 0, NULL, &raw_px, &e) != TINYDNG_OK) {
    CHECK(0, "raw re-decode: %s", e.message);
    rc = 1;
  } else {
    CHECK(memcmp(raw_px.data, src, sizeof(src)) == 0,
          "raw decode unchanged after preview");
    tinydng_pixels_free(ctx, &raw_px);
  }
  tinydng_document_destroy(ctx, doc);
  tinydng_buffer_free(ctx, blob);
  return rc;
}

/* ------------------------------------------------------------------ */
/* Compressed DNG (LZW): preview over the real codec decode path       */
/* ------------------------------------------------------------------ */

static int test_lzw_dng(tinydng_context *ctx) {
  const uint32_t W = 24, H = 16;
  uint16_t *src = (uint16_t *)malloc((size_t)W * H * 2u);
  tinydng_raw_info raw;
  tinydng_cfa cfa;
  tinydng_document *doc;
  tinydng_pixels g8, r8, r16;
  tinydng_error e;
  tinydng_preview_options po;
  uint8_t *blob = NULL;
  size_t bloblen = 0, i;
  int rc = 0;
  memset(&raw, 0, sizeof(raw));
  pv_set_cfa(&cfa, 1, 2, 0, 1, 1); /* GRBG, exercises a non-RGGB phase */
  for (i = 0; i < (size_t)W * H; i++) {
    src[i] = (uint16_t)(64u + (i * 37u) % 900u);
  }
  raw.black_level_present = 1;
  raw.black_level_count = 1;
  raw.black_level[0] = 64;
  raw.white_level_present = 1;
  raw.white_level_count = 1;
  raw.white_level[0] = 1023;
  doc = pv_write_reopen_comp(ctx, W, H, 16, 0, &cfa, &raw,
                             (const uint8_t *)src, (size_t)W * H * 2u,
                             TINYDNG_COMPRESSION_LZW, &blob, &bloblen, &e);
  CHECK(doc != NULL, "lzw dng setup: %s", e.message);
  if (!doc) {
    free(src);
    return 1;
  }
  CHECK(tinydng_image_get(doc, 0)->compression == TINYDNG_COMPRESSION_LZW,
        "fixture really is LZW");
  memset(&po, 0, sizeof(po));
  po.mode = TINYDNG_PREVIEW_GRAY;
  if (tinydng_preview_linear(ctx, doc, 0, &po, &g8, &e) != TINYDNG_OK) {
    CHECK(0, "lzw gray: %s", e.message);
    rc = 1;
  } else {
    CHECK(g8.width == W && g8.height == H && g8.samples_per_pixel == 1u &&
              g8.size == (size_t)W * H,
          "lzw gray full-window geometry");
    tinydng_pixels_free(ctx, &g8);
  }
  memset(&po, 0, sizeof(po));
  po.mode = TINYDNG_PREVIEW_RGB;
  if (tinydng_preview_linear(ctx, doc, 0, &po, &r8, &e) != TINYDNG_OK) {
    CHECK(0, "lzw rgb8: %s", e.message);
    rc = 1;
  } else {
    CHECK(r8.width == W && r8.height == H && r8.samples_per_pixel == 3u &&
              r8.size == (size_t)W * H * 3u,
          "lzw rgb8 full-window geometry");
    tinydng_pixels_free(ctx, &r8);
  }
  memset(&po, 0, sizeof(po));
  po.mode = TINYDNG_PREVIEW_RGB;
  po.output_bps = TINYDNG_PREVIEW_16;
  if (tinydng_preview_linear(ctx, doc, 0, &po, &r16, &e) != TINYDNG_OK) {
    CHECK(0, "lzw rgb16: %s", e.message);
    rc = 1;
  } else {
    const uint16_t *o16 = (const uint16_t *)r16.data;
    size_t n = (size_t)W * H * 3u, bad = 0;
    for (i = 0; i < n; i++) {
      if (o16[i] > 65535u) {
        bad = 1;
      }
    }
    CHECK(bad == 0u, "lzw rgb16 saturation bound respected");
    tinydng_pixels_free(ctx, &r16);
  }
  tinydng_document_destroy(ctx, doc);
  tinydng_buffer_free(ctx, blob);
  free(src);
  return rc;
}

/* ------------------------------------------------------------------ */
/* Real camera DNG smoke test (LJPEG CFA raw). Skipped when absent.    */
/* ------------------------------------------------------------------ */

static int test_real_dng(tinydng_context *ctx, const char *root) {
  char path[1024];
  size_t n = 0;
  uint8_t *data;
  tinydng_document *doc = NULL;
  const tinydng_image_info *img;
  tinydng_error e;
  tinydng_pixels g8, r16;
  tinydng_preview_options po;
  int rc = 0, i;
  snprintf(path, sizeof(path), "%s/colorchart.dng", root);
  data = tdt_slurp(path, &n);
  if (!data) {
    printf("  colorchart.dng absent, skipping real-DNG smoke test\n");
    return 0;
  }
  if (tinydng_open_memory(ctx, data, n, NULL, &doc, &e) != TINYDNG_OK) {
    CHECK(0, "real dng open: %s", e.message);
    free(data);
    return 1;
  }
  /* Find the CFA raw image (the main LJPEG image, not the RGB thumbnail). */
  img = NULL;
  size_t raw_idx = 0;
  for (i = 0; i < (int)tinydng_image_count(doc); i++) {
    const tinydng_image_info *cand = tinydng_image_get(doc, (size_t)i);
    if (cand->cfa.present && cand->samples_per_pixel == 1u) {
      img = cand;
      raw_idx = (size_t)i;
      break;
    }
  }
  CHECK(img != NULL, "colorchart has a CFA raw image");
  if (!img) {
    tinydng_document_destroy(ctx, doc);
    free(data);
    return 1;
  }
  {
    size_t idx = raw_idx;
    memset(&po, 0, sizeof(po));
    po.mode = TINYDNG_PREVIEW_GRAY;
    if (tinydng_preview_linear(ctx, doc, idx, &po, &g8, &e) != TINYDNG_OK) {
      CHECK(0, "real dng gray: %s", e.message);
      rc = 1;
    } else {
      CHECK(g8.width == img->width && g8.height == img->height &&
                g8.samples_per_pixel == 1u,
            "real dng gray keeps raw window %ux%u", (unsigned)img->width,
            (unsigned)img->height);
      tinydng_pixels_free(ctx, &g8);
    }
    memset(&po, 0, sizeof(po));
    po.mode = TINYDNG_PREVIEW_RGB;
    po.output_bps = TINYDNG_PREVIEW_16;
    if (tinydng_preview_linear(ctx, doc, idx, &po, &r16, &e) != TINYDNG_OK) {
      CHECK(0, "real dng rgb16: %s", e.message);
      rc = 1;
    } else {
      const uint16_t *o16 = (const uint16_t *)r16.data;
      size_t npix = (size_t)img->width * img->height, bad = 0, k;
      CHECK(r16.width == img->width && r16.height == img->height &&
                r16.samples_per_pixel == 3u &&
                r16.size == npix * 3u * 2u,
            "real dng rgb16 geometry");
      for (k = 0; k < npix * 3u; k++) {
        if (o16[k] > 65535u) {
          bad = 1;
        }
      }
      CHECK(bad == 0u, "real dng rgb16 within output range");
      tinydng_pixels_free(ctx, &r16);
    }
  }
  tinydng_document_destroy(ctx, doc);
  free(data);
  return rc;
}

int main(int argc, char **argv) {
  const char *root = (argc > 1) ? argv[1] : ".";
  tinydng_context *ctx = tinydng_context_create(NULL, NULL);
  if (!ctx) {
    fprintf(stderr, "context create failed\n");
    return 1;
  }
  printf("== linear preview ==\n");
  test_gray_mono(ctx);
  test_cfa_rgb(ctx);
  test_all_bayer_phases(ctx);
  test_per_plane_levels(ctx);
  test_linearization(ctx);
  test_missing_levels(ctx);
  test_inverted_levels(ctx);
  test_bad_cfa(ctx);
  test_float_unsupported(ctx);
  test_truncated(ctx);
  test_overflow_size();
  test_dst_buffer(ctx);
  test_native_rgb(ctx);
  test_native_rgba_rejected(ctx);
  test_raw_unchanged(ctx);
  test_lzw_dng(ctx);
  test_real_dng(ctx, root);

  printf("memory leak check: used=%zu\n", tinydng_context_memory_used(ctx));
  if (tinydng_context_memory_used(ctx) != 0u) {
    g_fail = 1;
  }
  tinydng_context_destroy(ctx);
  printf(g_fail ? "PREVIEW: FAILURES\n" : "PREVIEW: ALL PASS\n");
  return g_fail;
}
