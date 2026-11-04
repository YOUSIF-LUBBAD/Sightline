/* Copyright (c) 2026 Yousif Lubbad. See LICENSE. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_WIN32)
#define SL_EXPORT __declspec(dllexport)
#else
#define SL_EXPORT __attribute__((visibility("default")))
#endif

SL_EXPORT unsigned sl_abi_version(void) { return 1; }

/* Each result byte represents one tile, including the short tiles at the edges. */
SL_EXPORT int sl_diff(const uint8_t *a, const uint8_t *b, size_t width, size_t height,
                      size_t tile, uint8_t threshold, uint8_t *result, size_t capacity) {
    if (!a || !b || !result || !width || !height || !tile || width > SIZE_MAX / 3 / height)
        return -1;
    size_t cols = 1 + (width - 1) / tile;
    size_t rows = 1 + (height - 1) / tile;
    if (capacity < cols * rows) return -1;

    for (size_t ty = 0; ty < rows; ++ty) {
        size_t top = ty * tile;
        size_t height_here = height - top < tile ? height - top : tile;
        for (size_t tx = 0; tx < cols; ++tx) {
            size_t left = tx * tile;
            size_t width_here = width - left < tile ? width - left : tile;
            size_t bytes = 3 * width_here;
            int changed = 0;
            for (size_t y = top; y < top + height_here && !changed; ++y) {
                size_t offset = 3 * (y * width + left);
                if (threshold == 0) {
                    changed = memcmp(a + offset, b + offset, bytes) != 0;
                } else {
                    for (size_t i = 0; i < bytes; ++i) {
                        int delta = (int)a[offset + i] - (int)b[offset + i];
                        if (delta > threshold || delta < -(int)threshold) {
                            changed = 1;
                            break;
                        }
                    }
                }
            }
            result[ty * cols + tx] = (uint8_t)changed;
        }
    }
    return 0;
}
