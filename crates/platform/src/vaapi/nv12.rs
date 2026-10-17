//! Planar 8-bit 4:2:0 pictures into an NV12 image of a (macroblock-padded) encode surface. Safe
//! code, compiled everywhere so it is tested everywhere; `va::EncodeSession::upload` maps the
//! image and hands its bytes here.

/// Write planar 4:2:0 (`w`×`h`, tight planes, chroma `ceil(w/2)`×`ceil(h/2)`) into an NV12 image of
/// `pw`×`ph` with the given plane pitches and offsets, replicating the right / bottom edges into
/// the padding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill(
    data: &mut [u8],
    pitches: [u32; 2],
    offsets: [u32; 2],
    y: &[u8],
    u: &[u8],
    v: &[u8],
    w: usize,
    h: usize,
    pw: usize,
    ph: usize,
) -> Result<(), String> {
    let (p0, p1) = (pitches[0] as usize, pitches[1] as usize);
    let (o0, o1) = (offsets[0] as usize, offsets[1] as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    if w == 0 || h == 0 || pw < w || ph < h {
        return Err(format!("a {w}x{h} picture for a {pw}x{ph} surface"));
    }
    if p0 < pw || p1 < pw {
        return Err("image pitch smaller than the surface".into());
    }
    if y.len() < w.saturating_mul(h) || u.len() < cw.saturating_mul(ch) || v.len() < cw.saturating_mul(ch) {
        return Err("picture planes shorter than the picture".into());
    }
    for row in 0..ph {
        let r = row.min(h - 1);
        let src = y.get(r * w..r * w + w).ok_or("luma row")?;
        let at = o0.saturating_add(row.saturating_mul(p0));
        let dst = data.get_mut(at..at.saturating_add(pw)).ok_or("image too small (luma)")?;
        let (left, right) = dst.split_at_mut(w);
        left.copy_from_slice(src);
        right.fill(src.last().copied().unwrap_or(16));
    }
    for row in 0..ph.div_ceil(2) {
        let r = row.min(ch - 1);
        let (su, sv) = (u.get(r * cw..r * cw + cw).ok_or("cb row")?, v.get(r * cw..r * cw + cw).ok_or("cr row")?);
        let at = o1.saturating_add(row.saturating_mul(p1));
        let dst = data.get_mut(at..at.saturating_add(pw)).ok_or("image too small (chroma)")?;
        for (i, pair) in dst.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let c = i.min(cw - 1);
            *pair = [su.get(c).copied().unwrap_or(128), sv.get(c).copied().unwrap_or(128)];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::fill;

    #[test]
    fn pads_by_replicating_edges() {
        // 3x2 picture into a 4x4 surface, pitch 4, chroma at offset 16
        let (y, u, v) = ([1u8, 2, 3, 4, 5, 6], [10u8, 11], [20u8, 21]);
        let mut data = vec![0u8; 24];
        fill(&mut data, [4, 4], [0, 16], &y, &u, &v, 3, 2, 4, 4).unwrap();
        assert_eq!(&data[0..16], &[1, 2, 3, 3, 4, 5, 6, 6, 4, 5, 6, 6, 4, 5, 6, 6]);
        assert_eq!(&data[16..24], &[10, 20, 11, 21, 10, 20, 11, 21]);
    }

    #[test]
    fn rejects_what_does_not_fit() {
        let mut data = vec![0u8; 24];
        assert!(fill(&mut data, [4, 4], [0, 16], &[], &[], &[], 0, 0, 4, 4).is_err(), "empty picture");
        assert!(fill(&mut data, [4, 4], [0, 16], &[0; 10], &[0; 3], &[0; 3], 5, 2, 4, 4).is_err(), "wider than the surface");
        assert!(fill(&mut data, [2, 2], [0, 16], &[0; 4], &[0; 1], &[0; 1], 2, 2, 4, 4).is_err(), "pitch below the surface width");
        assert!(fill(&mut data[..8], [4, 4], [0, 16], &[0; 4], &[0; 1], &[0; 1], 2, 2, 4, 4).is_err(), "image smaller than its layout");
        assert!(fill(&mut data, [4, 4], [0, 16], &[0; 3], &[0; 1], &[0; 1], 2, 2, 4, 4).is_err(), "short luma plane");
    }
}

/// Convert straight RGBA8 directly into a pitched, padded NV12 image.
/// Matches rgba_to_yuv420_8 followed by fill, including BT.709 rounding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_rgba(data: &mut [u8], pitches: [u32; 2], offsets: [u32; 2], rgba: &[u8], w: usize, h: usize, pw: usize, ph: usize) -> Result<(), String> {
    use rayon::prelude::*;

    let (p0, p1) = (pitches[0] as usize, pitches[1] as usize);
    let (o0, o1) = (offsets[0] as usize, offsets[1] as usize);
    let cw = w.div_ceil(2);

    if w == 0 || h == 0 || pw < w || ph < h {
        return Err("invalid RGBA picture dimensions".into());
    }
    let uv_width = pw.div_ceil(2)
        .checked_mul(2)
        .ok_or("NV12 chroma width overflow")?;

    if p0 < pw || p1 < uv_width || p0 == 0 || p1 == 0 {
        return Err("NV12 pitch smaller than surface".into());
    }

    let rgba_size = w.checked_mul(h)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or("RGBA picture size overflow")?;

    if rgba.len() < rgba_size {
        return Err("RGBA buffer too small".into());
    }

    let y_end = o0.checked_add(p0.checked_mul(ph).ok_or("Y size overflow")?).ok_or("Y offset overflow")?;
    let uv_end = o1.checked_add(p1.checked_mul(ph.div_ceil(2)).ok_or("UV size overflow")?).ok_or("UV offset overflow")?;

    if o1 < y_end || uv_end > data.len() {
        return Err("invalid or overlapping NV12 planes".into());
    }

    let (before_uv, after_y) = data.split_at_mut(y_end);
    let y_plane = &mut before_uv[o0..y_end];
    let uv_plane = &mut after_y[o1 - y_end..uv_end - y_end];


    y_plane
        .par_chunks_mut(p0 * 2)
        .zip(uv_plane.par_chunks_mut(p1))
        .enumerate()
        .try_for_each(|(cy, (y_rows, uv_row))| -> Result<(), String> {
            if cy >= h.div_ceil(2) {
                return Ok(());
            }

            let row0 = cy * 2;

            for cx in 0..cw {
                let x0 = cx * 2;

                let mut us = 0.0f32;
                let mut vs = 0.0f32;
                let mut count = 0.0f32;

                for dy in 0..2 {
                    let row = row0 + dy;
                    if row >= h {
                        break;
                    }

                    for dx in 0..2 {
                        let x = x0 + dx;
                        if x >= w {
                            break;
                        }

                        let i = (row * w + x) * 4;
                        let r = rgba[i] as f32 / 255.0;
                        let g = rgba[i + 1] as f32 / 255.0;
                        let b = rgba[i + 2] as f32 / 255.0;

                        let yy = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                        y_rows[dy * p0 + x] =
                            (16.0 + 219.0 * yy).round().clamp(1.0, 254.0) as u8;

                        us += (b - yy) / 1.8556;
                        vs += (r - yy) / 1.5748;
                        count += 1.0;
                    }
                }

                uv_row[cx * 2] =
                    (128.0 + 224.0 * us / count).round().clamp(1.0, 254.0) as u8;
                uv_row[cx * 2 + 1] =
                    (128.0 + 224.0 * vs / count).round().clamp(1.0, 254.0) as u8;
            }

            for dy in 0..2 {
                let row = row0 + dy;
                if row >= h {
                    break;
                }
                let dst = &mut y_rows[dy * p0..dy * p0 + pw];
                let last = dst[w - 1];
                dst[w..pw].fill(last);
            }

            for cx in cw..pw / 2 {
                let last = (cw - 1) * 2;
                uv_row[cx * 2] = uv_row[last];
                uv_row[cx * 2 + 1] = uv_row[last + 1];
            }

            Ok(())
        })?;

    for row in h..ph {
        let src = (h - 1) * p0;
        let dst = row * p0;
        y_plane.copy_within(src..src + pw, dst);
    }

    for row in h.div_ceil(2)..ph.div_ceil(2) {
        let src = (h.div_ceil(2) - 1) * p1;
        let dst = row * p1;
        uv_plane.copy_within(src..src + pw, dst);
    }

    Ok(())
}

#[cfg(test)]
mod direct_rgba_tests {
    use super::{fill, fill_rgba};

    #[test]
    fn direct_rgba_rejects_invalid_layouts() {
        let rgba = vec![128u8; 3 * 2 * 4];
        let mut data = vec![0u8; 64];

        // Odd surface width: UV requires 4 bytes, not 3.
        assert!(fill_rgba(
            &mut data, [4, 3], [0, 16],
            &rgba, 3, 2, 3, 2
        ).is_err());

        // RGBA input shorter than the declared picture.
        assert!(fill_rgba(
            &mut data, [4, 4], [0, 16],
            &rgba[..8], 3, 2, 4, 2
        ).is_err());

        // Declared picture dimensions overflow usize.
        assert!(fill_rgba(
            &mut data, [4, 4], [0, 16],
            &rgba, usize::MAX, 2, usize::MAX, 2
        ).is_err());

        // NV12 planes overlap.
        assert!(fill_rgba(
            &mut data, [4, 4], [0, 4],
            &rgba, 3, 2, 4, 2
        ).is_err());
    }

    #[test]
    fn direct_rgba_matches_planar_path() {
        for (w, h) in [(2usize, 2usize), (3, 2), (5, 3), (16, 16), (1920, 1080), (3840, 2160)] {
            let pw = w.div_ceil(16) * 16;
            let ph = h.div_ceil(16) * 16;
            let pitch = pw + 16;
            let uv_offset = pitch * ph;
            let size = uv_offset + pitch * ph.div_ceil(2);

            let rgba: Vec<u8> = (0..w * h * 4).map(|i| ((i * 37 + i / 17) % 256) as u8).collect();

            let (mut y, mut u, mut v) = (Vec::new(), Vec::new(), Vec::new());
            filmcraft_export::rgba_to_yuv420_8(&rgba, w, h, &mut y, &mut u, &mut v);

            let mut original = vec![0u8; size];
            let mut direct = vec![0u8; size];

            let pitches = [pitch as u32, pitch as u32];
            let offsets = [0, uv_offset as u32];

            fill(&mut original, pitches, offsets, &y, &u, &v, w, h, pw, ph).unwrap();
            fill_rgba(&mut direct, pitches, offsets, &rgba, w, h, pw, ph).unwrap();

            assert_eq!(original, direct, "NV12 mismatch at {w}x{h}");
            println!("PASS: {w}x{h}");
        }
    }
}
