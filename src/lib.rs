use image::*;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use thiserror::Error;
#[macro_use]
extern crate log;

type HistType = [u32];

#[derive(Error, Debug)]
pub enum ClaheError {
    #[error("C-Contiguous array is required")]
    InvalidMemoryLayout,
    #[error("Tile sample is too small. It must be greater than 1.0")]
    TileSampleTooSmall,
}

fn clip_hist(hist: &mut HistType, limit: u32) {
    let mut clipped: u32 = 0;

    hist.iter_mut().for_each(|count| {
        if *count > limit {
            clipped += *count - limit;
            *count = limit;
        }
    });

    if clipped > 0 {
        let hist_size = hist.len() as u32;
        let redist_batch = clipped / hist_size;

        if redist_batch > 0 {
            hist.iter_mut().for_each(|count| {
                *count += redist_batch;
            });
        }

        let residual = (clipped - redist_batch * hist_size) as usize;
        if residual > 0 {
            let step = usize::max(1, hist_size as usize / residual);
            let end = step * residual;
            hist.iter_mut().take(end).step_by(step).for_each(|count| {
                *count += 1;
            });
        }
    }
}

fn clip_hist_w_end(hist: &mut HistType, limit: u32, calc_end: usize) {
    let mut clipped: u32 = 0;

    hist.iter_mut().for_each(|count| {
        if *count > limit {
            clipped += *count - limit;
            *count = limit;
        }
    });

    if clipped > 0 {
        let hist_size = hist.len() as u32;
        let redist_batch = clipped / hist_size;

        if redist_batch > 0 {
            hist.iter_mut().take(calc_end).for_each(|count| {
                *count += redist_batch;
            });
        }

        let residual = (clipped - redist_batch * hist_size) as usize;
        if residual > 0 {
            let step = usize::max(1, hist_size as usize / residual);
            let end = usize::min(calc_end, step * residual);
            hist.iter_mut().take(end).step_by(step).for_each(|count| {
                *count += 1;
            });
        }
    }
}

/// Round half to even
#[cfg(target_arch = "x86_64")]
unsafe fn round(f: f32) -> i32 {
    let tmp = core::arch::x86_64::_mm_set_ss(f);
    core::arch::x86_64::_mm_cvtss_si32(tmp)
}

/// f32::round
#[cfg(not(target_arch = "x86_64"))]
unsafe fn round(f: f32) -> f32 {
    f.round()
}

pub trait RoundFrom {
    fn round_from(f: f32) -> Self;
}

impl RoundFrom for u8 {
    fn round_from(f: f32) -> Self {
        unsafe { round(f) as u8 }
    }
}

impl RoundFrom for u16 {
    fn round_from(f: f32) -> Self {
        unsafe { round(f) as u16 }
    }
}

fn calc_lut<T: RoundFrom>(hist: &HistType, lut: &mut [T], scale: f32) {
    let mut cumsum: u32 = 0;
    unsafe {
        for index in 0..hist.len() {
            cumsum += *hist.get_unchecked(index);
            *lut.get_unchecked_mut(index) = T::round_from(cumsum as f32 * scale);
        }
    }
}

fn _clahe_ndarray<A, B, S, T>(
    (original_input_width, original_input_height): (u32, u32),
    input: ArrayBase<S, Ix2>,
    grid_width: u32,
    grid_height: u32,
    clip_limit: u32,
    tile_sample: f64,
) -> ArrayBase<T, Ix2>
where
    S: Clone + ndarray::Data<Elem = A>,
    A: Copy + PartialOrd,
    T: Clone + ndarray::DataOwned<Elem = B> + ndarray::RawDataMut<Elem = B>,
    B: num_traits::Bounded + RoundFrom + Clone + Copy + num_traits::Zero,
    f32: From<A> + From<B>,
    u32: From<A>,
    usize: From<A>,
{
    let (input_width, input_height) = (input.ncols() as u32, input.nrows() as u32);
    let (
        tile_width,
        tile_height,
        tile_step_width,
        tile_step_height,
        sampled_grid_width,
        sampled_grid_height,
    ) = calculate_tile_params(
        original_input_width,
        grid_width,
        original_input_height,
        grid_height,
        tile_sample,
        input_width,
        input_height,
    );

    debug!("Input size {input_width} x {input_height}");
    debug!("Grid size {} x {}", grid_width, grid_height);
    debug!("Tile size {} x {}", tile_width, tile_height);

    let mut output = ArrayBase::zeros((
        original_input_height as usize,
        original_input_width as usize,
    ));
    debug!(
        "Sampled grid size {} x {}",
        sampled_grid_width, sampled_grid_height
    );

    debug!("Tile step size {} x {}", tile_step_width, tile_step_height);

    debug!("Calculate lookup tables");
    let lookup_tables: Array3<B> = if tile_sample >= 2.0 {
        calculate_luts(
            (sampled_grid_height, sampled_grid_width),
            tile_step_width,
            tile_step_height,
            tile_width,
            tile_height,
            &input,
            clip_limit,
        )
    } else {
        calculate_luts_naive(
            (sampled_grid_height, sampled_grid_width),
            tile_step_width,
            tile_step_height,
            tile_width,
            tile_height,
            &input,
            clip_limit,
        )
    };
    type Float = f32;

    debug!("Apply interpolations");

    // pre calculate x positions and weights
    let lr_luts_x_weights = calculate_lut_and_weights(
        original_input_width,
        tile_width,
        tile_step_width,
        sampled_grid_width,
        1,
    );
    info!("Max lut index {}", lr_luts_x_weights.last().unwrap().1);
    // perform interpolation
    unsafe {
        let output_ptr: *mut B = output.as_mut_ptr();
        for y in 0..(original_input_height as usize) {
            let (top_y, bottom_y, y_weight) = calculate_lut_weights_for_position(
                y,
                tile_height,
                tile_step_height,
                sampled_grid_height,
                1,
            );
            let output_row_ptr = output_ptr.add(y * original_input_width as usize);
            let top_lut = lookup_tables.slice(s![top_y as usize, .., ..]);
            let bottom_lut = lookup_tables.slice(s![bottom_y as usize, .., ..]);
            for x in 0..(original_input_width as usize) {
                let input_pixel: u32 = (*input.get((y, x)).unwrap()).into();
                let (left, right, x_weight) = *lr_luts_x_weights.get_unchecked(x);

                let top_left = *top_lut.get((left as usize, input_pixel as usize)).unwrap();
                let bottom_left = *bottom_lut
                    .get((left as usize, input_pixel as usize))
                    .unwrap();
                let top_right = *top_lut.get((right as usize, input_pixel as usize)).unwrap();
                let bottom_right = *bottom_lut
                    .get((right as usize, input_pixel as usize))
                    .unwrap();

                #[inline]
                fn interpolate<T: Into<Float>>(left: T, right: T, right_weight: Float) -> Float {
                    let left: Float = left.into();
                    let right: Float = right.into();
                    left as Float * (1.0 - right_weight) + right as Float * right_weight
                }
                let intermediate_1 = interpolate(top_left, top_right, x_weight);
                let intermediate_2 = interpolate(bottom_left, bottom_right, x_weight);
                let interpolated = interpolate(intermediate_1, intermediate_2, y_weight);
                let interpolated = B::round_from(interpolated);
                *output_row_ptr.add(x) = interpolated;
            }
        }
    }

    output
}

/// Public for benchmarking purpose only.
pub fn calculate_tile_params(
    original_input_width: u32,
    grid_width: u32,
    original_input_height: u32,
    grid_height: u32,
    tile_sample: f64,
    input_width: u32,
    input_height: u32,
) -> (u32, u32, u32, u32, u32, u32) {
    let tile_width = original_input_width / grid_width;
    let tile_height = original_input_height / grid_height;

    let tile_step_width = tile_width / tile_sample as u32;
    let tile_step_height = tile_height / tile_sample as u32;
    let sampled_grid_width = (input_width - tile_width) / tile_step_width + 1;
    let sampled_grid_height = (input_height - tile_height) / tile_step_height + 1;
    (
        tile_width,
        tile_height,
        tile_step_width,
        tile_step_height,
        sampled_grid_width,
        sampled_grid_height,
    )
}

/// hist_size is either u8::MAX or u16::MAX in OpenCV's implementation but not in this implementation.
fn calculate_clip_limit(clip_limit: u32, tile_width: u32, tile_height: u32, hist_size: u32) -> u32 {
    if clip_limit > 0 {
        let new_limit = u32::max(1, clip_limit * (tile_width * tile_height) / hist_size);
        debug!("New clip limit {}", new_limit);
        new_limit
    } else {
        0
    }
}

/// Calculate lookup tables for each tile.
/// Naive implementation without optimization for overlapping tiles.
/// Public for benchmarking purpose only.
pub fn calculate_luts_naive<A, B, S>(
    luts_shape: (u32, u32),
    tile_step_width: u32,
    tile_step_height: u32,
    tile_width: u32,
    tile_height: u32,
    input: &ArrayBase<S, Dim<[usize; 2]>>,
    clip_limit: u32,
) -> ArrayBase<ndarray::OwnedRepr<B>, Dim<[usize; 3]>>
where
    A: Copy + PartialOrd,
    B: num_traits::Bounded + RoundFrom + Clone + Copy + num_traits::Zero,
    S: Clone + ndarray::Data<Elem = A>,
    f32: From<B>,
    usize: From<A>,
{
    let max_pix_value = input.max().unwrap();

    // max_pix_value + 1 is used as the size of the histogram to reduce the computation for clip_hist and calc_lut.
    // This is different from OpenCV's size (T::Max + 1).
    // This difference does not affect test images in tests directories,
    let hist_size: usize = usize::max(u8::MAX as usize, usize::from(*max_pix_value)) + 1;
    debug!("Hist size {}", hist_size);
    let lut_size = hist_size as u32;

    let clip_limit = calculate_clip_limit(clip_limit, tile_width, tile_height, hist_size as u32);

    let lut_scale = f32::from(B::max_value()) / (tile_width * tile_height) as f32;
    let (sampled_grid_height, sampled_grid_width) = luts_shape;
    let mut lookup_tables: Array3<B> = Array3::zeros((
        sampled_grid_height as usize,
        sampled_grid_width as usize,
        lut_size as usize,
    ));
    let mut hist = vec![0; hist_size];
    for tile_y in 0..sampled_grid_height {
        for tile_x in 0..sampled_grid_width {
            let (left, top, width, height) = (
                (tile_step_width * tile_x) as usize,
                (tile_step_height * tile_y) as usize,
                tile_width as usize,
                tile_height as usize,
            );
            hist.fill(0);

            input
                .slice(s![top..top + height, left..left + width])
                .iter()
                .for_each(|pix| {
                    let hist_index: usize = (*pix).into();
                    hist[hist_index] += 1;
                });

            if clip_limit >= 1 {
                clip_hist(hist.as_mut_slice(), clip_limit);
            }
            let mut lut = lookup_tables.slice_mut(s![tile_y as usize, tile_x as usize, ..]);
            calc_lut(hist.as_slice(), lut.as_slice_mut().unwrap(), lut_scale);
        }
    }
    lookup_tables
}

/// Calculate lookup tables for each tile.
/// Public for benchmarking purpose only.
pub fn calculate_luts<A, B, S>(
    luts_shape: (u32, u32),
    tile_step_width: u32,
    tile_step_height: u32,
    tile_width: u32,
    tile_height: u32,
    input: &ArrayBase<S, Dim<[usize; 2]>>,
    clip_limit: u32,
) -> ArrayBase<ndarray::OwnedRepr<B>, Dim<[usize; 3]>>
where
    A: Copy + PartialOrd,
    B: num_traits::Bounded + RoundFrom + Clone + Copy + num_traits::Zero,
    S: Clone + ndarray::Data<Elem = A>,
    f32: From<B>,
    usize: From<A>,
{
    let max_pix_value = input.max().unwrap();

    // max_pix_value + 1 is used as the size of the histogram to reduce the computation for clip_hist and calc_lut.
    // This is different from OpenCV's size (T::Max + 1).
    // This difference does not affect test images in tests directories,
    let hist_size: usize = usize::max(u8::MAX as usize, usize::from(*max_pix_value)) + 1;
    debug!("Hist size {}", hist_size);
    let lut_size = hist_size;

    let clip_limit = calculate_clip_limit(clip_limit, tile_width, tile_height, hist_size as u32);

    let lut_scale = f32::from(B::max_value()) / (tile_width * tile_height) as f32;
    let (sampled_grid_height, sampled_grid_width) = luts_shape;
    let mut lookup_tables: Array3<B> = Array3::zeros((
        sampled_grid_height as usize,
        sampled_grid_width as usize,
        lut_size,
    ));
    let mut histograms: Array2<u32> = Array2::zeros((sampled_grid_width as usize, lut_size));
    for tile_x in 0..sampled_grid_width {
        if tile_x == 0 {
            let mut hist = histograms.index_axis_mut(Axis(0), 0);
            input
                .slice(s![0..tile_height as usize, 0..tile_width as usize])
                .iter()
                .for_each(|pix| {
                    let hist_index: usize = (*pix).into();
                    *unsafe { hist.uget_mut(hist_index) } += 1;
                });
        } else {
            let (mut cur_hist, prev_hist) = histograms
                .multi_slice_mut((s![tile_x as usize, ..], s![(tile_x - 1) as usize, ..]));
            cur_hist.assign(&prev_hist);
            let mut hist = histograms.index_axis_mut(Axis(0), tile_x as usize);

            // remove pixels on the left
            let (left, top, width, height) = (
                (tile_step_width * (tile_x - 1)) as usize,
                0,
                tile_step_width as usize,
                tile_height as usize,
            );
            input
                .slice(s![top..top + height, left..left + width])
                .iter()
                .for_each(|pix| {
                    let hist_index: usize = (*pix).into();
                    *unsafe { hist.uget_mut(hist_index) } -= 1;
                });
            // add pixels on the right
            let (left, top, width, height) = (
                (tile_step_width * (tile_x - 1) + tile_width) as usize,
                0,
                tile_step_width as usize,
                tile_height as usize,
            );
            input
                .slice(s![top..top + height, left..left + width])
                .iter()
                .for_each(|pix| {
                    let hist_index: usize = (*pix).into();
                    *unsafe { hist.uget_mut(hist_index) } += 1;
                });
        }

        // calculate lut
        let mut hist = histograms
            .index_axis_mut(Axis(0), tile_x as usize)
            .to_owned();
        if clip_limit >= 1 {
            clip_hist(hist.as_slice_mut().unwrap(), clip_limit);
        }
        let mut lut = lookup_tables.slice_mut(s![0, tile_x as usize, ..]);
        calc_lut(
            hist.as_slice().unwrap(),
            lut.as_slice_mut().unwrap(),
            lut_scale,
        );
    }
    for tile_y in 1..sampled_grid_height {
        for tile_x in 0..sampled_grid_width {
            let mut hist = histograms.index_axis_mut(Axis(0), tile_x as usize);

            // remove pixels on the top
            let (left, top, width, height) = (
                (tile_step_width * tile_x) as usize,
                (tile_step_height * (tile_y - 1)) as usize,
                tile_width as usize,
                tile_step_height as usize,
            );
            input
                .slice(s![top..top + height, left..left + width])
                .iter()
                .for_each(|pix| {
                    let hist_index: usize = (*pix).into();
                    *unsafe { hist.uget_mut(hist_index) } -= 1;
                });

            // add pixels on the bottom
            let (left, top, width, height) = (
                (tile_step_width * tile_x) as usize,
                (tile_step_height * (tile_y - 1) + tile_height) as usize,
                tile_width as usize,
                tile_step_height as usize,
            );
            input
                .slice(s![top..top + height, left..left + width])
                .iter()
                .for_each(|pix| {
                    let hist_index: usize = (*pix).into();
                    *unsafe { hist.uget_mut(hist_index) } += 1;
                });

            let mut hist = histograms
                .index_axis_mut(Axis(0), tile_x as usize)
                .to_owned();
            if clip_limit >= 1 {
                clip_hist(hist.as_slice_mut().unwrap(), clip_limit);
            }
            let mut lut = lookup_tables.slice_mut(s![tile_y as usize, tile_x as usize, ..]);
            calc_lut(
                hist.as_slice().unwrap(),
                lut.as_slice_mut().unwrap(),
                lut_scale,
            );
        }
    }
    lookup_tables
}

pub fn clahe_wo_interpolation<S, T>(
    input: ArrayView2<S>,
    tile_width: u32,
    tile_height: u32,
    clip_limit: u32,
) -> Result<Array2<T>, ClaheError>
where
    S: Copy + PartialOrd + num_traits::Zero,
    T: num_traits::Bounded + RoundFrom + Clone + Copy + num_traits::Zero,
    f32: From<S> + From<T>,
    u32: From<S>,
    usize: From<S>,
{
    let (input_width, input_height) = (input.ncols() as u32, input.nrows() as u32);
    debug!("Input size {input_width} x {input_height}");
    debug!("Tile size {} x {}", tile_width, tile_height);

    let mut output = Array2::zeros((input_height as usize, input_width as usize));
    let max_pix_value = input.max().unwrap();
    let hist_size: usize = usize::max(u8::MAX as usize, usize::from(*max_pix_value)) + 1;
    debug!("Hist size {}", hist_size);
    let lut_scale = f32::from(T::max_value()) / (tile_width * tile_height) as f32;
    let clip_limit = calculate_clip_limit(clip_limit, tile_width, tile_height, hist_size as u32);

    let mut hist: Array1<u32> = Array1::zeros(hist_size);
    let mut hist_clipped: Array1<u32> = Array1::zeros(hist_size);
    let mut hist_prev_row: Array1<u32> = Array1::zeros(hist_size);

    // use full tile for the first histogram
    let top_left_tile = input.slice(s![0..tile_height as usize, 0..tile_width as usize]);
    top_left_tile.for_each(|pix| {
        let hist_index: usize = (*pix).into();
        *unsafe { hist_prev_row.uget_mut(hist_index) } += 1;
    });
    for y in 0..input_height {
        hist.assign(&hist_prev_row);
        let tile_top = if y < tile_height / 2 {
            0
        } else {
            y - tile_height / 2
        };
        if tile_top > 0 && tile_top < input_height - tile_height {
            // remove top row
            let row = input.slice(s![(tile_top - 1) as usize, 0..tile_width as usize]);
            row.for_each(|pix| {
                let hist_index: usize = (*pix).into();
                *unsafe { hist.uget_mut(hist_index) } -= 1;
            });
            // add bottom row
            let row = input.slice(s![
                (tile_top + tile_height - 1) as usize,
                0..tile_width as usize
            ]);
            row.for_each(|pix| {
                let hist_index: usize = (*pix).into();
                *unsafe { hist.uget_mut(hist_index) } += 1;
            });
            hist_prev_row.assign(&hist);
        }

        for x in 0..input_width {
            let tile_left = if x < tile_width / 2 {
                0
            } else {
                x - tile_width / 2
            };
            let tile_top = u32::min(tile_top, input_height - tile_height - 1);
            if tile_left > 0 && tile_left < input_width - tile_width {
                // remove left column
                let col = input.slice(s![
                    tile_top as usize..(tile_top + tile_height) as usize,
                    tile_left as usize - 1
                ]);
                col.for_each(|pix| {
                    let hist_index: usize = (*pix).into();
                    *unsafe { hist.uget_mut(hist_index) } -= 1;
                });
                // add right column
                let col = input.slice(s![
                    tile_top as usize..(tile_top + tile_height) as usize,
                    (tile_left + tile_width - 1) as usize
                ]);
                col.for_each(|pix| {
                    let hist_index: usize = (*pix).into();
                    *unsafe { hist.uget_mut(hist_index) } += 1;
                });
            }
            let input_pixel: u32 = (*input.get((y as usize, x as usize)).unwrap()).into();
            let ref_hist = if clip_limit >= 1 {
                hist_clipped.assign(&hist);
                // clip_hist(hist_clipped.as_slice_mut().unwrap(), clip_limit);
                clip_hist_w_end(
                    hist_clipped.as_slice_mut().unwrap(),
                    clip_limit,
                    input_pixel as usize + 1,
                );
                hist_clipped.as_slice().unwrap()
            } else {
                hist.as_slice().unwrap()
            };
            let cumsum = ref_hist.iter().take(input_pixel as usize + 1).sum::<u32>();
            let output_pixel = T::round_from(cumsum as f32 * lut_scale);
            *output.get_mut((y as usize, x as usize)).unwrap() = output_pixel;
        }
    }
    Ok(output)
}

pub fn clahe_image_wo_interpolation<T, S>(
    input: &ImageBuffer<Luma<T>, Vec<T>>,
    tile_width: u32,
    tile_height: u32,
    clip_limit: u32,
) -> Result<ImageBuffer<Luma<S>, Vec<S>>, ClaheError>
where
    T: image::Primitive + Into<usize> + Into<u32> + Ord + RoundFrom + Default + 'static,
    S: image::Primitive + Into<usize> + Into<u32> + Ord + RoundFrom + 'static,
    f32: From<T> + From<S>,
    u32: From<T>,
    usize: From<T>,
{
    let (input_width, input_height) = input.dimensions();
    let arr = clahe_wo_interpolation(image2array_view(input), tile_width, tile_height, clip_limit)?;
    let (vec, _offset) = arr.into_raw_vec_and_offset();
    Ok(
        ImageBuffer::<Luma<S>, Vec<S>>::from_vec(input_width, input_height, vec)
            .unwrap(),
    )
}

pub fn clahe_ndarray<S, T>(
    input: ArrayView2<S>,
    grid_width: u32,
    grid_height: u32,
    clip_limit: u32,
    tile_sample: f64,
) -> Result<Array2<T>, ClaheError>
where
    S: Copy + PartialOrd + num_traits::Zero,
    T: num_traits::Bounded + RoundFrom + Clone + Copy + num_traits::Zero,
    f32: From<S> + From<T>,
    u32: From<S>,
    usize: From<S>,
{
    if tile_sample < 1.0 {
        return Err(ClaheError::TileSampleTooSmall);
    }
    if !input.is_standard_layout() {
        return Err(ClaheError::InvalidMemoryLayout);
    }
    let (input_width, input_height) = (input.ncols() as u32, input.nrows() as u32);
    let (
        _tile_width,
        _tile_height,
        _tile_step_width,
        _tile_step_height,
        sampled_grid_width,
        sampled_grid_height,
    ) = calculate_tile_params(
        input_width,
        grid_width,
        input_height,
        grid_height,
        tile_sample,
        input_width,
        input_height,
    );

    if input_width % sampled_grid_width != 0 || input_height % grid_height != 0 {
        let pad_width =
            (sampled_grid_width - input_width % sampled_grid_width) % sampled_grid_width;
        let pad_height =
            (sampled_grid_height - input_height % sampled_grid_height) % sampled_grid_height;
        debug!(
            "Padding image by {} in width and {} in height",
            pad_width, pad_height
        );
        let padded: Array2<S> = pad_array(input, 0, pad_height as usize, 0, pad_width as usize);
        Ok(_clahe_ndarray(
            (input_width, input_height),
            padded,
            grid_width,
            grid_height,
            clip_limit,
            tile_sample,
        ))
    } else {
        Ok(_clahe_ndarray(
            (input_width, input_height),
            input,
            grid_width,
            grid_height,
            clip_limit,
            tile_sample,
        ))
    }
}

fn calculate_lut_and_weights(
    original_input_width: u32,
    tile_width: u32,
    tile_step_width: u32,
    sampled_grid_width: u32,
    lut_size: u32,
) -> Vec<(u32, u32, f32)> {
    (0..(original_input_width as usize))
        .map(|x| {
            calculate_lut_weights_for_position(
                x,
                tile_width,
                tile_step_width,
                sampled_grid_width,
                lut_size,
            )
        })
        .collect::<Vec<_>>()
}

fn calculate_lut_weights_for_position(
    index: usize,
    tile_size: u32,
    step_size: u32,
    sampled_grid_size: u32,
    lut_dimension: u32,
) -> (u32, u32, f32) {
    if (index as u32) <= (tile_size / 2) {
        (0, 0, 0.0)
    } else {
        let lut_position = (index - (tile_size / 2) as usize) as f64 / step_size as f64;
        let lower_bound = lut_position.floor().min((sampled_grid_size - 1) as f64);
        let upper_bound = std::cmp::min((lower_bound + 1.0) as u32, sampled_grid_size - 1);
        let position_weight = 0.0f64.max(lut_position - lower_bound);
        let lower_bound = lower_bound as u32;
        (
            lower_bound * lut_dimension,
            upper_bound * lut_dimension,
            position_weight as f32,
        )
    }
}

pub fn image2array_view<T>(input: &ImageBuffer<Luma<T>, Vec<T>>) -> ArrayView2<T>
where
    T: image::Primitive + Into<usize> + Into<u32> + Ord + 'static,
{
    unsafe {
        ArrayView2::from_shape_ptr(
            (input.height() as usize, input.width() as usize),
            input.as_ptr(),
        )
    }
}

pub fn array2image<T>(input: Array2<T>) -> ImageBuffer<Luma<T>, Vec<T>>
where
    T: image::Primitive + Into<usize> + Into<u32> + Ord + 'static,
{
    ImageBuffer::from_vec(
        input.ncols() as u32,
        input.nrows() as u32,
        input.into_raw_vec_and_offset().0,
    )
    .unwrap()
}

/// Contrast Limited Adaptive Histogram Equalization (CLAHE)
/// Interpolation is performed for efficiency.
///
/// # Arguments
/// * `input` - GrayImage or Gray16Image
///
/// The CLAHE implementation is based on OpenCV, which is licensed under Apache 2 License.
pub fn clahe_image<T, S>(
    input: &ImageBuffer<Luma<T>, Vec<T>>,
    grid_width: u32,
    grid_height: u32,
    clip_limit: u32,
    tile_sample: f64,
) -> Result<ImageBuffer<Luma<S>, Vec<S>>, ClaheError>
where
    T: image::Primitive + Into<usize> + Into<u32> + Ord + RoundFrom + Default + 'static,
    S: image::Primitive + Into<usize> + Into<u32> + Ord + RoundFrom + 'static,
    f32: From<T> + From<S>,
    u32: From<T>,
    usize: From<T>,
{
    let (input_width, input_height) = input.dimensions();
    let arr = clahe_ndarray(
        image2array_view(input),
        grid_width,
        grid_height,
        clip_limit,
        tile_sample,
    )?;
    Ok(
        ImageBuffer::<Luma<S>, Vec<S>>::from_vec(input_width, input_height, arr.into_raw_vec_and_offset().0)
            .unwrap(),
    )
}

pub fn pad_array<A, S>(
    input: ArrayBase<S, Ix2>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
) -> Array2<A>
where
    S: ndarray::Data<Elem = A> + ndarray::RawData<Elem = A>,
    A: num_traits::Zero + Clone + Copy,
{
    let (input_width, input_height) = (input.ncols(), input.nrows());
    let (output_width, output_height) = (input_width + left + right, input_height + top + bottom);
    let mut output = ArrayBase::zeros((output_height, output_width));

    let output_ptr: *mut A = output.as_mut_ptr();
    let input_ptr = input.as_ptr();

    for y in 0..input_height {
        for x in 0..input_width {
            unsafe {
                output_ptr
                    .add(x + left + (y + top) * output_width)
                    .write(input_ptr.add(x + y * input_width).read());
                // output.unsafe_put_pixel(x + left, y + top, input.unsafe_get_pixel(x, y));
            }
        }
    }

    let calc_src_x = |dst_x: usize| -> usize {
        let src_x = dst_x as i64 - left as i64;
        if src_x < 0 {
            // left
            src_x.unsigned_abs() as usize
        } else if (src_x as usize) < input_width {
            // middle
            src_x as usize
        } else {
            // right
            2 * input_width - src_x as usize - 2
        }
    };
    // top
    for y in 0..top {
        let dst_y = y;
        let src_y = top - y;
        for dst_x in 0..output_width {
            let src_x = calc_src_x(dst_x);
            unsafe {
                output_ptr
                    .add(dst_x + dst_y * output_width)
                    .write(input_ptr.add(src_x + src_y * input_width).read());
            }
        }
    }
    // bottom
    for y in 0..bottom {
        let dst_y = y + top + input_height;
        let src_y = input_height - y - 2;
        for dst_x in 0..output_width {
            let src_x = calc_src_x(dst_x);
            unsafe {
                output_ptr
                    .add(dst_x + dst_y * output_width)
                    .write(input_ptr.add(src_x + src_y * input_width).read());
            }
        }
    }
    // middle
    for dst_y in top..(input_height + top) {
        let src_y = dst_y - top;
        for dst_x in 0..left {
            let src_x = calc_src_x(dst_x);
            unsafe {
                output_ptr
                    .add(dst_x + dst_y * output_width)
                    .write(input_ptr.add(src_x + src_y * input_width).read());
            }
        }
        for x in 0..right {
            let dst_x = x + left + input_width;
            let src_x = calc_src_x(dst_x);
            unsafe {
                output_ptr
                    .add(dst_x + dst_y * output_width)
                    .write(input_ptr.add(src_x + src_y * input_width).read());
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use pretty_assertions::assert_eq;
    use std::sync::Once;

    use super::*;

    static INIT: Once = Once::new();
    fn setup() {
        INIT.call_once(|| {
            env_logger::init();
        });
    }

    #[test]
    fn test_clip_hist() {
        setup();
        // u8
        let input = array![[0, 1, 2], [2, 4, 2], [2, 5, 2]];
        let mut hist = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        input.for_each(|i| hist[*i as usize] += 1);
        assert_eq!(hist, vec![1, 1, 5, 0, 1, 1]);
        clip_hist(hist.as_mut_slice(), 3);
        assert_eq!(hist, vec![2, 1, 3, 1, 1, 1]);

        // u16
        let input = array![[0, 1, 256], [256, 4, 256], [2, 5, 2], [256, 258, 256]];
        let mut hist = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        input.for_each(|i| hist[*i as usize] += 1);
        let mut right = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        for (i, v) in vec![1, 1, 2, 0, 1, 1].into_iter().enumerate() {
            right[i] = v;
        }
        right[256] = 5;
        right[258] = 1;
        assert_eq!(hist, right);
        clip_hist(hist.as_mut_slice(), 3);
        let mut right = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        for (i, v) in vec![2, 1, 2, 0, 1, 1].into_iter().enumerate() {
            right[i] = v;
        }
        right[129] = 1; // redistributed residual
        right[256] = 3;
        right[258] = 1;
        assert_eq!(hist, right);
    }

    #[test]
    fn test_calc_lut() {
        setup();
        // u8
        let input = array![[0, 1, 2], [2, 4, 2]];
        let mut hist = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        input.for_each(|i| hist[*i as usize] += 1);
        assert_eq!(hist, vec![1, 1, 3, 0, 1]);
        let mut lut = vec![0u8; *input.iter().max().unwrap() as usize + 1];
        let scale: f64 = 255.0 / hist.iter().sum::<u32>() as f64;
        calc_lut(hist.as_slice(), lut.as_mut_slice(), scale as f32);
        #[cfg(target_arch = "x86_64")]
        {
            // round to even
            assert_eq!(lut, vec![42, 85, 212, 212, 255]);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // round half up
            assert_eq!(lut, vec![43, 85, 213, 213, 255]);
        }

        // u16
        let input = array![[0, 1, 256], [256, 4, 256], [2, 5, 2], [256, 258, 256]];
        let mut hist = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        input.for_each(|i| hist[*i as usize] += 1);

        let mut lut = vec![0u8; *input.iter().max().unwrap() as usize + 1];
        let scale: f64 = 255.0 / hist.iter().sum::<u32>() as f64;
        calc_lut(hist.as_slice(), lut.as_mut_slice(), scale as f32);

        let mut right = vec![0u8; *input.iter().max().unwrap() as usize + 1];
        #[cfg(target_arch = "x86_64")]
        {
            // round to even
            for (i, v) in vec![21, 42, 85, 85, 106, 128].into_iter().enumerate() {
                right[i] = v;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // round half up
            for (i, v) in vec![21, 43, 85, 85, 106, 128].into_iter().enumerate() {
                right[i] = v;
            }
        }
        for r in right.iter_mut().take(256).skip(6) {
            *r = 128;
        }
        right[256] = 234;
        right[257] = 234;
        right[258] = 255;
        assert_eq!(lut, right);
    }

    #[test]
    fn test_pad_array() {
        setup();
        let input = array![[0u16, 1, 2], [3, 4, 5]];
        let output = pad_array(input, 1, 1, 1, 1);
        let expected = array![
            [4u16, 3, 4, 5, 4],
            [1, 0, 1, 2, 1],
            [4, 3, 4, 5, 4],
            [1, 0, 1, 2, 1]
        ];
        assert_eq!(output, expected);
    }

    fn _test_clahe_size(width: u32, height: u32, tile_sample: f64) -> Result<()> {
        let input = image::GrayImage::new(width, height);
        println!("width, height, tile_sample: {width}, {height}, {tile_sample}");
        let output = clahe_image::<u8, u8>(&input, 8, 8, 8, tile_sample)?;
        assert_eq!(output.width(), input.width());
        assert_eq!(output.height(), input.height());
        Ok(())
    }

    fn _test_clahe_size_smaple(tile_sample: f64) -> Result<()> {
        _test_clahe_size(512, 512, tile_sample)?;
        // _test_clahe_size(848, 1024, tile_sample)?;
        // _test_clahe_size(848, 1020, tile_sample)?;
        // _test_clahe_size(1234, 567, tile_sample)?;
        // maybe more random sizes
        Ok(())
    }

    #[test]
    fn test_clahe_size_1() -> Result<()> {
        setup();
        _test_clahe_size_smaple(1.0)?;
        Ok(())
    }

    #[test]
    fn test_clahe_size_2() -> Result<()> {
        setup();
        _test_clahe_size_smaple(2.0)?;
        Ok(())
    }

    #[test]
    fn test_clahe_size_4() -> Result<()> {
        setup();
        _test_clahe_size_smaple(4.0)?;
        Ok(())
    }

    #[test]
    fn test_calculate_lut_and_weights() {
        let original_input_width = 8;
        let tile_width = 4;
        let tile_step_width = 4;
        let sampled_grid_width = 2;
        let lut_size = 1;
        let result = calculate_lut_and_weights(
            original_input_width,
            tile_width,
            tile_step_width,
            sampled_grid_width,
            lut_size,
        );
        assert_eq!(
            result,
            vec![
                (0, 0, 0.0),
                (0, 0, 0.0),
                (0, 0, 0.0),
                (0, 1, 0.25),
                (0, 1, 0.5),
                (0, 1, 0.75),
                (1, 1, 0.0),
                (1, 1, 0.25),
            ]
        );

        let original_input_width = 8;
        let tile_width = 4;
        let tile_step_width = 2;
        let sampled_grid_width = 3;
        let lut_size = 1;
        let result = calculate_lut_and_weights(
            original_input_width,
            tile_width,
            tile_step_width,
            sampled_grid_width,
            lut_size,
        );
        assert_eq!(
            result,
            vec![
                (0, 0, 0.0),
                (0, 0, 0.0),
                (0, 0, 0.0),
                (0, 1, 0.5),
                (1, 2, 0.0),
                (1, 2, 0.5),
                (2, 2, 0.0),
                (2, 2, 0.5),
            ]
        );
    }

    #[test]
    fn test_luts() -> Result<()> {
        let size = 512;
        let arr: Array2<u8> =
            Array2::random((size, size), Uniform::new(0., 255.)).mapv(|e| e as u8);
        let grid_width = 8;
        let grid_height = 8;
        let clip_limit = 40;
        for tile_sample in [1.0, 2.0, 3.0] {
            let (
                tile_width,
                tile_height,
                tile_step_width,
                tile_step_height,
                sampled_grid_width,
                sampled_grid_height,
            ) = calculate_tile_params(
                size as u32,
                grid_width,
                size as u32,
                grid_height,
                tile_sample,
                size as u32,
                size as u32,
            );
            let expected: Array3<u8> = calculate_luts_naive(
                (sampled_grid_height, sampled_grid_width),
                tile_step_width,
                tile_step_height,
                tile_width,
                tile_height,
                &arr,
                clip_limit,
            );
            let calculated: Array3<u8> = calculate_luts(
                (sampled_grid_height, sampled_grid_width),
                tile_step_width,
                tile_step_height,
                tile_width,
                tile_height,
                &arr,
                clip_limit,
            );
            assert_eq!(expected, calculated);
        }
        Ok(())
    }
}
