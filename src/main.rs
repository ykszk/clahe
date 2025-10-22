use clahe::{clahe_image, clahe_image_wo_interpolation};
use std::path::PathBuf;

#[macro_use]
extern crate log;
use anyhow::{bail, Context, Result};
use clap::Parser;
use image::DynamicImage;
use ndarray::prelude::*;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input image filename
    input: PathBuf,
    /// Output image filename
    output: PathBuf,

    /// Grid width
    #[clap(long = "width", default_value_t = 8)]
    grid_width: u32,
    /// Grid height
    #[clap(long = "height", default_value_t = 8)]
    grid_height: u32,
    /// Clip limit
    #[clap(long = "limit", default_value_t = 40)]
    clip_limit: u32,
    /// Sampling rate for tile. Specify 0 for no interpolation
    #[clap(long = "sample", default_value_t = 1.0)]
    tile_sample: f64,

    /// Force 8 bits output even for 16 bits input
    #[clap(short, long)]
    eight: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let im = image::open(&args.input).with_context(|| format!("Loading {:?}", args.input))?;
    let wo_interpolation = args.tile_sample == 0.0;

    info!("CLAHE");
    let output = match im {
        DynamicImage::ImageLuma8(img) => {
            debug!("u8 input");
            let clahed = if wo_interpolation {
                clahe_image_wo_interpolation(
                    &img,
                    img.width() / args.grid_width,
                    img.height() / args.grid_height,
                    args.clip_limit,
                )?
            } else {
                clahe_image(
                    &img,
                    args.grid_width,
                    args.grid_height,
                    args.clip_limit,
                    args.tile_sample,
                )?
            };
            DynamicImage::ImageLuma8(clahed)
        }
        DynamicImage::ImageLuma16(img) => {
            if args.eight {
                debug!("u16 input and u8 output");
                let clahed = if wo_interpolation {
                    clahe_image_wo_interpolation(
                        &img,
                        img.width() / args.grid_width,
                        img.height() / args.grid_height,
                        args.clip_limit,
                    )?
                } else {
                    clahe_image(
                        &img,
                        args.grid_width,
                        args.grid_height,
                        args.clip_limit,
                        args.tile_sample,
                    )?
                };
                DynamicImage::ImageLuma8(clahed)
            } else {
                debug!("u16 input and u16 output");
                let clahed = if wo_interpolation {
                    clahe_image_wo_interpolation(
                        &img,
                        img.width() / args.grid_width,
                        img.height() / args.grid_height,
                        args.clip_limit,
                    )?
                } else {
                    clahe_image(
                        &img,
                        args.grid_width,
                        args.grid_height,
                        args.clip_limit,
                        args.tile_sample,
                    )?
                };
                DynamicImage::ImageLuma16(clahed)
            }
        }
        DynamicImage::ImageRgb8(img) => {
            debug!("rgb input");
            let image_shape = (img.height() as usize, img.width() as usize);
            let arr_rgb = Array2::from_shape_vec(
                (img.height() as usize * img.width() as usize, 3usize),
                img.into_vec(),
            )
            .context("Image to array")?;
            let mut arr_hsl = arr_rgb.map_axis(Axis(1), |rgb| {
                let rgb = coolor::Rgb::new(rgb[0], rgb[1], rgb[2]);
                rgb.to_hsl()
            });
            let arr_lum = arr_hsl.map(|hsl| (hsl.l * 255.0).round() as u8);
            let arr_lum = Array2::from_shape_vec(image_shape, arr_lum.into_raw_vec_and_offset().0)
                .context("Rgb array to luminosity array")?;
            let new_lum: Array2<u8> = if wo_interpolation {
                clahe::clahe_wo_interpolation(
                    arr_lum.view(),
                    args.grid_width,
                    args.grid_height,
                    args.clip_limit,
                )?
            } else {
                clahe::clahe_ndarray(
                    arr_lum.view(),
                    args.grid_width,
                    args.grid_height,
                    args.clip_limit,
                    args.tile_sample,
                )?
            };
            let new_lum = Array1::from_shape_vec(new_lum.len(), new_lum.into_raw_vec_and_offset().0)?;
            arr_hsl.zip_mut_with(&new_lum, |hsl, lum| {
                hsl.l = *lum as f32 / 255.0;
            });

            // Use `Vec` as intermediate container because I could not figure out
            // a way to directly convert Array to Image
            let mut vec_rgb = arr_rgb.into_raw_vec_and_offset().0; //vec![0; image_shape.0 * image_shape.1 * 3];
            vec_rgb.clear();
            arr_hsl.for_each(|hsl| {
                let rgb = hsl.to_rgb();
                vec_rgb.push(rgb.r);
                vec_rgb.push(rgb.g);
                vec_rgb.push(rgb.b);
            });

            let rgb_img =
                image::RgbImage::from_raw(image_shape.1 as u32, image_shape.0 as u32, vec_rgb)
                    .unwrap();
            DynamicImage::ImageRgb8(rgb_img)
        }
        invalid_image => {
            bail!(
                "L8, L16, or RGB image is expected. Found {:?}",
                invalid_image.color()
            );
        }
    };

    output
        .save(&args.output)
        .with_context(|| format!("Saving {:?}", args.output))?;
    Ok(())
}
