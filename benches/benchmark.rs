use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array2, Array3};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

fn exec_clahe(size: usize, tile_sample: f64, random_input: bool) -> Array2<u8> {
    let arr: Array2<u8> = if random_input {
        Array2::random((size, size), Uniform::new(0., 255.).unwrap()).mapv(|e| e as u8)
    } else {
        Array2::zeros((size, size))
    };

    let grid_width = 8;
    let grid_height = 8;
    let clip_limit = 40;
    clahe::clahe_ndarray(arr.view(), grid_width, grid_height, clip_limit, tile_sample).unwrap()
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Input size");
    for size in [256, 512, 768, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| exec_clahe(size, 1.0, true));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("LUT sample");
    for tile_sample in 1..=8 {
        group.bench_with_input(
            BenchmarkId::from_parameter(tile_sample),
            &tile_sample,
            |b, &tile_sample| {
                b.iter(|| exec_clahe(512, tile_sample as f64, true));
            },
        );
    }
    group.finish();

    for size in [512] {
        let arr: Array2<u8> =
            Array2::random((size, size), Uniform::new(0., 255.).unwrap()).mapv(|e| e as u8);
        let grid_width = 8;
        let grid_height = 8;
        let clip_limit = 40;
        let mut group = c.benchmark_group(format!("LUT {size}"));
        for tile_sample in [1.0, 2.0, 4.0, 8.0, 16.0] {
            let (
                tile_width,
                tile_height,
                tile_step_width,
                tile_step_height,
                sampled_grid_width,
                sampled_grid_height,
            ) = clahe::calculate_tile_params(
                size as u32,
                grid_width,
                size as u32,
                grid_height,
                tile_sample,
                size as u32,
                size as u32,
            );
            group.bench_with_input(
                BenchmarkId::new("Naive", tile_sample),
                &tile_sample,
                |b, _tile_sample| {
                    b.iter(|| {
                        let a: Array3<u8> = clahe::calculate_luts_naive(
                            (sampled_grid_height, sampled_grid_width),
                            tile_step_width,
                            tile_step_height,
                            tile_width,
                            tile_height,
                            &arr,
                            clip_limit,
                        );
                        a
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("Optimized", tile_sample),
                &tile_sample,
                |b, _tile_sample| {
                    b.iter(|| {
                        let a: Array3<u8> = clahe::calculate_luts(
                            (sampled_grid_height, sampled_grid_width),
                            tile_step_width,
                            tile_step_height,
                            tile_width,
                            tile_height,
                            &arr,
                            clip_limit,
                        );
                        a
                    })
                },
            );
        }
        group.finish();
    }
}

fn criterion_u16(c: &mut Criterion) {
    let mut group = c.benchmark_group("clahe u16");
    for max_value in [256u16, 1 << 9, 1 << 10, 1 << 11, 1 << 12] {
        let arr: Array2<u16> =
            Array2::random((128, 128), Uniform::new(0., max_value as f64).unwrap())
                .mapv(|e| e as u16);

        let grid_width = 8;
        let grid_height = 8;
        let clip_limit = 40;
        group.bench_with_input(
            BenchmarkId::new("Sampled", max_value),
            &max_value,
            |b, &_max_value| {
                b.iter(|| {
                    let result: Array2<u16> =
                        clahe::clahe_ndarray(arr.view(), grid_width, grid_height, clip_limit, 2.0)
                            .unwrap();
                    result
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("W/o interpolation", max_value),
            &max_value,
            |b, &_max_value| {
                b.iter(|| {
                    let result: Array2<u16> = clahe::clahe_wo_interpolation(
                        arr.view(),
                        u32::try_from(arr.ncols()).unwrap() / grid_width,
                        u32::try_from(arr.nrows()).unwrap() / grid_height,
                        clip_limit,
                    )
                    .unwrap();
                    result
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark, criterion_u16);
criterion_main!(benches);
