mod util;

use crate::util::{
    SplatPoint, compute_ssim, concat_splats, load_scene_batch, points_to_splats, project_pixel,
    save_splat, save_ssim_map, splat_means, structure_tensor_texture,
};
use brush_process::{burn_init_setup, wait_for_device};
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::gaussian_splats::SplatRenderMode;
use brush_render::kernels::camera_model::CameraModel;
use brush_render::{AlphaMode, Splats};
use brush_train::config::TrainConfig;
use brush_train::eval::eval_stats;
use brush_train::train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds};
use burn::module::AutodiffModule;
use burn::prelude::Device;
use clap::{Parser, ValueEnum};
use image::DynamicImage;
use rayon::prelude::*;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, ValueEnum)]
enum InitMode {
    Depth,
    Texture,
}

async fn init_splat_from_depth(
    camera: Camera,
    image: &DynamicImage,
    depth: &[f32],
    initial_cov_scale: f32,
    sh_degree: u32,
    render_mode: SplatRenderMode,
    initial_opacity: f32,
    stride: usize,
    added_splats: &mut [bool],
    device: &Device,
) -> Splats {
    let w = image.width() as usize;
    let h = image.height() as usize;
    let img_size = glam::UVec2::new(image.width(), image.height());
    let raw_img = image.as_rgba8().unwrap().as_raw();
    let focal = camera.focal(img_size);

    let candidates: Vec<(usize, SplatPoint)> = (0..h * w)
        .into_par_iter()
        .filter_map(|idx| {
            let d = depth[idx];
            if d <= 0.01 {
                return None;
            }

            let u = idx % w;
            let v = idx / w;

            if !u.is_multiple_of(stride) || !v.is_multiple_of(stride) {
                return None;
            }

            Some((
                idx,
                project_pixel(
                    u,
                    v,
                    d,
                    raw_img,
                    &camera,
                    img_size,
                    focal,
                    initial_cov_scale,
                ),
            ))
        })
        .collect();

    let points: Vec<SplatPoint> = candidates
        .into_iter()
        .map(|(idx, point)| {
            added_splats[idx] = true;
            point
        })
        .collect();

    points_to_splats(
        &points,
        initial_opacity,
        sh_degree,
        render_mode,
        None,
        device,
    )
}

#[allow(clippy::too_many_arguments)]
async fn init_splat_from_texture(
    camera: Camera,
    image: &DynamicImage,
    depth: &[f32],
    initial_cov_scale: f32,
    sh_degree: u32,
    render_mode: SplatRenderMode,
    initial_opacity: f32,
    num_samples: usize,
    window: usize,
    floor: f32,
    added_splats: &mut [bool],
    device: &Device,
) -> Splats {
    let w = image.width() as usize;
    let h = image.height() as usize;
    let img_size = glam::UVec2::new(image.width(), image.height());
    let raw_img = image.as_rgba8().unwrap().as_raw();
    let focal = camera.focal(img_size);

    let texture = structure_tensor_texture(image, window);
    let max_tex = texture.iter().copied().fold(0.0f32, f32::max).max(1e-12);

    let weights: Vec<f32> = (0..w * h)
        .map(|idx| {
            if depth[idx] <= 0.01 || added_splats[idx] {
                0.0
            } else {
                texture[idx] / max_tex + floor
            }
        })
        .collect();

    let valid = weights.iter().filter(|&&weight| weight > 0.0).count();
    let n = num_samples.min(valid);

    let mut rng = rand::rng();
    let sampled = rand::seq::index::sample_weighted(&mut rng, w * h, |i| weights[i], n)
        .expect("Failed to sample texture-weighted pixels");

    let points: Vec<SplatPoint> = sampled
        .iter()
        .map(|idx| {
            added_splats[idx] = true;
            project_pixel(
                idx % w,
                idx / w,
                depth[idx],
                raw_img,
                &camera,
                img_size,
                focal,
                initial_cov_scale,
            )
        })
        .collect();

    points_to_splats(
        &points,
        initial_opacity,
        sh_degree,
        render_mode,
        None,
        device,
    )
}

#[allow(clippy::too_many_arguments)]
fn ssim_injection_points(
    ssim: &[f32],
    added_splats: &mut [bool],
    depth: &[f32],
    raw_img: &[u8],
    camera: &Camera,
    img_size: glam::UVec2,
    focal: glam::Vec2,
    img_width: usize,
    img_height: usize,
    initial_cov_scale: f32,
    num_samples: usize,
    floor: f32,
) -> Vec<SplatPoint> {
    let pixels = img_width * img_height;

    let weights: Vec<f32> = (0..pixels)
        .map(|idx| {
            if added_splats[idx] || depth[idx] < 0.1 {
                0.0
            } else {
                (1.0 - ssim[idx]).max(0.0) + floor
            }
        })
        .collect();

    let valid = weights.iter().filter(|&&weight| weight > 0.0).count();
    let n = num_samples.min(valid);

    let mut rng = rand::rng();
    let sampled = rand::seq::index::sample_weighted(&mut rng, pixels, |i| weights[i], n)
        .expect("Failed to sample ssim-weighted pixels");

    sampled
        .iter()
        .map(|idx| {
            added_splats[idx] = true;
            project_pixel(
                idx % img_width,
                idx / img_width,
                depth[idx],
                raw_img,
                camera,
                img_size,
                focal,
                initial_cov_scale,
            )
        })
        .collect()
}

#[derive(clap::Parser)]
struct Args {
    /// Path to the dataset (COLMAP-style folder).
    #[arg(
        long,
        default_value = "/datasets/oak_d_lite/living_room1/colmap_with_depth"
    )]
    dataset_path: PathBuf,

    /// Image path within the dataset VFS.
    #[arg(long, default_value = "images/174893340692.png")]
    image_path: PathBuf,

    /// Depth path within the dataset VFS.
    #[arg(long, default_value = "depth/174893340692.tiff")]
    depth_path: PathBuf,

    /// Where to write the per-step training CSV.
    #[arg(long, default_value = "/tmp/train.csv")]
    out_csv_path: PathBuf,

    /// Initial covariance scale for the depth-initialised splats.
    #[arg(long, default_value = "0.005")]
    initial_cov_scale: f32,

    /// SH degree of the initial splats.
    #[arg(long, default_value = "3")]
    sh_degree: u32,

    /// Initial opacity of the depth-initialised splats.
    #[arg(long, default_value = "0.5")]
    initial_opacity: f32,

    #[arg(long, default_value = "1")]
    stride: usize,

    /// How the initial splats are placed: a strided depth grid, or pixels
    /// sampled weighted by a structure-tensor texture map.
    #[arg(long, value_enum, default_value = "depth")]
    init_mode: InitMode,

    /// Number of texture-weighted samples for --init-mode texture.
    #[arg(long, default_value = "20000")]
    texture_samples: usize,

    /// Smoothing radius for the structure tensor texture map.
    #[arg(long, default_value = "2")]
    texture_window: usize,

    /// Baseline weight added to every (normalised) texture value so flat
    /// regions still receive a few samples.
    #[arg(long, default_value = "0.05")]
    texture_floor: f32,

    /// Number of pixels sampled per refinement, weighted by SSIM error.
    #[arg(long, default_value = "5000")]
    refine_samples: usize,

    /// Baseline weight added to every pixel's SSIM error so well-reconstructed
    /// regions still have a small chance of receiving samples.
    #[arg(long, default_value = "0.0")]
    refine_floor: f32,

    /// Compute the scales of refinement-added gaussians with the KNN density
    /// trick (as in to_init_splats), querying against the existing + new
    /// gaussians together, instead of the depth-derived scale. Existing
    /// gaussians are left unchanged.
    #[arg(long, default_value = "false")]
    refine_knn_scales: bool,

    /// Folder to write intermediate .ply splats into.
    #[arg(long, default_value = "/tmp/splats")]
    splats_folder: PathBuf,

    /// Folder to write per-step SSIM maps (.npy) into.
    #[arg(long, default_value = "/tmp/ssim")]
    ssim_folder: PathBuf,

    /// Store the per-step SSIM maps (.npy) into --ssim-folder. Off by default.
    #[arg(long, default_value = "false")]
    save_ssim: bool,

    /// All training options (--total-train-iters, --lr-mean, --render-mode, ...).
    #[command(flatten)]
    train_config: TrainConfig,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    burn_init_setup().await;

    let wgpu_device = wait_for_device().await;
    let device: Device = wgpu_device.clone().into();

    let dataset_path = args.dataset_path.clone();
    let image_path = args.image_path.clone();
    let depth_path = args.depth_path.clone();

    let out_csv_path = args.out_csv_path.clone();

    let train_config = args.train_config.clone();

    let img_width = 640;
    let img_height = 480;
    let fx = 483.97064436383135;
    let fy = 484.0299202830771;
    let cx = 327.25;
    let cy = 238.22;
    let camera_model = CameraModel::Pinhole;

    let initial_cov_scale = args.initial_cov_scale;
    let sh_degree = args.sh_degree;
    let render_mode = args
        .train_config
        .render_mode
        .unwrap_or(SplatRenderMode::Default);
    let initial_opacity = args.initial_opacity;

    let camera = Camera::new(
        glam::Vec3::ZERO,
        glam::Quat::IDENTITY,
        focal_to_fov(fx, img_width, &camera_model),
        focal_to_fov(fy, img_height, &camera_model),
        glam::vec2(cx / img_width as f32, cy / img_height as f32),
        camera_model,
    );

    let (scene_batch, image, depth) =
        load_scene_batch(dataset_path, image_path, depth_path, camera).await;

    let mut added_splats = vec![false; (img_width * img_height) as usize];
    let init_start = Instant::now();
    let mut splats = match args.init_mode {
        InitMode::Depth => {
            init_splat_from_depth(
                camera,
                &image,
                &depth,
                initial_cov_scale,
                sh_degree,
                render_mode,
                initial_opacity,
                args.stride,
                &mut added_splats,
                &device,
            )
            .await
        }
        InitMode::Texture => {
            init_splat_from_texture(
                camera,
                &image,
                &depth,
                initial_cov_scale,
                sh_degree,
                render_mode,
                initial_opacity,
                args.texture_samples,
                args.texture_window,
                args.texture_floor,
                &mut added_splats,
                &device,
            )
            .await
        }
    };
    let init_duration = init_start.elapsed();

    save_splat(&args.splats_folder, "init.ply", splats.clone()).await;

    let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&train_config, &device, bounds);

    let mut csv =
        String::from("step,init_secs,refine_secs,train_secs,lr_mean,loss,psnr,ssim,num_splats\n");

    let mut refine_times: Vec<f64> = Vec::new();
    let mut train_times: Vec<f64> = Vec::new();

    let raw_img = image.as_rgba8().unwrap().as_raw();
    let img_size = glam::UVec2::new(image.width(), image.height());
    let focal = camera.focal(img_size);

    let start = Instant::now();
    for train_step in 1..train_config.total_train_iters + 1 {
        print!(
            "\rTrain step: {}/{}                  ",
            train_step, train_config.total_train_iters
        );
        std::io::stdout().flush().unwrap();

        let ssim_map = compute_ssim(
            splats.clone(),
            &camera,
            image.clone(),
            AlphaMode::Masked,
            &device,
        )
        .await
        .unwrap()
        .mean_dims(&[2]);

        let mut refine_duration = Duration::ZERO;
        if train_step.is_multiple_of(args.train_config.refine_every) {
            let refine_start = Instant::now();
            let ssim_map_cpu = ssim_map.clone().into_data_async().await.unwrap();
            let ssim_slice: &[f32] = ssim_map_cpu.as_slice().unwrap();

            let points = ssim_injection_points(
                ssim_slice,
                &mut added_splats,
                &depth,
                raw_img,
                &camera,
                img_size,
                focal,
                img_width as usize,
                img_height as usize,
                initial_cov_scale,
                args.refine_samples,
                args.refine_floor,
            );

            if !points.is_empty() {
                let knn_context = if args.refine_knn_scales {
                    Some(splat_means(&splats).await)
                } else {
                    None
                };
                let new_splat = points_to_splats(
                    &points,
                    initial_opacity,
                    sh_degree,
                    render_mode,
                    knn_context.as_deref(),
                    &device,
                );
                splats = concat_splats(&splats, &new_splat, render_mode);

                let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
                trainer = SplatTrainer::new(&train_config, &device, bounds);
            }
            refine_duration = refine_start.elapsed();
            refine_times.push(refine_duration.as_secs_f64());
        }

        let start = Instant::now();

        let diff_splats = brush_render_bwd::burn_glue::lift_splats_to_autodiff(splats.clone());
        let (new_diff, stat) = trainer.step(scene_batch.clone(), diff_splats).await;
        splats = new_diff.valid();

        let step_duration = start.elapsed();
        train_times.push(step_duration.as_secs_f64());

        let loss = stat
            .loss
            .clone()
            .into_scalar_async::<f32>()
            .await
            .expect("Failed to read loss scalar");

        let eval_stat = eval_stats(
            splats.clone(),
            &camera,
            image.clone(),
            AlphaMode::Masked,
            &device,
        )
        .await
        .unwrap();

        let psnr = eval_stat.psnr.into_scalar_async::<f32>().await.unwrap();
        let ssim = eval_stat.ssim.into_scalar_async::<f32>().await.unwrap();

        if args.save_ssim {
            save_ssim_map(
                &args.ssim_folder,
                format!("ssim_{train_step}.npy"),
                ssim_map,
            )
            .await;
        }

        let num_splats = splats.num_splats();

        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{}\n",
            train_step,
            init_duration.as_secs_f64(),
            refine_duration.as_secs_f64(),
            step_duration.as_secs_f64(),
            stat.lr_mean,
            loss,
            psnr,
            ssim,
            num_splats
        ));
    }
    let total_duration = start.elapsed();

    save_splat(&args.splats_folder, "final.ply", splats.clone()).await;

    tokio::fs::write(&out_csv_path, csv).await.unwrap();

    let mean = |times: &[f64]| {
        if times.is_empty() {
            0.0
        } else {
            times.iter().sum::<f64>() / times.len() as f64
        }
    };
    println!(
        "\n\nMean initialization time: {:.6} s",
        init_duration.as_secs_f64()
    );
    println!(
        "Mean refinement step time: {:.6} s ({} refinements)",
        mean(&refine_times),
        refine_times.len()
    );
    println!(
        "Mean train step time: {:.6} s ({} steps)",
        mean(&train_times),
        train_times.len()
    );

    println!("Total time: {:.6} s", total_duration.as_secs_f64());
}
