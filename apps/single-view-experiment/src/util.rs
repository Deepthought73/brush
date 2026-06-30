use std::path::{Path, PathBuf};
use std::sync::Arc;
use burn::Tensor;
use burn::backend::s;
use burn::prelude::{Device, Int};
use glam::Vec3;
use image::DynamicImage;
use brush_dataset::load_depth::LoadDepth;
use brush_dataset::load_image::LoadImage;
use brush_dataset::scene::{
    sample_to_packed_data, sample_to_packed_data_without_copy, view_to_sample_image, SceneBatch,
};
use brush_loss::{image_loss_eval, ImageLossConfig};
use brush_render::camera::Camera;
use brush_render::shaders::SH_C0;
use brush_render::{render_splats, AlphaMode, Splats, TextureMode};
use brush_render::gaussian_splats::{inverse_sigmoid, SplatRenderMode};
use brush_serde::SplatData;
use brush_train::{knn_scales_with_context, to_init_splats};
use brush_vfs::BrushVfs;

pub struct SplatPoint {
    pub mean: Vec3,
    pub color: f32,
    pub log_scale: f32,
}

pub fn project_pixel(
    u: usize,
    v: usize,
    depth: f32,
    raw_img: &[u8],
    camera: &Camera,
    img_size: glam::UVec2,
    focal: glam::Vec2,
    initial_cov_scale: f32,
) -> SplatPoint {
    let idx = v * img_size.x as usize + u;
    let uv = glam::Vec2::new(u as f32 + 0.5, v as f32 + 0.5);
    let pos_cam = camera.unproject(uv, depth, img_size);
    let pos_world = camera.transform(pos_cam);
    let color = (raw_img[idx * 4] as f32 / 255.0 - 0.5) / SH_C0;
    let log_scale = (initial_cov_scale * depth / focal.x).ln();
    SplatPoint {
        mean: pos_world,
        color,
        log_scale,
    }
}

pub fn points_to_splats(
    points: &[SplatPoint],
    initial_opacity: f32,
    sh_degree: u32,
    render_mode: SplatRenderMode,
    knn_context: Option<&[f32]>,
    device: &Device,
) -> Splats {
    let mut means = Vec::with_capacity(points.len() * 3);
    let mut sh_coeffs = Vec::with_capacity(points.len() * 3);
    let mut log_scales = Vec::with_capacity(points.len() * 3);
    for p in points {
        means.extend_from_slice(&[p.mean.x, p.mean.y, p.mean.z]);
        sh_coeffs.extend_from_slice(&[p.color, p.color, p.color]);
        log_scales.extend_from_slice(&[p.log_scale, p.log_scale, p.log_scale]);
    }
    // With a KNN context, scales come from the new points' density relative to
    // the existing gaussians; otherwise the per-pixel depth-derived scale.
    let log_scales = match knn_context {
        Some(existing) => knn_scales_with_context(existing, &means),
        None => log_scales,
    };
    to_init_splats(
        SplatData {
            means,
            rotations: None,
            log_scales: Some(log_scales),
            sh_coeffs: Some(sh_coeffs),
            raw_opacities: Some(vec![inverse_sigmoid(initial_opacity); points.len()]),
        },
        render_mode,
        device,
    )
    .with_sh_degree(sh_degree)
}

pub async fn splat_means(splats: &Splats) -> Vec<f32> {
    splats
        .means()
        .into_data_async()
        .await
        .expect("Failed to fetch splat means")
        .into_vec::<f32>()
        .expect("Failed to convert splat means")
}

fn box_blur(data: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    if radius == 0 {
        return data.to_vec();
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius).min(h - 1);
        for x in 0..w {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius).min(w - 1);
            let mut sum = 0.0;
            let mut count = 0.0;
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    sum += data[yy * w + xx];
                    count += 1.0;
                }
            }
            out[y * w + x] = sum / count;
        }
    }
    out
}

/// Per-pixel "texture" map: the determinant of the (window-smoothed) structure
/// tensor `J = [[Ix², IxIy], [IxIy, Iy²]]`. It is large where intensity varies
/// strongly in both directions (corners/texture) and ~0 on flat surfaces.
pub fn structure_tensor_texture(image: &DynamicImage, window: usize) -> Vec<f32> {
    let w = image.width() as usize;
    let h = image.height() as usize;
    let raw = image.as_rgba8().unwrap().as_raw();

    let gray: Vec<f32> = (0..w * h)
        .map(|i| {
            let r = raw[i * 4] as f32;
            let g = raw[i * 4 + 1] as f32;
            let b = raw[i * 4 + 2] as f32;
            (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        })
        .collect();

    let mut jxx = vec![0.0f32; w * h];
    let mut jyy = vec![0.0f32; w * h];
    let mut jxy = vec![0.0f32; w * h];
    for y in 0..h {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(h - 1);
        for x in 0..w {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            let ix = (gray[y * w + xp] - gray[y * w + xm]) * 0.5;
            let iy = (gray[yp * w + x] - gray[ym * w + x]) * 0.5;
            let idx = y * w + x;
            jxx[idx] = ix * ix;
            jyy[idx] = iy * iy;
            jxy[idx] = ix * iy;
        }
    }

    let sxx = box_blur(&jxx, w, h, window);
    let syy = box_blur(&jyy, w, h, window);
    let sxy = box_blur(&jxy, w, h, window);

    (0..w * h)
        .map(|i| (sxx[i] * syy[i] - sxy[i] * sxy[i]).max(0.0))
        .collect()
}

pub async fn compute_ssim(
    splats: Splats,
    gt_cam: &Camera,
    gt_img: DynamicImage,
    alpha_mode: AlphaMode,
    device: &Device,
) -> anyhow::Result<Tensor<3>> {
    let res = glam::uvec2(gt_img.width(), gt_img.height());

    let (gt_packed_data, _has_alpha) =
        sample_to_packed_data(view_to_sample_image(gt_img.clone(), alpha_mode));
    let gt_packed: Tensor<2, Int> = Tensor::from_data(gt_packed_data, device);

    let (img, _) = render_splats(splats, gt_cam, res, Vec3::ZERO, None, TextureMode::Float).await;
    let render_rgb = img.slice(s![.., .., 0..3]);

    let ssim = image_loss_eval(
        render_rgb.clone(),
        gt_packed,
        ImageLossConfig {
            l1_weight: 0.0,
            ssim_weight: 1.0,
            composite_bg: None,
            mask: false,
        },
    );

    Ok(ssim)
}

pub async fn load_scene_batch(
    dataset_path: PathBuf,
    image_path: PathBuf,
    depth_path: PathBuf,
    camera: Camera,
) -> (SceneBatch, DynamicImage, Vec<f32>) {
    let vfs = Arc::new(BrushVfs::from_path(&dataset_path).await.unwrap());

    let load_image = LoadImage::new(vfs.clone(), image_path.clone(), None, 4000, None);
    let load_depth = LoadDepth::new(vfs.clone(), depth_path.clone());

    let image = load_image.load().await.unwrap();
    let image = DynamicImage::ImageRgba8(image.to_rgba8());

    let depth = load_depth
        .load(image.height() as usize, image.width() as usize)
        .await
        .unwrap();
    let depth_vec = load_depth
        .load_vec(image.height() as usize, image.width() as usize)
        .await
        .unwrap();

    let (img_packed, has_alpha) = sample_to_packed_data_without_copy(&image);

    (
        SceneBatch {
            img_packed,
            has_alpha,
            alpha_mode: AlphaMode::Masked,
            camera,
            depth: Some(depth),
        },
        image,
        depth_vec,
    )
}

pub async fn save_splat<A: AsRef<Path>, B: AsRef<Path>>(folder: A, filename: B, splats: Splats) {
    let folder = folder.as_ref().to_owned();
    tokio::fs::create_dir_all(&folder).await.unwrap();
    let splat_data = brush_serde::splat_to_ply(splats).await.unwrap();
    tokio::fs::write(folder.join(filename.as_ref()), splat_data)
        .await
        .unwrap();
}


/// Save a `[H, W, 1]` SSIM map as a 2D `[H, W]` little-endian float32 `.npy`
/// file, so it can be loaded with `numpy.load` and shown with matplotlib.
pub async fn save_ssim_map<P: AsRef<std::path::Path>>(
    folder: P,
    filename: String,
    ssim_map: Tensor<3>,
) {
    let folder = folder.as_ref().to_owned();
    tokio::fs::create_dir_all(&folder).await.unwrap();

    let [h, w, _] = ssim_map.dims();
    let data = ssim_map
        .into_data_async()
        .await
        .expect("Failed to read SSIM map")
        .into_vec::<f32>()
        .expect("Failed to convert SSIM map");

    // Minimal NPY v1.0 container for a C-contiguous f32 [H, W] array.
    let mut bytes = Vec::with_capacity(128 + data.len() * 4);
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({h}, {w}), }}"
    );
    // Pad with spaces + trailing newline so the 10-byte preamble plus header
    // length is a multiple of 64 (the NPY alignment requirement).
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    header.push('\n');
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in &data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    tokio::fs::write(folder.join(filename), bytes)
        .await
        .unwrap();
}

pub fn concat_splats(a: &Splats, b: &Splats, mode: SplatRenderMode) -> Splats {
    let means = Tensor::cat(vec![a.means(), b.means()], 0);
    let rotations = Tensor::cat(vec![a.rotations(), b.rotations()], 0);
    let log_scales = Tensor::cat(vec![a.log_scales(), b.log_scales()], 0);
    let sh_coeffs = Tensor::cat(vec![a.sh_coeffs.val(), b.sh_coeffs.val()], 0);
    let opacities = Tensor::cat(vec![a.raw_opacities.val(), b.raw_opacities.val()], 0);
    Splats::from_tensor_data(means, rotations, log_scales, sh_coeffs, opacities, mode)
}
