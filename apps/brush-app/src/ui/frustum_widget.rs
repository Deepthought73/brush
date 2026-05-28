use brush_dataset::Dataset;
use brush_render::camera::Camera;
use eframe::egui_wgpu::{self, RenderState, wgpu};
use egui::Rect;
use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    // params.x = global frustum scale.
    params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FrustumInstance {
    transform: [[f32; 4]; 4],
    // (tan(fov_x / 2), tan(fov_y / 2), unused, unused)
    cam_params: [f32; 4],
    color: [f32; 4],
}

impl FrustumInstance {
    const ATTRIBS: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        0 => Float32x4,
        1 => Float32x4,
        2 => Float32x4,
        3 => Float32x4,
        4 => Float32x4,
        5 => Float32x4,
    ];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct CameraFrustumWidget {
    instances: Vec<FrustumInstance>,
}

impl CameraFrustumWidget {
    pub fn new(state: &RenderState) -> Self {
        state
            .renderer
            .write()
            .callback_resources
            .insert(CameraFrustumWidgetResources::new(
                &state.device,
                state.target_format,
            ));
        Self {
            instances: Vec::new(),
        }
    }

    pub fn set_dataset(&mut self, dataset: &Dataset) {
        const TRAIN_COLOR: [f32; 4] = [1.0, 0.55, 0.1, 0.95];
        const EVAL_COLOR: [f32; 4] = [0.2, 0.7, 1.0, 0.95];

        self.instances.clear();

        let train = dataset.train.views.iter().map(|v| (v, TRAIN_COLOR));
        let eval = dataset
            .eval
            .iter()
            .flat_map(|s| s.views.iter().map(|v| (v, EVAL_COLOR)));

        for (view, color) in train.chain(eval) {
            let transform = Mat4::from(view.camera.local_to_world()).to_cols_array_2d();
            let tan_half_x = (view.camera.fov_x as f32 * 0.5).tan();
            let tan_half_y = (view.camera.fov_y as f32 * 0.5).tan();
            self.instances.push(FrustumInstance {
                transform,
                cam_params: [tan_half_x, tan_half_y, 0.0, 0.0],
                color,
            });
        }
    }

    pub fn clear(&mut self) {
        self.instances.clear();
    }

    pub fn paint(&self, rect: Rect, camera: Camera, ui: &egui::Ui, scale: f32) {
        if self.instances.is_empty() {
            return;
        }
        ui.painter()
            .add(eframe::egui_wgpu::Callback::new_paint_callback(
                rect,
                CameraFrustumWidgetPainter {
                    camera,
                    scale,
                    instances: self.instances.clone(),
                },
            ));
    }
}

pub struct CameraFrustumWidgetResources {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u64,
}

impl CameraFrustumWidgetResources {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Frustum Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/frustum.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Frustum Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Frustum Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Frustum Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Frustum Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Frustum Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[FrustumInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        const INITIAL_CAPACITY: u64 = 64;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Frustum Instance Buffer"),
            size: INITIAL_CAPACITY * std::mem::size_of::<FrustumInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            instance_buffer,
            instance_capacity: INITIAL_CAPACITY,
        }
    }

    fn ensure_capacity(&mut self, device: &wgpu::Device, count: u64) {
        if count > self.instance_capacity {
            let new_capacity = count.next_power_of_two().max(64);
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Frustum Instance Buffer"),
                size: new_capacity * std::mem::size_of::<FrustumInstance>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_capacity;
        }
    }
}

struct CameraFrustumWidgetPainter {
    camera: Camera,
    scale: f32,
    instances: Vec<FrustumInstance>,
}

impl egui_wgpu::CallbackTrait for CameraFrustumWidgetPainter {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(resources) = resources.get_mut::<CameraFrustumWidgetResources>() else {
            return Vec::new();
        };

        let aspect =
            screen_descriptor.size_in_pixels[0] as f32 / screen_descriptor.size_in_pixels[1] as f32;
        let proj = Mat4::perspective_lh(self.camera.fov_y as f32, aspect, 0.1, 1000.0);
        let y_flip = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));
        // Camera position/rotation here are in the model-local frame
        // (see scene.rs view_eff swap), so world_to_local goes model -> camera.
        // The per-instance transform maps view-camera-local -> model frame.
        let view = Mat4::from(self.camera.world_to_local());
        let view_proj = proj * y_flip * view;

        queue.write_buffer(
            &resources.uniform_buffer,
            0,
            bytemuck::cast_slice(&[Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                params: [self.scale, 0.0, 0.0, 0.0],
            }]),
        );

        if !self.instances.is_empty() {
            resources.ensure_capacity(device, self.instances.len() as u64);
            queue.write_buffer(
                &resources.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instances),
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(resources) = resources.get::<CameraFrustumWidgetResources>() else {
            return;
        };
        if self.instances.is_empty() {
            return;
        }
        render_pass.set_pipeline(&resources.pipeline);
        render_pass.set_bind_group(0, &resources.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, resources.instance_buffer.slice(..));
        render_pass.draw(0..16, 0..self.instances.len() as u32);
    }
}
