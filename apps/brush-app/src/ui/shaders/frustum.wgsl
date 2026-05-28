struct Uniforms {
    view_proj: mat4x4<f32>,
    // params.x = global frustum scale (length along camera +Z).
    params: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct InstanceInput {
    @location(0) tx0: vec4<f32>,
    @location(1) tx1: vec4<f32>,
    @location(2) tx2: vec4<f32>,
    @location(3) tx3: vec4<f32>,
    // tan(fov_x / 2), tan(fov_y / 2), unused, unused
    @location(4) cam_params: vec4<f32>,
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
}

const WIREFRAME_VERTICES = array<vec3<f32>, 16>(
    // 4 edges from apex to far corners
    vec3(0., 0., 0.),  vec3(-1., -1., 1.),
    vec3(0., 0., 0.),  vec3(-1.,  1., 1.),
    vec3(0., 0., 0.),  vec3( 1.,  1., 1.),
    vec3(0., 0., 0.),  vec3( 1., -1., 1.),
    // far plane rectangle
    vec3(-1., -1., 1.), vec3( 1., -1., 1.),
    vec3( 1., -1., 1.), vec3( 1.,  1., 1.),
    vec3( 1.,  1., 1.), vec3(-1.,  1., 1.),
    vec3(-1.,  1., 1.), vec3(-1., -1., 1.),
);

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, inst: InstanceInput) -> VertexOutput {
    let transform = mat4x4<f32>(inst.tx0, inst.tx1, inst.tx2, inst.tx3);
    let tan_half_x = inst.cam_params.x;
    let tan_half_y = inst.cam_params.y;
    let scale = uniforms.params.x;

    var pos = WIREFRAME_VERTICES[vertex_index];
    pos.x *= tan_half_x * scale;
    pos.y *= tan_half_y * scale;
    pos.z *= scale;

    let world_pos = transform * vec4<f32>(pos, 1.0);

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * world_pos;
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}