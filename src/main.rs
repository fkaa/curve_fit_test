trait CatmullRom {
    fn distance(a: Self, b: Self) -> f32;
    fn interpolate(a: Self, b: Self, c: Self, d: Self, t: f32) -> Self;
}

impl CatmullRom for f32 {
    fn distance(a: Self, b: Self) -> f32 {
        a - b
    }
    
    #[inline]
    fn interpolate(p0: Self, p1: Self, p2: Self, p3: Self, t: f32) -> Self {
        let t2 = t * t;
        let t3 = t2 * t;

        let f1 = -0.5f32 * t3 + t2 - 0.5f32 * t;
        let f2 = 1.5f32 * t3 - 2.5f32 * t2 + 1f32;
        let f3 = -1.5f32 * t3 + 2.0f32 * t2 + 0.5f32 * t;
        let f4 = 0.5f32 * t3 - 0.5f32 * t2;

        p0 * f1 + p1 * f2 + p2 * f3 + p3 * f4
    }
}

struct CurveFitter<'a, T> {
    keys: &'a [(f32, T)],
    curve: Vec<(f32, T)>,
    // spline_
}

impl<'a, T: CatmullRom + Clone + std::fmt::Debug> CurveFitter<'a, T> {
    fn new(keys: &'a [(f32, T)]) -> Self {
        let mut curve = Vec::new();
        
        let begin = &keys[0];
        let end = &keys[keys.len() - 1];
        curve.push((begin.0 - 0.5, begin.1.clone()));
        curve.push((begin.0, begin.1.clone()));
        curve.push((end.0, end.1.clone()));
        curve.push((end.0 + 0.5, end.1.clone()));
        
        CurveFitter {
            keys,
            curve,
        }
    }
    
    fn subdivide(&mut self, threshold: f32) -> bool {
        if let Some(max_idx) = self.errors()
            .iter()
            .map(|e| e.abs())
            .enumerate()
            .filter(|&(i, e)| e > threshold)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, e)| i)
        {
            let key = &self.keys[max_idx];
            let idx = self.get_index_at(key.0);
            let new_point = (key.0, key.1.clone());
            
            self.curve.insert(idx + 1, new_point);

            println!("Inserting new point at max_idx={}, idx={}", max_idx, idx);

            true
        } else {
            false
        }
    }

    fn get_index_at(&self, t: f32) -> usize {
        let mut idx = 0;
        for i in 1..(self.curve.len() ) {
            let point = &self.curve[i];
            
            if point.0 > t {
                break;
            }

            idx += 1;
        }

        idx = idx.min(self.curve.len() - 3);

        idx
    }

    fn eval_at(&self, t: f32) -> T {
        let idx = self.get_index_at(t);

        // println!("{}: {}", t, idx);
        
        let a = self.curve[idx - 1].clone();
        let b = self.curve[idx - 0].clone();
        let c = self.curve[idx + 1].clone();
        let d = self.curve[idx + 2].clone();

        // println!("t={}, idx={}, ({}, {}, {}, {})", t, idx, a.0, b.0, c.0, d.0);

        let t1 = b.0;
        let t2 = c.0;
        
        //let t = t1 + t * (t2 - t1);
        let tz = (t - t1) / (t2 - t1);
        // println!("t={}", t);
        T::interpolate(a.1, b.1, c.1, d.1, tz)
    }

    fn errors(&self) -> Vec<f32> {
        let mut errors = Vec::new();

        for key in self.keys {
            let t = key.0;

            let key_val = key.1.clone();
            let curve_val = self.eval_at(t);

            let err = T::distance(key_val, curve_val);

            errors.push(err);
        }
        
        errors
    }
}

struct CSpline {
    points: Vec<f32>
}

const DATA: [(f32, f32); 60] = [
    (0.0, 0.84),
    (0.016666666666666666, 0.7857870141056448),
    (0.03333333333333333, 0.731968711129689),
    (0.05, 0.6805936774715144),
    (0.06666666666666667, 0.6335352079688814),
    (0.08333333333333333, 0.5924180650636451),
    (0.1, 0.5585565317895961),
    (0.11666666666666667, 0.5329062097728754),
    (0.13333333333333333, 0.5160314215806813),
    (0.15, 0.5080894115320483),
    (0.16666666666666666, 0.5088318288874735),
    (0.18333333333333332, 0.517623252274559),
    (0.2, 0.5334758048326134),
    (0.21666666666666667, 0.5550982455572299),
    (0.23333333333333334, 0.5809573312697498),
    (0.25, 0.6093487498310115),
    (0.26666666666666666, 0.6384745486907532),
    (0.2833333333333333, 0.6665237385416289),
    (0.3, 0.6917526489609029),
    (0.31666666666666665, 0.7125616546553352),
    (0.3333333333333333, 0.7275650743167048),
    (0.35, 0.735651360218063),
    (0.36666666666666664, 0.7360311310599335),
    (0.38333333333333336, 0.7282711338495464),
    (0.4, 0.7123128293775124),
    (0.4166666666666667, 0.6884749537597485),
    (0.43333333333333335, 0.6574400873024205),
    (0.45, 0.620225932756329),
    (0.4666666666666667, 0.5781426395901123),
    (0.48333333333333334, 0.532738082767275),
    (0.5, 0.4857334901144181),
    (0.5166666666666667, 0.43895219206506614),
    (0.5333333333333333, 0.394244526400116),
    (0.55, 0.3534120589187171),
    (0.5666666666666667, 0.31813427475281525),
    (0.5833333333333334, 0.2899007560452129),
    (0.6, 0.26995159734265634),
    (0.6166666666666667, 0.25922843297337994),
    (0.6333333333333333, 0.2583379782104164),
    (0.65, 0.2675294393700301),
    (0.6666666666666666, 0.2866865512844134),
    (0.6833333333333333, 0.31533437980045326),
    (0.7, 0.3526604087917452),
    (0.7166666666666667, 0.39754884189116635),
    (0.7333333333333333, 0.4486265134552387),
    (0.75, 0.5043183432492434),
    (0.7666666666666667, 0.5629099035529965),
    (0.7833333333333333, 0.6226144101012914),
    (0.8, 0.6816413089064934),
    (0.8166666666666667, 0.7382636137786008),
    (0.8333333333333334, 0.7908812531565124),
    (0.85, 0.838077903385995),
    (0.8666666666666667, 0.878669107654112),
    (0.8833333333333333, 0.9117398899309168),
    (0.9, 0.9366705524058959),
    (0.9166666666666666, 0.9531498712604115),
    (0.9333333333333333, 0.9611754556876873),
    (0.95, 0.9610415846848144),
    (0.9666666666666667, 0.9533153615150783),
    (1.0, 0.938802504518808),
];

fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;

    let f1 = -0.5f32 * t3 + t2 - 0.5f32 * t;
    let f2 = 1.5f32 * t3 - 2.5f32 * t2 + 1f32;
    let f3 = -1.5f32 * t3 + 2.0f32 * t2 + 0.5f32 * t;
    let f4 = 0.5f32 * t3 - 0.5f32 * t2;

    p0 * f1 + p1 * f2 + p2 * f3 + p3 * f4
}

fn main() {
    let mut width = 800f32;
    let mut height = 600f32;
    
    let mut ev_loop = glutin::EventsLoop::new();
    let winit_window = glutin::WindowBuilder::new()
        .with_title("Curve Fitness")
        .with_dimensions(LogicalSize::new(width as _, height as _));
    let context = glutin::ContextBuilder::new()
        .with_gl(glutin::GlRequest::Latest)
        .with_gl_profile(glutin::GlProfile::Core)
        .with_vsync(true);
    let mut wnd = glutin::GlWindow::new(winit_window, context, &ev_loop).unwrap();

    unsafe {
        wnd.make_current().unwrap();
        gl::load_with(|symbol| wnd.get_proc_address(symbol) as *const _);
        gl::ClearColor(0.05, 0.05, 0.05, 1.0);
        gl::ClearDepth(1.0);
        gl::Viewport(0, 0, width as _, height as _);
        gl::Enable(gl::DEPTH_TEST);
        gl::Enable(gl::BLEND);
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
    }

    let mut renderer = RgOpenGlRenderer::new(width as f32, height as f32).unwrap();
    let mut cxt = rg::Context::new(Box::new(renderer));

    let family = font_kit::sources::fs::FsSource::new()
        .select_family_by_name("Times New Roman")
        .unwrap();
    let handle = &family.fonts()[0];
    let mut big_font = rg::Font::new(String::from("Test"), handle, 38f32).unwrap();
    let mut smol_font = rg::Font::new(String::from("Test"), handle, 14f32).unwrap();


    let mut fitter = CurveFitter::new(&DATA[..]);
    let mut running = true;
    
    while running {
        
        ev_loop.poll_events(|event| {
            match event {
                glutin::Event::WindowEvent{ event, window_id } => match event {
                    glutin::WindowEvent::CloseRequested => running = false,
                    glutin::WindowEvent::Resized(logical_size) => {
                        let dpi_factor = wnd.get_hidpi_factor();
                        let new_size = logical_size.to_physical(dpi_factor);

                        wnd.resize(new_size);
                        width = new_size.width as f32;
                        height = new_size.height as f32;
                        
                        unsafe {
                            wnd.make_current().unwrap();  
                        }
                        
                        
                        unsafe {
                            gl::Viewport(0, 0, new_size.width as i32, new_size.height as i32);
                        }
                    },
                    glutin::WindowEvent::KeyboardInput { input, .. } => {
                         match (input.virtual_keycode, input.state) {
                             (Some(glutin::VirtualKeyCode::S), glutin::ElementState::Released) => {
                                 println!("Attemting to subdivide");
                                 fitter.subdivide(0.03f32);
                            },
                            _ => {}
                        }
                    }
                    glutin::WindowEvent::MouseInput {
                        device_id,
                        state,
                        button,
                        modifiers
                    } => {

                    },
                    glutin::WindowEvent::CursorMoved { position, .. } => {
                        let dpi_factor = wnd.get_hidpi_factor();
                        let new_pos = position.to_physical(dpi_factor);
                    }

                    _ => {}
                }
                _ => {}
            }
        });

        unsafe {
            gl::ClearColor(0.2f32, 0.2f32, 0.2f32, 1f32);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        let mut list = &mut cxt.draw_list;
        cxt.renderer.resize(width, height);

        let area = rg::Rect::new(
            rg::float2(0f32, 0f32),
            rg::float2(width, height)
        ).pad_sides(10f32, 45f32, 10f32, 25f32);

        list.add_text_wrapped(
            &mut *cxt.renderer,
            &mut big_font,
            &format!("Curve Fitting Test: {} Control Points", fitter.curve.len() - 2),
            rg::float2(
                area.min.0,
                area.min.1 - 47f32,
            ),
            area.max.0,
            0xffa3e6ff
        );

        // grid
        for x in 0..10 {
            let fx = x as f32 / 10f32;
            let nx = area.min.0 + fx * area.width();
            
            list.add_line(
                rg::float2(nx, area.min.1),
                rg::float2(nx, area.max.1),
                0xaa555555
            );
            
            for y in 0..10 {
                let fy = y as f32 / 10f32;
                let ny = area.min.1 + fy * area.height();
                
                list.add_line(
                    rg::float2(area.min.0, ny),
                    rg::float2(area.max.0, ny),
                    0xaa555555
                );
            }
        }

        
        let max_t = DATA.iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap().0;
        let l = DATA.len() - 1;

        let w = 37f32;
        for x in 0..11 {
            let fx = x as f32 / 10f32;
            let nx = area.min.0 + fx * area.width();

            list.add_text_wrapped(
                &mut *cxt.renderer,
                &mut smol_font,
                &format!("{:.3}", fx * max_t),
                rg::float2(nx - w * fx, area.max.1 + 3f32),
                area.max.0 + 200f32,
                0xaa555555
            );
        }


        for i in 0..DATA.len() {
            let v = DATA[i];
            let x1 = area.min.0 + (v.0 / max_t) * area.width();
            let y1 = area.max.1 - v.1 * area.height();

            if i < l {
                let v2 = DATA[i + 1];
                
                let x2 = area.min.0 + (v2.0 / max_t) * area.width();
                let y2 = area.max.1 - v2.1 * area.height();

                list.add_line(
                    rg::float2(x1, y1),
                    rg::float2(x2, y2),
                    0xffa3e6ff
                );
            }
            
            let sz = 4f32;
            list.add_rect_filled(
                rg::float2(x1 - sz, y1 - sz),
                rg::float2(x1 + sz, y1 + sz),
                0f32,
                0xff73b5ec
            );
        }

        let n = &fitter.curve;
        let errors = fitter.errors();

        /*n.push((-0.5f32, 0.0f32));
        n.push((0.0f32, 0.0f32));
        n.push((1f32, 1f32));
        n.push((1.5f32, 1f32));*/

        let mut points = Vec::new();
        
        let mut path = list.path();
        for i in 1..(n.len() - 2) {
            let a = n[i - 1];
            let b = n[i - 0];
            let c = n[i + 1];
            let d = n[i + 2];

            let t1 = b.0;
            let t2 = c.0;

            // control points
            let sz = 4f32;
            let x1 = area.min.0 + b.0 * area.width();
            let y1 = area.max.1 - b.1 * area.height();
            points.push((x1, y1));
            
            if i == n.len() - 3 {
                let x2 = area.min.0 + c.0 * area.width();
                let y2 = area.max.1 - c.1 * area.height();
                points.push((x2, y2));
            }
            
            let max_iter = 20;
            for i in 0..max_iter {
                let f = i as f32 / (max_iter - 1) as f32;
                let t = t1 + f * (t2 - t1);
                let tz = (t - t1) / (t2 - t1);

                //let n = catmull_rom(a.1, b.1, c.1, d.1, tz);
                let n = fitter.eval_at(t);

                // println!("{}: t={}, tz={}, n={}", i, t, tz, n);
                path = path.line(rg::float2(
                    area.min.0 + t * area.width(),
                    area.max.1 - n * area.height(),
                ));
            }            
        }
        path.stroke(2f32, false, 0x881d1dd1);

        let err = fitter.errors();
        for (err, (t, v)) in err.iter().zip(&DATA[..]) {
            let x1 = area.min.0 + t * area.width();
            let y1 = area.max.1 - v * area.height();

            let eval = fitter.eval_at(*t);
            let x2 = area.min.0 + t * area.width();
            let y2 = area.max.1 - eval * area.height();

            let tx = (x1 + x2) * 0.5f32;
            let ty = (y1 + y2) * 0.5f32;

            list.add_line(rg::float2(x1, y1), rg::float2(x2, y2), 0x881d1dd1);
            list.add_text_wrapped(
                &mut *cxt.renderer,
                &mut smol_font,
                &format!("{:.3}", err),
                rg::float2(tx, ty),
                area.max.0 + 200f32,
                0xaa999999
            );
        }
        
        for point in points {
            let sz = 8f32;
            list.add_rect_filled(
                rg::float2(point.0 - sz, point.1 - sz),
                rg::float2(point.0 + sz, point.1 + sz),
                0f32,
                0x881d1dd1
            );
        }
                
        list.add_rect(area.min, area.max, 0f32, 2f32, 0xffa3e6ff);
        
        cxt.renderer.render(list);
        list.clear();
        
        wnd.swap_buffers().unwrap();
    }
}








use glutin::dpi::*;
use glutin::GlContext;

use rg::TextureHandle;

use std::slice;
use std::ptr;
use std::mem;
use std::ffi::CString;

pub struct InputElement {
    pub shader_slot: u32,
    pub buffer_slot: u32,
    pub format: u32,
    pub components: u32,
    pub normalized: u8,
    pub stride: u32,
    pub offset: u32,
}

// TODO: 
pub struct InputLayout {
    pub elements: Vec<InputElement>,
    pub vao: u32,
}

pub struct LayoutGuard {
    vao: u32
}

impl Drop for LayoutGuard {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteVertexArrays(1, &mut self.vao);
        }
    }
}

impl InputLayout {
    pub fn new(elements: Vec<InputElement>) -> Self {
        let mut vao = 0;
        
        unsafe {
            gl::GenVertexArrays(1, &mut vao);
        }
        
        InputLayout {
            elements,
            vao,
        }
    }

    pub fn bind(&mut self, buffers: &[(u32, &Buffer)], index_buffer: Option<&Buffer>) -> LayoutGuard {
        let mut guard = LayoutGuard {
            vao: 0
        };
        
        unsafe {
            gl::GenVertexArrays(1, &mut guard.vao);
            gl::BindVertexArray(guard.vao);
        }
        
        for &(slot, ref buffer) in buffers {
            unsafe {
                gl::BindBuffer(buffer.ty(), buffer.handle());
            }
            
            for attr in self.elements.iter().filter(|a| a.buffer_slot == slot) {
                unsafe {
                    gl::EnableVertexAttribArray(attr.shader_slot);
                    gl::VertexAttribPointer(
                        attr.shader_slot,
                        attr.components as i32,
                        attr.format,
                        attr.normalized,
                        attr.stride as i32,
                        attr.offset as *const _
                    );
                }
            }
        }

        if let Some(buf) = index_buffer {
            unsafe {
                gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, buf.handle());
            }
        }

        guard
    }
}

#[derive(Debug)]
pub struct Texture {
    pub handle: u32,
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteTextures(1, &self.handle);
        }
    }
}

impl Texture {
    pub fn with_data_2d(
        data: &[u8],
        width: u32,
        height: u32,
        internal_format: u32,
        format: u32,
        filter: u32
    ) -> Self {
        let mut handle = 0;
        
        unsafe {
            gl::GenTextures(1, &mut handle);
            
            gl::BindTexture(gl::TEXTURE_2D, handle);
            gl::TexImage2D(gl::TEXTURE_2D, 0, internal_format as i32, width as i32, height as i32, 0, format as u32, gl::UNSIGNED_BYTE, data.as_ptr() as *const _);

            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, filter as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, filter as i32);
        }

        Texture {
            handle
        }
    }
}

#[derive(Debug)]
pub enum VariableBinding {
    Attribute(String, u32),
    Uniform(String, u32),
    UniformBlock(String, u32),
    Sampler(String, u32),
}

#[derive(Debug)]
pub struct Shader {
    pub program: u32,
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteProgram(self.program);
        }
    }
}

impl Shader {
    pub fn new(
        vs: &str,
        ps: Option<&str>,
        bindings: Vec<VariableBinding>
    ) -> Result<Self, String> {
        unsafe fn compile_shader(ty: u32, shdr: &str) -> Result<u32, String> {
            let shader = gl::CreateShader(ty);
            let len = shdr.len() as i32;
            let shdr = shdr.as_ptr() as *const i8;
            gl::ShaderSource(shader, 1, &shdr, &len);
            gl::CompileShader(shader);

            let mut success = 0i32;
            gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success as _);

            if success == gl::FALSE as i32 {
                let mut log_size = 0i32;
                gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut log_size as _);

                let mut log = vec![0u8; log_size as usize];
                gl::GetShaderInfoLog(shader, log_size, ptr::null_mut(), log.as_mut_ptr() as _);

                gl::DeleteShader(shader);
                Err(String::from_utf8_unchecked(log))
            } else {
                Ok(shader)
            }
        }
        
        let vs = unsafe { compile_shader(gl::VERTEX_SHADER, vs)? };
        let ps = if let Some(ps) = ps {
            Some(unsafe { compile_shader(gl::FRAGMENT_SHADER, ps)? })
        } else {
            None
        };

        unsafe {
            let program = gl::CreateProgram();
            
            gl::AttachShader(program, vs);
            if let Some(ps) = ps {
                gl::AttachShader(program, ps);
            }

            for bind in &bindings {
                match bind {
                    &VariableBinding::Attribute(ref name, id) => {
                        let c_str = CString::new(name.clone()).unwrap();
                        gl::BindAttribLocation(program, id, c_str.to_bytes_with_nul().as_ptr() as *const _);     
                    },
                    _ => {}
                }
            }

            gl::LinkProgram(program);

            let mut success = 0i32;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
            if success == gl::FALSE as i32 {
                let mut log_size = 0i32;
                gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut log_size as _);

                let mut log = vec![0u8; log_size as usize];
                gl::GetProgramInfoLog(program, log_size, ptr::null_mut(), log.as_mut_ptr() as _);

                gl::DeleteProgram(program);
                return Err(String::from_utf8_unchecked(log));
            }

            gl::DetachShader(program, vs);
            if let Some(ps) = ps {
                gl::DetachShader(program, ps);
            }

            gl::UseProgram(program);

            // after linking we setup sampler bindings as specified in the shader
            for bind in bindings {
                match bind {
                    VariableBinding::Uniform(name, id) => {
                        // TODO: impl for block?
                    },
                    VariableBinding::UniformBlock(name, id) => {
                        let c_str = CString::new(name).unwrap();
                        let index = gl::GetUniformBlockIndex(program, c_str.to_bytes_with_nul().as_ptr() as *const _);

                        gl::UniformBlockBinding(program, index, id);
                    }
                    VariableBinding::Sampler(name, id) => {
                        let c_str = CString::new(name).unwrap();
                        let index = gl::GetUniformLocation(program, c_str.to_bytes_with_nul().as_ptr() as *const _);
                        
                        gl::Uniform1i(index, id as i32);
                    },
                    _ => {}
                }
            }

            Ok(Shader {
                program
            })
        }
    }

    pub fn bind(&self) {
        unsafe {
            gl::UseProgram(self.program);
        }
    }

    pub fn bind_uniform_block<T>(
        &self,
        idx: u32,
        buffer: &UniformBuffer<T>
    ) {
        self.bind();
        
        unsafe {
            gl::BindBufferBase(
                gl::UNIFORM_BUFFER,
                idx,
                buffer.handle()
            );
        }
    }

    pub fn bind_texture(&self, index: u32, texture: &Texture) {
        self.bind();
        
        unsafe {
            gl::ActiveTexture(gl::TEXTURE0 + index);
            gl::BindTexture(gl::TEXTURE_2D, texture.handle);
        }
    }
}

// TODO: add size field to buffers to check for overflow on writes..
#[derive(Debug)]
pub struct Buffer {
    pub handle: u32,
    pub ty: u32,
    pub size: usize,
}

impl Buffer {
    fn empty(
        ty: u32,
        usage: u32,
        size: isize
    ) -> Buffer {
        let mut buffer = 0;

        unsafe {
            gl::GenBuffers(1, &mut buffer);
            gl::BindBuffer(ty, buffer);
            gl::BufferData(ty, size, ptr::null_mut(), usage);
        }

        Buffer {
            handle: buffer,
            ty,
            size: size as usize,
        }
    }
    
    fn with_data(
        ty: u32,
        usage: u32,
        data: &[u8]
    ) -> Buffer {
        let mut buffer = 0;

        unsafe {
            gl::GenBuffers(1, &mut buffer);
            gl::BindBuffer(ty, buffer);
            gl::BufferData(ty, data.len() as isize, data.as_ptr() as _, usage);
        }

        Buffer {
            handle: buffer,
            ty,
            size: data.len()
        }
    }

    pub fn ty(&self) -> u32 {
        self.ty
    }

    pub fn handle(&self) -> u32 {
        self.handle
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn write(&self, data: &[u8]) {
        unsafe {
            gl::BindBuffer(self.ty, self.handle);
            gl::BufferSubData(self.ty, 0, data.len() as isize, data.as_ptr() as *const _ as *const _);
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.handle);
        }
    }
}

pub struct UniformBuffer<T> {
    pub buffer: Buffer,
    _phantom: ::std::marker::PhantomData<T>,
}

impl<T> UniformBuffer<T> {
    pub fn new() -> Self {
        UniformBuffer {
            buffer: Buffer::empty(gl::UNIFORM_BUFFER, gl::DYNAMIC_DRAW, mem::size_of::<T>() as isize),
            _phantom: ::std::marker::PhantomData,
        }
    }

    pub fn write(&self, value: &T) {
        let buffer = &self.buffer;
        
        unsafe {
            gl::BindBuffer(buffer.ty(), buffer.handle());
            gl::BufferSubData(buffer.ty(), 0, mem::size_of::<T>() as isize, value as *const _ as *const _);
        }
    }

    pub fn handle(&self) -> u32 {
        self.buffer.handle
    }
}

struct CommonUniforms {
    projection: [f32; 16],
}

struct Vertex {
    pos: [f32; 2],
    
}

struct RgOpenGlRenderer {
    texture: Texture,
    layout: InputLayout,
    shader: Shader,
    width: f32,
    height: f32,

    common_uniforms: UniformBuffer<CommonUniforms>,

    vertex_buffer: Buffer,
    index_buffer: Buffer,
}

impl RgOpenGlRenderer {
    pub fn new(width: f32, height: f32) -> Result<Self, String> {
        let texture = Texture::with_data_2d(
            &[0xff, 0xff, 0xff, 0xff],
            1,
            1,
            gl::RGBA8,
            gl::RGBA,
            gl::NEAREST,
        );

        let layout = InputLayout::new(vec![
            InputElement { shader_slot: 0, buffer_slot: 0, format: gl::FLOAT, normalized: gl::FALSE, components: 2, stride: 20, offset: 0 },
            InputElement { shader_slot: 1, buffer_slot: 0, format: gl::FLOAT, normalized: gl::FALSE, components: 2, stride: 20, offset: 8 },
            InputElement { shader_slot: 2, buffer_slot: 0, format: gl::UNSIGNED_BYTE, normalized: gl::TRUE, components: 4, stride: 20, offset: 16 },
        ]);

        let shader = Shader::new(
            include_str!("../shaders/main.vs"),
            Some(include_str!("../shaders/main.fs")),
            vec![
                VariableBinding::Attribute(String::from("Position"), 0),
                VariableBinding::Attribute(String::from("Uv"), 1),
                VariableBinding::Attribute(String::from("Color"), 2),
                VariableBinding::UniformBlock(String::from("Common"), 0),
            ]
        )?;
        
        Ok(RgOpenGlRenderer {
            texture,
            layout,
            shader,
            width,
            height,
            common_uniforms: UniformBuffer::new(),
            vertex_buffer: Buffer::empty(gl::ARRAY_BUFFER, gl::DYNAMIC_DRAW, 20000 * mem::size_of::<rg::Vertex>() as isize),
            index_buffer: Buffer::empty(gl::ELEMENT_ARRAY_BUFFER, gl::DYNAMIC_DRAW, 30000 * mem::size_of::<u16>() as isize),
        })
    }
}


impl rg::Renderer for RgOpenGlRenderer {
    fn resize(&mut self, width: f32, height: f32) {
        self.width = width;
        self.height = height;
    }
    
    fn render(&mut self, list: &rg::DrawList) {
        unsafe {
            gl::Disable(gl::DEPTH_TEST);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        }
        
        let ortho = {
            let l = 0f32;
            let r = self.width as f32;
            let b = self.height as f32;
            let t = 0f32;

            [
                2f32 / (r - l),    0f32,              0f32,   0f32,
                0f32,              2f32 / (t - b),    0f32,   0f32,
                0f32,              0f32,              0.5f32, 0f32,
                (r + l) / (l - r), (t + b) / (b - t), 0.5f32, 1f32,
            ]
        };
        self.common_uniforms.write(&CommonUniforms {
            projection: ortho
        });

        let vertices = unsafe {
            slice::from_raw_parts(
                list.vertices.as_ptr() as *const u8,
                list.vertices.len() * mem::size_of::<rg::Vertex>(),
            )
        };
        self.vertex_buffer.write(vertices);

        let indices = unsafe {
            slice::from_raw_parts(
                list.indices.as_ptr() as *const u8,
                list.indices.len() * mem::size_of::<u16>(),
            )
        };
        self.index_buffer.write(indices);

        let _guard = self.layout.bind(&[
            (0, &self.vertex_buffer),
        ], Some(&self.index_buffer));
        
        self.shader.bind();
        self.shader.bind_uniform_block(0, &self.common_uniforms);

        let mut index_offset = 0;
        for command in list.commands() {
            unsafe {
                gl::BindTexture(gl::TEXTURE_2D, command.texture_id as _);
                gl::DrawElements(gl::TRIANGLES, command.index_count as i32, gl::UNSIGNED_SHORT, index_offset as *const _);
            }
            
            index_offset += command.index_count * 2;
        }
    }

    
    fn create_texture_a8(&mut self, width: u32, height: u32) -> (TextureHandle, TextureHandle) {
        let zeroed = vec![0u8; (width * height) as usize];
        let texture = Texture::with_data_2d(
            &zeroed,
            width,
            height,
            gl::R8,
            gl::RED,
            gl::LINEAR
        );

        let handle = texture.handle;

        ::std::mem::forget(texture);

        (handle as TextureHandle, handle as TextureHandle)
    }
    
    fn upload_a8(&mut self, handle: TextureHandle, x: u32, y: u32, width: u32, height: u32, data: &[u8], stride: u32) {
        let handle = handle as u32;

        unsafe {
            gl::PixelStorei(gl::UNPACK_ROW_LENGTH, stride as i32);
            gl::BindTexture(gl::TEXTURE_2D, handle);
            gl::TexSubImage2D(
                gl::TEXTURE_2D,
                0,
                x as i32, y as i32,
                width as i32, height as i32,
                gl::RED,
                gl::UNSIGNED_BYTE,
                data.as_ptr() as *const _
            );
        }
        /*unsafe {
            (*self.context).UpdateSubresource(
                handle as *mut _,
                0,
                &D3D11_BOX {
                    left: x,
                    top: y,
                    front: 0,
                    right: x + width,
                    bottom: y + height,
                    back: 1,
                },
                data.as_ptr() as *const _,
                stride,
                0
            );
        }*/
    }
}
