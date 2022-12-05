#![allow(clippy::all)]

use std::sync::{Arc, Mutex};

use graphical_engine::{BaseEngine, GraphicalEngine, LogicalDevice, Vertex};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassContents,
    },
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{self, AcquireError, PresentInfo},
    sync::{self, FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::{PhysicalSize, Pixel},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[cfg(test)]
mod tests;

mod shader_vertex {
    vulkano_shaders::shader! {ty: "vertex", path: "src/shader.vert"}
}

mod shader_fragment {
    vulkano_shaders::shader! {ty: "fragment", path: "src/shader.frag"}
}

fn create_viewport<T: Pixel>(physical_size: PhysicalSize<T>) -> Viewport {
    Viewport {
        origin: [0.0, 0.0],
        dimensions: physical_size.into(),
        depth_range: 0.0..1.0,
    }
}

fn create_pipeline<T: Pixel>(
    vertex_shader: Arc<ShaderModule>,
    fragment_shader: Arc<ShaderModule>,
    physical_size: PhysicalSize<T>,
    render_pass: Arc<RenderPass>,
    logical_device: Arc<LogicalDevice>,
) -> Arc<GraphicsPipeline> {
    let viewport = create_viewport(physical_size);

    GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vertex_shader.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fragment_shader.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(logical_device.get_device())
        .unwrap()
}

fn create_command_buffers(
    frame_buffers: Vec<Arc<Framebuffer>>,
    graphical_engine: Arc<Mutex<GraphicalEngine>>,
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    frame_buffers
        .iter()
        .map(|framebuffer| {
            let engine_arc = graphical_engine.lock().unwrap();

            let mut builder = AutoCommandBufferBuilder::primary(
                engine_arc.get_logical_device().get_device(),
                engine_arc.get_logical_device().get_queue_family_index(),
                CommandBufferUsage::MultipleSubmit, // don't forget to write the correct buffer usage
            )
            .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

pub fn entrypoint() {
    // Vulkan instance
    let instance = GraphicalEngine::make_instance();

    // Window
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone()) // Not all Winit versions are compatible with vulkano-win apparently. Make sure they work together or imports won't work!
        .expect("failed to create window surface");

    // Engine
    let graphical_engine = Arc::new(Mutex::new(GraphicalEngine::new(instance, surface.clone())));

    GraphicalEngine::print_api_information(
        graphical_engine.lock().unwrap().get_instance(),
        log::Level::Info,
    );

    // Set vertices for triangle
    let vertex1 = Vertex {
        position: [-0.5, -0.5],
    };
    let vertex2 = Vertex {
        position: [0.0, 0.5],
    };
    let vertex3 = Vertex {
        position: [0.5, -0.25],
    };

    // Create vertex buffer
    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        graphical_engine
            .lock()
            .unwrap()
            .get_logical_device()
            .get_device(),
        BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        },
        false,
        vec![vertex1, vertex2, vertex3].into_iter(),
    )
    .unwrap();

    // RenderPass
    let render_pass = graphical_engine.lock().unwrap().create_render_pass();

    // Shaders
    let vertex_shader = shader_vertex::load(
        graphical_engine
            .lock()
            .unwrap()
            .get_logical_device()
            .get_device(),
    )
    .expect("failed to create vertex shader module");
    let fragment_shader = shader_fragment::load(
        graphical_engine
            .lock()
            .unwrap()
            .get_logical_device()
            .get_device(),
    )
    .expect("failed to create fragment shader module");

    // Pipeline
    let mut pipeline = create_pipeline(
        vertex_shader.clone(),
        fragment_shader.clone(),
        surface.window().inner_size(),
        render_pass.clone(),
        graphical_engine.lock().unwrap().get_logical_device(),
    );

    // Framebuffer
    let mut frame_buffers = graphical_engine
        .lock()
        .unwrap()
        .create_frame_buffers(render_pass.clone());

    // Command Buffers
    let mut command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>> = create_command_buffers(
        frame_buffers.clone(),
        graphical_engine.clone(),
        pipeline.clone(),
        vertex_buffer.clone(),
    );

    // Window variables
    let mut window_resized = false;
    let mut recreate_swapchain = false;

    // Hijack thread and open window
    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;

                // Kills the engine (and this main thread) and frees resources.
                // Otherwise, SEGFAULT's will occur on exit.
                graphical_engine.lock().unwrap().kill();
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                window_resized = true;
            }
            Event::RedrawEventsCleared => {
                log::debug!("RedrawEventsCleared");
                log::debug!("Resized: {}", window_resized);
                log::debug!("Recreate: {}", recreate_swapchain);

                if window_resized || recreate_swapchain {
                    recreate_swapchain = false;

                    match graphical_engine
                        .lock()
                        .unwrap()
                        .recreate_swap_chain_and_images(render_pass.clone())
                    {
                        Some(new_frame_buffers) => frame_buffers = new_frame_buffers,
                        None => {
                            // Something did go wrong while recreating the swapchain.
                            // There is no ideal way of handling this, our best bet is that this is a single occurrence.
                            // If it is, we just need to recreate the swapchain again and run the 'resize window' code again.
                            // If not, this error will probably repeat forever and crash the program eventually.

                            log::error!("Failed recreating SwapChain!");
                            window_resized = true;
                            recreate_swapchain = true;
                            return;
                        }
                    };

                    if window_resized {
                        window_resized = false;

                        pipeline = create_pipeline(
                            vertex_shader.clone(),
                            fragment_shader.clone(),
                            surface.window().inner_size(),
                            render_pass.clone(),
                            graphical_engine.lock().unwrap().get_logical_device(),
                        );

                        command_buffers = create_command_buffers(
                            frame_buffers.clone(),
                            graphical_engine.clone(),
                            pipeline.clone(),
                            vertex_buffer.clone(),
                        );
                    }
                }

                let (image_i, suboptimal, acquire_future) = match swapchain::acquire_next_image(
                    graphical_engine.lock().unwrap().get_swap_chain().clone(),
                    None,
                ) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let engine_arc = graphical_engine.lock().unwrap();
                let execution = sync::now(engine_arc.get_logical_device().get_device())
                    // Wait for the image to actually become available
                    .join(acquire_future)
                    // Run `CommandBuffer` for that image
                    .then_execute(
                        engine_arc.get_logical_device().get_first_queue(),
                        command_buffers[image_i].clone(),
                    )
                    .unwrap()
                    // Finish drawing and present the image on the swapchain
                    .then_swapchain_present(
                        engine_arc.get_logical_device().get_first_queue(),
                        PresentInfo {
                            index: image_i,
                            ..PresentInfo::swapchain(engine_arc.get_swap_chain())
                        },
                    )
                    .then_signal_fence_and_flush();

                match execution {
                    Ok(future) => future.wait(None).unwrap(), // Wait for the GPU to finish
                    Err(FlushError::OutOfDate) => {
                        // Something did go wrong, recreate swapchain
                        recreate_swapchain = true;
                    }
                    Err(e) => {
                        // Unknown error
                        log::error!("Failed to flush future: {:?}", e);
                    }
                }
            }
            Event::MainEventsCleared => {}
            _ => (),
        }
    });
}
