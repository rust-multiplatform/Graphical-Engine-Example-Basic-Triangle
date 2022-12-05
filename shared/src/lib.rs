#![allow(clippy::all)]

use std::sync::{Arc, Mutex};

use graphical_engine::{BaseEngine, GraphicalEngine};
use vulkano_win::VkSurfaceBuild;
use winit::{event_loop::EventLoop, window::WindowBuilder};

#[cfg(test)]
mod tests;

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
}
