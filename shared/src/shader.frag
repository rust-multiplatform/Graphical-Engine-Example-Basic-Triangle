#version 450

layout(location = 0) out vec4 f_color;

// Is called for each pixel inside the shape and asks for a color for this pixel.
void main() { f_color = vec4(1.0, 0.0, 0.0, 1.0); }
