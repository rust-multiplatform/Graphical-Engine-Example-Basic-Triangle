#version 450

layout(location = 0) in vec2 position;

// Is called for each vertex of the triangle and defines the position of the
// shape we want to draw.
void main() { gl_Position = vec4(position, 0.0, 1.0); }
