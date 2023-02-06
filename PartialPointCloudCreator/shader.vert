#version 400 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
uniform mat4 MVP;
out vec4 out_color;

void main(void){
    gl_Position = MVP * vec4(position, 1);
    out_color = color;
}
