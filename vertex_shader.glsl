#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 modelViewProjectionMatrix;

void main() {
    gl_Position = modelViewProjectionMatrix * vec4(aPos, 1.0);
}
