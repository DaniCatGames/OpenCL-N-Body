#version 330 core

layout (location = 0) in vec3 vPos;
layout (location = 1) in vec2 vUv;

out vec2 fUv;

/*uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;*/

void main()
{
    //gl_Position = uProjection * uView * uModel * vec4(vPos, 1.0);
    gl_Position = vec4(vPos, 1.0);
    fUv = vUv;
}