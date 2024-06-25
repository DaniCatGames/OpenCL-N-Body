#version 330 core

in vec2 fUv;

out vec4 FragColor; //Out Color

//uniform float uTime; // elapsed in sec
//uniform vec2 uResolution;
//vec2 uPosition;
//uniform float uRadius;

uniform sampler2D uTexture0;

void main()
{
    FragColor = texture(uTexture0, fUv);
}