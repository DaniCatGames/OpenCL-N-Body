#version 330 core

in vec2 fUv;

out vec4 FragColor; //Out Color

uniform float uTime; // elapsed in sec
uniform vec2 uResolution;

vec3 palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.263, 0.416, 0.557);

    return a + b * cos(6.28318 * (c * t + d));
}


void main()
{
    vec2 uv = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;
    vec2 uv0 = uv;
    vec3 finalColor = vec3(0.0);

    for (float i = 0.0; i < 4.0; i++) {
        uv = fract(uv * 1.5) - 0.5;
        float dis = length(uv) * exp(length(uv0));

        vec3 color = palette(length(uv0) + i * 1.0 + uTime * 0.2);

        dis = sin(dis * 8. + uTime) / 8.0;
        dis = abs(dis);
        dis = pow(0.01 / dis, 2.0);

        finalColor += color * dis;
    }

    FragColor = vec4(finalColor, 1.0);
}