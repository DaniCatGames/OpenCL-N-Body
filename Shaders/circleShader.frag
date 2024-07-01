#version 330 core
#define pi 3.1415926538

in vec2 fUv;

out vec4 FragColor; //Out Color

uniform float uTime; // elapsed in sec
uniform vec2 uResolution;

//uniform sampler2D uTexture0;

float checkerAA(vec2 p) {
    vec2 q = sin(pi * p * vec2(20, 10));
    float m = q.x * q.y;

    return 0.5 - m / fwidth(m);
}

void main() {
    vec2 uv = (2.0 * gl_FragCoord.xy - uResolution.xy) / uResolution.x;
    float hfov = 2.3;
    float dist = 5.0;

    vec3 vel = normalize(vec3(1, -uv * tan(hfov / 2.0)));

    vec3 pos = vec3(-dist, 0.0, 0.0);
    float r = length(pos);
    float dtau = 0.2;

    while (r < dist * 2.0 && r > 1.0) {
        float ddtau = dtau * r;
        pos += vel * ddtau;
        r = length(pos);
        vec3 er = pos / r;
        vec3 c = cross(vel, er);
        vel -= ddtau * dot(c, c) * er / r / r;
    }

    float phi1 = 1.0 - atan(vel.y, vel.x) / (2. * pi);
    float theta1 = 1.0 - atan(length(vel.xy), vel.z) / pi;
    vec2 UV = vec2(phi1, theta1) + vec2(uTime * 0.01, 0.0);
    vec3 rgb = vec3(checkerAA(UV * 180.0 / pi / 30.0));
    rgb *= float(r > 1.0);

    FragColor = vec4(rgb, 1.0);
}