#include <optix_helpers/display/MeshRenderer.h>

namespace optix_helpers { namespace display {

const Source MeshRendererObj::vertexShader = Source( R"(
#version 430 core

in vec3 point;
in vec3 normal;

uniform mat4 view;
uniform mat4 projection;
uniform vec3 color;

out vec3 c;

void main()
{
    gl_Position = view*vec4(point, 1.0f);
    c = color * dot(mat3(view)*normal, normalize(vec3(gl_Position)));
    gl_Position = projection * gl_Position;
}
)", "vertex");

const Source MeshRendererObj::fragmentShader = Source(R"(
#version 430 core

in vec3 c;
out vec4 outColor;

void main()
{
    outColor = vec4(c, 1.0f);
}
)", "fragment");

MeshRendererObj::MeshRendererObj(const View3D& view) :
    RendererObj(vertexShader, fragmentShader, view),
    points_(0),
    faces_(0),
    normals_(0)
{
}

void MeshRendererObj::draw()
{
}

}; //namespace display
}; //namespace optix_helpers
