#include <rtac_display/renderers/PointCloudRenderer.h>

namespace rtac { namespace display {

const std::string PointCloudRendererBase::vertexShader = std::string( R"(
#version 430 core

in vec3 point;

uniform mat4 view;
uniform vec4 color;

out vec4 c;

void main()
{
    gl_Position = view*vec4(point, 1.0f);
    c = color;
}
)");

const std::string PointCloudRendererBase::fragmentShader = std::string(R"(
#version 430 core

in vec4 c;
out vec4 outColor;

void main()
{
    outColor = c;
}
)");

PointCloudRendererBase::PointCloudRendererBase(const View3D::Ptr& view, const Color::RGBAf& color) :
    Renderer(vertexShader, fragmentShader, view),
    pose_(Pose()),
    color_(color)
{
    std::cout << "Request : " << color << std::endl;
    std::cout << "Color : " << color_ << std::endl;
    this->set_color(color);
    std::cout << "Color : " << color_ << std::endl;
}

void PointCloudRendererBase::set_pose(const Pose& pose)
{
    pose_ = pose;
}

void PointCloudRendererBase::set_color(const Color::RGBAf& color)
{
    color_.r = std::max(0.0f, std::min(1.0f, color.r));
    color_.g = std::max(0.0f, std::min(1.0f, color.g));
    color_.b = std::max(0.0f, std::min(1.0f, color.b));
    color_.a = std::max(0.0f, std::min(1.0f, color.a));
}

}; //namespace display
}; //namespace rtac

