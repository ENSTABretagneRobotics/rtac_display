#include <rtac_display/renderers/PointCloudRenderer.h>

namespace rtac { namespace display {

const std::string PointCloudRendererBase::vertexShader = std::string( R"(
#version 430 core

in vec3 point;

uniform mat4 view;
uniform vec3 color;

out vec3 c;

void main()
{
    gl_Position = view*vec4(point, 1.0f);
    c = color;
}
)");

const std::string PointCloudRendererBase::fragmentShader = std::string(R"(
#version 430 core

in vec3 c;
out vec4 outColor;

void main()
{
    outColor = vec4(c, 1.0f);
}
)");

PointCloudRendererBase::PointCloudRendererBase(const View3D::Ptr& view, const Color& color) :
    Renderer(vertexShader, fragmentShader, view),
    pose_(Pose()),
    color_(color)
{
    std::cout << "Request : " << color[0] << ", " << color[1] << ", " << color[2] << std::endl;
    std::cout << "Color : " << color_[0] << ", " << color_[1] << ", " << color_[2] << std::endl;
    this->set_color(color);
    std::cout << "Color : " << color_[0] << ", " << color_[1] << ", " << color_[2] << std::endl;
}

void PointCloudRendererBase::set_pose(const Pose& pose)
{
    pose_ = pose;
}

void PointCloudRendererBase::set_color(const Color& color)
{
    color_[0] = std::max(0.0f, std::min(1.0f, color[0]));
    color_[1] = std::max(0.0f, std::min(1.0f, color[1]));
    color_[2] = std::max(0.0f, std::min(1.0f, color[2]));
}

}; //namespace display
}; //namespace rtac

