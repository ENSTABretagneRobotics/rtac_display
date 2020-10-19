#ifndef _DEF_OPTIX_HELPERS_DISPLAY_POINTCLOUD_RENDERER_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_POINTCLOUD_RENDERER_H_

#include <iostream>
#include <array>
#include <algorithm>

#include <rtac_base/types/PointCloud.h>

#include <optix_helpers/Handle.h>

#include <optix_helpers/display/RenderBufferGL.h>
#include <optix_helpers/display/Renderer.h>
#include <optix_helpers/display/View3D.h>

namespace optix_helpers { namespace display {


class PointCloudRendererObj : public RendererObj
{
    public:

    using Mat4    = View3DObj::Mat4;
    using Shape   = View3DObj::Shape;
    using Pose    = View3DObj::Pose;
    using Color   = std::array<float,3>;

    static const std::string vertexShader;
    static const std::string fragmentShader;

    protected:
    
    size_t numPoints_;
    GLuint points_;
    Pose   pose_;
    Color  color_;
    
    void allocate_points(size_t numPoints);
    void delete_points();

    public:
    
    PointCloudRendererObj(const View3D& view,
                          const Color& color = {0.7,0.7,1.0});
    ~PointCloudRendererObj();
    
    template <typename PointCloudT>
    void set_points(const rtac::types::PointCloud<PointCloudT>& pc);
    void set_points(size_t numPoints, const float* data);
    void set_points(const RenderBufferGL& buffer);
    template <typename Derived>
    void set_points(const Eigen::DenseBase<Derived>& points);
    void set_pose(const Pose& pose);
    void set_color(const Color& color);

    virtual void draw();
};
using PointCloudRenderer = Handle<PointCloudRendererObj>;

// implementation
template <typename PointCloudT>
void PointCloudRendererObj::set_points(const rtac::types::PointCloud<PointCloudT>& pc)
{
    this->allocate_points(pc.size());
    glBindBuffer(GL_ARRAY_BUFFER, points_);
    auto deviceData = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    int i = 0;
    for(auto& point : pc) {
        deviceData[i]     = point.x;
        deviceData[i + 1] = point.y;
        deviceData[i + 2] = point.z;
        i += 3;
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    numPoints_ = pc.size();
}

template <typename Derived>
void PointCloudRendererObj::set_points(const Eigen::DenseBase<Derived>& points)
{
    //expects points on rows.
    if(points.cols() != 3) {
        throw std::runtime_error("PointCloudRenderer.set_points : Wrong matrix shape");
    }
    size_t numPoints = points.rows();

    this->allocate_points(numPoints);
    glBindBuffer(GL_ARRAY_BUFFER, points_);
    auto deviceData = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    for(int i = 0; i < numPoints; i++) {
        deviceData[3*i]     = points(i,0);
        deviceData[3*i + 1] = points(i,1);
        deviceData[3*i + 2] = points(i,2);
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    numPoints_ = numPoints;
}

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_POINTCLOUD_RENDERER_H_
