#ifndef _DEF_RTAC_DISPLAY_POINTCLOUD_RENDERER_H_
#define _DEF_RTAC_DISPLAY_POINTCLOUD_RENDERER_H_

#include <iostream>
#include <array>
#include <algorithm>

#include <rtac_base/types/Handle.h>
#include <rtac_base/types/PointCloud.h>

#include <rtac_display/Color.h>
#include <rtac_display/GLVector.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/views/View3D.h>

namespace rtac { namespace display {

class PointCloudRendererBase : public Renderer
{
    public:

    using Ptr      = rtac::types::Handle<PointCloudRendererBase>;
    using ConstPtr = rtac::types::Handle<const PointCloudRendererBase>;

    using Mat4    = View3D::Mat4;
    using Shape   = View3D::Shape;
    using Pose    = View3D::Pose;

    static const std::string vertexShader;
    static const std::string fragmentShader;

    protected:
    
    Pose  pose_;
    Color::RGBAf color_;

    public:

    PointCloudRendererBase(const View3D::Ptr& view,
                           const Color::RGBAf& color = {0.7,0.7,1.0,1.0});

    void set_pose(const Pose& pose);
    void set_color(const Color::RGBAf& color);
};

template <typename PointT>
class PointCloudRenderer : public PointCloudRendererBase
{
    public:

    static_assert(sizeof(PointT) >= 3*sizeof(float),
                 "PointT must be at least the size of 3 floats");

    using Ptr      = rtac::types::Handle<PointCloudRenderer<PointT>>;
    using ConstPtr = rtac::types::Handle<const PointCloudRenderer<PointT>>;

    using Mat4    = PointCloudRendererBase::Mat4;
    using Shape   = PointCloudRendererBase::Shape;
    using Pose    = PointCloudRendererBase::Pose;

    protected:
    
    GLVector<PointT> points_;

    public:
    
    static Ptr New(const View3D::Ptr& view,
                   const Color::RGBAf& color = {0.7,0.7,1.0,1.0});
    PointCloudRenderer(const View3D::Ptr& view,
                       const Color::RGBAf& color = {0.7,0.7,1.0,1.0});

    GLVector<PointT>& points();
    const GLVector<PointT>& points() const;
    
    void set_points(size_t numPoints, const PointT* data);
    template <typename PointCloudT>
    void set_points(const rtac::types::PointCloud<PointCloudT>& pc);
    template <typename Derived>
    void set_points(const Eigen::DenseBase<Derived>& points);
    
    virtual void draw();
};

// implementation
template <typename PointT>
typename PointCloudRenderer<PointT>::Ptr PointCloudRenderer<PointT>::New(
    const View3D::Ptr& view, const Color::RGBAf& color)
{
    return Ptr(new PointCloudRenderer<PointT>(view, color));
}

template <typename PointT>
PointCloudRenderer<PointT>::PointCloudRenderer(const View3D::Ptr& view,
                                               const Color::RGBAf& color) :
    PointCloudRendererBase(view, color)
{}

template <typename PointT>
GLVector<PointT>& PointCloudRenderer<PointT>::points()
{
    return points_;
}

template <typename PointT>
const GLVector<PointT>& PointCloudRenderer<PointT>::points() const
{
    return points_;
}

template <typename PointT>
void PointCloudRenderer<PointT>::set_points(size_t numPoints, const PointT* data)
{
    points_.set_data(numPoints, data);
}

template <typename PointT> template <typename PointCloudT>
void PointCloudRenderer<PointT>::set_points(const rtac::types::PointCloud<PointCloudT>& pc)
{
    points_.resize(pc.size());
    auto devicePtr = points_.map();
    int i = 0;
    for(auto& point : pc) {
        devicePtr[i].x = point.x;
        devicePtr[i].y = point.y;
        devicePtr[i].z = point.z;
        i++;
    }
}

template <typename PointT> template <typename Derived>
void PointCloudRenderer<PointT>::set_points(const Eigen::DenseBase<Derived>& points)
{
    //expects points on rows.
    if(points.cols() != 3) {
        throw std::runtime_error("PointCloudRenderer.set_points : Wrong matrix shape");
    }
    size_t numPoints = points.rows();

    points_.resize(numPoints);
    auto devicePtr = points_.map();
    for(int i = 0; i < numPoints; i++) {
        devicePtr[i].x = points(i,0);
        devicePtr[i].y = points(i,1);
        devicePtr[i].z = points(i,2);
    }
}


template <typename PointT>
void PointCloudRenderer<PointT>::draw()
{
    if(points_.size() == 0)
        return;
    
    glDisable(GL_DEPTH_TEST);
    Mat4 view = view_->view_matrix() * pose_.homogeneous_matrix();

    GLfloat pointSize;
    glGetFloatv(GL_POINT_SIZE, &pointSize);
    glPointSize(1);

    glUseProgram(renderProgram_);
    
    points_.bind(GL_ARRAY_BUFFER);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PointT), NULL);
    glEnableVertexAttribArray(0);

    //color_[0] = 1.0;
    //color_[1] = 0.0;
    //color_[2] = 0.0;

    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "view"),
        1, GL_FALSE, view.data());
    glUniform4fv(glGetUniformLocation(renderProgram_, "color"),
        1, reinterpret_cast<const float*>(&color_));

    glDrawArrays(GL_POINTS, 0, points_.size());
    
    glDisableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);

    glPointSize(pointSize);
}

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_POINTCLOUD_RENDERER_H_
