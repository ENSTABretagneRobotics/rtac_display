#ifndef _DEF_RTAC_DISPLAY_MESH_RENDERER_H_
#define _DEF_RTAC_DISPLAY_MESH_RENDERER_H_

#include <rtac_base/types/common.h>
#include <rtac_base/types/Handle.h>
#include <rtac_base/types/Mesh.h>

#include <rtac_display/GLContext.h>
#include <rtac_display/Color.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/views/View3D.h>

namespace rtac { namespace display {

class MeshRenderer : public Renderer
{
    public:

    using Ptr      = rtac::types::Handle<MeshRenderer>;
    using ConstPtr = rtac::types::Handle<const MeshRenderer>;
    
    using Mat4  = View3D::Mat4;
    using Mesh  = rtac::types::Mesh<>;
    using Pose  = View3D::Pose;

    protected:

    static const std::string vertexShader;
    static const std::string fragmentShader;

    size_t       numPoints_;
    GLuint       points_;
    GLuint       normals_;
    Pose         pose_;
    Color::RGBAf color_;

    protected:

    void allocate_points(size_t numPoints);
    void delete_points();

    MeshRenderer(const GLContext::Ptr& context,
                 const View3D::Ptr& view = nullptr,
                 const Color::RGBAf& color = {1.0,1.0,1.0,1.0});
    MeshRenderer(const View3D::Ptr& view,
                 const Color::RGBAf& color = {1.0,1.0,1.0,1.0});

    public:

    static Ptr Create(const GLContext::Ptr& context,
                      const View3D::Ptr& view = nullptr,
                      const Color::RGBAf& color = {1.0,1.0,1.0,1.0});
    static Ptr New(const View3D::Ptr& view,
                   const Color::RGBAf& color = {1.0,1.0,1.0,1.0});

    //void set_mesh(const Mesh& mesrh);
    void set_pose(const Pose& pose);
    void set_color(const Color::RGBAf& color);

    virtual void draw() const;
    virtual void draw(const View::ConstPtr& view) const;
    
    template <typename Tp, typename Tf>
    void set_mesh(const types::Mesh<Tp,Tf>& mesh);
};

template <typename Tp, typename Tf>
void MeshRenderer::set_mesh(const types::Mesh<Tp,Tf>& mesh)
{
    using namespace rtac::types;
    using namespace rtac::types::indexing;

    this->allocate_points(3*mesh.num_faces());
    
    glBindBuffer(GL_ARRAY_BUFFER, points_);
    auto points  = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    glBindBuffer(GL_ARRAY_BUFFER, normals_);
    auto normals = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));

    for(int nf = 0; nf < mesh.num_faces(); nf++) {
        auto f = mesh.face(nf);
        Map<const Vector3<float>> p0(reinterpret_cast<const float*>(&mesh.point(f.x)));
        Map<const Vector3<float>> p1(reinterpret_cast<const float*>(&mesh.point(f.y)));
        Map<const Vector3<float>> p2(reinterpret_cast<const float*>(&mesh.point(f.z)));
        Vector3<float> n = ((p1 - p0).cross(p2 - p0)).normalized();
        
        int i = 9*nf;
        points[i]     = p0(0); points[i + 1] = p0(1); points[i + 2] = p0(2);
        points[i + 3] = p1(0); points[i + 4] = p1(1); points[i + 5] = p1(2);
        points[i + 6] = p2(0); points[i + 7] = p2(1); points[i + 8] = p2(2);
        normals[i]     = n(0); normals[i + 1] = n(1); normals[i + 2] = n(2);
        normals[i + 3] = n(0); normals[i + 4] = n(1); normals[i + 5] = n(2);
        normals[i + 6] = n(0); normals[i + 7] = n(1); normals[i + 8] = n(2);
    }
    
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, points_);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    numPoints_ = 9*mesh.num_faces();

    GL_CHECK_LAST(); 
}

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_MESH_RENDERER_H_
