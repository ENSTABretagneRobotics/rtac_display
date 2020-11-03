#ifndef _DEF_RTAC_DISPLAY_NORMALS_RENDERER_H_
#define _DEF_RTAC_DISPLAY_NORMALS_RENDERER_H_

#include <iostream>

#include <rtac_base/types/Handle.h>
#include <rtac_base/types/Pose.h>

#include <rtac_display/utils.h>
#include <rtac_display/renderers/Renderer.h>
#include <rtac_display/views/View.h>
#include <rtac_display/views/View3D.h>

namespace rtac { namespace display {

class NormalsRenderer : public Renderer
{
    public:

    using Ptr      = rtac::types::Handle<NormalsRenderer>;
    using ConstPtr = rtac::types::Handle<const NormalsRenderer>;

    using Mat4     = View3D::Mat4;
    using Pose     = View3D::Pose;
    using Color    = std::array<float,3>;

    static const std::string vertexShader;
    static const std::string fragmentShader;
    static const std::string generateLineDataShader;

    protected:
    
    size_t numPoints_;
    GLuint displayData_; // contains points to draw segments
    GLuint generateLineProgram_;
    Pose   pose_;
    Color  color_;

    void allocate_data(size_t numPoints);
    void delete_data();

    public:

    static Ptr New(const View::Ptr& view = View::New(),
                   const Color& = Color({0.3,0.5,1.0}));
    NormalsRenderer(const View::Ptr& view = View::New(),
                    const Color& = Color({0.3,0.5,1.0}));
    ~NormalsRenderer();

    void set_normals(size_t numPoints, GLuint points, GLuint normals,
                     bool normalizeNormals = true);

    void set_pose(const Pose& pose);
    void set_color(const Color& color);

    virtual void draw();
};

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_NORMALS_RENDERER_H_
