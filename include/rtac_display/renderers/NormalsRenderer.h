#ifndef _DEF_RTAC_DISPLAY_NORMALS_RENDERER_H_
#define _DEF_RTAC_DISPLAY_NORMALS_RENDERER_H_

#include <iostream>

#include <rtac_base/types/Handle.h>
#include <rtac_base/types/Pose.h>
#include <rtac_base/types/Point.h>

#include <rtac_display/utils.h>
#include <rtac_display/Color.h>
#include <rtac_display/GLVector.h>
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

    static const std::string vertexShader;
    static const std::string fragmentShader;
    static const std::string generateLineDataShader;
    static const std::string generateLineDataShader2;

    protected:
    
    size_t        numPoints_;
    GLuint        displayData_; // contains points to draw segments
    GLuint        generateLineProgram_;
    GLuint        generateLineProgram2_;
    Pose          pose_;
    Color::RGBAf  color_;

    void allocate_data(size_t numPoints);
    void delete_data();

    public:

    static Ptr New(const View::Ptr& view = View::New(),
                   const Color::RGBAf& = {0.3,0.5,1.0,1.0});
    NormalsRenderer(const View::Ptr& view = View::New(),
                    const Color::RGBAf& = {0.3,0.5,1.0,1.0});
    ~NormalsRenderer();

    void set_normals(size_t numPoints, GLuint points, GLuint normals,
                     bool normalizeNormals = true);

    void set_normals(size_t numPoints, GLuint input, bool normalizeNormals = true);

    template <typename Derived1, typename Derived2>
    void set_normals(const Eigen::DenseBase<Derived1>& points,
                     const Eigen::DenseBase<Derived2>& normals,
                     bool normalizeNormals = true);
    

    void set_pose(const Pose& pose);
    void set_color(const Color::RGBAf& color);

    virtual void draw();
};

template <typename Derived1, typename Derived2>
void NormalsRenderer::set_normals(const Eigen::DenseBase<Derived1>& points,
                                  const Eigen::DenseBase<Derived2>& normals,
                                  bool normalizeNormals)
{
    if(points.rows() != normals.rows()) {
        std::ostringstream oss;
        oss << "NormalsRenderer::set_points : number of points not "
            << "equal to the number of normals";
        throw std::runtime_error(oss.str());
    }

    if(points.cols() < 3 || normals.cols() < 3) {
        std::ostringstream oss;
        oss << "NormalsRenderer::set_points : invalid data shape.";
        throw std::runtime_error(oss.str());
    }

    GLVector<types::Point3<float>> pdata(points.rows());
    GLVector<types::Point3<float>> ndata(normals.rows());
    {
        auto p = pdata.map();
        auto n = ndata.map();
        for(int i = 0; i < pdata.size(); i++) {
            p[i].x = points(i,0);
            p[i].y = points(i,1);
            p[i].z = points(i,2);
            n[i].x = normals(i,0);
            n[i].y = normals(i,1);
            n[i].z = normals(i,2);
        }
    }
    this->set_normals(points.rows(), pdata.gl_id(), ndata.gl_id(),
                      normalizeNormals);
}

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_NORMALS_RENDERER_H_
