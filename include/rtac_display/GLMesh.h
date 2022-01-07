#ifndef _DEF_RTAC_DISPLAY_GL_MESH_H_
#define _DEF_RTAC_DISPLAY_GL_MESH_H_

#include <rtac_base/types/Handle.h>
#include <rtac_base/types/Point.h>
#include <rtac_base/types/Mesh.h>

#include <rtac_display/GLFormat.h>
#include <rtac_display/GLVector.h>

namespace rtac { namespace display {

template <typename P = types::Point3<float>,
          typename F = types::Point3<uint32_t>,
          typename U = types::Point2<float>,
          typename N = types::Point3<float>>
class GLMesh
{
    public:

    using Point    = P;
    using Face     = F;
    using UV       = U;
    using Normal   = N;

    using Ptr      = rtac::types::Handle<GLMesh>;
    using ConstPtr = rtac::types::Handle<const GLMesh>;

    using BaseMesh = types::Mesh<Point,Face>;

    protected:
    
    GLVector<Point>  points_;
    GLVector<Face>   faces_;
    GLVector<UV>     uvs_;
    GLVector<Normal> faceNormals_;
    GLVector<Normal> pointNormals_;

    public:

    static Ptr Create() { return Ptr(new GLMesh()); }

    GLMesh() {}
    GLMesh(GLMesh&& other);
    GLMesh& operator=(GLMesh&& other);

    template <template <typename> class VectorT>
    GLMesh(const types::Mesh<P,F,VectorT>& other);
    template <template <typename> class VectorT>
    GLMesh& operator=(const types::Mesh<P,F,VectorT>& other);

    GLVector<Point>&  points()        { return points_; }
    GLVector<Face>&   faces()         { return faces_; }
    GLVector<UV>&     uvs()           { return uvs_; }
    GLVector<Normal>& face_normals()  { return faceNormals_; }
    GLVector<Normal>& point_normals() { return pointNormals_; }

    const GLVector<Point>&  points()        const { return points_; }
    const GLVector<Face>&   faces()         const { return faces_; }
    const GLVector<UV>&     uvs()           const { return uvs_; }
    const GLVector<Normal>& face_normals()  const { return faceNormals_; }
    const GLVector<Normal>& point_normals() const { return pointNormals_; }

    static GLMesh::Ptr cube(float scale = 1.0f) {
        return Ptr(new GLMesh(BaseMesh::cube(scale)));
    }
};

template <typename P, typename F, typename U, typename N>
GLMesh<P,F,U,N>::GLMesh(GLMesh<P,F,U,N>&& other) :
    GLMesh()
{
    *this = std::move(other);
}

template <typename P, typename F, typename U, typename N>
GLMesh<P,F,U,N>& GLMesh<P,F,U,N>::operator=(GLMesh<P,F,U,N>&& other)
{
    points_       = std::move(other.points_);
    faces_        = std::move(other.faces_);
    uvs_          = std::move(other.uvs_);
    faceNormals_  = std::move(other.faceNormals_);
    pointNormals_ = std::move(other.pointNormals_);

    return *this;
}

template <typename P, typename F, typename U, typename N>
template <template <typename> class VectorT>
GLMesh<P,F,U,N>::GLMesh(const types::Mesh<P,F,VectorT>& other) :
    GLMesh()
{
    *this = other;
}

template <typename P, typename F, typename U, typename N>
template <template <typename> class VectorT>
GLMesh<P,F,U,N>& GLMesh<P,F,U,N>::operator=(const types::Mesh<P,F,VectorT>& other)
{
    points_ = other.points();
    faces_  = other.faces();

    uvs_.resize(0);
    faceNormals_.resize(0);
    pointNormals_.resize(0);

    return *this;
}

}; //namespace display
}; //namespace rtac

template <typename P, typename F, typename U, typename N>
inline std::ostream& operator<<(std::ostream& os,
                                const rtac::display::GLMesh<P,F,U,N>& mesh)
{
    os << "GLMesh :"
       << "\n- " << mesh.points().size()       << " points"
       << "\n- " << mesh.faces().size()        << " faces"
       << "\n- " << mesh.uvs().size()          << " uvs"
       << "\n- " << mesh.face_normals().size()  << " faceNormals"
       << "\n- " << mesh.point_normals().size() << " pointNormals";
    return os;
}

#endif //_DEF_RTAC_DISPLAY_GL_MESH_H_
